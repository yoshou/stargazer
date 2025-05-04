#include "reconstruction_pipeline.hpp"

#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <spdlog/spdlog.h>
#include <sqlite3.h>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <regex>

#include "capture_pipeline.hpp"
#include "correspondance.hpp"
#include "glm_json.hpp"
#include "glm_serialize.hpp"
#include "graph_proc.h"
#include "graph_proc_cv.h"
#include "graph_proc_img.h"
#include "graph_proc_tensor.h"
#include "mvpose.hpp"
#include "parameters.hpp"
#include "point_data.hpp"
#include "sensor.grpc.pb.h"
#include "triangulation.hpp"
#include "utils.hpp"
#include "voxelpose.hpp"
#include "reconstruction.hpp"

using namespace coalsack;

static std::string read_text_file(const std::string &filename) {
  std::ifstream ifs(filename.c_str());
  return std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
}

struct se3 {
  glm::vec3 position;
  glm::quat rotation;

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(position, rotation);
  }
};

struct float2 {
  float x;
  float y;

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(x, y);
  }
};

struct float3 {
  float x;
  float y;
  float z;

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(x, y, z);
  }
};

using float2_list_message = frame_message<std::vector<float2>>;
using float3_list_message = frame_message<std::vector<float3>>;
using mat4_message = frame_message<glm::mat4>;
using se3_list_message = frame_message<std::vector<se3>>;

CEREAL_REGISTER_TYPE(float2_list_message)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::frame_message_base, float2_list_message)

CEREAL_REGISTER_TYPE(float3_list_message)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::frame_message_base, float3_list_message)

CEREAL_REGISTER_TYPE(mat4_message)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::frame_message_base, mat4_message)

CEREAL_REGISTER_TYPE(se3_list_message)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::frame_message_base, se3_list_message)

class SensorServiceImpl final : public stargazer::Sensor::Service {
  std::mutex mtx;
  std::unordered_map<std::string, grpc::ServerWriter<stargazer::SphereMessage> *> writers;
  std::vector<std::function<void(const std::string &, int64_t, const std::vector<se3> &)>>
      se3_received;

 public:
  void notify_sphere(const std::string &name, int64_t timestamp,
                     const std::vector<glm::vec3> &spheres) {
    stargazer::SphereMessage response;
    response.set_name(name);
    response.set_timestamp(timestamp);
    const auto mutable_values = response.mutable_values();
    for (const auto &sphere : spheres) {
      const auto mutable_value = mutable_values->Add();
      mutable_value->mutable_point()->set_x(sphere.x);
      mutable_value->mutable_point()->set_y(sphere.y);
      mutable_value->mutable_point()->set_z(sphere.z);
      mutable_value->set_radius(0.02);
    }
    {
      std::lock_guard<std::mutex> lock(mtx);
      for (const auto &[name, writer] : writers) {
        writer->Write(response);
      }
    }
  }

  void receive_se3(std::function<void(const std::string &, int64_t, const std::vector<se3> &)> f) {
    se3_received.push_back(f);
  }

  grpc::Status SubscribeSphere(grpc::ServerContext *context,
                               const stargazer::SubscribeRequest *request,
                               grpc::ServerWriter<stargazer::SphereMessage> *writer) override {
    {
      std::lock_guard<std::mutex> lock(mtx);
      writers.insert(std::make_pair(request->name(), writer));
    }
    while (!context->IsCancelled()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    {
      std::lock_guard<std::mutex> lock(mtx);
      if (const auto iter = writers.find(request->name()); iter != writers.end()) {
        writers.erase(iter);
      }
    }
    return grpc::Status::OK;
  }

  grpc::Status PublishSE3(grpc::ServerContext *context,
                          grpc::ServerReader<stargazer::SE3Message> *reader,
                          google::protobuf::Empty *response) override {
    stargazer::SE3Message data;
    while (reader->Read(&data)) {
      const auto name = data.name();
      const auto timestamp = data.timestamp();
      const auto &values = data.values();
      std::vector<se3> se3;
      for (const auto &value : values) {
        se3.push_back({{static_cast<float>(value.t().x()), static_cast<float>(value.t().y()),
                        static_cast<float>(value.t().z())},
                       {static_cast<float>(value.q().x()), static_cast<float>(value.q().y()),
                        static_cast<float>(value.q().z()), static_cast<float>(value.q().w())}});
      }
      for (const auto &f : se3_received) {
        f(name, timestamp, se3);
      }
    }
    return grpc::Status::OK;
  }
};

class grpc_server {
  std::string server_address;
  std::atomic_bool running;
  std::shared_ptr<std::thread> server_th;
  std::unique_ptr<grpc::Server> server;
  std::unique_ptr<SensorServiceImpl> service;

 public:
  grpc_server(const std::string &server_address);
  ~grpc_server();

  void run();
  void stop();

  void notify_sphere(const std::string &name, int64_t timestamp,
                     const std::vector<glm::vec3> &spheres);
  void receive_se3(std::function<void(const std::string &, int64_t, const std::vector<se3> &)> f);
};

grpc_server::grpc_server(const std::string &server_address)
    : server_address(server_address),
      running(false),
      server_th(),
      server(),
      service(std::make_unique<SensorServiceImpl>()) {}

grpc_server::~grpc_server() {}

void grpc_server::run() {
  running = true;

  grpc::ServerBuilder builder;

#ifdef USE_SECURE_CREDENTIALS
  std::string ca_crt_content = read_text_file("../data/ca.crt");
  std::string server_crt_content = read_text_file("../data/server.crt");
  std::string server_key_content = read_text_file("../data/server.key");

  grpc::SslServerCredentialsOptions ssl_options;
  grpc::SslServerCredentialsOptions::PemKeyCertPair key_cert = {server_key_content,
                                                                server_crt_content};
  ssl_options.pem_root_certs = ca_crt_content;
  ssl_options.pem_key_cert_pairs.push_back(key_cert);
  builder.AddListeningPort(server_address, grpc::SslServerCredentials(ssl_options));
#else
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
#endif
  builder.RegisterService(service.get());
  server = builder.BuildAndStart();
  spdlog::info("Server listening on " + server_address);
  server_th.reset(new std::thread([this]() { server->Wait(); }));
}

void grpc_server::stop() {
  if (running.load()) {
    running.store(false);
  }
  if (server) {
    server->Shutdown(std::chrono::system_clock::now());
    if (server_th && server_th->joinable()) {
      server_th->join();
    }
  }
}

void grpc_server::notify_sphere(const std::string &name, int64_t timestamp,
                                const std::vector<glm::vec3> &spheres) {
  if (running && service) {
    service->notify_sphere(name, timestamp, spheres);
  }
}

void grpc_server::receive_se3(
    std::function<void(const std::string &, int64_t, const std::vector<se3> &)> f) {
  service->receive_se3(f);
}

class grpc_server_node : public graph_node {
  std::unique_ptr<grpc_server> server;
  graph_edge_ptr output;

  std::string address;

 public:
  grpc_server_node()
      : graph_node(),
        server(),
        output(std::make_shared<graph_edge>(this)),
        address("0.0.0.0:50051") {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "grpc_server_node"; }

  void set_address(const std::string &value) { address = value; }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(address);
  }

  virtual void run() override {
    server.reset(new grpc_server(address));

    server->receive_se3(
        [this](const std::string &name, int64_t timestamp, const std::vector<se3> &se3) {
          auto msg = std::make_shared<frame_message<object_message>>();
          auto obj_msg = object_message();
          auto se3_msg = std::make_shared<se3_list_message>();
          se3_msg->set_data(se3);
          se3_msg->set_timestamp(static_cast<double>(timestamp));
          obj_msg.set_field(name, se3_msg);
          msg->set_data(obj_msg);
          msg->set_timestamp(static_cast<double>(timestamp));
          output->send(msg);
        });

    server->run();
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (const auto msg = std::dynamic_pointer_cast<float3_list_message>(message)) {
      std::vector<glm::vec3> spheres;
      for (const auto &data : msg->get_data()) {
        spheres.push_back(glm::vec3(data.x, data.y, data.z));
      }

      server->notify_sphere(input_name, static_cast<int64_t>(msg->get_timestamp()), spheres);
    }
  }

  virtual void stop() override {
    server->stop();
    server.reset();
  }
};

CEREAL_REGISTER_TYPE(grpc_server_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, grpc_server_node)

CEREAL_REGISTER_TYPE(frame_message<object_message>)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::frame_message_base, frame_message<object_message>)

class callback_node;

class callback_list : public resource_base {
  using callback_func = std::function<void(const callback_node *, std::string, graph_message_ptr)>;
  std::vector<callback_func> callbacks;

 public:
  virtual std::string get_name() const { return "callback_list"; }

  void add(callback_func callback) { callbacks.push_back(callback); }

  void invoke(const callback_node *node, std::string input_name, graph_message_ptr message) const {
    for (auto &callback : callbacks) {
      callback(node, input_name, message);
    }
  }
};

class callback_node : public graph_node {
  std::string name;

 public:
  callback_node() : graph_node() {}

  virtual std::string get_proc_name() const override { return "callback_node"; }

  void set_name(const std::string &value) { name = value; }
  std::string get_name() const { return name; }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(name);
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (const auto resource = resources->get("callback_list")) {
      if (const auto callbacks = std::dynamic_pointer_cast<callback_list>(resource)) {
        callbacks->invoke(this, input_name, message);
      }
    }
  }
};

CEREAL_REGISTER_TYPE(callback_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, callback_node)

class camera_message : public graph_message {
  stargazer::camera_t camera;

 public:
  camera_message() : graph_message(), camera() {}

  camera_message(const stargazer::camera_t &camera) : graph_message(), camera(camera) {}

  stargazer::camera_t get_camera() const { return camera; }

  void set_camera(const stargazer::camera_t &value) { camera = value; }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(camera);
  }
};

CEREAL_REGISTER_TYPE(camera_message)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_message, camera_message)

class epipolar_reconstruct_node : public graph_node {
  mutable std::mutex cameras_mtx;
  std::map<std::string, stargazer::camera_t> cameras;
  glm::mat4 axis;
  graph_edge_ptr output;

 public:
  epipolar_reconstruct_node()
      : graph_node(), cameras(), axis(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "epipolar_reconstruct_node"; }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(cameras, axis);
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (input_name == "cameras") {
      if (auto camera_msg = std::dynamic_pointer_cast<object_message>(message)) {
        for (const auto &[name, field] : camera_msg->get_fields()) {
          if (auto camera_msg = std::dynamic_pointer_cast<camera_message>(field)) {
            std::lock_guard lock(cameras_mtx);
            cameras[name] = camera_msg->get_camera();
          }
        }
      }

      return;
    }
    if (input_name == "axis") {
      if (auto mat4_msg = std::dynamic_pointer_cast<mat4_message>(message)) {
        axis = mat4_msg->get_data();
      }

      return;
    }

    if (auto frame_msg = std::dynamic_pointer_cast<frame_message<object_message>>(message)) {
      const auto obj_msg = frame_msg->get_data();

      std::vector<std::vector<glm::vec2>> camera_pts;
      std::vector<stargazer::camera_t> camera_list;

      std::map<std::string, stargazer::camera_t> cameras;
      {
        std::lock_guard lock(cameras_mtx);
        cameras = this->cameras;
      }

      for (const auto &[name, field] : obj_msg.get_fields()) {
        if (auto points_msg = std::dynamic_pointer_cast<float2_list_message>(field)) {
          if (cameras.find(name) == cameras.end()) {
            continue;
          }
          const auto &camera = cameras.at(name);
          std::vector<glm::vec2> pts;
          for (const auto &pt : points_msg->get_data()) {
            pts.push_back(glm::vec2(pt.x, pt.y));
          }
          camera_pts.push_back(pts);
          camera_list.push_back(camera);
        }
      }

      const auto markers = stargazer::reconstruction::reconstruct(camera_list, camera_pts, axis);

      auto marker_msg = std::make_shared<float3_list_message>();
      std::vector<float3> marker_data;
      for (const auto &marker : markers) {
        marker_data.push_back({marker.x, marker.y, marker.z});
      }
      marker_msg->set_data(marker_data);
      marker_msg->set_frame_number(frame_msg->get_frame_number());
      output->send(marker_msg);
    }
  }
};

CEREAL_REGISTER_TYPE(epipolar_reconstruct_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, epipolar_reconstruct_node)

class frame_number_numbering_node : public graph_node {
  uint64_t frame_number;
  graph_edge_ptr output;

 public:
  frame_number_numbering_node()
      : graph_node(), frame_number(0), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "frame_number_numbering_node"; }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(frame_number);
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (auto msg = std::dynamic_pointer_cast<frame_message_base>(message)) {
      msg->set_frame_number(frame_number++);
      output->send(msg);
    }
  }
};

CEREAL_REGISTER_TYPE(frame_number_numbering_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, frame_number_numbering_node)

template <typename Task>
class task_queue {
  const uint32_t thread_count;
  std::unique_ptr<std::thread[]> threads;
  std::queue<Task> tasks{};
  std::mutex tasks_mutex;
  std::condition_variable condition;
  std::atomic_bool running{true};

  void worker() {
    for (;;) {
      Task task;

      {
        std::unique_lock<std::mutex> lock(tasks_mutex);
        condition.wait(lock, [&] { return !tasks.empty() || !running; });
        if (!running) {
          return;
        }

        task = std::move(tasks.front());
        tasks.pop();
      }

      task();
    }
  }

 public:
  task_queue(const uint32_t thread_count = std::thread::hardware_concurrency())
      : thread_count(thread_count), threads(std::make_unique<std::thread[]>(thread_count)) {
    for (uint32_t i = 0; i < thread_count; ++i) {
      threads[i] = std::thread(&task_queue::worker, this);
    }
  }

  void push_task(const Task &task) {
    {
      const std::lock_guard<std::mutex> lock(tasks_mutex);

      if (!running) {
        throw std::runtime_error("Cannot schedule new task after shutdown.");
      }

      tasks.push(task);
    }

    condition.notify_one();
  }
  ~task_queue() {
    {
      std::lock_guard<std::mutex> lock(tasks_mutex);
      running = false;
    }

    condition.notify_all();

    for (uint32_t i = 0; i < thread_count; ++i) {
      threads[i].join();
    }
  }
  size_t size() const { return tasks.size(); }
};

class parallel_queue_node : public graph_node {
  std::unique_ptr<task_queue<std::function<void()>>> workers;
  graph_edge_ptr output;

  uint32_t num_threads;

 public:
  parallel_queue_node()
      : graph_node(),
        workers(),
        output(std::make_shared<graph_edge>(this)),
        num_threads(std::thread::hardware_concurrency()) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "parallel_queue_node"; }

  void set_num_threads(uint32_t value) { num_threads = value; }
  uint32_t get_num_threads() const { return num_threads; }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(num_threads);
  }

  virtual void run() override { workers.reset(new task_queue<std::function<void()>>(num_threads)); }

  virtual void stop() override { workers.reset(); }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (auto msg = std::dynamic_pointer_cast<frame_message_base>(message)) {
      workers->push_task([this, msg]() { output->send(msg); });
    }
  }
};

CEREAL_REGISTER_TYPE(parallel_queue_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, parallel_queue_node)

class greater_graph_message_ptr {
 public:
  bool operator()(const graph_message_ptr &lhs, const graph_message_ptr &rhs) const {
    return std::dynamic_pointer_cast<frame_message_base>(lhs)->get_frame_number() >
           std::dynamic_pointer_cast<frame_message_base>(rhs)->get_frame_number();
  }
};

class frame_number_ordering_node : public graph_node {
  graph_edge_ptr output;
  std::mutex mtx;

  std::priority_queue<graph_message_ptr, std::deque<graph_message_ptr>, greater_graph_message_ptr>
      messages;

  std::shared_ptr<std::thread> th;
  std::atomic_bool running;
  std::condition_variable cv;
  std::uint32_t max_size;
  std::atomic_ullong frame_number;

 public:
  frame_number_ordering_node()
      : graph_node(), output(std::make_shared<graph_edge>(this)), max_size(100), frame_number(0) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "frame_number_ordering_node"; }

  void set_max_size(std::uint32_t value) { max_size = value; }
  std::uint32_t get_max_size() const { return max_size; }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(max_size, frame_number);
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (!running) {
      return;
    }

    if (input_name == "default") {
      std::lock_guard<std::mutex> lock(mtx);

      if (messages.size() >= max_size) {
        std::cout << "Fifo overflow" << std::endl;
        spdlog::error("Fifo overflow");
      } else {
        messages.push(message);
        cv.notify_one();
      }
    }
  }

  virtual void run() override {
    running = true;
    th.reset(new std::thread([this]() {
      while (running.load()) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] {
          return (!messages.empty() && std::dynamic_pointer_cast<frame_message_base>(messages.top())
                                               ->get_frame_number() == frame_number) ||
                 !running;
        });

        if (!running) {
          break;
        }
        if (!messages.empty() &&
            std::dynamic_pointer_cast<frame_message_base>(messages.top())->get_frame_number() ==
                frame_number) {
          const auto message = messages.top();
          messages.pop();
          output->send(message);

          frame_number++;
        }
      }
    }));
  }

  virtual void stop() override {
    if (running.load()) {
      {
        std::lock_guard<std::mutex> lock(mtx);
        running.store(false);
      }
      cv.notify_one();
      if (th && th->joinable()) {
        th->join();
      }
    }
  }
};

class multiview_point_reconstruction_pipeline {
  graph_proc graph;

  std::atomic_bool running;

  mutable std::mutex markers_mtx;
  std::vector<glm::vec3> markers;
  std::vector<std::function<void(const std::vector<glm::vec3> &)>> markers_received;

  std::shared_ptr<epipolar_reconstruct_node> reconstruct_node;
  std::shared_ptr<graph_node> input_node;

 public:
  void add_markers_received(std::function<void(const std::vector<glm::vec3> &)> f) {
    std::lock_guard lock(markers_mtx);
    markers_received.push_back(f);
  }

  void clear_markers_received() {
    std::lock_guard lock(markers_mtx);
    markers_received.clear();
  }

  multiview_point_reconstruction_pipeline()
      : graph(), running(false), markers(), markers_received(), reconstruct_node(), input_node() {}

  void set_camera(const std::string &name, const stargazer::camera_t &camera) {
    auto camera_msg = std::make_shared<camera_message>(camera);
    camera_msg->set_camera(camera);

    auto obj_msg = std::make_shared<object_message>();
    obj_msg->add_field(name, camera_msg);

    if (reconstruct_node) {
      graph.process(reconstruct_node.get(), "cameras", obj_msg);
    }
  }

  void set_axis(const glm::mat4 &axis) {
    auto mat4_msg = std::make_shared<mat4_message>();
    mat4_msg->set_data(axis);

    if (reconstruct_node) {
      graph.process(reconstruct_node.get(), "axis", mat4_msg);
    }
  }

  using frame_type = std::map<std::string, std::vector<stargazer::point_data>>;

  void push_frame(const frame_type &frame) {
    if (!running) {
      return;
    }

    auto msg = std::make_shared<object_message>();
    for (const auto &[name, field] : frame) {
      auto float2_msg = std::make_shared<float2_list_message>();
      std::vector<float2> float2_data;
      for (const auto &pt : field) {
        float2_data.push_back({pt.point.x, pt.point.y});
      }
      float2_msg->set_data(float2_data);
      msg->add_field(name, float2_msg);
    }

    auto frame_msg = std::make_shared<frame_message<object_message>>();
    frame_msg->set_data(*msg);

    if (input_node) {
      graph.process(input_node.get(), frame_msg);
    }
  }

  void run() {
    std::shared_ptr<subgraph> g(new subgraph());

    std::shared_ptr<frame_number_numbering_node> n4(new frame_number_numbering_node());
    g->add_node(n4);

    input_node = n4;

    std::shared_ptr<parallel_queue_node> n6(new parallel_queue_node());
    n6->set_input(n4->get_output());
    g->add_node(n6);

    std::shared_ptr<epipolar_reconstruct_node> n1(new epipolar_reconstruct_node());
    n1->set_input(n6->get_output());
    g->add_node(n1);

    reconstruct_node = n1;

    std::shared_ptr<frame_number_ordering_node> n5(new frame_number_ordering_node());
    n5->set_input(n1->get_output());
    g->add_node(n5);

    std::shared_ptr<callback_node> n2(new callback_node());
    n2->set_input(n5->get_output());
    g->add_node(n2);

    n2->set_name("markers");

    std::shared_ptr<grpc_server_node> n3(new grpc_server_node());
    n3->set_input(n5->get_output(), "sphere");
    g->add_node(n3);

    const auto callbacks = std::make_shared<callback_list>();

    callbacks->add(
        [this](const callback_node *node, std::string input_name, graph_message_ptr message) {
          if (node->get_name() == "markers") {
            if (const auto markers_msg = std::dynamic_pointer_cast<float3_list_message>(message)) {
              std::vector<glm::vec3> markers;
              for (const auto &marker : markers_msg->get_data()) {
                markers.push_back(glm::vec3(marker.x, marker.y, marker.z));
              }

              {
                std::lock_guard lock(markers_mtx);
                this->markers = markers;
              }

              for (const auto &f : markers_received) {
                f(markers);
              }
            }
          }
        });

    graph.deploy(g);
    graph.get_resources()->add(callbacks);
    graph.run();

    running = true;
  }

  void stop() {
    running.store(false);
    graph.stop();
  }

  std::vector<glm::vec3> get_markers() const {
    std::vector<glm::vec3> result;

    {
      std::lock_guard lock(markers_mtx);
      result = this->markers;
    }

    return result;
  }
};

class multiview_point_reconstruction::impl {
 public:
  std::shared_ptr<multiview_point_reconstruction_pipeline> pipeline;

  impl() : pipeline(std::make_shared<multiview_point_reconstruction_pipeline>()) {}

  ~impl() = default;

  void run() { pipeline->run(); }

  void stop() { pipeline->stop(); }
};

multiview_point_reconstruction::multiview_point_reconstruction() : pimpl(new impl()) {}
multiview_point_reconstruction::~multiview_point_reconstruction() = default;

void multiview_point_reconstruction::push_frame(const frame_type &frame) {
  pimpl->pipeline->push_frame(frame);
}

void multiview_point_reconstruction::run() { pimpl->run(); }

void multiview_point_reconstruction::stop() { pimpl->stop(); }

std::vector<glm::vec3> multiview_point_reconstruction::get_markers() const {
  return pimpl->pipeline->get_markers();
}

void multiview_point_reconstruction::set_camera(const std::string &name,
                                                const stargazer::camera_t &camera) {
  cameras[name] = camera;
  pimpl->pipeline->set_camera(name, camera);
}
void multiview_point_reconstruction::set_axis(const glm::mat4 &axis) {
  this->axis = axis;
  pimpl->pipeline->set_axis(axis);
}

#define PANOPTIC

class image_reconstruct_node : public graph_node {
 public:
  virtual std::map<std::string, cv::Mat> get_features() const = 0;
};

CEREAL_REGISTER_TYPE(image_reconstruct_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, image_reconstruct_node)

class voxelpose_reconstruct_node : public image_reconstruct_node {
  mutable std::mutex cameras_mtx;
  std::map<std::string, stargazer::camera_t> cameras;
  glm::mat4 axis;
  graph_edge_ptr output;

  std::vector<std::string> names;
  coalsack::tensor<float, 4> features;
  mutable std::mutex features_mtx;

  stargazer::voxelpose::voxelpose pose_estimator;

 public:
  voxelpose_reconstruct_node()
      : image_reconstruct_node(), cameras(), axis(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "voxelpose_reconstruct_node"; }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(cameras, axis);
  }

  std::vector<glm::vec3> reconstruct(const std::map<std::string, stargazer::camera_t> &cameras,
                                     const std::map<std::string, cv::Mat> &frame,
                                     const glm::mat4 &axis) {
    using namespace stargazer::voxelpose;

    std::vector<std::string> names;
    std::vector<cv::Mat> images_list;
    std::vector<camera_data> cameras_list;

    if (frame.size() <= 1) {
      return std::vector<glm::vec3>();
    }

    for (const auto &[camera_name, image] : frame) {
      names.push_back(camera_name);
    }

    for (size_t i = 0; i < frame.size(); i++) {
      const auto name = names[i];
      camera_data camera;

      const auto &src_camera = cameras.at(name);

      camera.fx = src_camera.intrin.fx;
      camera.fy = src_camera.intrin.fy;
      camera.cx = src_camera.intrin.cx;
      camera.cy = src_camera.intrin.cy;
      camera.k[0] = src_camera.intrin.coeffs[0];
      camera.k[1] = src_camera.intrin.coeffs[1];
      camera.k[2] = src_camera.intrin.coeffs[4];
      camera.p[0] = src_camera.intrin.coeffs[2];
      camera.p[1] = src_camera.intrin.coeffs[3];

      glm::mat4 gl_to_cv(1.f);
      gl_to_cv[0] = glm::vec4(1.f, 0.f, 0.f, 0.f);
      gl_to_cv[1] = glm::vec4(0.f, -1.f, 0.f, 0.f);
      gl_to_cv[2] = glm::vec4(0.f, 0.f, -1.f, 0.f);

      glm::mat4 m(1.f);
      m[0] = glm::vec4(1.f, 0.f, 0.f, 0.f);
      m[1] = glm::vec4(0.f, 0.f, 1.f, 0.f);
      m[2] = glm::vec4(0.f, -1.f, 0.f, 0.f);

      const auto camera_pose =
          axis * glm::inverse(src_camera.extrin.transform_matrix() * gl_to_cv * m);

      for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
          camera.rotation[i][j] = camera_pose[i][j];
        }
        camera.translation[i] = camera_pose[3][i] * 1000.0;
      }

      cameras_list.push_back(camera);
      images_list.push_back(frame.at(name));
    }

#ifdef PANOPTIC
    std::array<float, 3> grid_center = {0.0f, -500.0f, 800.0f};
#else
    std::array<float, 3> grid_center = {0.0f, 0.0f, 0.0f};
#endif

    pose_estimator.set_grid_center(grid_center);

    const auto points = pose_estimator.inference(images_list, cameras_list);

    coalsack::tensor<float, 4> heatmaps(
        {pose_estimator.get_heatmap_width(), pose_estimator.get_heatmap_height(),
         pose_estimator.get_num_joints(), (uint32_t)images_list.size()});
    pose_estimator.copy_heatmap_to(images_list.size(), heatmaps.get_data());

    {
      std::lock_guard lock(features_mtx);
      this->names = names;
      this->features = std::move(heatmaps);
    }

    return points;
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (input_name == "cameras") {
      if (auto camera_msg = std::dynamic_pointer_cast<object_message>(message)) {
        for (const auto &[name, field] : camera_msg->get_fields()) {
          if (auto camera_msg = std::dynamic_pointer_cast<camera_message>(field)) {
            std::lock_guard lock(cameras_mtx);
            cameras[name] = camera_msg->get_camera();
          }
        }
      }

      return;
    }
    if (input_name == "axis") {
      if (auto mat4_msg = std::dynamic_pointer_cast<mat4_message>(message)) {
        axis = mat4_msg->get_data();
      }

      return;
    }

    if (auto frame_msg = std::dynamic_pointer_cast<frame_message<object_message>>(message)) {
      const auto obj_msg = frame_msg->get_data();

      std::map<std::string, stargazer::camera_t> cameras;
      {
        std::lock_guard lock(cameras_mtx);
        cameras = this->cameras;
      }

      std::map<std::string, cv::Mat> images;

      for (const auto &[name, field] : obj_msg.get_fields()) {
        if (auto img_msg = std::dynamic_pointer_cast<image_message>(field)) {
          if (cameras.find(name) == cameras.end()) {
            continue;
          }
          const auto &image = img_msg->get_image();
          cv::Mat img(image.get_height(), image.get_width(), convert_to_cv_type(image.get_format()),
                      (void *)image.get_data(), image.get_stride());
          images[name] = img;
        }
      }

      const auto markers = reconstruct(cameras, images, axis);

      auto marker_msg = std::make_shared<float3_list_message>();
      std::vector<float3> marker_data;
      for (const auto &marker : markers) {
        marker_data.push_back({marker.x, marker.y, marker.z});
      }
      marker_msg->set_data(marker_data);
      marker_msg->set_frame_number(frame_msg->get_frame_number());

      output->send(marker_msg);
    }
  }

  std::map<std::string, cv::Mat> get_features() const override {
    coalsack::tensor<float, 4> features;
    std::vector<std::string> names;
    {
      std::lock_guard lock(features_mtx);
      features = this->features;
      names = this->names;
    }
    std::map<std::string, cv::Mat> result;
    if (features.get_size() == 0) {
      return result;
    }
    for (size_t i = 0; i < names.size(); i++) {
      const auto name = names[i];
      const auto heatmap =
          features
              .view<3>({features.shape[0], features.shape[1], features.shape[2], 0},
                       {0, 0, 0, static_cast<uint32_t>(i)})
              .contiguous()
              .sum<1>({2});

      cv::Mat heatmap_mat;
      cv::Mat(heatmap.shape[1], heatmap.shape[0], CV_32FC1, (float *)heatmap.get_data())
          .clone()
          .convertTo(heatmap_mat, CV_8U, 255);
      cv::resize(heatmap_mat, heatmap_mat, cv::Size(960, 540));
      cv::cvtColor(heatmap_mat, heatmap_mat, cv::COLOR_GRAY2BGR);

      result[name] = heatmap_mat;
    }
    return result;
  }
};

CEREAL_REGISTER_TYPE(voxelpose_reconstruct_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(image_reconstruct_node, voxelpose_reconstruct_node)

class mvpose_reconstruct_node : public image_reconstruct_node {
  mutable std::mutex cameras_mtx;
  std::map<std::string, stargazer::camera_t> cameras;
  glm::mat4 axis;
  graph_edge_ptr output;

  std::vector<std::string> names;
  coalsack::tensor<float, 4> features;
  mutable std::mutex features_mtx;

  stargazer::mvpose::mvpose pose_estimator;

 public:
  mvpose_reconstruct_node()
      : image_reconstruct_node(), cameras(), axis(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "mvpose_reconstruct_node"; }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(cameras, axis);
  }

  std::vector<glm::vec3> reconstruct(const std::map<std::string, stargazer::camera_t> &cameras,
                                     const std::map<std::string, cv::Mat> &frame,
                                     const glm::mat4 &axis) {
    using namespace stargazer::mvpose;

    std::vector<std::string> names;
    coalsack::tensor<float, 4> heatmaps;
    std::vector<cv::Mat> images_list;
    std::vector<stargazer::camera_t> cameras_list;

    if (frame.size() <= 1) {
      return std::vector<glm::vec3>();
    }

    for (const auto &[camera_name, image] : frame) {
      names.push_back(camera_name);
    }

    for (size_t i = 0; i < frame.size(); i++) {
      const auto name = names[i];

      stargazer::camera_t camera;

      const auto &src_camera = cameras.at(name);

      camera.intrin.fx = src_camera.intrin.fx;
      camera.intrin.fy = src_camera.intrin.fy;
      camera.intrin.cx = src_camera.intrin.cx;
      camera.intrin.cy = src_camera.intrin.cy;
      camera.intrin.coeffs[0] = src_camera.intrin.coeffs[0];
      camera.intrin.coeffs[1] = src_camera.intrin.coeffs[1];
      camera.intrin.coeffs[2] = src_camera.intrin.coeffs[2];
      camera.intrin.coeffs[3] = src_camera.intrin.coeffs[3];
      camera.intrin.coeffs[4] = src_camera.intrin.coeffs[4];

      const auto camera_pose = src_camera.extrin.transform_matrix() * glm::inverse(axis);

      for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
          camera.extrin.rotation[i][j] = camera_pose[i][j];
        }
        camera.extrin.translation[i] = camera_pose[3][i];
      }

      cameras_list.push_back(camera);
      images_list.push_back(frame.at(name));
    }

    const auto points = pose_estimator.inference(images_list, cameras_list);

    {
      std::lock_guard lock(features_mtx);
      this->names = names;
      this->features = std::move(heatmaps);
    }

    return points;
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (input_name == "cameras") {
      if (auto camera_msg = std::dynamic_pointer_cast<object_message>(message)) {
        for (const auto &[name, field] : camera_msg->get_fields()) {
          if (auto camera_msg = std::dynamic_pointer_cast<camera_message>(field)) {
            std::lock_guard lock(cameras_mtx);
            cameras[name] = camera_msg->get_camera();
          }
        }
      }

      return;
    }
    if (input_name == "axis") {
      if (auto mat4_msg = std::dynamic_pointer_cast<mat4_message>(message)) {
        axis = mat4_msg->get_data();
      }

      return;
    }

    if (auto frame_msg = std::dynamic_pointer_cast<frame_message<object_message>>(message)) {
      const auto obj_msg = frame_msg->get_data();

      std::map<std::string, stargazer::camera_t> cameras;
      {
        std::lock_guard lock(cameras_mtx);
        cameras = this->cameras;
      }

      std::map<std::string, cv::Mat> images;

      for (const auto &[name, field] : obj_msg.get_fields()) {
        if (auto img_msg = std::dynamic_pointer_cast<image_message>(field)) {
          if (cameras.find(name) == cameras.end()) {
            continue;
          }
          const auto &image = img_msg->get_image();
          cv::Mat img(image.get_height(), image.get_width(), convert_to_cv_type(image.get_format()),
                      (void *)image.get_data(), image.get_stride());
          images[name] = img;
        }
      }

      const auto markers = reconstruct(cameras, images, axis);

      auto marker_msg = std::make_shared<float3_list_message>();
      std::vector<float3> marker_data;
      for (const auto &marker : markers) {
        marker_data.push_back({marker.x, marker.y, marker.z});
      }
      marker_msg->set_data(marker_data);
      marker_msg->set_frame_number(frame_msg->get_frame_number());

      output->send(marker_msg);
    }
  }

  std::map<std::string, cv::Mat> get_features() const override {
    coalsack::tensor<float, 4> features;
    std::vector<std::string> names;
    {
      std::lock_guard lock(features_mtx);
      features = this->features;
      names = this->names;
    }
    std::map<std::string, cv::Mat> result;
    if (features.get_size() == 0) {
      return result;
    }
    for (size_t i = 0; i < names.size(); i++) {
      const auto name = names[i];
      const auto heatmap =
          features
              .view<3>({features.shape[0], features.shape[1], features.shape[2], 0},
                       {0, 0, 0, static_cast<uint32_t>(i)})
              .contiguous()
              .sum<1>({2});

      cv::Mat heatmap_mat;
      cv::Mat(heatmap.shape[1], heatmap.shape[0], CV_32FC1, (float *)heatmap.get_data())
          .clone()
          .convertTo(heatmap_mat, CV_8U, 255);
      cv::resize(heatmap_mat, heatmap_mat, cv::Size(960, 540));
      cv::cvtColor(heatmap_mat, heatmap_mat, cv::COLOR_GRAY2BGR);

      result[name] = heatmap_mat;
    }
    return result;
  }
};

CEREAL_REGISTER_TYPE(mvpose_reconstruct_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(image_reconstruct_node, mvpose_reconstruct_node)

class dump_se3_node : public graph_node {
  std::string db_path;
  std::string name;

  sqlite3 *db;
  sqlite3_stmt *stmt;
  int topic_id;

  std::deque<std::tuple<double, std::string>> queue;

 public:
  dump_se3_node() : graph_node(), db(nullptr), stmt(nullptr), topic_id(-1) {}

  void set_db_path(std::string value) { db_path = value; }

  void set_name(std::string value) { name = value; }

  virtual std::string get_proc_name() const override { return "dump_se3_node"; }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(db_path);
    archive(name);
  }

  virtual void initialize() override {
    if (sqlite3_open(db_path.c_str(), &db) != SQLITE_OK) {
      throw std::runtime_error("Failed to open database");
    }

    if (sqlite3_exec(db,
                     "CREATE TABLE IF NOT EXISTS topics(id INTEGER PRIMARY KEY, name TEXT NOT "
                     "NULL, type TEXT NOT NULL)",
                     nullptr, nullptr, nullptr) != SQLITE_OK) {
      throw std::runtime_error("Failed to create table");
    }

    if (sqlite3_exec(db,
                     "CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, topic_id INTEGER "
                     "NOT NULL, timestamp INTEGER NOT NULL, data BLOB NOT NULL)",
                     nullptr, nullptr, nullptr) != SQLITE_OK) {
      throw std::runtime_error("Failed to create table");
    }

    if (sqlite3_exec(db, "CREATE INDEX IF NOT EXISTS timestamp_idx ON messages (timestamp ASC)",
                     nullptr, nullptr, nullptr) != SQLITE_OK) {
      throw std::runtime_error("Failed to create index");
    }

    {
      sqlite3_stmt *stmt;
      if (sqlite3_prepare_v2(db, "SELECT id FROM topics WHERE name = ?", -1, &stmt, nullptr) !=
          SQLITE_OK) {
        throw std::runtime_error("Failed to prepare statement");
      }

      if (sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_STATIC) != SQLITE_OK) {
        throw std::runtime_error("Failed to bind text");
      }

      if (sqlite3_step(stmt) == SQLITE_ROW) {
        topic_id = sqlite3_column_int(stmt, 0);
      } else {
        topic_id = -1;
      }

      sqlite3_finalize(stmt);
    }

    if (topic_id == -1) {
      sqlite3_stmt *stmt;
      if (sqlite3_prepare_v2(db, "INSERT INTO topics (name, type) VALUES (?, ?)", -1, &stmt,
                             nullptr) != SQLITE_OK) {
        throw std::runtime_error("Failed to prepare statement");
      }

      if (sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_STATIC) != SQLITE_OK) {
        throw std::runtime_error("Failed to bind text");
      }

      if (sqlite3_bind_text(stmt, 2, "se3", -1, SQLITE_STATIC) != SQLITE_OK) {
        throw std::runtime_error("Failed to bind text");
      }

      auto result = sqlite3_step(stmt);

      while (result == SQLITE_BUSY) {
        result = sqlite3_step(stmt);
      }

      if (result != SQLITE_DONE) {
        throw std::runtime_error("Failed to step");
      }

      sqlite3_finalize(stmt);

      if (sqlite3_prepare_v2(db, "SELECT last_insert_rowid()", -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error("Failed to prepare statement");
      }

      if (sqlite3_step(stmt) == SQLITE_ROW) {
        topic_id = sqlite3_column_int(stmt, 0);
      } else {
        throw std::runtime_error("Failed to get last insert id");
      }

      sqlite3_finalize(stmt);
    }

    if (sqlite3_prepare_v2(db, "INSERT INTO messages (topic_id, timestamp, data) VALUES (?, ?, ?)",
                           -1, &stmt, nullptr) != SQLITE_OK) {
      throw std::runtime_error("Failed to prepare statement");
    }
  }

  virtual void finalize() override {
    flush_queue();

    if (stmt) {
      sqlite3_finalize(stmt);
      stmt = nullptr;
    }
    if (db) {
      sqlite3_close(db);
      db = nullptr;
    }
  }

  void flush_queue() {
    if (sqlite3_exec(db, "BEGIN TRANSACTION", nullptr, nullptr, nullptr) != SQLITE_OK) {
      throw std::runtime_error("Failed to begin transaction");
    }

    while (queue.size() > 0) {
      const auto &[timestamp, str] = queue.front();

      if (sqlite3_reset(stmt) != SQLITE_OK) {
        throw std::runtime_error("Failed to reset");
      }

      if (sqlite3_bind_int64(stmt, 1, topic_id) != SQLITE_OK) {
        throw std::runtime_error("Failed to bind int64");
      }

      if (sqlite3_bind_int64(stmt, 2, timestamp) != SQLITE_OK) {
        throw std::runtime_error("Failed to bind int64");
      }

      if (sqlite3_bind_blob(stmt, 3, str.c_str(), str.size(), SQLITE_STATIC) != SQLITE_OK) {
        throw std::runtime_error("Failed to bind text");
      }

      auto result = sqlite3_step(stmt);

      while (result == SQLITE_BUSY) {
        result = sqlite3_step(stmt);
      }

      if (result != SQLITE_DONE) {
        throw std::runtime_error("Failed to step");
      }

      queue.pop_front();
    }

    auto result = sqlite3_exec(db, "END TRANSACTION", nullptr, nullptr, nullptr);

    while (result == SQLITE_BUSY) {
      result = sqlite3_exec(db, "END TRANSACTION", nullptr, nullptr, nullptr);
    }

    if (result != SQLITE_OK) {
      throw std::runtime_error("Failed to end transaction");
    }
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (auto frame_msg = std::dynamic_pointer_cast<se3_list_message>(message)) {
      const auto &data = frame_msg->get_data();

      nlohmann::json j_frame;
      std::vector<nlohmann::json> j_se3_list;
      for (const auto &se3 : data) {
        nlohmann::json j_se3;
        j_se3["position"]["x"] = se3.position.x;
        j_se3["position"]["y"] = se3.position.y;
        j_se3["position"]["z"] = se3.position.z;
        j_se3["rotation"]["x"] = se3.rotation.x;
        j_se3["rotation"]["y"] = se3.rotation.y;
        j_se3["rotation"]["z"] = se3.rotation.z;
        j_se3["rotation"]["w"] = se3.rotation.w;
        j_se3_list.push_back(j_se3);
      }

      j_frame["se3_list"] = j_se3_list;
      j_frame["timestamp"] = frame_msg->get_timestamp();
      j_frame["frame_number"] = frame_msg->get_frame_number();

      const auto j_str = j_frame.dump(2);

      queue.emplace_back(frame_msg->get_timestamp(), j_str);

      if (queue.size() >= 200) {
        flush_queue();
      }
    }
  }
};

CEREAL_REGISTER_TYPE(dump_se3_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, dump_se3_node)

class multiview_image_reconstruction_pipeline {
  graph_proc graph;

  std::atomic_bool running;

  mutable std::mutex markers_mtx;
  std::vector<glm::vec3> markers;
  std::vector<std::function<void(const std::vector<glm::vec3> &)>> markers_received;

  std::shared_ptr<image_reconstruct_node> reconstruct_node;
  std::shared_ptr<graph_node> input_node;

 public:
  void add_markers_received(std::function<void(const std::vector<glm::vec3> &)> f) {
    std::lock_guard lock(markers_mtx);
    markers_received.push_back(f);
  }

  void clear_markers_received() {
    std::lock_guard lock(markers_mtx);
    markers_received.clear();
  }

  multiview_image_reconstruction_pipeline()
      : graph(), running(false), markers(), markers_received(), reconstruct_node(), input_node() {}

  void set_camera(const std::string &name, const stargazer::camera_t &camera) {
    auto camera_msg = std::make_shared<camera_message>(camera);
    camera_msg->set_camera(camera);

    auto obj_msg = std::make_shared<object_message>();
    obj_msg->add_field(name, camera_msg);

    if (reconstruct_node) {
      graph.process(reconstruct_node.get(), "cameras", obj_msg);
    }
  }

  void set_axis(const glm::mat4 &axis) {
    auto mat4_msg = std::make_shared<mat4_message>();
    mat4_msg->set_data(axis);

    if (reconstruct_node) {
      graph.process(reconstruct_node.get(), "axis", mat4_msg);
    }
  }

  using frame_type = std::map<std::string, cv::Mat>;

  static image_format convert_to_image_format(int type) {
    switch (type) {
      case CV_8UC1:
        return image_format::Y8_UINT;
      case CV_8UC3:
        return image_format::B8G8R8_UINT;
      case CV_8UC4:
        return image_format::B8G8R8A8_UINT;
      default:
        throw std::runtime_error("Invalid image format");
    }
  }

  void push_frame(const frame_type &frame) {
    if (!running) {
      return;
    }

    auto msg = std::make_shared<object_message>();
    for (const auto &[name, field] : frame) {
      auto img_msg = std::make_shared<image_message>();

      image img(static_cast<std::uint32_t>(field.size().width),
                static_cast<std::uint32_t>(field.size().height),
                static_cast<std::uint32_t>(field.elemSize()),
                static_cast<std::uint32_t>(field.step), (const uint8_t *)field.data);
      img.set_format(convert_to_image_format(field.type()));

      img_msg->set_image(std::move(img));
      msg->add_field(name, img_msg);
    }

    auto frame_msg = std::make_shared<frame_message<object_message>>();
    frame_msg->set_data(*msg);

    if (input_node) {
      graph.process(input_node.get(), frame_msg);
    }
  }

  void run(const std::vector<node_info> &infos) {
    std::shared_ptr<subgraph> g(new subgraph());

    std::shared_ptr<frame_number_numbering_node> n4(new frame_number_numbering_node());
    g->add_node(n4);

    input_node = n4;

    std::shared_ptr<parallel_queue_node> n6(new parallel_queue_node());
    n6->set_input(n4->get_output());
    n6->set_num_threads(1);
    g->add_node(n6);

    for (const auto &info : infos) {
      if (info.type == node_type::voxelpose_reconstruction) {
        std::shared_ptr<voxelpose_reconstruct_node> n1(new voxelpose_reconstruct_node());
        n1->set_input(n6->get_output());
        g->add_node(n1);

        reconstruct_node = n1;
      }
      if (info.type == node_type::mvpose_reconstruction) {
        std::shared_ptr<mvpose_reconstruct_node> n1(new mvpose_reconstruct_node());
        n1->set_input(n6->get_output());
        g->add_node(n1);

        reconstruct_node = n1;
      }
    }

    std::shared_ptr<frame_number_ordering_node> n5(new frame_number_ordering_node());
    n5->set_input(reconstruct_node->get_output());
    g->add_node(n5);

    std::shared_ptr<callback_node> n2(new callback_node());
    n2->set_input(n5->get_output());
    g->add_node(n2);

    n2->set_name("markers");

    std::shared_ptr<grpc_server_node> n3(new grpc_server_node());
    n3->set_input(n5->get_output(), "sphere");
    n3->set_address("0.0.0.0:50052");
    g->add_node(n3);

    const std::vector<std::string> device_names = {
    };

    std::shared_ptr<grpc_server_node> n7(new grpc_server_node());
    n7->set_address("0.0.0.0:50053");
    g->add_node(n7);

    std::shared_ptr<frame_demux_node> n8(new frame_demux_node());
    n8->set_input(n7->get_output());

    for (const auto &device_name : device_names) {
      n8->add_output(device_name);
    }

    g->add_node(n8);

    for (auto &device_name : device_names) {
      std::shared_ptr<dump_se3_node> n9(new dump_se3_node());
      n9->set_db_path("../data/data_20250115_1/imu.db");
      n9->set_name(device_name);
      n9->set_input(n8->get_output(device_name), device_name);
      g->add_node(n9);
    }

    const auto callbacks = std::make_shared<callback_list>();

    callbacks->add(
        [this](const callback_node *node, std::string input_name, graph_message_ptr message) {
          if (node->get_name() == "markers") {
            if (const auto markers_msg = std::dynamic_pointer_cast<float3_list_message>(message)) {
              std::vector<glm::vec3> markers;
              for (const auto &marker : markers_msg->get_data()) {
                markers.push_back(glm::vec3(marker.x, marker.y, marker.z));
              }

              {
                std::lock_guard lock(markers_mtx);
                this->markers = markers;
              }

              for (const auto &f : markers_received) {
                f(markers);
              }
            }
          }
        });

    graph.deploy(g);
    graph.get_resources()->add(callbacks);
    graph.run();

    running = true;
  }

  void stop() {
    running.store(false);
    graph.stop();
  }

  std::vector<glm::vec3> get_markers() const {
    std::vector<glm::vec3> result;

    {
      std::lock_guard lock(markers_mtx);
      result = this->markers;
    }

    return result;
  }

  std::map<std::string, cv::Mat> get_features() const { return reconstruct_node->get_features(); }
};

class multiview_image_reconstruction::impl {
 public:
  std::unique_ptr<multiview_image_reconstruction_pipeline> pipeline;

  impl() : pipeline(std::make_unique<multiview_image_reconstruction_pipeline>()) {}

  ~impl() = default;

  void run(const std::vector<node_info> &infos) { pipeline->run(infos); }

  void stop() { pipeline->stop(); }
};

multiview_image_reconstruction::multiview_image_reconstruction() : pimpl(new impl()) {}
multiview_image_reconstruction::~multiview_image_reconstruction() = default;

void multiview_image_reconstruction::push_frame(const frame_type &frame) {
  pimpl->pipeline->push_frame(frame);
}

void multiview_image_reconstruction::run(const std::vector<node_info> &infos) { pimpl->run(infos); }

void multiview_image_reconstruction::stop() { pimpl->stop(); }

std::map<std::string, cv::Mat> multiview_image_reconstruction::get_features() const {
  return pimpl->pipeline->get_features();
}

std::vector<glm::vec3> multiview_image_reconstruction::get_markers() const {
  return pimpl->pipeline->get_markers();
}

void multiview_image_reconstruction::set_camera(const std::string &name,
                                                const stargazer::camera_t &camera) {
  pimpl->pipeline->set_camera(name, camera);
}

void multiview_image_reconstruction::set_axis(const glm::mat4 &axis) {
  pimpl->pipeline->set_axis(axis);
}