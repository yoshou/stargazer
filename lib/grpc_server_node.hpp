#pragma once

#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <spdlog/spdlog.h>

#include <atomic>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "graph_proc.h"
#include "messages.hpp"
#include "sensor.grpc.pb.h"

namespace stargazer {

using namespace coalsack;

static std::string read_text_file(const std::string& filename) {
  std::ifstream ifs(filename.c_str());
  return std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
}

class SensorServiceImpl final : public Sensor::Service {
  std::mutex mtx;
  std::unordered_map<std::string, grpc::ServerWriter<SphereMessage>*> writers;
  std::vector<std::function<void(const std::string&, int64_t, const std::vector<se3>&)>>
      se3_received;
  std::vector<std::function<void(const std::string&, int64_t, const std::vector<camera_image>&)>>
      camera_image_received;
  std::vector<std::function<void(const std::string&, int64_t, const std::vector<inertial>&)>>
      inertial_received;

 public:
  void notify_sphere(const std::string& name, int64_t timestamp,
                     const std::vector<glm::vec3>& spheres) {
    SphereMessage response;
    response.set_name(name);
    response.set_timestamp(timestamp);
    const auto mutable_values = response.mutable_values();
    for (const auto& sphere : spheres) {
      const auto mutable_value = mutable_values->Add();
      mutable_value->mutable_point()->set_x(sphere.x);
      mutable_value->mutable_point()->set_y(sphere.y);
      mutable_value->mutable_point()->set_z(sphere.z);
      mutable_value->set_radius(0.02);
    }
    {
      std::lock_guard<std::mutex> lock(mtx);
      for (const auto& [name, writer] : writers) {
        writer->Write(response);
      }
    }
  }

  void receive_se3(std::function<void(const std::string&, int64_t, const std::vector<se3>&)> f) {
    se3_received.push_back(f);
  }

  void receive_camera_image(
      std::function<void(const std::string&, int64_t, const std::vector<camera_image>&)> f) {
    camera_image_received.push_back(f);
  }

  void receive_inertial(
      std::function<void(const std::string&, int64_t, const std::vector<inertial>&)> f) {
    inertial_received.push_back(f);
  }

  grpc::Status SubscribeSphere(grpc::ServerContext* context, const SubscribeRequest* request,
                               grpc::ServerWriter<SphereMessage>* writer) override {
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

  grpc::Status PublishSE3(grpc::ServerContext* context, grpc::ServerReader<SE3Message>* reader,
                          google::protobuf::Empty* response) override {
    SE3Message data;
    while (reader->Read(&data)) {
      const auto name = data.name();
      const auto timestamp = data.timestamp();
      const auto& values = data.values();
      std::vector<se3> se3;
      for (const auto& value : values) {
        se3.push_back({{static_cast<float>(value.t().x()), static_cast<float>(value.t().y()),
                        static_cast<float>(value.t().z())},
                       {static_cast<float>(value.q().x()), static_cast<float>(value.q().y()),
                        static_cast<float>(value.q().z()), static_cast<float>(value.q().w())}});
      }
      for (const auto& f : se3_received) {
        f(name, timestamp, se3);
      }
    }
    return grpc::Status::OK;
  }

  grpc::Status PublishCameraImage(grpc::ServerContext* context,
                                  grpc::ServerReader<CameraImageMessage>* reader,
                                  google::protobuf::Empty* response) override {
    CameraImageMessage data;
    while (reader->Read(&data)) {
      const auto name = data.name();
      const auto timestamp = data.timestamp();
      const auto& values = data.values();
      std::vector<camera_image> images;
      for (const auto& value : values) {
        camera_image img;
        img.image_data.assign(value.image_data().begin(), value.image_data().end());
        img.image_size = glm::ivec2(value.image_size().x(), value.image_size().y());
        img.format = static_cast<image_data_format>(value.format());
        img.intrinsics.focal_length =
            glm::vec2(value.intrinsics().focal_length().x(), value.intrinsics().focal_length().y());
        img.intrinsics.principal_point = glm::vec2(value.intrinsics().principal_point().x(),
                                                   value.intrinsics().principal_point().y());
        img.intrinsics.image_size =
            glm::ivec2(value.intrinsics().image_size().x(), value.intrinsics().image_size().y());
        img.intrinsics.distortion.k1 = value.intrinsics().distortion().k1();
        img.intrinsics.distortion.k2 = value.intrinsics().distortion().k2();
        img.intrinsics.distortion.p1 = value.intrinsics().distortion().p1();
        img.intrinsics.distortion.p2 = value.intrinsics().distortion().p2();
        img.intrinsics.distortion.k3 = value.intrinsics().distortion().k3();
        images.push_back(img);
      }
      for (const auto& f : camera_image_received) {
        f(name, timestamp, images);
      }
    }
    return grpc::Status::OK;
  }

  grpc::Status PublishInertial(grpc::ServerContext* context,
                               grpc::ServerReader<InertialMessage>* reader,
                               google::protobuf::Empty* response) override {
    InertialMessage data;
    while (reader->Read(&data)) {
      const auto name = data.name();
      const auto timestamp = data.timestamp();
      const auto& values = data.values();
      std::vector<inertial> samples;
      for (const auto& value : values) {
        inertial sample;
        sample.acceleration = glm::vec3(static_cast<float>(value.acceleration().x()),
                                       static_cast<float>(value.acceleration().y()),
                                       static_cast<float>(value.acceleration().z()));
        sample.gyroscope = glm::vec3(static_cast<float>(value.gyroscope().x()),
                                    static_cast<float>(value.gyroscope().y()),
                                    static_cast<float>(value.gyroscope().z()));
        sample.magnetometer = glm::vec3(static_cast<float>(value.magnetometer().x()),
                                       static_cast<float>(value.magnetometer().y()),
                                       static_cast<float>(value.magnetometer().z()));
        sample.gravity = glm::vec3(static_cast<float>(value.gravity().x()),
                                  static_cast<float>(value.gravity().y()),
                                  static_cast<float>(value.gravity().z()));
        samples.push_back(sample);
      }
      for (const auto& f : inertial_received) {
        f(name, timestamp, samples);
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
  grpc_server(const std::string& server_address)
      : server_address(server_address),
        running(false),
        server_th(),
        server(),
        service(std::make_unique<SensorServiceImpl>()) {}

  ~grpc_server() {}

  void run() {
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

  void stop() {
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

  void notify_sphere(const std::string& name, int64_t timestamp,
                     const std::vector<glm::vec3>& spheres) {
    if (running && service) {
      service->notify_sphere(name, timestamp, spheres);
    }
  }

  void receive_se3(std::function<void(const std::string&, int64_t, const std::vector<se3>&)> f) {
    service->receive_se3(f);
  }

  void receive_camera_image(
      std::function<void(const std::string&, int64_t, const std::vector<camera_image>&)> f) {
    service->receive_camera_image(f);
  }

  void receive_inertial(
      std::function<void(const std::string&, int64_t, const std::vector<inertial>&)> f) {
    service->receive_inertial(f);
  }
};

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

  virtual std::string get_proc_name() const override { return "grpc_server"; }

  void set_address(const std::string& value) { address = value; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(address);
  }

  virtual void run() override {
    server.reset(new grpc_server(address));

    server->receive_se3(
        [this](const std::string& name, int64_t timestamp, const std::vector<se3>& se3) {
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

    server->receive_camera_image(
        [this](const std::string& name, int64_t timestamp,
               const std::vector<camera_image>& images) {
          auto msg = std::make_shared<camera_image_list_message>();
          msg->set_data(images);
          msg->set_timestamp(static_cast<double>(timestamp));
          output->send(msg);
          spdlog::info("Received camera images: name={}, timestamp={}, count={}", name, timestamp,
                       images.size());
        });

    server->receive_inertial(
        [this](const std::string& name, int64_t timestamp, const std::vector<inertial>& samples) {
          auto msg = std::make_shared<inertial_list_message>();
          msg->set_data(samples);
          msg->set_timestamp(static_cast<double>(timestamp));
          output->send(msg);
          spdlog::info("Received inertial data: name={}, timestamp={}, samples={}", name,
                       timestamp, samples.size());
        });

    server->run();
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (const auto msg = std::dynamic_pointer_cast<float3_list_message>(message)) {
      std::vector<glm::vec3> spheres;
      for (const auto& data : msg->get_data()) {
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

}  // namespace stargazer

COALSACK_REGISTER_NODE(stargazer::grpc_server_node, coalsack::graph_node)
