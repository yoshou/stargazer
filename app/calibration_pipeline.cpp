#include "calibration_pipeline.hpp"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <spdlog/spdlog.h>

#include <random>
#include <unordered_map>
#include <unordered_set>

#include "calibration.hpp"
#include "calibration_target.hpp"
#include "callback_node.hpp"
#include "glm_serialize.hpp"
#include "graph_proc.h"
#include "graph_proc_cv.h"
#include "graph_proc_img.h"
#include "messages.hpp"
#include "parameters.hpp"
#include "pattern_board_calibration_target_detector_node.hpp"
#include "reconstruction.hpp"
#include "triangulation.hpp"
#include "utils.hpp"

using namespace stargazer;
using namespace stargazer::calibration;
using namespace stargazer::reconstruction;
using namespace coalsack;

class three_point_bar_calibration_target_detector_node : public graph_node {
  std::unique_ptr<three_point_bar_calibration_target> detector;
  graph_edge_ptr output;

 public:
  three_point_bar_calibration_target_detector_node()
      : graph_node(), detector(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override {
    return "three_point_bar_calibration_target_detector";
  }

  template <typename Archive>
  void serialize(Archive& archive) {}

  virtual void run() override { detector = std::make_unique<three_point_bar_calibration_target>(); }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (const auto frame_msg = std::dynamic_pointer_cast<float2_list_message>(message)) {
      if (detector) {
        std::vector<point_data> markers;
        for (const auto& pt : frame_msg->get_data()) {
          markers.push_back({{pt.x, pt.y}, 0.0, 0.0});
        }

        const auto points = detector->detect_points(markers);

        std::vector<float2> float2_data;
        for (const auto& pt : points) {
          float2_data.push_back({pt.x, pt.y});
        }

        auto msg = std::make_shared<float2_list_message>();
        msg->set_data(float2_data);
        msg->set_frame_number(
            std::dynamic_pointer_cast<frame_message_base>(message)->get_frame_number());
        msg->set_timestamp(std::dynamic_pointer_cast<frame_message_base>(message)->get_timestamp());

        output->send(msg);
      }
    }
  }
};

COALSACK_REGISTER_NODE(three_point_bar_calibration_target_detector_node, graph_node)

class calibration_node : public graph_node {
  bool only_extrinsic;
  bool robust;

  observed_points_frames observed_frames;

  mutable std::mutex cameras_mtx;
  std::vector<std::string> camera_names;
  std::unordered_map<std::string, camera_t> cameras;
  std::unordered_map<std::string, camera_t> calibrated_cameras;

  graph_edge_ptr output;

 public:
  calibration_node()
      : graph_node(),
        only_extrinsic(true),
        robust(false),
        output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "calibration"; }

  void set_cameras(const std::unordered_map<std::string, camera_t>& cameras) {
    this->cameras = cameras;
  }

  void set_only_extrinsic(bool only_extrinsic) { this->only_extrinsic = only_extrinsic; }

  void set_robust(bool robust) { this->robust = robust; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cameras, only_extrinsic, robust);
  }

  size_t get_num_frames(std::string name) const { return observed_frames.get_num_points(name); }

  const std::vector<observed_points_t> get_observed_points(std::string name) const {
    return observed_frames.get_observed_points(name);
  }

  void calibrate() {
    if (!initialize_cameras(cameras, camera_names, observed_frames)) {
      spdlog::error("Failed to initialize cameras");
      return;
    }

    bundle_adjust_data ba_data;

    prepare_bundle_adjustment(camera_names, cameras, observed_frames, ba_data);

    bundle_adjustment(ba_data, only_extrinsic, robust);

    {
      calibrated_cameras = cameras;

      {
        for (size_t i = 0; i < camera_names.size(); i++) {
          calibrated_cameras[camera_names[i]].extrin.rotation =
              glm::mat3(ba_data.get_camera_extrinsic(i));
          calibrated_cameras[camera_names[i]].extrin.translation =
              glm::vec3(ba_data.get_camera_extrinsic(i)[3]);
        }
      }

      if (!only_extrinsic) {
        for (size_t i = 0; i < camera_names.size(); i++) {
          const auto intrin = &ba_data.mutable_camera(i)[6];
          calibrated_cameras[camera_names[i]].intrin.fx = intrin[0];
          calibrated_cameras[camera_names[i]].intrin.fy = intrin[1];
          calibrated_cameras[camera_names[i]].intrin.cx = intrin[2];
          calibrated_cameras[camera_names[i]].intrin.cy = intrin[3];
          calibrated_cameras[camera_names[i]].intrin.coeffs[0] = intrin[4];
          calibrated_cameras[camera_names[i]].intrin.coeffs[1] = intrin[5];
          calibrated_cameras[camera_names[i]].intrin.coeffs[4] = intrin[6];
          calibrated_cameras[camera_names[i]].intrin.coeffs[2] = intrin[7];
          calibrated_cameras[camera_names[i]].intrin.coeffs[3] = intrin[8];
        }
      }
    }
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (input_name == "calibrate") {
      camera_names.clear();

      if (auto camera_msg = std::dynamic_pointer_cast<object_message>(message)) {
        for (const auto& [name, field] : camera_msg->get_fields()) {
          if (auto camera_msg = std::dynamic_pointer_cast<camera_message>(field)) {
            std::lock_guard lock(cameras_mtx);
            cameras[name] = camera_msg->get_camera();
            camera_names.push_back(name);
          }
        }
      }

      calibrate();

      std::shared_ptr<object_message> msg(new object_message());
      for (const auto& [name, camera] : calibrated_cameras) {
        std::shared_ptr<camera_message> camera_msg(new camera_message(camera));
        msg->add_field(name, camera_msg);
      }
      output->send(msg);

      return;
    }

    if (auto points_msg = std::dynamic_pointer_cast<float2_list_message>(message)) {
      std::vector<glm::vec2> points;
      for (const auto& pt : points_msg->get_data()) {
        points.emplace_back(pt.x, pt.y);
      }
      observed_frames.add_frame_points(points_msg->get_frame_number(), input_name, points);
    }
  }
};

COALSACK_REGISTER_NODE(calibration_node, graph_node)

class object_map_node : public graph_node {
 public:
  object_map_node() : graph_node() {}

  virtual std::string get_proc_name() const override { return "object_map"; }

  template <typename Archive>
  void save(Archive& archive) const {
    std::vector<std::string> output_names;
    auto outputs = get_outputs();
    for (auto output : outputs) {
      output_names.push_back(output.first);
    }
    archive(output_names);
  }

  template <typename Archive>
  void load(Archive& archive) {
    std::vector<std::string> output_names;
    archive(output_names);
    for (auto output_name : output_names) {
      set_output(std::make_shared<graph_edge>(this), output_name);
    }
  }

  graph_edge_ptr add_output(const std::string& name) {
    auto outputs = get_outputs();
    auto it = outputs.find(name);
    if (it == outputs.end()) {
      auto output = std::make_shared<graph_edge>(this);
      set_output(output, name);
      return output;
    }
    return it->second;
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message)) {
      for (const auto& [name, field] : obj_msg->get_fields()) {
        if (auto frame_msg = std::dynamic_pointer_cast<frame_message_base>(field)) {
          try {
            const auto output = get_output(name);
            output->send(field);
          } catch (const std::exception& e) {
            spdlog::error(e.what());
          }
        }
      }
    }
  }
};

COALSACK_REGISTER_NODE(object_map_node, graph_node)

class sentinel_message : public graph_message {
 public:
  sentinel_message() {}

  static std::string get_type() { return "sentinel"; }

  template <typename Archive>
  void serialize(Archive& archive) {}
};

class object_mux_node : public graph_node {
  graph_edge_ptr output;

 public:
  object_mux_node() : graph_node(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "object_mux"; }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message)) {
      for (const auto& [name, field] : obj_msg->get_fields()) {
        auto msg = std::make_shared<object_message>();
        msg->add_field(name, field);
        output->send(msg);
      }
      auto msg = std::make_shared<sentinel_message>();
      output->send(msg);
    }
  }
};

COALSACK_REGISTER_NODE(object_mux_node, graph_node)

class calibration_pipeline::impl {
 public:
  graph_proc graph;

  std::atomic_bool running;

  std::shared_ptr<calibration_node> calib_node;
  std::shared_ptr<graph_node> input_node;

  std::unordered_map<std::string, camera_t> cameras;
  std::unordered_map<std::string, camera_t> calibrated_cameras;

  std::vector<std::function<void(const std::unordered_map<std::string, camera_t>&)>> calibrated;

  void add_calibrated(
      std::function<void(const std::unordered_map<std::string, camera_t>&)> callback) {
    calibrated.push_back(callback);
  }

  void clear_calibrated() { calibrated.clear(); }

  void push_frame(const std::map<std::string, std::vector<point_data>>& frame) {
    if (!running) {
      return;
    }

    auto msg = std::make_shared<object_message>();
    for (const auto& [name, field] : frame) {
      auto float2_msg = std::make_shared<float2_list_message>();
      std::vector<float2> float2_data;
      for (const auto& pt : field) {
        float2_data.push_back({pt.point.x, pt.point.y});
      }
      float2_msg->set_data(float2_data);
      msg->add_field(name, float2_msg);
    }

    if (input_node) {
      graph.process(input_node.get(), msg);
    }
  }

  void calibrate(const std::unordered_map<std::string, camera_t>& cameras) {
    std::shared_ptr<object_message> msg(new object_message());
    for (const auto& [name, camera] : cameras) {
      std::shared_ptr<camera_message> camera_msg(new camera_message(camera));
      msg->add_field(name, camera_msg);
    }
    graph.process(calib_node.get(), "calibrate", msg);
  }

  void run(const std::vector<node_info>& infos) {
    std::shared_ptr<subgraph> g(new subgraph());

    std::shared_ptr<frame_number_numbering_node> n4(new frame_number_numbering_node());
    g->add_node(n4);

    input_node = n4;

    std::shared_ptr<object_map_node> n5(new object_map_node());
    n5->set_input(n4->get_output());
    g->add_node(n5);

    std::unordered_map<std::string, graph_node_ptr> detector_nodes;

    for (const auto& info : infos) {
      if (info.get_type() == node_type::pattern_board_calibration_target_detector) {
        for (const auto& [name, camera] : cameras) {
          std::shared_ptr<pattern_board_calibration_target_detector_node> n1(
              new pattern_board_calibration_target_detector_node());
          n1->set_input(n5->add_output(name));
          n1->set_camera(camera);
          g->add_node(n1);

          detector_nodes[name] = n1;
        }
      }
      if (info.get_type() == node_type::three_point_bar_calibration_target_detector) {
        for (const auto& [name, camera] : cameras) {
          std::shared_ptr<three_point_bar_calibration_target_detector_node> n1(
              new three_point_bar_calibration_target_detector_node());
          n1->set_input(n5->add_output(name));
          g->add_node(n1);

          detector_nodes[name] = n1;
        }
      }
    }

    for (const auto& info : infos) {
      if (info.get_type() == node_type::calibration) {
        std::shared_ptr<calibration_node> n1(new calibration_node());
        for (const auto& [name, node] : detector_nodes) {
          n1->set_input(node->get_output(), name);
        }
        n1->set_cameras(cameras);
        n1->set_only_extrinsic(info.get_param<bool>("only_extrinsic"));
        n1->set_robust(info.get_param<bool>("robust"));
        g->add_node(n1);

        calib_node = n1;
      }
    }

    if (calib_node == nullptr) {
      spdlog::error("Calibration node not found");
      return;
    }

    std::shared_ptr<callback_node> n2(new callback_node());
    n2->set_input(calib_node->get_output());
    g->add_node(n2);

    n2->set_callback_name("cameras");

    const auto callbacks = std::make_shared<callback_list>();

    callbacks->add(
        [this](const callback_node* node, std::string input_name, graph_message_ptr message) {
          if (node->get_callback_name() == "cameras") {
            if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message)) {
              for (const auto& [name, field] : obj_msg->get_fields()) {
                if (auto camera_msg = std::dynamic_pointer_cast<camera_message>(field)) {
                  calibrated_cameras[name] = camera_msg->get_camera();
                }
              }
              for (const auto& f : calibrated) {
                f(calibrated_cameras);
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

  size_t get_num_frames(std::string name) const {
    if (!calib_node) {
      return 0;
    }
    return calib_node->get_num_frames(name);
  }

  const std::vector<observed_points_t> get_observed_points(std::string name) const {
    if (!calib_node) {
      static std::vector<observed_points_t> empty;
      return empty;
    }
    return calib_node->get_observed_points(name);
  }
};

calibration_pipeline::calibration_pipeline() : pimpl(std::make_unique<impl>()) {}

calibration_pipeline::~calibration_pipeline() = default;

void calibration_pipeline::set_camera(const std::string& name, const camera_t& camera) {
  pimpl->cameras[name] = camera;
}

size_t calibration_pipeline::get_camera_size() const { return pimpl->cameras.size(); }

const std::unordered_map<std::string, camera_t>& calibration_pipeline::get_cameras() const {
  return pimpl->cameras;
}

std::unordered_map<std::string, camera_t>& calibration_pipeline::get_cameras() {
  return pimpl->cameras;
}

void calibration_pipeline::run(const std::vector<node_info>& infos) { pimpl->run(infos); }

void calibration_pipeline::stop() { pimpl->stop(); }

void calibration_pipeline::add_calibrated(
    std::function<void(const std::unordered_map<std::string, camera_t>&)> callback) {
  pimpl->add_calibrated(callback);
}

void calibration_pipeline::clear_calibrated() { pimpl->clear_calibrated(); }

size_t calibration_pipeline::get_num_frames(std::string name) const {
  return pimpl->get_num_frames(name);
}

const std::vector<observed_points_t> calibration_pipeline::get_observed_points(
    std::string name) const {
  return pimpl->get_observed_points(name);
}

const std::unordered_map<std::string, camera_t>& calibration_pipeline::get_calibrated_cameras()
    const {
  return pimpl->calibrated_cameras;
}

void calibration_pipeline::push_frame(const std::map<std::string, std::vector<point_data>>& frame) {
  pimpl->push_frame(frame);
}

void calibration_pipeline::calibrate() { pimpl->calibrate(pimpl->cameras); }

static std::vector<size_t> create_random_indices(size_t size) {
  std::vector<size_t> data(size);
  for (size_t i = 0; i < size; i++) {
    data[i] = i;
  }

  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());

  std::shuffle(data.begin(), data.end(), engine);

  return data;
}

class intrinsic_calibration_node : public graph_node {
  std::shared_ptr<graph_edge> output;

  std::vector<std::vector<point_data>> frames;
  camera_t initial_camera;
  camera_t calibrated_camera;
  double rms = 0.0;

 public:
  intrinsic_calibration_node() : graph_node(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "intrinsic_calibration"; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(initial_camera);
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (input_name == "calibrate") {
      if (auto camera_msg = std::dynamic_pointer_cast<camera_message>(message)) {
        initial_camera = camera_msg->get_camera();
      }

      calibrate();

      auto camera_msg = std::make_shared<camera_message>(calibrated_camera);
      output->send(camera_msg);
    }

    if (auto points_msg = std::dynamic_pointer_cast<float2_list_message>(message)) {
      std::vector<point_data> points;
      for (const auto& pt : points_msg->get_data()) {
        points.push_back(point_data{glm::vec2(pt.x, pt.y), 0, 0});
      }
      push_frame(points);
    }

    if (auto image_msg = std::dynamic_pointer_cast<image_message>(message)) {
      const auto& image = image_msg->get_image();
      cv::Mat img(image.get_height(), image.get_width(), convert_to_cv_type(image.get_format()),
                  (void*)image.get_data(), image.get_stride());
      push_frame(img);
    }
  }

  double get_rms() const { return rms; }

  void set_initial_camera(const camera_t& camera) { initial_camera = camera; }

  const camera_t& get_calibrated_camera() const { return calibrated_camera; }

  size_t get_num_frames() const { return frames.size(); }

  void push_frame(const std::vector<point_data>& frame) { frames.push_back(frame); }

  void push_frame(const cv::Mat& frame) {
    std::vector<cv::Point2f> board;
    if (detect_calibration_board(frame, board)) {
      std::vector<point_data> points;
      for (const auto& point : board) {
        points.push_back(point_data{glm::vec2(point.x, point.y), 0, 0});
      }
      push_frame(points);
    }
  }

  void calibrate() {
    const auto image_width = initial_camera.width;
    const auto image_height = initial_camera.height;
    const auto square_size = cv::Size2f(2.41, 2.4);  // TODO: Define as config
    const auto board_size = cv::Size(10, 7);         // TODO: Define as config
    const auto image_size = cv::Size(image_width, image_height);

    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points;

    std::vector<cv::Point3f> object_point;
    calc_board_corner_positions(board_size, square_size, object_point);

    const auto max_num_frames = 100;  // TODO: Define as config

    const auto frame_indices =
        create_random_indices(std::min(frames.size(), static_cast<size_t>(max_num_frames)));

    for (const auto& frame_index : frame_indices) {
      const auto& frame = frames.at(frame_index);

      object_points.push_back(object_point);

      std::vector<cv::Point2f> image_point;
      for (const auto& point : frame) {
        image_point.push_back(cv::Point2f(point.point.x, point.point.y));
      }

      image_points.push_back(image_point);
    }

    std::vector<cv::Mat> rvecs, tvecs;
    cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);

    rms = cv::calibrateCamera(object_points, image_points, image_size, camera_matrix, dist_coeffs,
                              rvecs, tvecs);

    calibrated_camera.intrin.fx = camera_matrix.at<double>(0, 0);
    calibrated_camera.intrin.fy = camera_matrix.at<double>(1, 1);
    calibrated_camera.intrin.cx = camera_matrix.at<double>(0, 2);
    calibrated_camera.intrin.cy = camera_matrix.at<double>(1, 2);
    calibrated_camera.intrin.coeffs[0] = dist_coeffs.at<double>(0);
    calibrated_camera.intrin.coeffs[1] = dist_coeffs.at<double>(1);
    calibrated_camera.intrin.coeffs[2] = dist_coeffs.at<double>(2);
    calibrated_camera.intrin.coeffs[3] = dist_coeffs.at<double>(3);
    calibrated_camera.intrin.coeffs[4] = dist_coeffs.at<double>(4);
    calibrated_camera.width = image_width;
    calibrated_camera.height = image_height;
  }
};

COALSACK_REGISTER_NODE(intrinsic_calibration_node, graph_node)

class intrinsic_calibration_pipeline::impl {
 public:
  graph_proc graph;

  std::atomic_bool running;

  std::shared_ptr<intrinsic_calibration_node> calib_node;
  std::shared_ptr<graph_node> input_node;

  impl() = default;

  void run(const std::vector<node_info>& infos) {
    std::shared_ptr<subgraph> g(new subgraph());

    std::shared_ptr<frame_number_numbering_node> n4(new frame_number_numbering_node());
    g->add_node(n4);

    input_node = n4;

    std::shared_ptr<object_mux_node> n5(new object_mux_node());
    n5->set_input(n4->get_output());
    g->add_node(n5);

    for (const auto& info : infos) {
      if (info.get_type() == node_type::calibration) {
        std::shared_ptr<intrinsic_calibration_node> n1(new intrinsic_calibration_node());
        n1->set_input(n5->get_output());
        g->add_node(n1);

        calib_node = n1;
      }
    }

    if (calib_node == nullptr) {
      spdlog::error("Calibration node not found");
      return;
    }

    std::shared_ptr<callback_node> n2(new callback_node());
    n2->set_input(calib_node->get_output());
    g->add_node(n2);

    n2->set_callback_name("camera");

    const auto callbacks = std::make_shared<callback_list>();

    callbacks->add(
        [](const callback_node* node, std::string input_name, graph_message_ptr message) {});

    graph.deploy(g);
    graph.get_resources()->add(callbacks);
    graph.run();

    running = true;
  }

  void stop() {
    running.store(false);
    graph.stop();
  }
};

intrinsic_calibration_pipeline::intrinsic_calibration_pipeline()
    : pimpl(std::make_unique<impl>()) {}

intrinsic_calibration_pipeline::~intrinsic_calibration_pipeline() = default;

void intrinsic_calibration_pipeline::run(const std::vector<node_info>& infos) { pimpl->run(infos); }
void intrinsic_calibration_pipeline::stop() { pimpl->stop(); }

double intrinsic_calibration_pipeline::get_rms() const {
  if (pimpl->calib_node) {
    return pimpl->calib_node->get_rms();
  }
  return 0.0;
}

void intrinsic_calibration_pipeline::set_image_size(int width, int height) {
  if (pimpl->calib_node) {
    camera_t initial_camera;
    initial_camera.width = width;
    initial_camera.height = height;
    pimpl->calib_node->set_initial_camera(initial_camera);
  }
}

const camera_t& intrinsic_calibration_pipeline::get_calibrated_camera() const {
  if (pimpl->calib_node) {
    return pimpl->calib_node->get_calibrated_camera();
  }
  static camera_t empty_camera;
  return empty_camera;
}

size_t intrinsic_calibration_pipeline::get_num_frames() const {
  if (pimpl->calib_node) {
    return pimpl->calib_node->get_num_frames();
  }
  return 0;
}

void intrinsic_calibration_pipeline::push_frame(const std::vector<point_data>& frame) {
  if (pimpl->calib_node) {
    pimpl->calib_node->push_frame(frame);
  }
}
void intrinsic_calibration_pipeline::push_frame(const cv::Mat& frame) {
  if (pimpl->calib_node) {
    pimpl->calib_node->push_frame(frame);
  }
}

void intrinsic_calibration_pipeline::calibrate() {
  if (pimpl->calib_node) {
    pimpl->calib_node->calibrate();
  }
}

class axis_reconstruction {
  glm::mat4 axis;
  std::map<std::string, camera_t> cameras;

 public:
  void set_camera(const std::string& name, const camera_t& camera);

  glm::mat4 get_axis() const { return axis; }

  void set_axis(const glm::mat4& axis) { this->axis = axis; }

  static bool compute_axis(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::mat4& axis);

  static bool detect_axis(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::mat4& axis,
                          float x_axis_length = 0.14f, float y_axis_length = 0.17f);

  void push_frame(const std::map<std::string, cv::Mat>& frame);

  void push_frame(const std::map<std::string, std::vector<point_data>>& points);
};

void axis_reconstruction::set_camera(const std::string& name, const camera_t& camera) {
  cameras[name] = camera;
}

bool axis_reconstruction::compute_axis(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::mat4& axis) {
  if (!(std::abs(glm::dot(p1 - p0, p2 - p0)) < 0.01)) {
    return false;
  }
  const auto origin = p0;
  const auto e1 = p1 - p0;
  const auto e2 = p2 - p0;

  glm::vec3 x_axis = e1;
  glm::vec3 y_axis = e2;

  auto z_axis = glm::cross(glm::normalize(x_axis), glm::normalize(y_axis));
  z_axis = glm::normalize(z_axis);

  const auto y_axis_length = 0.196f;
  const auto scale = y_axis_length / glm::length(y_axis);
  x_axis = glm::normalize(x_axis);
  y_axis = glm::normalize(y_axis);

  axis = glm::mat4(1.0f);

  axis[0] = glm::vec4(x_axis / scale, 0.0f);
  axis[1] = glm::vec4(y_axis / scale, 0.0f);
  axis[2] = glm::vec4(z_axis / scale, 0.0f);
  axis[3] = glm::vec4(origin, 1.0f);

  axis = glm::inverse(axis);

  return true;
}

bool axis_reconstruction::detect_axis(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::mat4& axis,
                                      float x_axis_length, float y_axis_length) {
  glm::vec3 origin;
  glm::vec3 e1, e2;

  // Find origin
  if (std::abs(glm::dot(p1 - p0, p2 - p0)) < 0.01) {
    origin = p0;
    e1 = p1 - p0;
    e2 = p2 - p0;
  } else if (std::abs(glm::dot(p0 - p1, p2 - p1)) < 0.01) {
    origin = p1;
    e1 = p0 - p1;
    e2 = p2 - p1;
  } else if (std::abs(glm::dot(p0 - p2, p1 - p2)) < 0.01) {
    origin = p2;
    e1 = p0 - p2;
    e2 = p1 - p2;
  } else {
    return false;
  }

  glm::vec3 x_axis, y_axis;
  if (glm::length(e1) < glm::length(e2)) {
    x_axis = e1;
    y_axis = e2;
  } else {
    x_axis = e2;
    y_axis = e1;
  }

  auto z_axis = glm::cross(x_axis, y_axis);
  z_axis = glm::normalize(z_axis);

  const auto scale = x_axis_length / glm::length(x_axis);
  if ((std::abs(scale - y_axis_length / glm::length(y_axis)) / scale) > 0.05) {
    return false;
  }
  x_axis = glm::normalize(x_axis);
  y_axis = glm::normalize(y_axis);

  axis[0] = glm::vec4(x_axis / scale, 0.0f);
  axis[1] = glm::vec4(y_axis / scale, 0.0f);
  axis[2] = glm::vec4(z_axis / scale, 0.0f);
  axis[3] = glm::vec4(origin, 1.0f);

  axis = glm::inverse(axis);

  return true;
}

static void detect_aruco_marker(cv::Mat image, std::vector<std::vector<cv::Point2f>>& points,
                                std::vector<int>& ids) {
  cv::aruco::DetectorParameters detector_params = cv::aruco::DetectorParameters();
  cv::aruco::RefineParameters refine_params = cv::aruco::RefineParameters();
  const auto dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
  const auto detector = cv::aruco::ArucoDetector(dictionary, detector_params, refine_params);

  points.clear();
  ids.clear();
  detector.detectMarkers(image, points, ids);
}

void axis_reconstruction::push_frame(const std::map<std::string, cv::Mat>& frame) {
  std::map<std::string, std::vector<point_data>> points;

  for (const auto& [name, image] : frame) {
    std::vector<int> marker_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners;
    detect_aruco_marker(image, marker_corners, marker_ids);

    for (size_t i = 0; i < marker_ids.size(); i++) {
      if (marker_ids[i] == 0) {
        auto& corner_points = points[name];
        for (size_t j = 0; j < 3; j++) {
          point_data point{};
          point.point.x = marker_corners[i][j].x;
          point.point.y = marker_corners[i][j].y;
          corner_points.push_back(point);
        }
      }
    }
  }

  std::vector<glm::vec3> markers;
  for (size_t j = 0; j < 3; j++) {
    std::vector<glm::vec2> pts;
    std::vector<camera_t> cams;

    for (const auto& [name, camera] : cameras) {
      pts.push_back(points[name][j].point);
      cams.push_back(camera);
    }
    const auto marker = triangulate(pts, cams);
    markers.push_back(marker);
  }

  if (markers.size() == 3) {
    if (!compute_axis(markers[1], markers[0], markers[2], axis)) {
      std::cout << "Failed to compute axis" << std::endl;
      return;
    }
  }

  glm::mat4 basis(1.f);
  basis[0] = glm::vec4(-1.f, 0.f, 0.f, 0.f);
  basis[1] = glm::vec4(0.f, 0.f, 1.f, 0.f);
  basis[2] = glm::vec4(0.f, 1.f, 0.f, 0.f);

  glm::mat4 cv_to_gl(1.f);
  cv_to_gl[0] = glm::vec4(1.f, 0.f, 0.f, 0.f);
  cv_to_gl[1] = glm::vec4(0.f, -1.f, 0.f, 0.f);
  cv_to_gl[2] = glm::vec4(0.f, 0.f, -1.f, 0.f);

  // z down -> z up -> opengl
  axis = basis * axis;
  // axis = basis * cv_to_gl * extrinsic_calib.axis;
}

void axis_reconstruction::push_frame(const std::map<std::string, std::vector<point_data>>& frame) {
  const auto markers = reconstruct(cameras, frame);

  glm::mat4 axis;
  if (markers.size() == 3) {
    if (detect_axis(markers[0], markers[1], markers[2], axis)) {
      glm::mat4 basis(1.f);
      basis[0] = glm::vec4(-1.f, 0.f, 0.f, 0.f);
      basis[1] = glm::vec4(0.f, 0.f, 1.f, 0.f);
      basis[2] = glm::vec4(0.f, 1.f, 0.f, 0.f);

      // z up -> opengl
      axis = basis * axis;

      this->axis = axis;

      return;
    }
  }

  {
    constexpr size_t grid_width = 2;
    constexpr size_t grid_height = 9;
    constexpr size_t num_points = grid_width * grid_height;

    std::vector<std::vector<cv::Point2f>> image_points_nview;
    std::vector<camera_t> cams;
    for (const auto& [name, camera] : cameras) {
      std::vector<cv::Point2f> image_points;
      for (const auto& pt : frame.at(name)) {
        image_points.push_back(cv::Point2f(pt.point.x, pt.point.y));
      }
      if (image_points.size() == num_points) {
        image_points_nview.push_back(image_points);
        cams.push_back(camera);
      }
    }

    if (cams.size() < 2) {
      return;
    }

    std::vector<glm::vec3> markers;
    for (size_t i = 0; i < num_points; i++) {
      std::vector<glm::vec2> pts;

      for (size_t j = 0; j < cams.size(); j++) {
        pts.push_back(glm::vec2(image_points_nview[j][i].x, image_points_nview[j][i].y));
      }

      const auto marker = triangulate(pts, cams);
      markers.push_back(marker);
    }

    const auto grid_size_x = 0.01825f;
    const auto grid_size_y = 0.01825f;
    const auto x_axis_length = grid_size_x * 2.0f * (grid_width - 1);
    const auto y_axis_length = grid_size_y * (grid_height - 1);

    // Points are expected to be in the following order:
    // 0 1
    //  2 3
    // 4 5
    //  6 7
    // ...

    if (detect_axis(markers[num_points - grid_width], markers[num_points - 1], markers[0], axis,
                    x_axis_length, y_axis_length)) {
      glm::mat4 basis(1.f);
      basis[0] = glm::vec4(-1.f, 0.f, 0.f, 0.f);
      basis[1] = glm::vec4(0.f, 0.f, 1.f, 0.f);
      basis[2] = glm::vec4(0.f, 1.f, 0.f, 0.f);

      // z up -> opengl
      axis = basis * axis;

      this->axis = axis;

      return;
    }
  }
}

class axis_calibration_node : public graph_node {
  observed_points_frames observed_frames;

  mutable std::mutex cameras_mtx;
  std::vector<std::string> camera_names;
  std::unordered_map<std::string, camera_t> cameras;

  axis_reconstruction reconstructor;

  graph_edge_ptr output;

 public:
  axis_calibration_node() : graph_node(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "calibration"; }

  void set_cameras(const std::unordered_map<std::string, camera_t>& cameras) {
    this->cameras = cameras;
  }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cameras);
  }

  size_t get_num_frames(std::string name) const { return observed_frames.get_num_points(name); }

  const std::vector<observed_points_t> get_observed_points(std::string name) const {
    return observed_frames.get_observed_points(name);
  }

  void calibrate() {
    for (const auto& [name, camera] : cameras) {
      reconstructor.set_camera(name, camera);
    }

    {
      for (size_t f = 0; f < observed_frames.get_num_frames(); f++) {
        std::map<std::string, std::vector<point_data>> frame;
        for (const auto& camera_name : camera_names) {
          std::vector<point_data> points_data;
          const auto point = observed_frames.get_observed_point(camera_name, f);
          for (const auto& pt : point.points) {
            points_data.push_back(point_data{pt, 0, 0});
          }
          frame[camera_name] = points_data;
        }

        reconstructor.push_frame(frame);
      }
    }
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (input_name == "calibrate") {
      camera_names.clear();

      if (auto camera_msg = std::dynamic_pointer_cast<object_message>(message)) {
        for (const auto& [name, field] : camera_msg->get_fields()) {
          if (auto camera_msg = std::dynamic_pointer_cast<camera_message>(field)) {
            std::lock_guard lock(cameras_mtx);
            cameras[name] = camera_msg->get_camera();
            camera_names.push_back(name);
          }
        }
      }

      calibrate();
      scene_t scene;
      scene.axis = reconstructor.get_axis();

      std::shared_ptr<scene_message> msg(new scene_message(scene));
      output->send(msg);

      return;
    }

    if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message)) {
      for (const auto& [name, field] : obj_msg->get_fields()) {
        if (auto points_msg = std::dynamic_pointer_cast<float2_list_message>(field)) {
          std::vector<glm::vec2> points;
          for (const auto& pt : points_msg->get_data()) {
            points.emplace_back(pt.x, pt.y);
          }
          observed_frames.add_frame_points(points_msg->get_frame_number(), name, points);
        }
      }
    }
  }
};

COALSACK_REGISTER_NODE(axis_calibration_node, graph_node)

class axis_calibration_pipeline::impl {
 public:
  graph_proc graph;

  std::atomic_bool running;

  std::shared_ptr<axis_calibration_node> calib_node;
  std::shared_ptr<graph_node> input_node;

  using callback_func_type = std::function<void(const scene_t&)>;

  std::vector<callback_func_type> calibrated;

  mutable std::mutex cameras_mtx;
  std::vector<std::string> camera_names;
  std::unordered_map<std::string, camera_t> cameras;

  std::shared_ptr<parameters_t> parameters;

  void add_calibrated(callback_func_type callback) { calibrated.push_back(callback); }

  void clear_calibrated() { calibrated.clear(); }

  void push_frame(const std::map<std::string, std::vector<point_data>>& frame) {
    if (!running) {
      return;
    }

    auto msg = std::make_shared<object_message>();
    for (const auto& [name, field] : frame) {
      auto float2_msg = std::make_shared<float2_list_message>();
      std::vector<float2> float2_data;
      for (const auto& pt : field) {
        float2_data.push_back({pt.point.x, pt.point.y});
      }
      float2_msg->set_data(float2_data);
      msg->add_field(name, float2_msg);
    }

    if (input_node) {
      graph.process(input_node.get(), msg);
    }
  }

  void calibrate() {
    std::shared_ptr<object_message> msg(new object_message());
    for (const auto& [name, camera] : cameras) {
      std::shared_ptr<camera_message> camera_msg(new camera_message(camera));
      msg->add_field(name, camera_msg);
    }
    graph.process(calib_node.get(), "calibrate", msg);
  }

  void run(const std::vector<node_info>& infos) {
    std::shared_ptr<subgraph> g(new subgraph());

    std::shared_ptr<frame_number_numbering_node> n4(new frame_number_numbering_node());
    g->add_node(n4);

    input_node = n4;

    std::shared_ptr<object_mux_node> n5(new object_mux_node());
    n5->set_input(n4->get_output());
    g->add_node(n5);

    for (const auto& info : infos) {
      if (info.get_type() == node_type::calibration) {
        std::shared_ptr<axis_calibration_node> n1(new axis_calibration_node());
        n1->set_input(n5->get_output());
        g->add_node(n1);

        calib_node = n1;
      }
    }

    if (calib_node == nullptr) {
      spdlog::error("Calibration node not found");
      return;
    }

    std::shared_ptr<callback_node> n2(new callback_node());
    n2->set_input(calib_node->get_output());
    g->add_node(n2);

    n2->set_callback_name("scene");

    const auto callbacks = std::make_shared<callback_list>();

    callbacks->add(
        [this](const callback_node* node, std::string input_name, graph_message_ptr message) {
          if (node->get_callback_name() == "scene") {
            if (auto scene_msg = std::dynamic_pointer_cast<scene_message>(message)) {
              for (const auto& f : calibrated) {
                f(scene_msg->get_scene());
              }

              auto& scene = std::get<scene_t>(parameters->at("scene"));
              scene = scene_msg->get_scene();
              parameters->save();
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

  size_t get_num_frames(std::string name) const {
    if (!calib_node) {
      return 0;
    }
    return calib_node->get_num_frames(name);
  }

  const std::vector<observed_points_t> get_observed_points(std::string name) const {
    if (!calib_node) {
      static std::vector<observed_points_t> empty;
      return empty;
    }
    return calib_node->get_observed_points(name);
  }
};

axis_calibration_pipeline::axis_calibration_pipeline(std::shared_ptr<parameters_t> parameters)
    : pimpl(std::make_unique<impl>()) {
  pimpl->parameters = parameters;
}

axis_calibration_pipeline::~axis_calibration_pipeline() = default;

void axis_calibration_pipeline::set_camera(const std::string& name, const camera_t& camera) {
  pimpl->cameras[name] = camera;
}

size_t axis_calibration_pipeline::get_camera_size() const { return pimpl->cameras.size(); }

const std::unordered_map<std::string, camera_t>& axis_calibration_pipeline::get_cameras() const {
  return pimpl->cameras;
}

std::unordered_map<std::string, camera_t>& axis_calibration_pipeline::get_cameras() {
  return pimpl->cameras;
}

size_t axis_calibration_pipeline::get_num_frames(std::string name) const {
  return pimpl->get_num_frames(name);
}

const std::vector<observed_points_t> axis_calibration_pipeline::get_observed_points(
    std::string name) const {
  return pimpl->get_observed_points(name);
}

void axis_calibration_pipeline::push_frame(
    const std::map<std::string, std::vector<point_data>>& frame) {
  pimpl->push_frame(frame);
}

void axis_calibration_pipeline::run(const std::vector<node_info>& infos) { pimpl->run(infos); }
void axis_calibration_pipeline::stop() { pimpl->stop(); }

void axis_calibration_pipeline::calibrate() { pimpl->calibrate(); }
