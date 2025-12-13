#include "calibration_pipeline.hpp"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <spdlog/spdlog.h>

#include <random>
#include <unordered_map>
#include <unordered_set>

#include "axis_calibration_node.hpp"
#include "calibration.hpp"
#include "calibration_node.hpp"
#include "calibration_target.hpp"
#include "callback_node.hpp"
#include "glm_serialize.hpp"
#include "graph_builder.hpp"
#include "graph_proc.h"
#include "graph_proc_cv.h"
#include "graph_proc_img.h"
#include "intrinsic_calibration_node.hpp"
#include "messages.hpp"
#include "object_map_node.hpp"
#include "object_mux_node.hpp"
#include "parameters.hpp"
#include "pattern_board_calibration_target_detector_node.hpp"
#include "reconstruction.hpp"
#include "three_point_bar_calibration_target_detector_node.hpp"
#include "triangulation.hpp"
#include "utils.hpp"

using namespace stargazer;
using namespace stargazer::calibration;
using namespace stargazer::reconstruction;
using namespace coalsack;

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
    std::unordered_map<std::string, graph_node_ptr> node_map;
    std::map<std::string, std::shared_ptr<subgraph>> subgraphs;
    subgraphs[""] = g;

    // Build graph using common function
    stargazer::build_graph_from_json(infos, subgraphs, node_map);

    // Extract specific nodes from the graph
    for (const auto& info : infos) {
      if (info.get_type() == node_type::frame_number_numbering) {
        input_node = std::dynamic_pointer_cast<frame_number_numbering_node>(node_map.at(info.name));
      } else if (info.get_type() == node_type::calibration) {
        calib_node = std::dynamic_pointer_cast<calibration_node>(node_map.at(info.name));
        // Set cameras for calibration node
        if (calib_node) {
          calib_node->set_cameras(cameras);
        }
      } else if (info.get_type() == node_type::pattern_board_calibration_target_detector) {
        // Set camera from cameras map if camera_name parameter is provided
        if (info.contains_param("camera_name")) {
          const auto camera_name = info.get_param<std::string>("camera_name");
          if (cameras.find(camera_name) != cameras.end()) {
            auto detector_node =
                std::dynamic_pointer_cast<pattern_board_calibration_target_detector_node>(
                    node_map.at(info.name));
            if (detector_node) {
              detector_node->set_camera(cameras.at(camera_name));
            }
          }
        }
      }
    }

    if (calib_node == nullptr) {
      spdlog::error("Calibration node not found");
      return;
    }

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

class intrinsic_calibration_pipeline::impl {
 public:
  graph_proc graph;

  std::atomic_bool running;

  std::shared_ptr<intrinsic_calibration_node> calib_node;
  std::shared_ptr<graph_node> input_node;

  impl() = default;

  void run(const std::vector<node_info>& infos) {
    std::shared_ptr<subgraph> g(new subgraph());
    std::unordered_map<std::string, graph_node_ptr> node_map;
    std::map<std::string, std::shared_ptr<subgraph>> subgraphs;
    subgraphs[""] = g;

    // Build graph using common function
    stargazer::build_graph_from_json(infos, subgraphs, node_map);

    // Extract specific nodes from the graph
    for (const auto& info : infos) {
      if (info.get_type() == node_type::frame_number_numbering) {
        input_node = std::dynamic_pointer_cast<frame_number_numbering_node>(node_map.at(info.name));
      } else if (info.get_type() == node_type::intrinsic_calibration) {
        calib_node = std::dynamic_pointer_cast<intrinsic_calibration_node>(node_map.at(info.name));
      }
    }

    if (calib_node == nullptr) {
      spdlog::error("Calibration node not found");
      return;
    }

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
    std::unordered_map<std::string, graph_node_ptr> node_map;
    std::map<std::string, std::shared_ptr<subgraph>> subgraphs;
    subgraphs[""] = g;

    // Build graph using common function
    stargazer::build_graph_from_json(infos, subgraphs, node_map);

    // Extract specific nodes from the graph
    for (const auto& info : infos) {
      if (info.get_type() == node_type::frame_number_numbering) {
        input_node = std::dynamic_pointer_cast<frame_number_numbering_node>(node_map.at(info.name));
      } else if (info.get_type() == node_type::axis_calibration) {
        calib_node = std::dynamic_pointer_cast<axis_calibration_node>(node_map.at(info.name));
      }
    }

    if (calib_node == nullptr) {
      spdlog::error("Calibration node not found");
      return;
    }

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
