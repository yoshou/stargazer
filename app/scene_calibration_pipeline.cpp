#include "scene_calibration_pipeline.hpp"

#include <spdlog/spdlog.h>

#include <unordered_map>

#include "calibration.hpp"
#include "callback_node.hpp"
#include "coalsack/core/graph_proc.h"
#include "coalsack/image/graph_proc_cv.h"
#include "coalsack/image/image_nodes.h"
#include "glm_serialize.hpp"
#include "graph_builder.hpp"
#include "messages.hpp"
#include "object_map_node.hpp"
#include "object_mux_node.hpp"
#include "parameters.hpp"
#include "reconstruction.hpp"
#include "scene_calibration_node.hpp"
#include "triangulation.hpp"
#include "utils.hpp"

using namespace stargazer;
using namespace stargazer::calibration;
using namespace stargazer::reconstruction;
using namespace coalsack;

class scene_calibration_pipeline::impl {
 public:
  graph_proc graph;
  std::unordered_map<std::string, graph_node_ptr> node_map;

  std::atomic_bool running;

  std::shared_ptr<scene_calibration_node> calib_node;
  std::shared_ptr<graph_node> input_node;

  using callback_func_type = std::function<void(const scene_t&)>;

  std::vector<callback_func_type> calibrated;

  mutable std::mutex cameras_mtx;
  std::vector<std::string> camera_names;
  std::unordered_map<std::string, stargazer::camera_t> cameras;

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

  void run(const std::vector<node_def>& nodes) {
    // Group nodes by subgraph instance
    std::map<std::string, std::vector<node_def>> nodes_by_subgraph;
    for (const auto& node : nodes) {
      nodes_by_subgraph[node.subgraph_instance].push_back(node);
    }

    // Create empty subgraphs first
    std::map<std::string, std::shared_ptr<subgraph>> subgraphs;
    for (const auto& [subgraph_name, nodes] : nodes_by_subgraph) {
      subgraphs[subgraph_name] = std::make_shared<subgraph>();
    }

    std::unordered_map<std::string, graph_node_ptr> built_node_map;

    // Build graph using common function
    stargazer::build_graph_from_json(nodes, subgraphs, built_node_map);
    node_map = built_node_map;

    // Extract specific nodes from the graph
    for (const auto& node : nodes) {
      if (node.get_type() == node_type::frame_number_numbering) {
        input_node =
            std::dynamic_pointer_cast<frame_number_numbering_node>(built_node_map.at(node.name));
      } else if (node.get_type() == node_type::scene_calibration) {
        calib_node =
            std::dynamic_pointer_cast<scene_calibration_node>(built_node_map.at(node.name));
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

    // Deploy all subgraphs
    for (const auto& [subgraph_name, subgraph_ptr] : subgraphs) {
      graph.deploy(subgraph_ptr);
    }
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

  std::optional<property_value> get_node_property(const std::string& node_name,
                                                  const std::string& key) const {
    const auto found = node_map.find(node_name);
    if (found == node_map.end() || !found->second) {
      return std::nullopt;
    }
    return found->second->get_property(key);
  }
};

scene_calibration_pipeline::scene_calibration_pipeline(std::shared_ptr<parameters_t> parameters)
    : pimpl(std::make_unique<impl>()) {
  pimpl->parameters = parameters;
}

scene_calibration_pipeline::~scene_calibration_pipeline() = default;

void scene_calibration_pipeline::add_calibrated(std::function<void(const scene_t&)> f) {
  pimpl->add_calibrated(f);
}

void scene_calibration_pipeline::clear_calibrated() { pimpl->clear_calibrated(); }

void scene_calibration_pipeline::set_camera(const std::string& name,
                                            const stargazer::camera_t& camera) {
  pimpl->cameras[name] = camera;
}

size_t scene_calibration_pipeline::get_camera_size() const { return pimpl->cameras.size(); }

const std::unordered_map<std::string, stargazer::camera_t>&
scene_calibration_pipeline::get_cameras() const {
  return pimpl->cameras;
}

std::unordered_map<std::string, stargazer::camera_t>& scene_calibration_pipeline::get_cameras() {
  return pimpl->cameras;
}

size_t scene_calibration_pipeline::get_num_frames(std::string name) const {
  return pimpl->get_num_frames(name);
}

const std::vector<observed_points_t> scene_calibration_pipeline::get_observed_points(
    std::string name) const {
  return pimpl->get_observed_points(name);
}

void scene_calibration_pipeline::push_frame(
    const std::map<std::string, std::vector<point_data>>& frame) {
  pimpl->push_frame(frame);
}

void scene_calibration_pipeline::run(const std::vector<node_def>& nodes) { pimpl->run(nodes); }
void scene_calibration_pipeline::stop() { pimpl->stop(); }

void scene_calibration_pipeline::calibrate() { pimpl->calibrate(); }

std::optional<property_value> scene_calibration_pipeline::get_node_property(
    const std::string& node_name, const std::string& key) const {
  return pimpl->get_node_property(node_name, key);
}
