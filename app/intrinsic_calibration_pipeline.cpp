#include "intrinsic_calibration_pipeline.hpp"

#include <spdlog/spdlog.h>

#include <unordered_map>

#include "calibration.hpp"
#include "callback_node.hpp"
#include "coalsack/core/graph_proc.h"
#include "coalsack/image/graph_proc_cv.h"
#include "coalsack/image/image_nodes.h"
#include "glm_serialize.hpp"
#include "graph_builder.hpp"
#include "intrinsic_calibration_node.hpp"
#include "messages.hpp"
#include "parameters.hpp"

using namespace stargazer;
using namespace stargazer::calibration;
using namespace coalsack;

class intrinsic_calibration_pipeline::impl {
 public:
  graph_proc graph;
  std::unordered_map<std::string, graph_node_ptr> node_map;

  std::atomic_bool running;

  std::shared_ptr<intrinsic_calibration_node> calib_node;
  std::shared_ptr<graph_node> input_node;

  impl() = default;

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
      } else if (node.get_type() == node_type::intrinsic_calibration) {
        calib_node =
            std::dynamic_pointer_cast<intrinsic_calibration_node>(built_node_map.at(node.name));
      }
    }

    if (calib_node == nullptr) {
      spdlog::error("Calibration node not found");
      return;
    }

    const auto callbacks = std::make_shared<callback_list>();

    callbacks->add(
        [](const callback_node* node, std::string input_name, graph_message_ptr message) {});

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

  std::optional<property_value> get_node_property(const std::string& node_name,
                                                  const std::string& key) const {
    const auto found = node_map.find(node_name);
    if (found == node_map.end() || !found->second) {
      return std::nullopt;
    }
    return found->second->get_property(key);
  }
};

intrinsic_calibration_pipeline::intrinsic_calibration_pipeline()
    : pimpl(std::make_unique<impl>()) {}

intrinsic_calibration_pipeline::~intrinsic_calibration_pipeline() = default;

void intrinsic_calibration_pipeline::run(const std::vector<node_def>& nodes) { pimpl->run(nodes); }
void intrinsic_calibration_pipeline::stop() { pimpl->stop(); }

double intrinsic_calibration_pipeline::get_rms() const {
  if (pimpl->calib_node) {
    return pimpl->calib_node->get_rms();
  }
  return 0.0;
}

void intrinsic_calibration_pipeline::set_image_size(int width, int height) {
  if (pimpl->calib_node) {
    stargazer::camera_t initial_camera;
    initial_camera.width = width;
    initial_camera.height = height;
    pimpl->calib_node->set_initial_camera(initial_camera);
  }
}

const stargazer::camera_t& intrinsic_calibration_pipeline::get_calibrated_camera() const {
  if (pimpl->calib_node) {
    return pimpl->calib_node->get_calibrated_camera();
  }
  static stargazer::camera_t empty_camera;
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

std::optional<property_value> intrinsic_calibration_pipeline::get_node_property(
    const std::string& node_name, const std::string& key) const {
  return pimpl->get_node_property(node_name, key);
}
