#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace stargazer {
enum class node_type {
  unknown,
  raspi,
  raspi_color,
  depthai_color,
  rs_d435,
  rs_d435_color,
  raspi_playback,
  panoptic,
  record,
  calibration,
  pattern_board_calibration_target_detector,
  three_point_bar_calibration_target_detector,
  voxelpose_reconstruction,
  mvpose_reconstruction,
  epipolar_reconstruction,
};

using node_param_t = std::variant<std::string, std::int64_t, float, bool>;

struct node_info {
  std::string name{};
  node_type type{node_type::unknown};
  std::unordered_map<std::string, std::string> inputs{};
  std::unordered_map<std::string, node_param_t> params{};

  template <typename T>
  const T &get_param(const std::string &key) const {
    if (params.find(key) == params.end()) {
      throw std::runtime_error("Parameter not found: " + key);
    }
    return std::get<T>(params.at(key));
  }

  bool is_camera() const {
    return type == node_type::raspi || type == node_type::raspi_color ||
           type == node_type::depthai_color || type == node_type::rs_d435 ||
           type == node_type::rs_d435_color || type == node_type::raspi_playback ||
           type == node_type::panoptic;
  }
};

class configuration {
  std::string path;
  std::unordered_map<std::string, std::vector<node_info>> pipeline_nodes;
  std::unordered_map<std::string, std::string> pipeline_names;

 public:
  configuration(const std::string &path);

  void update();

  const std::vector<node_info> &get_node_infos() const {
    return pipeline_nodes.at(pipeline_names.at("pipeline"));
  }
  std::vector<node_info> &get_node_infos() {
    return pipeline_nodes.at(pipeline_names.at("pipeline"));
  }

  const std::vector<node_info> &get_node_infos(const std::string &pipeline) const {
    return pipeline_nodes.at(pipeline_names.at(pipeline));
  }
  std::vector<node_info> &get_node_infos(const std::string &pipeline) {
    return pipeline_nodes.at(pipeline_names.at(pipeline));
  }
};
}  // namespace stargazer
