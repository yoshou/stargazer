#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>
#include <memory>
#include <optional>

namespace stargazer {
enum class node_type {
  unknown,
  raspi,
  raspi_color,
  depthai_color,
  rs_d435,
  rs_d435_color,
  playback,
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

class node_info {
  node_type type{node_type::unknown};
  std::vector<std::shared_ptr<node_info>> extends;

  friend class configuration;

  void get_type(node_type& result_type) const {
    if (type != node_type::unknown) {
      result_type = type;
    } else {
      for (const auto& extend : extends) {
        extend->get_type(result_type);
      }
    }
  }

  template <typename T>
  void get_param(const std::string& key, std::optional<T>& value) const {
    if (params.find(key) != params.end()) {
      value = std::get<T>(params.at(key));
    } else {
      for (const auto& extend : extends) {
        extend->get_param(key, value);
      }
    }
  }

 public:
  std::string name{};
  std::unordered_map<std::string, std::string> inputs{};
  std::unordered_map<std::string, node_param_t> params{};

  void set_type(node_type type) { this->type = type; }
  node_type get_type() const {
    node_type result_type = type;
    get_type(result_type);
    if (result_type == node_type::unknown) {
      throw std::runtime_error("Node type is unknown");
    }
    return result_type;
  }
  
  bool contains_param(const std::string &key) const {
    if (params.find(key) != params.end()) {
      return true;
    }
    for (const auto &extend : extends) {
      if (extend->contains_param(key)) {
        return true;
      }
    }
    return false;
  }

  template <typename T>
  T get_param(const std::string &key) const {
    std::optional<T> value;
    get_param(key, value);
    if (!value.has_value()) {
      throw std::runtime_error("Parameter not found: " + key);
    }
    return value.value();
  }

  bool is_camera() const {
    const auto type = get_type();
    return type == node_type::raspi || type == node_type::raspi_color ||
           type == node_type::depthai_color || type == node_type::rs_d435 ||
           type == node_type::rs_d435_color || type == node_type::playback ||
           type == node_type::panoptic;
  }
};

class configuration {
  std::string path;
  std::unordered_map<std::string, std::vector<node_info>> pipeline_nodes;
  std::unordered_map<std::string, std::string> pipeline_names;
  std::unordered_map<std::string, std::shared_ptr<node_info>> nodes;

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
