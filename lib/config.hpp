#pragma once

#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace stargazer {
enum class node_type {
  unknown,
  record,
  calibration,
  pattern_board_calibration_target_detector,
  three_point_bar_calibration_target_detector,
  voxelpose_reconstruction,
  mvpose_reconstruction,
  mvp_reconstruction,
  epipolar_reconstruction,
  approximate_time_sync,
  frame_number_numbering,
  parallel_queue,
  frame_number_ordering,
  callback,
  grpc_server,
  frame_demux,
  dump_se3,
  libcamera_capture,
  timestamp,
  broadcast_talker,
  broadcast_listener,
  encode_jpeg,
  decode_jpeg,
  scale,
  resize,
  gaussian_blur,
  mask,
  p2p_tcp_talker,
  p2p_tcp_listener,
  fifo,
  video_time_sync_control,
  fast_blob_detector,
  detect_circle_grid,
  load_blob,
  load_marker,
  load_panoptic,
  charuco_detector,
  depthai_color_camera,
  rs_d435,
  object_map,
  object_mux,
  intrinsic_calibration,
  axis_calibration,
};

using node_param_t = std::variant<std::string, std::int64_t, double, float, bool>;

class node_def {
  node_type type{node_type::unknown};
  std::vector<std::shared_ptr<node_def>> extends;

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
      const auto& param = params.at(key);
      // Try direct get first
      if (std::holds_alternative<T>(param)) {
        value = std::get<T>(param);
      } else {
        // Handle numeric type conversions
        if constexpr (std::is_same_v<T, double>) {
          if (std::holds_alternative<std::int64_t>(param)) {
            value = static_cast<double>(std::get<std::int64_t>(param));
          } else if (std::holds_alternative<float>(param)) {
            value = static_cast<double>(std::get<float>(param));
          } else {
            value = std::get<T>(param);  // Will throw if type mismatch
          }
        } else if constexpr (std::is_same_v<T, float>) {
          if (std::holds_alternative<std::int64_t>(param)) {
            value = static_cast<float>(std::get<std::int64_t>(param));
          } else if (std::holds_alternative<double>(param)) {
            value = static_cast<float>(std::get<double>(param));
          } else {
            value = std::get<T>(param);
          }
        } else if constexpr (std::is_same_v<T, std::int64_t>) {
          if (std::holds_alternative<double>(param)) {
            value = static_cast<std::int64_t>(std::get<double>(param));
          } else if (std::holds_alternative<float>(param)) {
            value = static_cast<std::int64_t>(std::get<float>(param));
          } else {
            value = std::get<T>(param);
          }
        } else {
          value = std::get<T>(param);
        }
      }
    } else {
      for (const auto& extend : extends) {
        extend->get_param(key, value);
      }
    }
  }

 public:
  std::string name{};
  std::string subgraph_instance{};  // Which subgraph instance this node belongs to
  std::unordered_map<std::string, std::string> inputs{};
  std::vector<std::string> outputs{};
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

  bool contains_param(const std::string& key) const {
    if (params.find(key) != params.end()) {
      return true;
    }
    for (const auto& extend : extends) {
      if (extend->contains_param(key)) {
        return true;
      }
    }
    return false;
  }

  template <typename T>
  T get_param(const std::string& key) const {
    std::optional<T> value;
    get_param(key, value);
    if (!value.has_value()) {
      throw std::runtime_error("Parameter not found: " + key);
    }
    return value.value();
  }

  bool is_camera() const {
    const auto type = get_type();
    // For callback nodes, check if it's an image callback with camera_name set
    if (type == node_type::callback) {
      // Check callback_type parameter ("image")
      return contains_param("callback_type") &&
             get_param<std::string>("callback_type") == "image" && contains_param("camera_name") &&
             !get_param<std::string>("camera_name").empty();
    }
    return false;
  }

  std::string get_camera_name() const {
    const auto type = get_type();
    // For callback nodes, use camera_name parameter
    if (type == node_type::callback && contains_param("camera_name")) {
      return get_param<std::string>("camera_name");
    }
    return name;
  }
};

struct subgraph_def {
  std::string name;
  std::vector<node_def> nodes;
  std::vector<std::string> outputs;
  std::unordered_map<std::string, node_param_t> params;
  std::vector<std::string> extends;  // Template subgraph names to inherit from
};

struct pipeline_def {
  std::string name;
  std::vector<subgraph_def> subgraphs;  // Subgraph instances in this pipeline
};

class configuration {
  std::string path;
  std::unordered_map<std::string, pipeline_def> pipelines;
  std::unordered_map<std::string, std::string> pipeline_names;
  std::unordered_map<std::string, std::shared_ptr<node_def>> nodes;
  std::unordered_map<std::string, subgraph_def> subgraph_templates;  // Template definitions

 public:
  configuration(const std::string& path);

  void update();

  std::vector<node_def> get_nodes(const std::string& pipeline_key = "pipeline") const {
    const auto& pipeline_name = pipeline_names.at(pipeline_key);
    const auto& pipeline = pipelines.at(pipeline_name);
    std::vector<node_def> result;

    for (const auto& sg_instance : pipeline.subgraphs) {
      std::vector<node_def> sg_nodes = sg_instance.nodes;

      for (const auto& template_name : sg_instance.extends) {
        if (subgraph_templates.find(template_name) != subgraph_templates.end()) {
          const auto& sg_template = subgraph_templates.at(template_name);
          sg_nodes = sg_template.nodes;

          // Build a map of original node names in this subgraph
          std::unordered_set<std::string> local_node_names;
          for (const auto& n : sg_nodes) {
            local_node_names.insert(n.name);
          }

          for (auto& node : sg_nodes) {
            // Set subgraph instance name
            node.subgraph_instance = sg_instance.name;

            // Build parameter hierarchy (lowest to highest priority):
            // 1. Template subgraph params (lowest)
            // 2. Instance params
            // 3. Template node params (already in node.params)
            // 4. Instance node overrides (applied later, highest)

            std::unordered_map<std::string, node_param_t> merged_params;

            // Start with template subgraph parameters
            for (const auto& [key, value] : sg_template.params) {
              merged_params[key] = value;
            }

            // Override with instance parameters
            for (const auto& [key, value] : sg_instance.params) {
              merged_params[key] = value;
            }

            // Override with node's own parameters (from template)
            for (const auto& [key, value] : node.params) {
              merged_params[key] = value;
            }

            // Set the merged parameters
            node.params = merged_params;

            // Prefix node name with subgraph instance name if not already prefixed
            if (node.name.find(sg_instance.name) == std::string::npos) {
              node.name = sg_instance.name + "_" + node.name;
            }

            // Update input references to use prefixed names
            for (auto& [input_name, source_name] : node.inputs) {
              // Parse source_name to extract node reference and optional output name
              size_t colon_pos = source_name.find(':');
              std::string node_ref =
                  colon_pos != std::string::npos ? source_name.substr(0, colon_pos) : source_name;
              std::string output_ref =
                  colon_pos != std::string::npos ? source_name.substr(colon_pos) : "";

              // Check if this is a local reference (node exists in subgraph)
              if (local_node_names.count(node_ref) > 0) {
                // This is a local reference, prefix it
                if (node_ref.find(sg_instance.name) == std::string::npos) {
                  source_name = sg_instance.name + "_" + node_ref + output_ref;
                }
              }
            }
          }
        }
      }

      // Apply instance-specific node overrides
      for (const auto& override_node : sg_instance.nodes) {
        std::string target_node_name = override_node.name;

        // If the override node name doesn't include the instance prefix, add it
        if (target_node_name.find(sg_instance.name) == std::string::npos) {
          target_node_name = sg_instance.name + "_" + target_node_name;
        }

        // Find the node in sg_nodes and apply overrides
        for (auto& node : sg_nodes) {
          if (node.name == target_node_name) {
            // Override parameters
            for (const auto& [key, value] : override_node.params) {
              node.params[key] = value;
            }
            // Override inputs
            for (auto [key, value] : override_node.inputs) {
              // Replace '.' with '_' in cross-subgraph references
              std::replace(value.begin(), value.end(), '.', '_');
              node.inputs[key] = value;
            }
            break;
          }
        }
      }

      result.insert(result.end(), sg_nodes.begin(), sg_nodes.end());
    }
    return result;
  }

  bool has_subgraph(const std::string& name) const {
    return subgraph_templates.find(name) != subgraph_templates.end();
  }

  const subgraph_def& get_subgraph(const std::string& name) const {
    return subgraph_templates.at(name);
  }

  const pipeline_def& get_pipeline(const std::string& pipeline_key) const {
    return pipelines.at(pipeline_names.at(pipeline_key));
  }
};
}  // namespace stargazer
