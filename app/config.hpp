#pragma once

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
    return type == node_type::raspi || type == node_type::raspi_color ||
           type == node_type::depthai_color || type == node_type::rs_d435 ||
           type == node_type::rs_d435_color || type == node_type::playback ||
           type == node_type::panoptic;
  }
};

struct subgraph_def {
  std::string name;
  std::vector<node_info> nodes;
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
  std::unordered_map<std::string, std::shared_ptr<node_info>> nodes;
  std::unordered_map<std::string, subgraph_def> subgraph_templates;  // Template definitions

 public:
  configuration(const std::string& path);

  void update();

  std::vector<node_info> get_node_infos(const std::string& pipeline_key = "pipeline") const {
    const auto& pipeline_name = pipeline_names.at(pipeline_key);
    const auto& pipeline = pipelines.at(pipeline_name);
    std::vector<node_info> result;

    for (const auto& sg_instance : pipeline.subgraphs) {
      std::vector<node_info> sg_nodes = sg_instance.nodes;

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
            // Apply instance parameters
            for (const auto& [key, value] : sg_instance.params) {
              node.params[key] = value;
            }

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
