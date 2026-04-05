#pragma once

#include <algorithm>
#include <cstdint>
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
  extrinsic_calibration,
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
  grpc_server,
  frame_demux,
  dump_se3,
  dump_reconstruction,
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
  image_property,
  marker_property,
  feature_render,
  reconstruction_result_markers,
  intrinsic_calibration,
  scene_calibration,
  contrail_render,
  load_parameter,
  store_parameter,
  action,
  mask_generator,
  gate,
  keypoint_to_float2_map,
  object_to_frame,
  unframe_image_fields,
};

using node_param_t = std::variant<std::string, std::int64_t, double, float, bool>;

struct node_display_property {
  std::string id;
  std::string label;
  std::string source_key;
  std::string target;
  std::string resource_kind;
  std::string selector;
  std::string format;
  std::int64_t order = 0;
  std::optional<node_param_t> default_value;
};

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
  std::vector<node_display_property> properties{};

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
    if (type == node_type::image_property) {
      return contains_param("camera_name") && !get_param<std::string>("camera_name").empty();
    }
    if (type == node_type::contrail_render) {
      return contains_param("camera_name") && !get_param<std::string>("camera_name").empty();
    }
    return false;
  }

  std::string get_camera_name() const {
    const auto type = get_type();
    if ((type == node_type::image_property ||
         type == node_type::contrail_render) &&
        contains_param("camera_name")) {
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
  std::vector<std::string> extends;     // Template subgraph names to inherit from
  std::vector<subgraph_def> subgraphs;  // Nested subgraph instances (for group templates)
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

  // Expand a single subgraph instance into a flat list of node_def.
  // prefix      : full prefix to apply to node names (= the fully-qualified
  //               subgraph instance name, e.g. "camera1" or "camera1_receiver").
  // outer_params: params inherited from an enclosing scope (group template
  //               or outer pipeline instance).
  std::vector<node_def> expand_sg(
      const subgraph_def& sg_instance, const std::string& prefix,
      const std::unordered_map<std::string, node_param_t>& outer_params = {}) const {
    // ── Case 1: instance extends a template ────────────────────────────────
    if (!sg_instance.extends.empty()) {
      const auto& template_name = sg_instance.extends.front();
      if (subgraph_templates.find(template_name) != subgraph_templates.end()) {
        const auto& sg_template = subgraph_templates.at(template_name);

        // Build the params that the template/instance contribute to children
        std::unordered_map<std::string, node_param_t> scope_params;
        for (const auto& [k, v] : outer_params) scope_params[k] = v;
        for (const auto& [k, v] : sg_template.params) scope_params[k] = v;
        for (const auto& [k, v] : sg_instance.params) scope_params[k] = v;

        // ── Case 1a: template contains nested subgraph instances ───────────
        // The template defines a self-contained name space; inner subgraph
        // names are NOT prefixed with the outer instance prefix.  The outer
        // prefix is only used by Case 2 (inline nested in a pipeline instance)
        // so that two instances of the same group template can coexist.
        if (!sg_template.subgraphs.empty()) {
          std::vector<node_def> result;
          for (const auto& nested : sg_template.subgraphs) {
            // Propagate scope_params (nested's own params take priority)
            subgraph_def nested_instance = nested;
            for (const auto& [k, v] : scope_params) {
              if (nested_instance.params.find(k) == nested_instance.params.end()) {
                nested_instance.params[k] = v;
              }
            }
            auto sub = expand_sg(nested_instance, nested.name, {});
            result.insert(result.end(), sub.begin(), sub.end());
          }
          return result;
        }

        // ── Case 1b: template contains flat nodes ─────────────────────────
        std::vector<node_def> sg_nodes = sg_template.nodes;

        std::unordered_set<std::string> local_names;
        for (const auto& n : sg_nodes) local_names.insert(n.name);

        for (auto& node : sg_nodes) {
          node.subgraph_instance = prefix;

          // Param hierarchy: outer < template-sg < instance < node
          std::unordered_map<std::string, node_param_t> merged;
          for (const auto& [k, v] : scope_params) merged[k] = v;
          for (const auto& [k, v] : node.params) merged[k] = v;
          node.params = merged;

          // Prefix node name
          if (node.name.find(prefix) == std::string::npos) {
            node.name = prefix + "_" + node.name;
          }

          // Prefix local input references
          for (auto& [input_key, source] : node.inputs) {
            size_t colon = source.find(':');
            std::string ref = colon != std::string::npos ? source.substr(0, colon) : source;
            std::string out = colon != std::string::npos ? source.substr(colon) : "";
            if (local_names.count(ref) > 0 && ref.find(prefix) == std::string::npos) {
              source = prefix + "_" + ref + out;
            }
          }
        }

        // Apply instance-level node overrides
        for (const auto& ov : sg_instance.nodes) {
          std::string target = ov.name;
          if (target.find(prefix) == std::string::npos) target = prefix + "_" + target;
          for (auto& node : sg_nodes) {
            if (node.name == target) {
              for (const auto& [k, v] : ov.params) node.params[k] = v;
              for (auto [k, v] : ov.inputs) {
                std::replace(v.begin(), v.end(), '.', '_');
                node.inputs[k] = v;
              }
              break;
            }
          }
        }

        return sg_nodes;
      }
    }

    // ── Case 2: instance has inline nested subgraphs ───────────────────────
    if (!sg_instance.subgraphs.empty()) {
      std::vector<node_def> result;
      for (const auto& nested : sg_instance.subgraphs) {
        subgraph_def nested_instance = nested;
        for (const auto& [k, v] : sg_instance.params) {
          if (nested_instance.params.find(k) == nested_instance.params.end()) {
            nested_instance.params[k] = v;
          }
        }
        auto sub = expand_sg(nested_instance, prefix + "_" + nested.name, {});
        result.insert(result.end(), sub.begin(), sub.end());
      }
      return result;
    }

    // ── Case 3: direct nodes (no extends, no nested subgraphs) ────────────
    // Apply '.' → '_' replacement on all input values (cross-subgraph references).
    auto direct_nodes = sg_instance.nodes;
    for (auto& node : direct_nodes) {
      for (auto& [input_key, source] : node.inputs) {
        std::replace(source.begin(), source.end(), '.', '_');
      }
    }
    return direct_nodes;
  }

 public:
  configuration(const std::string& path);

  void update();

  std::vector<node_def> get_nodes() const {
    const auto& pipeline_name = pipeline_names.at("pipeline");
    const auto& pipeline = pipelines.at(pipeline_name);
    std::vector<node_def> result;
    for (const auto& sg_instance : pipeline.subgraphs) {
      auto expanded = expand_sg(sg_instance, sg_instance.name);
      result.insert(result.end(), expanded.begin(), expanded.end());
    }
    return result;
  }

  bool has_subgraph(const std::string& name) const {
    return subgraph_templates.find(name) != subgraph_templates.end();
  }

  const subgraph_def& get_subgraph(const std::string& name) const {
    return subgraph_templates.at(name);
  }

  const pipeline_def& get_pipeline() const {
    return pipelines.at(pipeline_names.at("pipeline"));
  }
};
}  // namespace stargazer
