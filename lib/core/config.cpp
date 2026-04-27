#include "config.hpp"

#include <fstream>
#include <nlohmann/json.hpp>
#include <unordered_set>

namespace nlohmann {
template <typename... Ts>
struct adl_serializer<std::variant<Ts...>> {
  static void to_json(nlohmann::json& j, const std::variant<Ts...>& data) {
    std::visit([&j](const auto& v) { j = v; }, data);
  }
  static void from_json(const nlohmann::json& j, std::variant<Ts...>& data) {
    if (j.is_string()) {
      data = j.get<std::string>();
    } else if (j.is_number_integer()) {
      data = j.get<std::int64_t>();
    } else if (j.is_number_float()) {
      data = j.get<float>();
    } else if (j.is_boolean()) {
      data = j.get<bool>();
    } else {
      throw std::runtime_error("Invalid variant type");
    }
  }
};
}  // namespace nlohmann

namespace stargazer {
static node_type get_node_type(const std::string& type) {
  if (type == "record") {
    return node_type::record;
  } else if (type == "extrinsic_calibration") {
    return node_type::extrinsic_calibration;
  } else if (type == "voxelpose_reconstruction") {
    return node_type::voxelpose_reconstruction;
  } else if (type == "mvpose_reconstruction") {
    return node_type::mvpose_reconstruction;
  } else if (type == "mvp_reconstruction") {
    return node_type::mvp_reconstruction;
  } else if (type == "epipolar_reconstruction") {
    return node_type::epipolar_reconstruction;
  } else if (type == "pattern_board_calibration_target_detector") {
    return node_type::pattern_board_calibration_target_detector;
  } else if (type == "three_point_bar_calibration_target_detector") {
    return node_type::three_point_bar_calibration_target_detector;
  } else if (type == "approximate_time_sync") {
    return node_type::approximate_time_sync;
  } else if (type == "frame_number_numbering") {
    return node_type::frame_number_numbering;
  } else if (type == "parallel_queue") {
    return node_type::parallel_queue;
  } else if (type == "frame_number_ordering") {
    return node_type::frame_number_ordering;
  } else if (type == "grpc_server") {
    return node_type::grpc_server;
  } else if (type == "frame_demux") {
    return node_type::frame_demux;
  } else if (type == "dump_se3") {
    return node_type::dump_se3;
  } else if (type == "dump_reconstruction") {
    return node_type::dump_reconstruction;
  } else if (type == "libcamera_capture") {
    return node_type::libcamera_capture;
  } else if (type == "timestamp") {
    return node_type::timestamp;
  } else if (type == "broadcast_talker") {
    return node_type::broadcast_talker;
  } else if (type == "broadcast_listener") {
    return node_type::broadcast_listener;
  } else if (type == "encode_jpeg") {
    return node_type::encode_jpeg;
  } else if (type == "decode_jpeg") {
    return node_type::decode_jpeg;
  } else if (type == "scale") {
    return node_type::scale;
  } else if (type == "resize") {
    return node_type::resize;
  } else if (type == "gaussian_blur") {
    return node_type::gaussian_blur;
  } else if (type == "mask") {
    return node_type::mask;
  } else if (type == "p2p_tcp_talker") {
    return node_type::p2p_tcp_talker;
  } else if (type == "p2p_tcp_listener") {
    return node_type::p2p_tcp_listener;
  } else if (type == "fifo") {
    return node_type::fifo;
  } else if (type == "video_time_sync_control") {
    return node_type::video_time_sync_control;
  } else if (type == "fast_blob_detector") {
    return node_type::fast_blob_detector;
  } else if (type == "detect_circle_grid") {
    return node_type::detect_circle_grid;
  } else if (type == "load_blob") {
    return node_type::load_blob;
  } else if (type == "load_marker") {
    return node_type::load_marker;
  } else if (type == "load_panoptic") {
    return node_type::load_panoptic;
  } else if (type == "charuco_detector") {
    return node_type::charuco_detector;
  } else if (type == "depthai_color_camera") {
    return node_type::depthai_color_camera;
  } else if (type == "rs_d435") {
    return node_type::rs_d435;
  } else if (type == "object_map") {
    return node_type::object_map;
  } else if (type == "object_mux") {
    return node_type::object_mux;
  } else if (type == "image_property") {
    return node_type::image_property;
  } else if (type == "marker_property") {
    return node_type::marker_property;
  } else if (type == "feature_render") {
    return node_type::feature_render;
  } else if (type == "reconstruction_result_markers") {
    return node_type::reconstruction_result_markers;
  } else if (type == "intrinsic_calibration") {
    return node_type::intrinsic_calibration;
  } else if (type == "scene_calibration") {
    return node_type::scene_calibration;
  } else if (type == "contrail_render") {
    return node_type::contrail_render;
  } else if (type == "load_parameter") {
    return node_type::load_parameter;
  } else if (type == "store_parameter") {
    return node_type::store_parameter;
  } else if (type == "action") {
    return node_type::action;
  } else if (type == "mask_generator") {
    return node_type::mask_generator;
  } else if (type == "gate") {
    return node_type::gate;
  } else if (type == "keypoint_to_float2_map") {
    return node_type::keypoint_to_float2_map;
  } else if (type == "object_to_frame") {
    return node_type::object_to_frame;
  } else if (type == "unframe_image_fields") {
    return node_type::unframe_image_fields;
  } else if (type == "dust3r_pose_estimation") {
    return node_type::dust3r_pose_estimation;
  } else if (type == "dust3r_calibration") {
    return node_type::dust3r_calibration;
  } else if (type == "mast3r_calibration") {
    return node_type::mast3r_calibration;
  }
  throw std::runtime_error("Invalid node type");
}

std::string get_node_type_name(node_type type) {
  switch (type) {
    case node_type::unknown:
      throw std::runtime_error("Invalid node type");
    case node_type::record:
      return "record";
    case node_type::extrinsic_calibration:
      return "extrinsic_calibration";
    case node_type::voxelpose_reconstruction:
      return "voxelpose_reconstruction";
    case node_type::mvpose_reconstruction:
      return "mvpose_reconstruction";
    case node_type::mvp_reconstruction:
      return "mvp_reconstruction";
    case node_type::epipolar_reconstruction:
      return "epipolar_reconstruction";
    case node_type::pattern_board_calibration_target_detector:
      return "pattern_board_calibration_target_detector";
    case node_type::three_point_bar_calibration_target_detector:
      return "three_point_bar_calibration_target_detector";
    case node_type::approximate_time_sync:
      return "approximate_time_sync";
    case node_type::frame_number_numbering:
      return "frame_number_numbering";
    case node_type::parallel_queue:
      return "parallel_queue";
    case node_type::frame_number_ordering:
      return "frame_number_ordering";
    case node_type::grpc_server:
      return "grpc_server";
    case node_type::frame_demux:
      return "frame_demux";
    case node_type::dump_se3:
      return "dump_se3";
    case node_type::dump_reconstruction:
      return "dump_reconstruction";
    case node_type::libcamera_capture:
      return "libcamera_capture";
    case node_type::timestamp:
      return "timestamp";
    case node_type::broadcast_talker:
      return "broadcast_talker";
    case node_type::broadcast_listener:
      return "broadcast_listener";
    case node_type::encode_jpeg:
      return "encode_jpeg";
    case node_type::decode_jpeg:
      return "decode_jpeg";
    case node_type::scale:
      return "scale";
    case node_type::resize:
      return "resize";
    case node_type::gaussian_blur:
      return "gaussian_blur";
    case node_type::mask:
      return "mask";
    case node_type::p2p_tcp_talker:
      return "p2p_tcp_talker";
    case node_type::p2p_tcp_listener:
      return "p2p_tcp_listener";
    case node_type::fifo:
      return "fifo";
    case node_type::video_time_sync_control:
      return "video_time_sync_control";
    case node_type::fast_blob_detector:
      return "fast_blob_detector";
    case node_type::detect_circle_grid:
      return "detect_circle_grid";
    case node_type::load_blob:
      return "load_blob";
    case node_type::load_marker:
      return "load_marker";
    case node_type::load_panoptic:
      return "load_panoptic";
    case node_type::charuco_detector:
      return "charuco_detector";
    case node_type::depthai_color_camera:
      return "depthai_color_camera";
    case node_type::rs_d435:
      return "rs_d435";
    case node_type::object_map:
      return "object_map";
    case node_type::object_mux:
      return "object_mux";
    case node_type::image_property:
      return "image_property";
    case node_type::marker_property:
      return "marker_property";
    case node_type::feature_render:
      return "feature_render";
    case node_type::reconstruction_result_markers:
      return "reconstruction_result_markers";
    case node_type::intrinsic_calibration:
      return "intrinsic_calibration";
    case node_type::scene_calibration:
      return "scene_calibration";
    case node_type::contrail_render:
      return "contrail_render";
    case node_type::load_parameter:
      return "load_parameter";
    case node_type::store_parameter:
      return "store_parameter";
    case node_type::action:
      return "action";
    case node_type::mask_generator:
      return "mask_generator";
    case node_type::gate:
      return "gate";
    case node_type::keypoint_to_float2_map:
      return "keypoint_to_float2_map";
    case node_type::object_to_frame:
      return "object_to_frame";
    case node_type::unframe_image_fields:
      return "unframe_image_fields";
    case node_type::dust3r_pose_estimation:
      return "dust3r_pose_estimation";
    case node_type::dust3r_calibration:
      return "dust3r_calibration";
    case node_type::mast3r_calibration:
      return "mast3r_calibration";
  }
  throw std::runtime_error("Invalid node type");
}

static node_param_t json_to_param(const nlohmann::json& value) {
  if (value.is_string()) {
    return value.get<std::string>();
  } else if (value.is_boolean()) {
    return value.get<bool>();
  } else if (value.is_number_integer()) {
    return value.get<std::int64_t>();
  } else if (value.is_number_float()) {
    return value.get<double>();
  }
  throw std::runtime_error("Unsupported JSON parameter type");
}

static node_display_property json_to_display_property(const nlohmann::json& value,
                                                      std::int64_t order) {
  if (!value.is_object()) {
    throw std::runtime_error("Node property must be an object");
  }

  node_display_property property;
  property.order = order;

  if (value.contains("id")) {
    property.id = value["id"].get<std::string>();
  }
  if (value.contains("label")) {
    property.label = value["label"].get<std::string>();
  }
  if (value.contains("source_key")) {
    property.source_key = value["source_key"].get<std::string>();
  }
  if (value.contains("target")) {
    property.target = value["target"].get<std::string>();
  }
  if (value.contains("resource_kind")) {
    property.resource_kind = value["resource_kind"].get<std::string>();
  }
  if (value.contains("format")) {
    property.format = value["format"].get<std::string>();
  }
  if (value.contains("order")) {
    property.order = value["order"].get<std::int64_t>();
  }
  if (value.contains("default_value")) {
    property.default_value = json_to_param(value["default_value"]);
  }

  if (property.id.empty()) {
    property.id = property.source_key;
  }
  if (property.label.empty()) {
    property.label = property.id;
  }
  if (property.source_key.empty()) {
    property.source_key = property.id;
  }

  return property;
}

static nlohmann::json display_property_to_json(const node_display_property& property) {
  nlohmann::json j_property;
  j_property["id"] = property.id;
  j_property["label"] = property.label;
  j_property["source_key"] = property.source_key;
  if (!property.target.empty()) {
    j_property["target"] = property.target;
  }
  if (!property.resource_kind.empty()) {
    j_property["resource_kind"] = property.resource_kind;
  }
  if (!property.format.empty()) {
    j_property["format"] = property.format;
  }
  j_property["order"] = property.order;
  if (property.default_value.has_value()) {
    j_property["default_value"] = property.default_value.value();
  }
  return j_property;
}

configuration::configuration(const std::string& path) : path(path) {
  std::ifstream ifs;
  ifs.open(path, std::ios::in);

  if (ifs) {
    const auto j = nlohmann::json::parse(ifs);

    if (j.contains("nodes")) {
      for (const auto& j_node : j["nodes"]) {
        auto node = std::make_shared<node_def>();
        for (const auto& [key, value] : j_node.items()) {
          if (key == "type") {
            node->type = get_node_type(value.get<std::string>());
          } else if (key == "name") {
            node->name = value.get<std::string>();
          } else if (key == "inputs") {
            node->inputs = value.get<std::unordered_map<std::string, std::string>>();
          } else if (key == "outputs") {
            node->outputs = value.get<std::vector<std::string>>();
          } else if (key == "extends") {
            const auto extends = value.get<std::vector<std::string>>();
            for (const auto& extend : extends) {
              if (nodes.find(extend) == nodes.end()) {
                throw std::runtime_error("Node not found: " + extend);
              }
              node->extends.push_back(nodes.at(extend));
            }
          } else if (key == "properties") {
            std::int64_t property_index = 0;
            for (const auto& j_property : value) {
              node->properties.push_back(json_to_display_property(j_property, property_index++));
            }
          } else {
            node->params[key] = json_to_param(value);
          }
        }
        if (nodes.find(node->name) != nodes.end()) {
          throw std::runtime_error("Duplicate node name");
        }
        nodes[node->name] = node;
      }
    }

    if (j.contains("subgraphs")) {
      for (const auto& j_subgraph : j["subgraphs"]) {
        subgraph_def subgraph;

        if (!j_subgraph.contains("name")) {
          throw std::runtime_error("Subgraph must have a name");
        }
        subgraph.name = j_subgraph["name"].get<std::string>();

        if (j_subgraph.contains("nodes")) {
          for (const auto& j_node : j_subgraph["nodes"]) {
            node_def node;
            for (const auto& [key, value] : j_node.items()) {
              if (key == "type") {
                node.type = get_node_type(value.get<std::string>());
              } else if (key == "name") {
                node.name = value.get<std::string>();
              } else if (key == "inputs") {
                node.inputs = value.get<std::unordered_map<std::string, std::string>>();
              } else if (key == "outputs") {
                node.outputs = value.get<std::vector<std::string>>();
              } else if (key == "extends") {
                const auto extends = value.get<std::vector<std::string>>();
                for (const auto& extend : extends) {
                  if (nodes.find(extend) == nodes.end()) {
                    throw std::runtime_error("Node not found: " + extend);
                  }
                  node.extends.push_back(nodes.at(extend));
                }
              } else if (key == "properties") {
                std::int64_t property_index = 0;
                for (const auto& j_property : value) {
                  node.properties.push_back(json_to_display_property(j_property, property_index++));
                }
              } else {
                node.params[key] = json_to_param(value);
              }
            }
            subgraph.nodes.push_back(node);
          }
        }

        if (j_subgraph.contains("outputs")) {
          subgraph.outputs = j_subgraph["outputs"].get<std::vector<std::string>>();
        }

        // Nested subgraph instances within this template
        if (j_subgraph.contains("subgraphs")) {
          for (const auto& j_nested : j_subgraph["subgraphs"]) {
            subgraph_def nested;
            if (j_nested.contains("name")) nested.name = j_nested["name"].get<std::string>();
            if (j_nested.contains("extends"))
              nested.extends = j_nested["extends"].get<std::vector<std::string>>();
            for (const auto& [key, value] : j_nested.items()) {
              if (key != "name" && key != "extends" && key != "nodes" && key != "outputs" &&
                  key != "subgraphs") {
                nested.params[key] = json_to_param(value);
              }
            }
            if (j_nested.contains("nodes")) {
              for (const auto& j_node : j_nested["nodes"]) {
                node_def node;
                for (const auto& [key, value] : j_node.items()) {
                  if (key == "type") {
                    node.type = get_node_type(value.get<std::string>());
                  } else if (key == "name") {
                    node.name = value.get<std::string>();
                  } else if (key == "inputs") {
                    node.inputs = value.get<std::unordered_map<std::string, std::string>>();
                  } else if (key == "outputs") {
                    node.outputs = value.get<std::vector<std::string>>();
                  } else {
                    node.params[key] = json_to_param(value);
                  }
                }
                nested.nodes.push_back(node);
              }
            }
            subgraph.subgraphs.push_back(nested);
          }
        }

        // Subgraph-level parameters (e.g., db_path, fps, etc.)
        for (const auto& [key, value] : j_subgraph.items()) {
          if (key != "name" && key != "nodes" && key != "outputs" && key != "extends" &&
              key != "subgraphs") {
            subgraph.params[key] = json_to_param(value);
          }
        }

        subgraph_templates[subgraph.name] = subgraph;
      }
    }

    for (const auto& [key, value] : j.items()) {
      if (!value.is_string()) {
        continue;
      }
      if (key == "pipeline") {
        pipeline_names.insert(std::make_pair(key, value.get<std::string>()));
      }
    }

    for (const auto& [pipeline_name, pipeline_json] : j["pipelines"].items()) {
      pipeline_def pipeline;
      pipeline.name = pipeline_name;

      // Parse subgraphs array in pipeline
      if (pipeline_json.contains("subgraphs")) {
        for (const auto& j_sg : pipeline_json["subgraphs"]) {
          subgraph_def sg_instance;

          // Subgraph instance must have a name
          if (j_sg.contains("name")) {
            sg_instance.name = j_sg["name"].get<std::string>();
          }

          // Check if it extends a template
          if (j_sg.contains("extends")) {
            sg_instance.extends = j_sg["extends"].get<std::vector<std::string>>();
          }

          // Instance-specific parameters (override template params)
          for (const auto& [key, value] : j_sg.items()) {
            if (key != "name" && key != "extends" && key != "nodes" && key != "outputs" &&
                key != "subgraphs") {
              sg_instance.params[key] = json_to_param(value);
            }
          }

          // Nested subgraph instances within this pipeline instance
          if (j_sg.contains("subgraphs")) {
            for (const auto& j_nested : j_sg["subgraphs"]) {
              subgraph_def nested;
              if (j_nested.contains("name")) nested.name = j_nested["name"].get<std::string>();
              if (j_nested.contains("extends"))
                nested.extends = j_nested["extends"].get<std::vector<std::string>>();
              for (const auto& [key, value] : j_nested.items()) {
                if (key != "name" && key != "extends" && key != "nodes" && key != "outputs" &&
                    key != "subgraphs") {
                  nested.params[key] = json_to_param(value);
                }
              }
              sg_instance.subgraphs.push_back(nested);
            }
          }

          // Instance can also have nodes directly defined
          if (j_sg.contains("nodes")) {
            for (const auto& j_node : j_sg["nodes"]) {
              node_def node;
              for (const auto& [key, value] : j_node.items()) {
                if (key == "type") {
                  node.type = get_node_type(value.get<std::string>());
                } else if (key == "name") {
                  node.name = value.get<std::string>();
                } else if (key == "inputs") {
                  node.inputs = value.get<std::unordered_map<std::string, std::string>>();
                } else if (key == "outputs") {
                  node.outputs = value.get<std::vector<std::string>>();
                } else if (key == "extends") {
                  const auto extends = value.get<std::vector<std::string>>();
                  for (const auto& extend : extends) {
                    if (nodes.find(extend) == nodes.end()) {
                      throw std::runtime_error("Node not found: " + extend);
                    }
                    node.extends.push_back(nodes.at(extend));
                  }
                } else if (key == "properties") {
                  std::int64_t property_index = 0;
                  for (const auto& j_property : value) {
                    node.properties.push_back(
                        json_to_display_property(j_property, property_index++));
                  }
                } else {
                  node.params[key] = json_to_param(value);
                }
              }
              sg_instance.nodes.push_back(node);
            }
          }

          pipeline.subgraphs.push_back(sg_instance);
        }
      }

      pipelines[pipeline_name] = pipeline;
    }
  }
}

void configuration::update() {
  nlohmann::json j;

  {
    std::vector<nlohmann::json> j_nodes;
    for (const auto& [name, node] : nodes) {
      nlohmann::json j_node;
      if (node->type != node_type::unknown) {
        j_node["type"] = get_node_type_name(node->type);
      }
      j_node["name"] = node->name;
      if (node->inputs.size() > 0) {
        j_node["inputs"] = node->inputs;
      }
      for (const auto& [key, value] : node->params) {
        j_node[key] = value;
      }
      if (!node->properties.empty()) {
        std::vector<nlohmann::json> j_properties;
        for (const auto& property : node->properties) {
          j_properties.push_back(display_property_to_json(property));
        }
        j_node["properties"] = j_properties;
      }
      if (node->extends.size() > 0) {
        std::vector<std::string> extends;
        for (const auto& extend : node->extends) {
          extends.push_back(extend->name);
        }
        j_node["extends"] = extends;
      }
      j_nodes.push_back(j_node);
    }
    j["nodes"] = j_nodes;
  }

  // Write subgraph templates
  {
    std::vector<nlohmann::json> j_subgraphs;
    for (const auto& [sg_name, sg] : subgraph_templates) {
      nlohmann::json j_subgraph;
      j_subgraph["name"] = sg.name;

      std::vector<nlohmann::json> j_nodes;
      for (const auto& node : sg.nodes) {
        nlohmann::json j_node;
        if (node.type != node_type::unknown) {
          j_node["type"] = get_node_type_name(node.type);
        }
        j_node["name"] = node.name;
        if (node.inputs.size() > 0) {
          j_node["inputs"] = node.inputs;
        }
        if (node.outputs.size() > 0) {
          j_node["outputs"] = node.outputs;
        }
        for (const auto& [key, value] : node.params) {
          j_node[key] = value;
        }
        if (!node.properties.empty()) {
          std::vector<nlohmann::json> j_properties;
          for (const auto& property : node.properties) {
            j_properties.push_back(display_property_to_json(property));
          }
          j_node["properties"] = j_properties;
        }
        j_nodes.push_back(j_node);
      }
      j_subgraph["nodes"] = j_nodes;

      if (sg.outputs.size() > 0) {
        j_subgraph["outputs"] = sg.outputs;
      }

      if (!sg.subgraphs.empty()) {
        std::vector<nlohmann::json> j_nested_sgs;
        for (const auto& nested : sg.subgraphs) {
          nlohmann::json j_nested;
          j_nested["name"] = nested.name;
          if (!nested.extends.empty()) j_nested["extends"] = nested.extends;
          for (const auto& [k, v] : nested.params) j_nested[k] = v;
          if (!nested.nodes.empty()) {
            std::vector<nlohmann::json> j_nodes;
            for (const auto& node : nested.nodes) {
              nlohmann::json j_node;
              if (node.type != node_type::unknown) j_node["type"] = get_node_type_name(node.type);
              j_node["name"] = node.name;
              if (!node.inputs.empty()) j_node["inputs"] = node.inputs;
              if (!node.outputs.empty()) j_node["outputs"] = node.outputs;
              for (const auto& [k, v] : node.params) j_node[k] = v;
              j_nodes.push_back(j_node);
            }
            j_nested["nodes"] = j_nodes;
          }
          j_nested_sgs.push_back(j_nested);
        }
        j_subgraph["subgraphs"] = j_nested_sgs;
      }

      for (const auto& [key, value] : sg.params) {
        j_subgraph[key] = value;
      }

      j_subgraphs.push_back(j_subgraph);
    }
    if (j_subgraphs.size() > 0) {
      j["subgraphs"] = j_subgraphs;
    }
  }

  // Write pipelines
  for (const auto& [pipeline_name, pipeline] : pipelines) {
    {
      std::vector<nlohmann::json> j_subgraphs;
      for (const auto& sg : pipeline.subgraphs) {
        nlohmann::json j_sg;
        j_sg["name"] = sg.name;

        if (sg.extends.size() > 0) {
          j_sg["extends"] = sg.extends;
        }

        for (const auto& [key, value] : sg.params) {
          j_sg[key] = value;
        }

        if (sg.nodes.size() > 0) {
          std::vector<nlohmann::json> j_nodes;
          for (const auto& node : sg.nodes) {
            nlohmann::json j_node;
            if (node.type != node_type::unknown) {
              j_node["type"] = get_node_type_name(node.type);
            }
            j_node["name"] = node.name;
            if (node.inputs.size() > 0) {
              j_node["inputs"] = node.inputs;
            }
            for (const auto& [key, value] : node.params) {
              j_node[key] = value;
            }
            if (!node.properties.empty()) {
              std::vector<nlohmann::json> j_properties;
              for (const auto& property : node.properties) {
                j_properties.push_back(display_property_to_json(property));
              }
              j_node["properties"] = j_properties;
            }
            j_nodes.push_back(j_node);
          }
          j_sg["nodes"] = j_nodes;
        }

        if (!sg.subgraphs.empty()) {
          std::vector<nlohmann::json> j_nested_sgs;
          for (const auto& nested : sg.subgraphs) {
            nlohmann::json j_nested;
            j_nested["name"] = nested.name;
            if (!nested.extends.empty()) j_nested["extends"] = nested.extends;
            for (const auto& [k, v] : nested.params) j_nested[k] = v;
            j_nested_sgs.push_back(j_nested);
          }
          j_sg["subgraphs"] = j_nested_sgs;
        }

        j_subgraphs.push_back(j_sg);
      }
      j["pipelines"][pipeline_name]["subgraphs"] = j_subgraphs;
    }
  }

  for (const auto& [pipeline, pipeline_name] : pipeline_names) {
    j[pipeline] = pipeline_name;
  }

  std::ofstream ofs;
  ofs.open(path, std::ios::out);
  ofs << j.dump(2);
}
}  // namespace stargazer
