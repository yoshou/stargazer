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
  } else if (type == "calibration") {
    return node_type::calibration;
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
  } else if (type == "callback") {
    return node_type::callback;
  } else if (type == "grpc_server") {
    return node_type::grpc_server;
  } else if (type == "frame_demux") {
    return node_type::frame_demux;
  } else if (type == "dump_se3") {
    return node_type::dump_se3;
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
  }
  throw std::runtime_error("Invalid node type");
}

static std::string get_node_type_name(node_type type) {
  switch (type) {
    case node_type::unknown:
      throw std::runtime_error("Invalid node type");
    case node_type::record:
      return "record";
    case node_type::calibration:
      return "calibration";
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
    case node_type::callback:
      return "callback";
    case node_type::grpc_server:
      return "grpc_server";
    case node_type::frame_demux:
      return "frame_demux";
    case node_type::dump_se3:
      return "dump_se3";
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

configuration::configuration(const std::string& path) : path(path) {
  std::ifstream ifs;
  ifs.open(path, std::ios::in);

  if (ifs) {
    const auto j = nlohmann::json::parse(ifs);

    if (j.contains("nodes")) {
      for (const auto& j_node : j["nodes"]) {
        auto node = std::make_shared<node_info>();
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
            node_info node;
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

        // Subgraph-level parameters (e.g., db_path, fps, etc.)
        for (const auto& [key, value] : j_subgraph.items()) {
          if (key != "name" && key != "nodes" && key != "outputs" && key != "extends") {
            subgraph.params[key] = json_to_param(value);
          }
        }

        subgraph_templates[subgraph.name] = subgraph;
      }
    }

    pipeline_names.insert(std::make_pair("pipeline", j["pipeline"].get<std::string>()));
    if (j.contains("static_pipeline")) {
      pipeline_names.insert(
          std::make_pair("static_pipeline", j["static_pipeline"].get<std::string>()));
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
            if (key != "name" && key != "extends" && key != "nodes" && key != "outputs") {
              sg_instance.params[key] = json_to_param(value);
            }
          }

          // Instance can also have nodes directly defined
          if (j_sg.contains("nodes")) {
            for (const auto& j_node : j_sg["nodes"]) {
              node_info node;
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
        j_nodes.push_back(j_node);
      }
      j_subgraph["nodes"] = j_nodes;

      if (sg.outputs.size() > 0) {
        j_subgraph["outputs"] = sg.outputs;
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
            j_nodes.push_back(j_node);
          }
          j_sg["nodes"] = j_nodes;
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
