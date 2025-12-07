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
  if (type == "raspi") {
    return node_type::raspi;
  } else if (type == "raspi_color") {
    return node_type::raspi_color;
  } else if (type == "rs_d435") {
    return node_type::rs_d435;
  } else if (type == "rs_d435_color") {
    return node_type::rs_d435_color;
  } else if (type == "depthai_color") {
    return node_type::depthai_color;
  } else if (type == "playback") {
    return node_type::playback;
  } else if (type == "panoptic") {
    return node_type::panoptic;
  } else if (type == "record") {
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
  }
  throw std::runtime_error("Invalid node type");
}

static std::string get_node_type_name(node_type type) {
  switch (type) {
    case node_type::unknown:
      throw std::runtime_error("Invalid node type");
    case node_type::raspi:
      return "raspi";
    case node_type::raspi_color:
      return "raspi_color";
    case node_type::rs_d435:
      return "rs_d435";
    case node_type::rs_d435_color:
      return "rs_d435_color";
    case node_type::depthai_color:
      return "depthai_color";
    case node_type::playback:
      return "playback";
    case node_type::panoptic:
      return "panoptic";
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
  }
  throw std::runtime_error("Invalid node type");
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
            node->params[key] = value.get<node_param_t>();
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
                node.params[key] = value.get<node_param_t>();
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
            subgraph.params[key] = value.get<node_param_t>();
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
              sg_instance.params[key] = value.get<node_param_t>();
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
                  node.params[key] = value.get<node_param_t>();
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
