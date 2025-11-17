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
  } else if (type == "epipolar_reconstruction") {
    return node_type::epipolar_reconstruction;
  } else if (type == "pattern_board_calibration_target_detector") {
    return node_type::pattern_board_calibration_target_detector;
  } else if (type == "three_point_bar_calibration_target_detector") {
    return node_type::three_point_bar_calibration_target_detector;
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
    case node_type::epipolar_reconstruction:
      return "epipolar_reconstruction";
    case node_type::pattern_board_calibration_target_detector:
      return "pattern_board_calibration_target_detector";
    case node_type::three_point_bar_calibration_target_detector:
      return "three_point_bar_calibration_target_detector";
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

    pipeline_names.insert(std::make_pair("pipeline", j["pipeline"].get<std::string>()));
    if (j.contains("static_pipeline")) {
      pipeline_names.insert(
          std::make_pair("static_pipeline", j["static_pipeline"].get<std::string>()));
    }

    for (const auto& [pipeline_name, pipeline] : j["pipelines"].items()) {
      std::unordered_set<std::string> node_names;

      std::vector<node_info> node_infos;
      for (const auto& j_node : pipeline["nodes"]) {
        node_info node;
        for (const auto& [key, value] : j_node.items()) {
          if (key == "type") {
            node.type = get_node_type(value.get<std::string>());
          } else if (key == "name") {
            node.name = value.get<std::string>();
          } else if (key == "inputs") {
            node.inputs = value.get<std::unordered_map<std::string, std::string>>();
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
        node_infos.push_back(node);

        if (node_names.find(node.name) != node_names.end()) {
          throw std::runtime_error("Duplicate node name");
        }
        node_names.insert(node.name);
      }

      pipeline_nodes.insert(std::make_pair(pipeline_name, node_infos));
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

  for (const auto& [pipeline_name, nodes] : pipeline_nodes) {
    std::vector<nlohmann::json> j_nodes;
    for (const auto& node : nodes) {
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
      if (node.extends.size() > 0) {
        std::vector<std::string> extends;
        for (const auto& extend : node.extends) {
          extends.push_back(extend->name);
        }
        j_node["extends"] = extends;
      }
      j_nodes.push_back(j_node);
    }

    j["pipelines"][pipeline_name]["nodes"] = j_nodes;
  }

  for (const auto& [pipeline, pipeline_name] : pipeline_names) {
    j[pipeline] = pipeline_name;
  }

  std::ofstream ofs;
  ofs.open(path, std::ios::out);
  ofs << j.dump(2);
}
}  // namespace stargazer
