#include "config_file.hpp"

#include <fstream>
#include <nlohmann/json.hpp>
#include <unordered_set>

namespace stargazer {
configuration_file::configuration_file(const std::string &path) : path(path) {
  std::ifstream ifs;
  ifs.open(path, std::ios::in);

  if (ifs) {
    const auto j = nlohmann::json::parse(ifs);

    pipeline_names.insert(std::make_pair("pipeline", j["pipeline"].get<std::string>()));
    if (j.contains("static_pipeline")) {
      pipeline_names.insert(
          std::make_pair("static_pipeline", j["static_pipeline"].get<std::string>()));
    }

    for (const auto &[pipeline_name, pipeline] : j["pipelines"].items()) {
      std::unordered_set<std::string> node_names;

      std::vector<node_info> node_infos;
      for (const auto &j_node : pipeline["nodes"]) {
        node_info node;
        const auto type = j_node["type"].get<std::string>();
        if (type == "raspi") {
          node.type = node_type::raspi;
        } else if (type == "raspi_color") {
          node.type = node_type::raspi_color;
        } else if (type == "rs_d435") {
          node.type = node_type::rs_d435;
        } else if (type == "rs_d435_color") {
          node.type = node_type::rs_d435_color;
        } else if (type == "depthai_color") {
          node.type = node_type::depthai_color;
        } else if (type == "raspi_playback") {
          node.type = node_type::raspi_playback;
        } else if (type == "panoptic") {
          node.type = node_type::panoptic;
        } else if (type == "record") {
          node.type = node_type::record;
        } else if (type == "calibration") {
          node.type = node_type::calibration;
        } else if (type == "pattern_board_calibration_target_detector") {
          node.type = node_type::pattern_board_calibration_target_detector;
        } else if (type == "three_point_bar_calibration_target_detector") {
          node.type = node_type::three_point_bar_calibration_target_detector;
        } else {
          throw std::runtime_error("Invalid node type");
        }

        node.name = j_node["name"].get<std::string>();
        if (type == "raspi_playback") {
          node.id = j_node["id"].get<std::string>();
          node.db_path = j_node["db_path"].get<std::string>();
          node.name = j_node["name"].get<std::string>();
        } else if (type == "panoptic") {
          node.id = j_node["id"].get<std::string>();
          node.db_path = j_node["db_path"].get<std::string>();
          node.name = j_node["name"].get<std::string>();
        } else if (type == "record") {
          node.db_path = j_node["db_path"].get<std::string>();
          node.name = j_node["name"].get<std::string>();
        } else if (type == "calibration") {
        } else if (type == "pattern_board_calibration_target_detector") {
        } else if (type == "three_point_bar_calibration_target_detector") {
        } else {
          node.id = j_node["id"].get<std::string>();
          node.address = j_node["address"].get<std::string>();
          node.endpoint = j_node["gateway"].get<std::string>();
        }
        if (j_node.contains("params")) {
          for (const auto &[key, value] : j_node["params"].items()) {
            if (value.is_number()) {
              node.params.insert(std::make_pair(key, value.get<float>()));
            } else if (value.is_boolean()) {
              node.params.insert(std::make_pair(key, value.get<bool>()));
            }
          }
        }
        if (j_node.contains("inputs")) {
          node.inputs = j_node["inputs"].get<std::unordered_map<std::string, std::string>>();
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

void configuration_file::update() {
  nlohmann::json j;
  for (const auto &[pipeline_name, nodes] : pipeline_nodes) {
    std::vector<nlohmann::json> j_nodes;
    for (const auto &node : nodes) {
      nlohmann::json j_node;

      std::string node_type_name;
      switch (node.type) {
        case node_type::raspi:
          node_type_name = "raspi";
          break;
        case node_type::raspi_color:
          node_type_name = "raspi_color";
          break;
        case node_type::rs_d435:
          node_type_name = "rs_d435";
          break;
        case node_type::rs_d435_color:
          node_type_name = "rs_d435_color";
          break;
        case node_type::depthai_color:
          node_type_name = "depthai_color";
          break;
        case node_type::raspi_playback:
          node_type_name = "raspi_playback";
          break;
        case node_type::panoptic:
          node_type_name = "panoptic";
          break;
        case node_type::record:
          node_type_name = "record";
          break;
        case node_type::calibration:
          node_type_name = "calibration";
          break;
        case node_type::pattern_board_calibration_target_detector:
          node_type_name = "pattern_board_calibration_target_detector";
          break;
        case node_type::three_point_bar_calibration_target_detector:
          node_type_name = "three_point_bar_calibration_target_detector";
          break;
        default:
          throw std::runtime_error("Invalid node type");
      }

      j_node["type"] = node_type_name;
      j_node["name"] = node.name;
      nlohmann::json j_params;
      for (const auto &[key, value] : node.params) {
        if (std::holds_alternative<float>(value)) {
          j_params[key] = std::get<float>(value);
        } else if (std::holds_alternative<bool>(value)) {
          j_params[key] = std::get<bool>(value);
        }
      }
      j_node["params"] = j_params;
      if (node.type == node_type::raspi_playback) {
        j_node["id"] = node.id;
        j_node["db_path"] = node.db_path;
      } else if (node.type == node_type::panoptic) {
        j_node["id"] = node.id;
        j_node["db_path"] = node.db_path;
      } else if (node.type == node_type::record) {
        j_node["db_path"] = node.db_path;
      } else if (node.type == node_type::calibration) {
      } else if (node.type == node_type::pattern_board_calibration_target_detector) {
      } else if (node.type == node_type::three_point_bar_calibration_target_detector) {
      } else {
        j_node["id"] = node.id;
        j_node["address"] = node.address;
        j_node["gateway"] = node.endpoint;
      }
      j_nodes.push_back(j_node);
    }

    j["pipelines"][pipeline_name] = j_nodes;
  }

  for (const auto &[pipeline, pipeline_name] : pipeline_names) {
    j[pipeline] = pipeline_name;
  }

  std::ofstream ofs;
  ofs.open(path, std::ios::out);
  ofs << j.dump(2);
}
}  // namespace stargazer
