#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "config.hpp"

namespace stargazer {

enum class config_tree_item_kind {
  pipeline,
  subgraph,
  node,
  detail,
};

struct config_tree_ref {
  std::string pipeline_key;
  std::string subgraph_instance;
  std::string node_name;
};

struct config_tree_runtime_status {
  bool is_streaming = false;
  size_t metric_value = 0;
};

struct runtime_node_property {
  std::string key;
  std::string value;
  bool read_only = true;
};

struct runtime_node_action {
  std::string id;
  std::string label;
  bool enabled = true;
};

struct runtime_node_handle {
  std::string stable_id;
  config_tree_ref ref;
  std::string label;
  std::string summary;
  bool is_camera = false;
  config_tree_runtime_status status;
  std::vector<std::string> badges;
  std::vector<runtime_node_property> properties;
  std::vector<runtime_node_action> actions;
};

struct config_tree_item {
  std::string stable_id;
  config_tree_item_kind kind = config_tree_item_kind::detail;
  std::string label;
  std::string summary;
  std::string runtime_node_id;
  std::vector<std::string> badges;
  std::vector<config_tree_item> children;
};

struct config_tree_render_options {
  bool allow_node_selection = true;
  bool show_runtime_summary = true;
  bool show_detail_rows = true;
};

struct config_tree_model {
  std::vector<config_tree_item> roots;
  std::unordered_map<std::string, runtime_node_handle> runtime_nodes;
  std::vector<std::string> camera_node_ids;
};

config_tree_model build_config_tree(const configuration& config,
                                    const std::string& pipeline_key = "pipeline");

}  // namespace stargazer