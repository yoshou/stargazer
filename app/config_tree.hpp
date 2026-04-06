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

enum class config_tree_detail_kind {
  param,
  input,
  output,
  property,
};

struct config_tree_ref {
  std::string node_name;
};

struct runtime_node_property {
  std::string key;
  std::string value;
  bool read_only = true;
};

struct runtime_node_action {
  std::string id;
  std::string label;
  std::string icon;
  bool enabled = true;
};

struct runtime_node_handle {
  std::string stable_id;
  config_tree_ref ref;
  std::string label;
  std::string summary;
  std::vector<std::string> badges;
  std::vector<node_display_property> display_properties;
  std::vector<runtime_node_property> properties;
  std::vector<runtime_node_action> actions;
};

struct config_tree_item {
  std::string stable_id;
  config_tree_item_kind kind = config_tree_item_kind::detail;
  config_tree_detail_kind detail_kind = config_tree_detail_kind::param;
  std::string label;
  std::string summary;
  std::string runtime_node_id;
  std::string property_source_key;
  std::string property_format;
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
};

config_tree_model build_config_tree(const configuration& config);

struct stream_source {
  std::string name;
  float width;
  float height;
  std::string target;
  std::string property_node_name;
  std::string property_key;
  std::string property_resource_kind;
  std::string property_selector;
};

struct stream_source_model {
  std::vector<stream_source> sources;
};

stream_source_model build_stream_source_model(const configuration& config);

struct pose_camera_source {
  std::string camera_name;
  config_tree_ref ref;
  std::string property_key;
};

struct pose_generic_source {
  config_tree_ref ref;
  std::string property_key;
};

struct pose_source_model {
  std::vector<pose_camera_source> camera_sources;
  pose_generic_source axis_source;
  std::vector<pose_generic_source> point_sources;
};

pose_source_model build_pose_source_model(const configuration& config);

}  // namespace stargazer