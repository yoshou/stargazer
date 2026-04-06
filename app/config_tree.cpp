#include "config_tree.hpp"

#include <cmath>
#include <sstream>

namespace stargazer {
namespace {

std::string node_param_to_string(const node_param_t& param) {
  return std::visit(
      [](const auto& value) {
        std::ostringstream stream;
        stream << value;
        return stream.str();
      },
      param);
}

std::string get_node_summary(const node_def& node) {
  if (node.get_type() == node_type::image_property) {
    return "image";
  }
  if (node.contains_param("callback_type")) {
    return node.get_param<std::string>("callback_type");
  }
  return {};
}

std::vector<std::string> get_node_badges(const node_def& node) {
  std::vector<std::string> badges;
  if (node.is_camera()) {
    badges.push_back("camera");
  }
  if (node.contains_param("callback_type")) {
    const auto callback_type = node.get_param<std::string>("callback_type");
    if (callback_type == "image" || callback_type == "marker") {
      badges.push_back(callback_type);
    }
  }
  if (node.get_type() == node_type::image_property) {
    badges.push_back("image");
  }
  return badges;
}

config_tree_item make_detail_item(
    const std::string& stable_id, const std::string& label, const std::string& summary,
    config_tree_detail_kind detail_kind = config_tree_detail_kind::param,
    const std::string& runtime_node_id = {}, const std::string& property_source_key = {},
    const std::string& property_format = {}) {
  config_tree_item item;
  item.stable_id = stable_id;
  item.kind = config_tree_item_kind::detail;
  item.detail_kind = detail_kind;
  item.label = label;
  item.summary = summary;
  item.runtime_node_id = runtime_node_id;
  item.property_source_key = property_source_key;
  item.property_format = property_format;
  return item;
}

std::vector<config_tree_item> build_detail_items(const node_def& node,
                                                 const std::string& runtime_id) {
  std::vector<config_tree_item> children;

  for (const auto& [key, value] : node.params) {
    children.push_back(make_detail_item(runtime_id + ".param." + key, key,
                                        node_param_to_string(value), config_tree_detail_kind::param,
                                        runtime_id));
  }

  for (const auto& [key, value] : node.inputs) {
    children.push_back(make_detail_item(runtime_id + ".input." + key, "input." + key, value,
                                        config_tree_detail_kind::input, runtime_id));
  }

  for (size_t index = 0; index < node.outputs.size(); ++index) {
    children.push_back(make_detail_item(runtime_id + ".output." + std::to_string(index), "output",
                                        node.outputs[index], config_tree_detail_kind::output,
                                        runtime_id));
  }

  auto properties = node.properties;
  std::sort(properties.begin(), properties.end(), [](const auto& left, const auto& right) {
    if (left.order != right.order) {
      return left.order < right.order;
    }
    return left.id < right.id;
  });
  for (const auto& property : properties) {
    if (!property.target.empty() && property.target != "detail") {
      continue;
    }
    std::string summary = "-";
    if (property.default_value.has_value()) {
      summary = node_param_to_string(property.default_value.value());
    }
    children.push_back(make_detail_item(runtime_id + ".property." + property.id, property.label,
                                        summary, config_tree_detail_kind::property, runtime_id,
                                        property.source_key, property.format));
  }

  return children;
}

void append_pipeline_tree(config_tree_model& model, const configuration& config) {
  config_tree_item pipeline_item;
  pipeline_item.stable_id = "pipeline:pipeline";
  pipeline_item.kind = config_tree_item_kind::pipeline;
  pipeline_item.label = "pipeline";
  pipeline_item.summary = config.get_pipeline().name;

  std::unordered_map<std::string, size_t> subgraph_indices;
  const auto nodes = config.get_nodes();

  for (const auto& node : nodes) {
    const auto subgraph_name =
        node.subgraph_instance.empty() ? std::string("pipeline") : node.subgraph_instance;
    size_t subgraph_index = 0;
    if (subgraph_indices.find(subgraph_name) == subgraph_indices.end()) {
      config_tree_item subgraph_item;
      subgraph_item.stable_id = pipeline_item.stable_id + ".subgraph." + subgraph_name;
      subgraph_item.kind = config_tree_item_kind::subgraph;
      subgraph_item.label = subgraph_name;
      subgraph_item.summary = "subgraph";
      pipeline_item.children.push_back(subgraph_item);
      subgraph_index = pipeline_item.children.size() - 1;
      subgraph_indices[subgraph_name] = subgraph_index;
    } else {
      subgraph_index = subgraph_indices[subgraph_name];
    }

    const auto runtime_id = "node:pipeline:" + node.name;

    runtime_node_handle runtime_node;
    runtime_node.stable_id = runtime_id;
    runtime_node.ref = config_tree_ref{node.name};
    runtime_node.label = node.name;
    runtime_node.summary = get_node_summary(node);
    runtime_node.badges = get_node_badges(node);
    runtime_node.display_properties = node.properties;
    for (const auto& [key, value] : node.params) {
      runtime_node.properties.push_back(
          runtime_node_property{key, node_param_to_string(value), true});
    }
    if (node.get_type() == node_type::action) {
      const auto action_id =
          node.contains_param("action_id") ? node.get_param<std::string>("action_id") : node.name;
      const auto label =
          node.contains_param("label") ? node.get_param<std::string>("label") : node.name;
      const auto icon = node.contains_param("icon") ? node.get_param<std::string>("icon")
                                                    : std::string("refresh");
      runtime_node.actions.push_back(runtime_node_action{action_id, label, icon, true});
    }
    model.runtime_nodes.insert_or_assign(runtime_id, runtime_node);

    config_tree_item node_item;
    node_item.stable_id = runtime_id;
    node_item.kind = config_tree_item_kind::node;
    node_item.label = node.name;
    node_item.summary = runtime_node.summary;
    node_item.runtime_node_id = runtime_id;
    node_item.badges = runtime_node.badges;
    node_item.children = build_detail_items(node, runtime_id);
    pipeline_item.children[subgraph_index].children.push_back(std::move(node_item));
  }

  model.roots.push_back(std::move(pipeline_item));
}

}  // namespace

config_tree_model build_config_tree(const configuration& config) {
  config_tree_model model;
  append_pipeline_tree(model, config);
  return model;
}

namespace {

static int get_node_dimension(const node_def& node, const std::string& key) {
  if (!node.contains_param(key)) {
    return 0;
  }
  try {
    return static_cast<int>(node.get_param<std::int64_t>(key));
  } catch (...) {
    return static_cast<int>(std::round(node.get_param<float>(key)));
  }
}

static std::optional<std::string> try_get_node_camera_name(const node_def& node) {
  if (!node.contains_param("camera_name")) {
    return std::nullopt;
  }
  const auto camera_name = node.get_param<std::string>("camera_name");
  if (camera_name.empty()) {
    return std::nullopt;
  }
  return camera_name;
}

}  // namespace

stream_source_model build_stream_source_model(const configuration& config) {
  stream_source_model model;
  const auto nodes = config.get_nodes();
  for (const auto& node : nodes) {
    bool has_stream_target = false;
    for (const auto& property : node.properties) {
      const auto& target = property.target;
      if (target == "image" || target == "point" || target == "contrail") {
        has_stream_target = true;
        break;
      }
    }
    if (!has_stream_target) {
      continue;
    }

    const float width = static_cast<float>(get_node_dimension(node, "width"));
    const float height = static_cast<float>(get_node_dimension(node, "height"));
    const auto stream_name = try_get_node_camera_name(node).value_or(std::string{});

    for (const auto& property : node.properties) {
      const auto& target = property.target;
      if (target != "image" && target != "point" && target != "contrail") {
        continue;
      }
      stream_source src;
      src.name = stream_name;
      src.width = width;
      src.height = height;
      src.target = target;
      src.property_node_name = node.name;
      src.property_key = property.source_key;
      src.property_resource_kind = property.resource_kind;
      src.property_selector = property.selector;
      model.sources.push_back(std::move(src));
    }
  }
  return model;
}

pose_source_model build_pose_source_model(const configuration& config) {
  pose_source_model model;
  const auto nodes = config.get_nodes();
  for (const auto& node : nodes) {
    if (node.get_type() == node_type::epipolar_reconstruction) {
      const config_tree_ref ref{node.name};
      model.axis_source = {ref, "axis"};
      for (const auto& camera_node : nodes) {
        const auto camera_name = try_get_node_camera_name(camera_node);
        if (!camera_name.has_value()) {
          continue;
        }
        model.camera_sources.push_back({*camera_name, ref, "camera." + *camera_name});
      }
    }

    if (node.get_type() == node_type::extrinsic_calibration) {
      const config_tree_ref ref{node.name};
      for (const auto& [input_name, _input] : node.inputs) {
        const std::string prefix{"camera."};
        if (input_name.rfind(prefix, 0) != 0) {
          continue;
        }
        const auto camera_name = input_name.substr(prefix.size());
        model.camera_sources.push_back({camera_name, ref, "calibrated." + camera_name});
      }
    }

    if (node.get_type() == node_type::marker_property) {
      const config_tree_ref ref{node.name};
      model.point_sources.push_back({ref, "markers"});
    }
  }
  return model;
}

}  // namespace stargazer