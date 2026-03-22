#include <GLFW/glfw3.h>
#include <signal.h>
#include <spdlog/spdlog.h>

#include <cmath>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <vulkan/vulkan.hpp>

#include "capture_pipeline.hpp"
#include "coalsack_math_conv.hpp"
#include "config.hpp"
#include "extrinsic_calibration_pipeline.hpp"
#include "gui.hpp"
#include "image_reconstruction_pipeline.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "intrinsic_calibration_pipeline.hpp"
#include "parameters.hpp"
#include "point_reconstruction_pipeline.hpp"
#include "render3d.hpp"
#include "scene_calibration_pipeline.hpp"
#include "viewer.hpp"
#include "views.hpp"

using namespace stargazer;

const int SCREEN_WIDTH = 1680;
const int SCREEN_HEIGHT = 1050;

// Helper function to collect all dependencies of a node
static std::vector<stargazer::node_def> collect_node_dependencies(
    const stargazer::node_def& target_node, const std::vector<stargazer::node_def>& all_nodes) {
  constexpr auto callback_suffix = "_callback_image";
  std::set<std::string> visited;
  std::vector<stargazer::node_def> required_nodes;

  std::function<void(const stargazer::node_def&)> collect_deps;
  collect_deps = [&](const stargazer::node_def& info) {
    if (visited.find(info.name) != visited.end()) return;
    visited.insert(info.name);

    for (const auto& [input_name, source_name] : info.inputs) {
      size_t pos = source_name.find(':');
      std::string source_node_name =
          (pos != std::string::npos) ? source_name.substr(0, pos) : source_name;

      auto source_it = std::find_if(all_nodes.begin(), all_nodes.end(),
                                    [&](const auto& n) { return n.name == source_node_name; });
      if (source_it != all_nodes.end()) {
        collect_deps(*source_it);
      }
    }

    const auto suffix_pos = info.name.rfind(callback_suffix);
    if (suffix_pos != std::string::npos &&
        suffix_pos + std::char_traits<char>::length(callback_suffix) == info.name.size()) {
      const auto display_node_name = info.name.substr(0, suffix_pos) + "_display_image";
      auto display_it = std::find_if(all_nodes.begin(), all_nodes.end(),
                                     [&](const auto& n) { return n.name == display_node_name; });
      if (display_it != all_nodes.end()) {
        collect_deps(*display_it);
      }
    }

    required_nodes.push_back(info);
  };

  collect_deps(target_node);
  return required_nodes;
}

static std::string format_double_value(double value, const std::string& format) {
  std::ostringstream stream;
  if (format == "fixed3") {
    stream << std::fixed << std::setprecision(3);
  }
  stream << value;
  return stream.str();
}

static std::string format_property_value(const coalsack::property_value& value,
                                         const std::string& format) {
  return std::visit(
      [&](const auto& raw_value) -> std::string {
        using value_t = std::decay_t<decltype(raw_value)>;
        if constexpr (std::is_same_v<value_t, std::string>) {
          return raw_value;
        } else if constexpr (std::is_same_v<value_t, std::int64_t>) {
          if (format == "fixed3") {
            return format_double_value(static_cast<double>(raw_value), format);
          }
          return std::to_string(raw_value);
        } else if constexpr (std::is_same_v<value_t, double>) {
          return format_double_value(raw_value, format);
        } else if constexpr (std::is_same_v<value_t, bool>) {
          return raw_value ? "true" : "false";
        } else if constexpr (std::is_same_v<value_t, std::shared_ptr<coalsack::image>>) {
          return raw_value ? "<image>" : "-";
        } else if constexpr (std::is_same_v<value_t, coalsack::camera_t>) {
          return std::string{"<camera>"};
        } else if constexpr (std::is_same_v<value_t, coalsack::mat4>) {
          return std::string{"<mat4>"};
        } else if constexpr (std::is_same_v<value_t, std::vector<coalsack::vec3>>) {
          return std::string{"<points:"} + std::to_string(raw_value.size()) + ">";
        }
        return std::string{"-"};
      },
      value);
}

static int image_format_to_cv_type(coalsack::image_format format) {
  switch (format) {
    case coalsack::image_format::Y8_UINT:
      return CV_8UC1;
    case coalsack::image_format::Y16_UINT:
    case coalsack::image_format::Z16_UINT:
      return CV_16UC1;
    case coalsack::image_format::R8G8B8_UINT:
    case coalsack::image_format::B8G8R8_UINT:
      return CV_8UC3;
    case coalsack::image_format::R8G8B8A8_UINT:
    case coalsack::image_format::B8G8R8A8_UINT:
      return CV_8UC4;
    default:
      return -1;
  }
}

class viewer_app : public window_base {
  ImFont* large_font;
  ImFont* default_font;
  graphics_context* gfx_ctx;  // Graphics context pointer

  std::unique_ptr<view_context> context;
  std::unique_ptr<top_bar_view> top_bar_view_;
  std::unique_ptr<capture_panel_view> capture_panel_view_;
  std::unique_ptr<calibration_panel_view> calibration_panel_view_;
  std::unique_ptr<reconstruction_panel_view> reconstruction_panel_view_;
  std::unique_ptr<image_tile_view> image_tile_view_;
  std::unique_ptr<image_tile_view> point_tile_view_;
  std::unique_ptr<image_tile_view> contrail_tile_view_;
  std::unique_ptr<pose_view> pose_view_;
  std::shared_ptr<azimuth_elevation> view_controller;

  std::shared_ptr<parameters_t> parameters;

  std::map<std::string, std::shared_ptr<capture_pipeline>> captures;
  std::shared_ptr<capture_pipeline> multiview_capture;

  std::unique_ptr<extrinsic_calibration_pipeline> extrinsic_calib;
  std::unique_ptr<intrinsic_calibration_pipeline> intrinsic_calib;
  std::unique_ptr<scene_calibration_pipeline> scene_calib;

  std::unique_ptr<multiview_point_reconstruction_pipeline> multiview_point_reconstruction_pipeline_;
  std::unique_ptr<multiview_image_reconstruction_pipeline> multiview_image_reconstruction_pipeline_;

  std::unique_ptr<configuration> capture_config;
  std::unique_ptr<configuration> point_reconstruction_config;
  std::unique_ptr<configuration> image_reconstruction_config;
  std::unique_ptr<configuration> extrinsic_calibration_config;
  std::unique_ptr<configuration> scene_calibration_config;
  std::unique_ptr<configuration> calibration_intrinsic_single_camera_config;

  std::string get_intrinsic_target_camera_name() const {
    for (const auto& node : calibration_intrinsic_single_camera_config->get_nodes()) {
      if (node.is_camera()) {
        return node.get_camera_name();
      }
    }
    throw std::runtime_error("No intrinsic target camera found in intrinsic calibration config");
  }

  std::optional<coalsack::property_value> query_capture_node_property(
      const std::string& node_name, const std::string& key) const {
    if (multiview_capture) {
      if (const auto value = multiview_capture->get_node_property(node_name, key);
          value.has_value()) {
        return value;
      }
    }
    for (const auto& [capture_name, capture] : captures) {
      (void)capture_name;
      if (!capture) {
        continue;
      }
      if (const auto value = capture->get_node_property(node_name, key); value.has_value()) {
        return value;
      }
    }
    return std::nullopt;
  }

  void rebuild_calibration_panel_nodes(const configuration& config) {
    if (!calibration_panel_view_) {
      return;
    }

    calibration_panel_view_->nodes.clear();
    for (const auto& node : config.get_nodes()) {
      if (!node.is_camera()) {
        continue;
      }

      std::string path;
      if (node.contains_param("address")) {
        path = node.get_param<std::string>("address");
      }
      if (node.contains_param("db_path")) {
        path = node.get_param<std::string>("db_path");
      }

      calibration_panel_view_->nodes.push_back(
          calibration_panel_view::node_def{node.get_camera_name(), path, node.params});
    }
  }

  std::optional<coalsack::property_value> query_runtime_node_property(
      const stargazer::config_tree_ref& ref, const std::string& key) const {
    if (ref.pipeline_key == "pipeline") {
      return query_capture_node_property(ref.node_name, key);
    }
    if (ref.pipeline_key == "extrinsic_calibration_pipeline" && extrinsic_calib) {
      return extrinsic_calib->get_node_property(ref.node_name, key);
    }
    if (ref.pipeline_key == "intrinsic_calibration_pipeline" && intrinsic_calib) {
      return intrinsic_calib->get_node_property(ref.node_name, key);
    }
    if (ref.pipeline_key == "scene_calibration_pipeline" && scene_calib) {
      return scene_calib->get_node_property(ref.node_name, key);
    }
    if (ref.pipeline_key == "point_reconstruction_pipeline" &&
        multiview_point_reconstruction_pipeline_) {
      return multiview_point_reconstruction_pipeline_->get_node_property(ref.node_name, key);
    }
    if (ref.pipeline_key == "image_reconstruction_pipeline" &&
        multiview_image_reconstruction_pipeline_) {
      return multiview_image_reconstruction_pipeline_->get_node_property(ref.node_name, key);
    }
    return std::nullopt;
  }

  // Bind pose view camera/axis/point sources from the active pipeline configs.
  void bind_pose_property() {
    if (!pose_view_) return;
    pose_view_->camera_sources.clear();
    pose_view_->point_sources.clear();
    pose_view_->axis_source = {};

    // Bind cameras from point_reconstruction_config (epipolar_reconstruct_node)
    if (point_reconstruction_config) {
      for (const auto& node :
           point_reconstruction_config->get_nodes("point_reconstruction_pipeline")) {
        if (node.get_type() != stargazer::node_type::epipolar_reconstruction) continue;
        const stargazer::config_tree_ref ref{"point_reconstruction_pipeline", "", node.name};
        // axis
        pose_view_->axis_source = {ref, "axis"};
        // cameras: derive names from point_reconstruction_config (is_camera nodes)
        for (const auto& cam_node : point_reconstruction_config->get_nodes()) {
          if (!cam_node.is_camera()) continue;
          const auto cam_name = cam_node.get_camera_name();
          pose_view_->camera_sources[cam_name] = {ref, "camera." + cam_name};
        }
      }
      // marker_property nodes → point_sources
      for (const auto& node :
           point_reconstruction_config->get_nodes("point_reconstruction_pipeline")) {
        if (node.get_type() != stargazer::node_type::marker_property) continue;
        const stargazer::config_tree_ref ref{"point_reconstruction_pipeline", "", node.name};
        pose_view_->point_sources.push_back({ref, "markers"});
      }
    }
    // image_reconstruction marker_property nodes
    if (image_reconstruction_config) {
      for (const auto& node :
           image_reconstruction_config->get_nodes("image_reconstruction_pipeline")) {
        if (node.get_type() != stargazer::node_type::marker_property) continue;
        const stargazer::config_tree_ref ref{"image_reconstruction_pipeline", "", node.name};
        pose_view_->point_sources.push_back({ref, "markers"});
      }
    }
  }

  // Bind extrinsic calibration pose property sources.
  void bind_extrinsic_calibration_pose_property() {
    if (!pose_view_) return;
    pose_view_->camera_sources.clear();
    pose_view_->point_sources.clear();
    pose_view_->axis_source = {};

    if (extrinsic_calibration_config) {
      for (const auto& node :
           extrinsic_calibration_config->get_nodes("extrinsic_calibration_pipeline")) {
        if (node.get_type() != stargazer::node_type::extrinsic_calibration) continue;
        const stargazer::config_tree_ref ref{"extrinsic_calibration_pipeline", "", node.name};
        // calibrated cameras: derive names from extrinsic_calibration node inputs
        for (const auto& [cam_name, _input] : node.inputs) {
          pose_view_->camera_sources[cam_name] = {ref, "calibrated." + cam_name};
        }
      }
    }
  }

  // Read pose properties from the bound sources and update pose_view_.
  void update_pose_from_properties() {
    if (!pose_view_) return;

    // Update cameras
    for (const auto& [cam_name, source] : pose_view_->camera_sources) {
      const auto value = query_runtime_node_property(source.ref, source.property_key);
      if (value && std::holds_alternative<coalsack::camera_t>(value.value())) {
        pose_view_->cameras[cam_name] = std::get<coalsack::camera_t>(value.value());
      }
    }

    // Update axis
    if (!pose_view_->axis_source.property_key.empty()) {
      const auto value = query_runtime_node_property(pose_view_->axis_source.ref,
                                                     pose_view_->axis_source.property_key);
      if (value && std::holds_alternative<coalsack::mat4>(value.value())) {
        pose_view_->axis = stargazer::to_glm(std::get<coalsack::mat4>(value.value()));
      }
    }

    // Update points
    pose_view_->points.clear();
    for (const auto& source : pose_view_->point_sources) {
      const auto value = query_runtime_node_property(source.ref, source.property_key);
      if (value && std::holds_alternative<std::vector<coalsack::vec3>>(value.value())) {
        for (const auto& p : std::get<std::vector<coalsack::vec3>>(value.value())) {
          pose_view_->points.push_back(stargazer::to_glm(p));
        }
      }
    }
  }

  void bind_capture_stream_property(const stargazer::runtime_node_handle& runtime_node,
                                    image_tile_view::stream_info& stream) const {
    stream.property_node_name.clear();
    stream.property_key.clear();
    stream.property_resource_kind.clear();
    stream.property_selector.clear();

    for (const auto& property : runtime_node.display_properties) {
      if (property.target != "image") {
        continue;
      }
      stream.property_node_name = runtime_node.ref.node_name;
      stream.property_key = property.source_key;
      stream.property_resource_kind = property.resource_kind;
      stream.property_selector = property.selector;
      break;
    }

    if (stream.property_key.empty() && runtime_node.is_camera) {
      constexpr auto callback_suffix = "_callback_image";
      const auto& node_name = runtime_node.ref.node_name;
      const auto suffix_pos = node_name.rfind(callback_suffix);
      if (suffix_pos != std::string::npos &&
          suffix_pos + std::char_traits<char>::length(callback_suffix) == node_name.size()) {
        stream.property_node_name = node_name.substr(0, suffix_pos) + "_display_image";
        stream.property_key = "image";
      }
    }
  }

  void bind_image_reconstruction_stream_property(const std::string& camera_name,
                                                 image_tile_view::stream_info& stream) const {
    stream.property_node_name.clear();
    stream.property_key.clear();
    stream.property_resource_kind.clear();
    stream.property_selector.clear();

    for (const auto& node :
         image_reconstruction_config->get_nodes("image_reconstruction_pipeline")) {
      if (!node.contains_param("camera_name") ||
          node.get_param<std::string>("camera_name") != camera_name) {
        continue;
      }
      for (const auto& property : node.properties) {
        if (property.target != "point") {
          continue;
        }
        stream.property_node_name = node.name;
        stream.property_key = property.source_key;
        stream.property_resource_kind = property.resource_kind;
        stream.property_selector = property.selector;
        return;
      }
    }
  }

  bool upload_capture_property_stream(
      const std::shared_ptr<image_tile_view::stream_info>& stream) const {
    if (!stream || stream->property_node_name.empty() || stream->property_key.empty()) {
      return false;
    }
    if (!stream->property_resource_kind.empty() && stream->property_resource_kind != "raw") {
      return false;
    }

    const auto value =
        query_capture_node_property(stream->property_node_name, stream->property_key);
    if (!value.has_value()) {
      return false;
    }

    const auto image_ptr = std::get_if<std::shared_ptr<coalsack::image>>(&value.value());
    if (!image_ptr || !(*image_ptr) || (*image_ptr)->empty()) {
      return false;
    }

    const auto type = image_format_to_cv_type((*image_ptr)->get_format());
    if (type < 0) {
      return false;
    }

    cv::Mat frame(static_cast<int>((*image_ptr)->get_height()),
                  static_cast<int>((*image_ptr)->get_width()), type,
                  const_cast<uint8_t*>((*image_ptr)->get_data()),
                  static_cast<size_t>((*image_ptr)->get_stride()));
    cv::Mat upload_image = frame.clone();

    if (upload_image.empty()) {
      return false;
    }

    switch ((*image_ptr)->get_format()) {
      case coalsack::image_format::Y8_UINT:
        cv::cvtColor(upload_image, upload_image, cv::COLOR_GRAY2RGB);
        break;
      case coalsack::image_format::B8G8R8_UINT:
        cv::cvtColor(upload_image, upload_image, cv::COLOR_BGR2RGB);
        break;
      case coalsack::image_format::B8G8R8A8_UINT:
        cv::cvtColor(upload_image, upload_image, cv::COLOR_BGRA2RGBA);
        break;
      default:
        break;
    }

    stream->texture.upload_image(upload_image.cols, upload_image.rows, upload_image.data, 0);
    return true;
  }

  bool upload_image_reconstruction_property_stream(
      const std::shared_ptr<image_tile_view::stream_info>& stream) const {
    if (!multiview_image_reconstruction_pipeline_ || !stream ||
        stream->property_node_name.empty() || stream->property_key.empty()) {
      return false;
    }
    if (!stream->property_resource_kind.empty() && stream->property_resource_kind != "feature") {
      return false;
    }

    const auto value = multiview_image_reconstruction_pipeline_->get_node_property(
        stream->property_node_name, stream->property_key);
    if (!value.has_value()) {
      return false;
    }

    const auto image_ptr = std::get_if<std::shared_ptr<coalsack::image>>(&value.value());
    if (!image_ptr || !(*image_ptr) || (*image_ptr)->empty()) {
      return false;
    }

    const auto type = image_format_to_cv_type((*image_ptr)->get_format());
    if (type < 0) {
      return false;
    }

    cv::Mat frame(static_cast<int>((*image_ptr)->get_height()),
                  static_cast<int>((*image_ptr)->get_width()), type,
                  const_cast<uint8_t*>((*image_ptr)->get_data()),
                  static_cast<size_t>((*image_ptr)->get_stride()));
    cv::Mat upload_image = frame.clone();
    if (upload_image.empty()) {
      return false;
    }

    switch ((*image_ptr)->get_format()) {
      case coalsack::image_format::Y8_UINT:
        cv::cvtColor(upload_image, upload_image, cv::COLOR_GRAY2RGB);
        break;
      case coalsack::image_format::B8G8R8_UINT:
        cv::cvtColor(upload_image, upload_image, cv::COLOR_BGR2RGB);
        break;
      case coalsack::image_format::B8G8R8A8_UINT:
        cv::cvtColor(upload_image, upload_image, cv::COLOR_BGRA2RGBA);
        break;
      default:
        break;
    }

    stream->texture.upload_image(upload_image.cols, upload_image.rows, upload_image.data, 0);
    return true;
  }

  bool upload_contrail_property_stream(
      const std::shared_ptr<image_tile_view::stream_info>& stream) const {
    if (!extrinsic_calib || !stream || stream->property_node_name.empty() ||
        stream->property_key.empty()) {
      return false;
    }

    const auto value =
        extrinsic_calib->get_node_property(stream->property_node_name, stream->property_key);
    if (!value.has_value()) {
      return false;
    }

    const auto image_ptr = std::get_if<std::shared_ptr<coalsack::image>>(&value.value());
    if (!image_ptr || !(*image_ptr) || (*image_ptr)->empty()) {
      return false;
    }

    const auto type = image_format_to_cv_type((*image_ptr)->get_format());
    if (type < 0) {
      return false;
    }

    cv::Mat frame(static_cast<int>((*image_ptr)->get_height()),
                  static_cast<int>((*image_ptr)->get_width()), type,
                  const_cast<uint8_t*>((*image_ptr)->get_data()),
                  static_cast<size_t>((*image_ptr)->get_stride()));
    cv::Mat upload_image = frame.clone();
    if (upload_image.empty()) {
      return false;
    }

    switch ((*image_ptr)->get_format()) {
      case coalsack::image_format::Y8_UINT:
        cv::cvtColor(upload_image, upload_image, cv::COLOR_GRAY2RGB);
        break;
      case coalsack::image_format::B8G8R8_UINT:
        cv::cvtColor(upload_image, upload_image, cv::COLOR_BGR2RGB);
        break;
      case coalsack::image_format::B8G8R8A8_UINT:
        cv::cvtColor(upload_image, upload_image, cv::COLOR_BGRA2RGBA);
        break;
      default:
        break;
    }

    stream->texture.upload_image(upload_image.cols, upload_image.rows, upload_image.data, 0);
    return true;
  }

  template <typename PanelT>
  std::optional<std::string> resolve_panel_detail_value(
      const PanelT* panel, const stargazer::config_tree_item& item) const {
    if (!panel || item.detail_kind != stargazer::config_tree_detail_kind::property ||
        item.runtime_node_id.empty()) {
      return std::nullopt;
    }

    const auto runtime_it = panel->tree.runtime_nodes.find(item.runtime_node_id);
    if (runtime_it == panel->tree.runtime_nodes.end()) {
      return std::nullopt;
    }

    const auto value =
        query_runtime_node_property(runtime_it->second.ref, item.property_source_key);
    if (!value.has_value()) {
      return std::nullopt;
    }
    return format_property_value(value.value(), item.property_format);
  }

  void sync_calibration_panel_state() {
    if (!calibration_panel_view_ || !top_bar_view_) {
      return;
    }

    const auto calibration_pipeline = top_bar_view_->calibration_pipeline;
    const int next_target_index =
        calibration_pipeline == top_bar_view::CalibrationPipeline::Extrinsic   ? 0
        : calibration_pipeline == top_bar_view::CalibrationPipeline::Intrinsic ? 1
                                                                               : 2;
    calibration_panel_view_->calibration_target_index = next_target_index;

    if (next_target_index == 0) {
      rebuild_calibration_panel_nodes(*extrinsic_calibration_config);
      calibration_panel_view_->tree =
          build_config_tree(*extrinsic_calibration_config,
                            std::vector<std::string>{"pipeline", "extrinsic_calibration_pipeline"});
      return;
    }

    if (next_target_index == 1) {
      rebuild_calibration_panel_nodes(*calibration_intrinsic_single_camera_config);
      calibration_panel_view_->tree =
          build_config_tree(*calibration_intrinsic_single_camera_config,
                            std::vector<std::string>{"pipeline", "intrinsic_calibration_pipeline"});

      const auto target_camera_name = get_intrinsic_target_camera_name();
      calibration_panel_view_->intrinsic_target_camera_name = target_camera_name;
      return;
    }

    rebuild_calibration_panel_nodes(*scene_calibration_config);
    calibration_panel_view_->tree =
        build_config_tree(*scene_calibration_config,
                          std::vector<std::string>{"pipeline", "scene_calibration_pipeline"});
  }

  void sync_reconstruction_panel_state() {
    if (!reconstruction_panel_view_ || !top_bar_view_) {
      return;
    }

    const auto reconstruction_pipeline = top_bar_view_->reconstruction_pipeline;
    const auto pipeline_name =
        reconstruction_pipeline == top_bar_view::ReconstructionPipeline::Marker
            ? std::string{"point_reconstruction_pipeline"}
            : std::string{"image_reconstruction_pipeline"};

    const auto& selected_recon_config =
        reconstruction_pipeline == top_bar_view::ReconstructionPipeline::Marker
            ? *point_reconstruction_config
            : *image_reconstruction_config;

    reconstruction_panel_view_->tree = build_config_tree(
        selected_recon_config, std::vector<std::string>{"pipeline", pipeline_name});
  }

  std::string generate_new_id() const {
    uint64_t max_id = 0;
    for (const auto& node : capture_config->get_nodes()) {
      size_t idx = 0;
      const auto id_str = node.get_param<std::string>("id");
      const auto id = std::stoull(id_str, &idx);

      if (id_str.size() == idx) {
        max_id = std::max(max_id, static_cast<uint64_t>(id + 1));
      }
    }
    return fmt::format("{:>012d}", max_id);
  }

  void init_capture_panel() {
    capture_panel_view_ = std::make_unique<capture_panel_view>();
    capture_panel_view_->resolve_detail_value =
        [this](const stargazer::config_tree_item& item) -> std::optional<std::string> {
      return resolve_panel_detail_value(capture_panel_view_.get(), item);
    };
    capture_panel_view_->tree = build_config_tree(*capture_config);

    capture_panel_view_->is_streaming_changed.push_back([this](const std::string& runtime_node_id,
                                                               bool is_streaming) {
      auto runtime_it = capture_panel_view_->tree.runtime_nodes.find(runtime_node_id);
      if (runtime_it == capture_panel_view_->tree.runtime_nodes.end()) {
        spdlog::error("Runtime node {} not found in capture tree", runtime_node_id);
        return false;
      }

      const auto& node_name = runtime_it->second.ref.node_name;
      if (is_streaming) {
        const auto& all_nodes = capture_config->get_nodes();
        auto found = std::find_if(all_nodes.begin(), all_nodes.end(),
                                  [&](const auto& x) { return x.name == node_name; });
        if (found == all_nodes.end()) {
          spdlog::error("Node {} not found in capture config", node_name);
          return false;
        }
        const auto& target_node = *found;

        // Collect all nodes that the target depends on
        auto required_nodes = collect_node_dependencies(target_node, all_nodes);

        const auto capture = std::make_shared<capture_pipeline>();

        try {
          capture->run(required_nodes);
        } catch (std::exception& e) {
          spdlog::error("Failed to start capture for {}: {}", node_name, e.what());
          return false;
        }
        captures.insert(std::make_pair(node_name, capture));
        const auto width = static_cast<int>(std::round(target_node.get_param<float>("width")));
        const auto height = static_cast<int>(std::round(target_node.get_param<float>("height")));

        const auto stream = std::make_shared<image_tile_view::stream_info>(
            node_name, float2{(float)width, (float)height}, gfx_ctx);
        bind_capture_stream_property(runtime_it->second, *stream);
        image_tile_view_->streams.push_back(stream);
      } else {
        auto it = captures.find(node_name);
        if (it == captures.end()) {
          return true;
        }
        it->second->stop();
        if (it != captures.end()) {
          captures.erase(captures.find(node_name));
        }

        const auto stream_it =
            std::find_if(image_tile_view_->streams.begin(), image_tile_view_->streams.end(),
                         [&](const auto& x) { return x->name == node_name; });

        if (stream_it != image_tile_view_->streams.end()) {
          image_tile_view_->streams.erase(stream_it);
        }
      }

      return true;
    });

    capture_panel_view_->is_all_streaming_changed.push_back([this](bool is_streaming) {
      if (is_streaming) {
        if (multiview_capture) {
          return false;
        }

        const auto& nodes = capture_config->get_nodes();

        multiview_capture.reset(new capture_pipeline());

        for (const auto& node : nodes) {
          if (node.is_camera()) {
            const auto camera_name = node.get_camera_name();
            multiview_capture->enable_marker_collecting(camera_name);
          }
        }

        multiview_capture->run(nodes);

        for (const auto& node : nodes) {
          if (node.is_camera()) {
            const auto camera_name = node.get_camera_name();
            int width, height;
            // Try to get as int64_t first (from JSON), then fall back to float
            if (node.contains_param("width")) {
              try {
                width = static_cast<int>(node.get_param<std::int64_t>("width"));
              } catch (...) {
                width = static_cast<int>(std::round(node.get_param<float>("width")));
              }
            } else {
              throw std::runtime_error("width parameter not found for camera: " + camera_name);
            }
            if (node.contains_param("height")) {
              try {
                height = static_cast<int>(node.get_param<std::int64_t>("height"));
              } catch (...) {
                height = static_cast<int>(std::round(node.get_param<float>("height")));
              }
            } else {
              throw std::runtime_error("height parameter not found for camera: " + camera_name);
            }
            const auto stream = std::make_shared<image_tile_view::stream_info>(
                camera_name, float2{(float)width, (float)height}, gfx_ctx);
            const auto runtime_id = std::string{"node:pipeline:"} + node.name;
            if (const auto runtime_it = capture_panel_view_->tree.runtime_nodes.find(runtime_id);
                runtime_it != capture_panel_view_->tree.runtime_nodes.end()) {
              bind_capture_stream_property(runtime_it->second, *stream);
            }
            image_tile_view_->streams.push_back(stream);
          }
        }
      } else {
        if (multiview_capture) {
          multiview_capture->stop();
          multiview_capture.reset();

          const auto& nodes = capture_config->get_nodes();

          for (const auto& node : nodes) {
            if (node.is_camera()) {
              const auto camera_name = node.get_camera_name();
              const auto stream_it =
                  std::find_if(image_tile_view_->streams.begin(), image_tile_view_->streams.end(),
                               [&](const auto& x) { return x->name == camera_name; });

              if (stream_it != image_tile_view_->streams.end()) {
                image_tile_view_->streams.erase(stream_it);
              }
            }
          }
        }
      }
      return true;
    });
  }

  void init_calibration_panel() {
    calibration_panel_view_ = std::make_unique<calibration_panel_view>();
    calibration_panel_view_->resolve_detail_value =
        [this](const stargazer::config_tree_item& item) -> std::optional<std::string> {
      return resolve_panel_detail_value(calibration_panel_view_.get(), item);
    };
    sync_calibration_panel_state();

    calibration_panel_view_->is_streaming_changed.push_back(
        [this](const std::vector<calibration_panel_view::node_def>& panel_nodes,
               bool is_streaming) {
          if (is_streaming) {
            if (multiview_capture) {
              return false;
            }

            if (calibration_panel_view_->calibration_target_index == 0) {
              // Extrinsic calibration
              const auto& nodes = extrinsic_calibration_config->get_nodes();

              multiview_capture.reset(new capture_pipeline());

              multiview_capture->add_marker_received(
                  [this](const std::map<std::string, marker_frame_data>& marker_frame) {
                    std::map<std::string, std::vector<point_data>> frame;
                    for (const auto& [name, markers] : marker_frame) {
                      std::vector<point_data> points;
                      for (const auto& marker : markers.markers) {
                        points.push_back(
                            point_data{glm::vec2(marker.x, marker.y), marker.r, markers.timestamp});
                      }
                      frame.insert(std::make_pair(name, points));
                    }
                    extrinsic_calib->push_frame(frame);
                  });

              multiview_capture->run(nodes);

              for (const auto& node : nodes) {
                if (node.is_camera()) {
                  const auto camera_name = node.get_camera_name();
                  int width, height;
                  if (node.contains_param("width")) {
                    try {
                      width = static_cast<int>(node.get_param<std::int64_t>("width"));
                    } catch (...) {
                      width = static_cast<int>(std::round(node.get_param<float>("width")));
                    }
                  } else {
                    throw std::runtime_error("width parameter not found for camera: " +
                                             camera_name);
                  }
                  if (node.contains_param("height")) {
                    try {
                      height = static_cast<int>(node.get_param<std::int64_t>("height"));
                    } catch (...) {
                      height = static_cast<int>(std::round(node.get_param<float>("height")));
                    }
                  } else {
                    throw std::runtime_error("height parameter not found for camera: " +
                                             camera_name);
                  }
                  const auto stream = std::make_shared<image_tile_view::stream_info>(
                      camera_name, float2{(float)width, (float)height}, gfx_ctx);
                  const auto runtime_id = std::string{"node:pipeline:"} + node.name;
                  if (const auto runtime_it =
                          calibration_panel_view_->tree.runtime_nodes.find(runtime_id);
                      runtime_it != calibration_panel_view_->tree.runtime_nodes.end()) {
                    bind_capture_stream_property(runtime_it->second, *stream);
                  }
                  image_tile_view_->streams.push_back(stream);
                }
              }
            } else if (calibration_panel_view_->calibration_target_index == 1) {
              // Intrinsic calibration
              const auto& nodes = calibration_intrinsic_single_camera_config->get_nodes();
              auto found = std::find_if(nodes.begin(), nodes.end(),
                                        [&](const auto& x) { return x.is_camera(); });
              if (found == nodes.end()) {
                return false;
              }
              const auto& node = *found;
              const auto& target_camera_name = node.get_camera_name();
              auto panel_node_it = std::find_if(
                  panel_nodes.begin(), panel_nodes.end(),
                  [&](const auto& panel_node) { return panel_node.name == target_camera_name; });
              if (panel_node_it == panel_nodes.end()) {
                return false;
              }
              const auto& panel_node = *panel_node_it;

              const auto capture = std::make_shared<capture_pipeline>();

              capture->add_image_received([this](const std::map<std::string, cv::Mat>& frames) {
                if (!frames.empty()) {
                  // For single camera, get the first (and only) frame
                  const auto& frame = frames.begin()->second;
                  intrinsic_calib->push_frame(frame);
                }
              });

              // Collect dependencies for intrinsic calibration
              auto required_nodes = collect_node_dependencies(node, nodes);

              try {
                capture->run(required_nodes);
              } catch (std::exception& e) {
                std::cout << "Failed to start capture: " << e.what() << std::endl;
                return false;
              }
              captures.insert(std::make_pair(panel_node.name, capture));
              const auto width = static_cast<int>(std::round(node.get_param<float>("width")));
              const auto height = static_cast<int>(std::round(node.get_param<float>("height")));

              const auto stream = std::make_shared<image_tile_view::stream_info>(
                  panel_node.name, float2{(float)width, (float)height}, gfx_ctx);
              const auto runtime_id = std::string{"node:pipeline:"} + node.name;
              if (const auto runtime_it =
                      calibration_panel_view_->tree.runtime_nodes.find(runtime_id);
                  runtime_it != calibration_panel_view_->tree.runtime_nodes.end()) {
                bind_capture_stream_property(runtime_it->second, *stream);
              }
              image_tile_view_->streams.push_back(stream);
            } else if (calibration_panel_view_->calibration_target_index == 2) {
              // Axis calibration
              const auto& nodes = scene_calibration_config->get_nodes();

              multiview_capture.reset(new capture_pipeline());

              multiview_capture->add_marker_received(
                  [this](const std::map<std::string, marker_frame_data>& marker_frame) {
                    std::map<std::string, std::vector<point_data>> frame;
                    for (const auto& [name, markers] : marker_frame) {
                      std::vector<point_data> points;
                      for (const auto& marker : markers.markers) {
                        points.push_back(
                            point_data{glm::vec2(marker.x, marker.y), marker.r, markers.timestamp});
                      }
                      frame.insert(std::make_pair(name, points));
                    }
                    scene_calib->push_frame(frame);
                  });

              for (const auto& node : nodes) {
                if (node.is_camera()) {
                  const auto camera_name = node.get_camera_name();
                  multiview_capture->enable_marker_collecting(camera_name);
                }
              }

              multiview_capture->run(nodes);

              for (const auto& node : nodes) {
                if (node.is_camera()) {
                  const auto camera_name = node.get_camera_name();
                  int width, height;
                  if (node.contains_param("width")) {
                    try {
                      width = static_cast<int>(node.get_param<std::int64_t>("width"));
                    } catch (...) {
                      width = static_cast<int>(std::round(node.get_param<float>("width")));
                    }
                  } else {
                    throw std::runtime_error("width parameter not found for camera: " +
                                             camera_name);
                  }
                  if (node.contains_param("height")) {
                    try {
                      height = static_cast<int>(node.get_param<std::int64_t>("height"));
                    } catch (...) {
                      height = static_cast<int>(std::round(node.get_param<float>("height")));
                    }
                  } else {
                    throw std::runtime_error("height parameter not found for camera: " +
                                             camera_name);
                  }
                  const auto stream = std::make_shared<image_tile_view::stream_info>(
                      camera_name, float2{(float)width, (float)height}, gfx_ctx);
                  const auto runtime_id = std::string{"node:pipeline:"} + node.name;
                  if (const auto runtime_it =
                          calibration_panel_view_->tree.runtime_nodes.find(runtime_id);
                      runtime_it != calibration_panel_view_->tree.runtime_nodes.end()) {
                    bind_capture_stream_property(runtime_it->second, *stream);
                  }
                  image_tile_view_->streams.push_back(stream);
                }
              }
            }
          } else {
            if (calibration_panel_view_->calibration_target_index == 0) {
              multiview_capture->stop();
              multiview_capture.reset();

              const auto& nodes = extrinsic_calibration_config->get_nodes();

              for (const auto& node : nodes) {
                if (node.is_camera()) {
                  const auto camera_name = node.get_camera_name();
                  const auto stream_it = std::find_if(
                      image_tile_view_->streams.begin(), image_tile_view_->streams.end(),
                      [&](const auto& x) { return x->name == camera_name; });

                  if (stream_it != image_tile_view_->streams.end()) {
                    image_tile_view_->streams.erase(stream_it);
                  }
                }
              }
            } else if (calibration_panel_view_->calibration_target_index == 1) {
              const auto target_camera_name = get_intrinsic_target_camera_name();
              auto panel_node_it = std::find_if(
                  panel_nodes.begin(), panel_nodes.end(),
                  [&](const auto& panel_node) { return panel_node.name == target_camera_name; });
              if (panel_node_it == panel_nodes.end()) {
                return false;
              }
              const auto& panel_node = *panel_node_it;

              auto it = captures.find(panel_node.name);
              if (it == captures.end()) {
                return true;
              }
              it->second->stop();
              captures.erase(it);

              const auto stream_it =
                  std::find_if(image_tile_view_->streams.begin(), image_tile_view_->streams.end(),
                               [&](const auto& x) { return x->name == panel_node.name; });

              if (stream_it != image_tile_view_->streams.end()) {
                image_tile_view_->streams.erase(stream_it);
              }
            } else if (calibration_panel_view_->calibration_target_index == 2) {
              multiview_capture->stop();
              multiview_capture.reset();

              const auto& nodes = scene_calibration_config->get_nodes();

              for (const auto& node : nodes) {
                if (node.is_camera()) {
                  const auto camera_name = node.get_camera_name();
                  const auto stream_it = std::find_if(
                      image_tile_view_->streams.begin(), image_tile_view_->streams.end(),
                      [&](const auto& x) { return x->name == camera_name; });

                  if (stream_it != image_tile_view_->streams.end()) {
                    image_tile_view_->streams.erase(stream_it);
                  }
                }
              }
            }

            calibration_panel_view_->is_marker_collecting = false;
          }
          return true;
        });

    calibration_panel_view_->is_marker_collecting_changed.push_back(
        [this](const std::vector<calibration_panel_view::node_def>& panel_nodes,
               bool is_marker_collecting) {
          if (calibration_panel_view_->calibration_target_index == 1) {
            return true;
          }
          if (!multiview_capture) {
            return false;
          }
          if (is_marker_collecting) {
            for (const auto& panel_node : panel_nodes) {
              const auto& nodes = extrinsic_calibration_config->get_nodes();
              auto found = std::find_if(nodes.begin(), nodes.end(), [&](const auto& x) {
                return x.is_camera() && x.get_camera_name() == panel_node.name;
              });
              if (found != nodes.end() && found->is_camera()) {
                multiview_capture->enable_marker_collecting(found->get_camera_name());
              }
            }
          } else {
            for (const auto& panel_node : panel_nodes) {
              const auto& nodes = extrinsic_calibration_config->get_nodes();
              auto found = std::find_if(nodes.begin(), nodes.end(), [&](const auto& x) {
                return x.is_camera() && x.get_camera_name() == panel_node.name;
              });
              if (found != nodes.end() && found->is_camera()) {
                multiview_capture->disable_marker_collecting(found->get_camera_name());
              }
            }
          }
          return true;
        });

    if (!calibration_panel_view_->nodes.empty()) {
      sync_calibration_panel_state();
    }

    calibration_panel_view_->on_action.push_back(
        [this](const std::string& node_id, const std::string& action_id) {
          if (multiview_capture) {
            multiview_capture->dispatch_action(action_id);
          }
          if (calibration_panel_view_->calibration_target_index == 0) {
            spdlog::info("Start calibration");
            extrinsic_calib->dispatch_action(action_id);
            spdlog::info("End calibration");
            return true;
          } else if (calibration_panel_view_->calibration_target_index == 1) {
            intrinsic_calib->dispatch_action(action_id);
            return true;
          } else if (calibration_panel_view_->calibration_target_index == 2) {
            spdlog::info("Start calibration");
            scene_calib->dispatch_action(action_id);
            spdlog::info("End calibration");
            return true;
          }
          return false;
        });
  }

  void init_reconstruction_panel() {
    reconstruction_panel_view_ = std::make_unique<reconstruction_panel_view>();
    reconstruction_panel_view_->resolve_detail_value =
        [this](const stargazer::config_tree_item& item) -> std::optional<std::string> {
      return resolve_panel_detail_value(reconstruction_panel_view_.get(), item);
    };
    auto add_nodes_from_config = [&](const configuration& cfg) {
      for (const auto& node : cfg.get_nodes()) {
        std::string path;
        if (node.contains_param("address")) {
          path = node.get_param<std::string>("address");
        }
        if (node.contains_param("db_path")) {
          path = node.get_param<std::string>("db_path");
        }
        reconstruction_panel_view_->nodes.push_back(
            reconstruction_panel_view::node_def{node.name, path});
      }
    };
    add_nodes_from_config(*point_reconstruction_config);
    add_nodes_from_config(*image_reconstruction_config);
    sync_reconstruction_panel_state();

    reconstruction_panel_view_->is_streaming_changed.push_back(
        [this](const std::vector<reconstruction_panel_view::node_def>& panel_nodes,
               bool is_streaming) {
          if (is_streaming) {
            if (top_bar_view_->reconstruction_pipeline ==
                top_bar_view::ReconstructionPipeline::Marker) {
              if (multiview_capture) {
                return false;
              }

              const auto& nodes = point_reconstruction_config->get_nodes();

              multiview_capture.reset(new capture_pipeline());

              multiview_capture->add_marker_received(
                  [this](const std::map<std::string, marker_frame_data>& marker_frame) {
                    std::map<std::string, std::vector<point_data>> frame;
                    for (const auto& [name, markers] : marker_frame) {
                      std::vector<point_data> points;
                      for (const auto& marker : markers.markers) {
                        points.push_back(
                            point_data{glm::vec2(marker.x, marker.y), marker.r, markers.timestamp});
                      }
                      frame.insert(std::make_pair(name, points));
                    }
                    multiview_point_reconstruction_pipeline_->push_frame(frame);
                  });

              for (const auto& node : nodes) {
                if (node.is_camera()) {
                  const auto camera_name = node.get_camera_name();
                  multiview_capture->enable_marker_collecting(camera_name);
                }
              }

              multiview_capture->run(nodes);

              for (const auto& node : nodes) {
                if (node.is_camera()) {
                  const auto camera_name = node.get_camera_name();
                  int width, height;
                  // Try to get as int64_t first (from JSON), then fall back to float
                  if (node.contains_param("width")) {
                    try {
                      width = static_cast<int>(node.get_param<std::int64_t>("width"));
                    } catch (...) {
                      width = static_cast<int>(std::round(node.get_param<float>("width")));
                    }
                  } else {
                    throw std::runtime_error("width parameter not found for camera: " +
                                             camera_name);
                  }
                  if (node.contains_param("height")) {
                    try {
                      height = static_cast<int>(node.get_param<std::int64_t>("height"));
                    } catch (...) {
                      height = static_cast<int>(std::round(node.get_param<float>("height")));
                    }
                  } else {
                    throw std::runtime_error("height parameter not found for camera: " +
                                             camera_name);
                  }
                  const auto stream = std::make_shared<image_tile_view::stream_info>(
                      camera_name, float2{(float)width, (float)height}, gfx_ctx);
                  const auto runtime_id = std::string{"node:pipeline:"} + node.name;
                  if (const auto runtime_it =
                          reconstruction_panel_view_->tree.runtime_nodes.find(runtime_id);
                      runtime_it != reconstruction_panel_view_->tree.runtime_nodes.end()) {
                    bind_capture_stream_property(runtime_it->second, *stream);
                  }
                  image_tile_view_->streams.push_back(stream);
                }
              }
            } else if (top_bar_view_->reconstruction_pipeline ==
                       top_bar_view::ReconstructionPipeline::Image) {
              if (multiview_capture) {
                return false;
              }

              const auto& nodes = image_reconstruction_config->get_nodes();

              multiview_capture.reset(new capture_pipeline());

              multiview_capture->add_image_received(
                  [this](const std::map<std::string, cv::Mat>& image_frame) {
                    std::map<std::string, cv::Mat> color_image_frame;
                    for (const auto& [name, image] : image_frame) {
                      if (image.channels() == 3 && image.depth() == cv::DataType<uchar>::depth) {
                        color_image_frame[name] = image;
                      }
                    }
                    multiview_image_reconstruction_pipeline_->push_frame(color_image_frame);
                  });

              multiview_capture->run(nodes);

              for (const auto& node : nodes) {
                if (node.is_camera()) {
                  const auto camera_name = node.get_camera_name();
                  int width, height;
                  // Try to get as int64_t first (from JSON), then fall back to float
                  if (node.contains_param("width")) {
                    try {
                      width = static_cast<int>(node.get_param<std::int64_t>("width"));
                    } catch (...) {
                      width = static_cast<int>(std::round(node.get_param<float>("width")));
                    }
                  } else {
                    throw std::runtime_error("width parameter not found for camera: " +
                                             camera_name);
                  }
                  if (node.contains_param("height")) {
                    try {
                      height = static_cast<int>(node.get_param<std::int64_t>("height"));
                    } catch (...) {
                      height = static_cast<int>(std::round(node.get_param<float>("height")));
                    }
                  } else {
                    throw std::runtime_error("height parameter not found for camera: " +
                                             camera_name);
                  }
                  const auto stream = std::make_shared<image_tile_view::stream_info>(
                      camera_name, float2{(float)width, (float)height}, gfx_ctx);
                  const auto runtime_id = std::string{"node:pipeline:"} + node.name;
                  if (const auto runtime_it =
                          reconstruction_panel_view_->tree.runtime_nodes.find(runtime_id);
                      runtime_it != reconstruction_panel_view_->tree.runtime_nodes.end()) {
                    bind_capture_stream_property(runtime_it->second, *stream);
                  }
                  image_tile_view_->streams.push_back(stream);
                  const auto point_stream = std::make_shared<image_tile_view::stream_info>(
                      camera_name, float2{(float)width, (float)height}, gfx_ctx);
                  bind_image_reconstruction_stream_property(camera_name, *point_stream);
                  point_tile_view_->streams.push_back(point_stream);
                }
              }
            }
          } else {
            if (multiview_capture) {
              multiview_capture->stop();
              multiview_capture.reset();

              const auto& nodes = (top_bar_view_->reconstruction_pipeline ==
                                   top_bar_view::ReconstructionPipeline::Marker)
                                      ? point_reconstruction_config->get_nodes()
                                      : image_reconstruction_config->get_nodes();

              for (const auto& node : nodes) {
                if (node.is_camera()) {
                  const auto camera_name = node.get_camera_name();
                  const auto stream_it = std::find_if(
                      image_tile_view_->streams.begin(), image_tile_view_->streams.end(),
                      [&](const auto& x) { return x->name == camera_name; });

                  if (stream_it != image_tile_view_->streams.end()) {
                    image_tile_view_->streams.erase(stream_it);
                  }
                }
              }
            }
          }
          return true;
        });
  }

  void init_gui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Disable ImGui software cursor rendering to use OS native cursor
    io.ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    // ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForVulkan((GLFWwindow*)get_handle(), true);

    static const ImWchar icons_ranges[] = {0xf000, 0xf999, 0};

    {
      const int OVERSAMPLE = true;

      ImFontConfig config_words;
      config_words.OversampleV = OVERSAMPLE;
      config_words.OversampleH = OVERSAMPLE;
      default_font =
          io.Fonts->AddFontFromFileTTF("../fonts/mplus/fonts/ttf/Mplus2-Regular.ttf", 16.0f,
                                       &config_words, io.Fonts->GetGlyphRangesJapanese());

      ImFontConfig config_glyphs;
      config_glyphs.MergeMode = true;
      config_glyphs.OversampleV = OVERSAMPLE;
      config_glyphs.OversampleH = OVERSAMPLE;
      default_font = io.Fonts->AddFontFromMemoryCompressedTTF(font_awesome_compressed_data,
                                                              font_awesome_compressed_size, 14.f,
                                                              &config_glyphs, icons_ranges);
    }
    IM_ASSERT(default_font != NULL);

    {
      const int OVERSAMPLE = true;

      ImFontConfig config_words;
      config_words.OversampleV = OVERSAMPLE;
      config_words.OversampleH = OVERSAMPLE;
      large_font =
          io.Fonts->AddFontFromFileTTF("../fonts/mplus/fonts/ttf/Mplus2-Regular.ttf", 20.0f,
                                       &config_words, io.Fonts->GetGlyphRangesJapanese());

      ImFontConfig config_glyphs;
      config_glyphs.MergeMode = true;
      config_glyphs.OversampleV = OVERSAMPLE;
      config_glyphs.OversampleH = OVERSAMPLE;
      large_font = io.Fonts->AddFontFromMemoryCompressedTTF(font_awesome_compressed_data,
                                                            font_awesome_compressed_size, 20.f,
                                                            &config_glyphs, icons_ranges);
    }
    IM_ASSERT(large_font != NULL);

    {
      ImGuiStyle& style = ImGui::GetStyle();

      style.WindowRounding = 0.0f;
      style.ScrollbarRounding = 0.0f;

      style.Colors[ImGuiCol_WindowBg] = dark_window_background;
      style.Colors[ImGuiCol_Border] = black;
      style.Colors[ImGuiCol_BorderShadow] = transparent;
      style.Colors[ImGuiCol_FrameBg] = dark_window_background;
      style.Colors[ImGuiCol_ScrollbarBg] = scrollbar_bg;
      style.Colors[ImGuiCol_ScrollbarGrab] = scrollbar_grab;
      style.Colors[ImGuiCol_ScrollbarGrabHovered] = scrollbar_grab + 0.1f;
      style.Colors[ImGuiCol_ScrollbarGrabActive] = scrollbar_grab + (-0.1f);
      // style.Colors[ImGuiCol_ComboBg] = dark_window_background;
      style.Colors[ImGuiCol_CheckMark] = regular_blue;
      style.Colors[ImGuiCol_SliderGrab] = regular_blue;
      style.Colors[ImGuiCol_SliderGrabActive] = regular_blue;
      style.Colors[ImGuiCol_Button] = button_color;
      style.Colors[ImGuiCol_ButtonHovered] = button_color + 0.1f;
      style.Colors[ImGuiCol_ButtonActive] = button_color + (-0.1f);
      style.Colors[ImGuiCol_Header] = header_color;
      style.Colors[ImGuiCol_HeaderActive] = header_color + (-0.1f);
      style.Colors[ImGuiCol_HeaderHovered] = header_color + 0.1f;
      style.Colors[ImGuiCol_TitleBg] = title_color;
      style.Colors[ImGuiCol_TitleBgCollapsed] = title_color;
      style.Colors[ImGuiCol_TitleBgActive] = header_color;
    }
  }

  void term_gui() {
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
  }

 public:
  viewer_app()
      : window_base("Stargazer", SCREEN_WIDTH, SCREEN_HEIGHT),
        gfx_ctx(nullptr),
        extrinsic_calib() {}

  void set_graphics_context(graphics_context* ctx) {
    gfx_ctx = ctx;

    // Set graphics context for all existing stream textures
    if (image_tile_view_) {
      for (auto& stream : image_tile_view_->streams) {
        stream->texture.set_context(gfx_ctx);
      }
    }
    if (point_tile_view_) {
      for (auto& stream : point_tile_view_->streams) {
        stream->texture.set_context(gfx_ctx);
      }
    }
    if (contrail_tile_view_) {
      for (auto& stream : contrail_tile_view_->streams) {
        stream->texture.set_context(gfx_ctx);
      }
    }
  }

  virtual void initialize() override {
    capture_config.reset(new configuration("../config/capture.json"));
    point_reconstruction_config.reset(new configuration("../config/point_reconstruction.json"));
    image_reconstruction_config.reset(new configuration("../config/image_reconstruction.json"));
    extrinsic_calibration_config.reset(new configuration("../config/calibration_extrinsic.json"));
    scene_calibration_config.reset(new configuration("../config/calibration_scene.json"));
    calibration_intrinsic_single_camera_config.reset(
        new configuration("../config/calibration_intrinsic_single_camera.json"));

    parameters = std::make_shared<parameters_t>("../config/parameters.json");
    parameters->load();

    init_gui();

    context = std::make_unique<view_context>();
    context->window = this;
    context->default_font = default_font;
    context->large_font = large_font;

    top_bar_view_ = std::make_unique<top_bar_view>();
    image_tile_view_ = std::make_unique<image_tile_view>();
    point_tile_view_ = std::make_unique<image_tile_view>();
    contrail_tile_view_ = std::make_unique<image_tile_view>();

    init_capture_panel();
    init_calibration_panel();
    init_reconstruction_panel();

    view_controller =
        std::make_shared<azimuth_elevation>(glm::u32vec2(0, 0), glm::u32vec2(width, height));

    pose_view_ = std::make_unique<pose_view>();

    // Initialize pose_view with Vulkan resources
    if (gfx_ctx) {
      pose_view_->initialize(gfx_ctx->device.get(), gfx_ctx->physical_device,
                             gfx_ctx->render_pass.get());
    }

    multiview_point_reconstruction_pipeline_ =
        std::make_unique<multiview_point_reconstruction_pipeline>(parameters);

    multiview_point_reconstruction_pipeline_->run(
        point_reconstruction_config->get_nodes("point_reconstruction_pipeline"));

    multiview_image_reconstruction_pipeline_ =
        std::make_unique<multiview_image_reconstruction_pipeline>(parameters);
    multiview_image_reconstruction_pipeline_->run(
        image_reconstruction_config->get_nodes("image_reconstruction_pipeline"));

    extrinsic_calib = std::make_unique<extrinsic_calibration_pipeline>(parameters);

    extrinsic_calib->run(extrinsic_calibration_config->get_nodes("extrinsic_calibration_pipeline"));

    contrail_tile_view_->streams.clear();
    for (const auto& node :
         extrinsic_calibration_config->get_nodes("extrinsic_calibration_pipeline")) {
      if (node.get_type() != stargazer::node_type::contrail_render) {
        continue;
      }
      if (!node.contains_param("camera_name")) {
        continue;
      }
      const auto camera_name = node.get_param<std::string>("camera_name");
      int width = 820;
      int height = 616;
      if (node.contains_param("width")) {
        width = static_cast<int>(node.get_param<std::int64_t>("width"));
      }
      if (node.contains_param("height")) {
        height = static_cast<int>(node.get_param<std::int64_t>("height"));
      }
      const auto stream = std::make_shared<image_tile_view::stream_info>(
          camera_name, float2{(float)width, (float)height}, gfx_ctx);
      stream->property_node_name = node.name;
      stream->property_key = "image";
      contrail_tile_view_->streams.push_back(stream);
    }

    scene_calib = std::make_unique<scene_calibration_pipeline>(parameters);

    scene_calib->run(scene_calibration_config->get_nodes("scene_calibration_pipeline"));

    intrinsic_calib = std::make_unique<intrinsic_calibration_pipeline>(parameters);

    intrinsic_calib->run(
        calibration_intrinsic_single_camera_config->get_nodes("intrinsic_calibration_pipeline"));

    bind_pose_property();

    window_base::initialize();
  }

  virtual void finalize() override {
    // Cleanup pose_view Vulkan resources
    if (pose_view_) {
      pose_view_->cleanup();
    }

    intrinsic_calib->stop();
    extrinsic_calib->stop();
    scene_calib->stop();
    multiview_point_reconstruction_pipeline_->stop();
    multiview_image_reconstruction_pipeline_->stop();

    window_base::finalize();
  }

  virtual void show() override { window_base::show(); }

  virtual void on_close() override {
    window_base::on_close();
    window_manager::get_instance()->exit();
  }

  virtual void on_scroll(double x, double y) override {
    if (view_controller) {
      view_controller->scroll(x, y);
    }
  }

  virtual void update() override {
    if (handle == nullptr) {
      return;
    }

    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    top_bar_view_->render(context.get());
    sync_calibration_panel_state();
    sync_reconstruction_panel_state();

    for (const auto& stream : image_tile_view_->streams) {
      upload_capture_property_stream(stream);
    }
    for (const auto& stream : point_tile_view_->streams) {
      upload_image_reconstruction_property_stream(stream);
    }

    if (top_bar_view_->view_mode == top_bar_view::Mode::Capture) {
      capture_panel_view_->render(context.get());
    } else if (top_bar_view_->view_mode == top_bar_view::Mode::Calibration) {
      calibration_panel_view_->render(context.get());
    } else if (top_bar_view_->view_mode == top_bar_view::Mode::Reconstruction) {
      reconstruction_panel_view_->render(context.get());
    }

    if (top_bar_view_->view_mode == top_bar_view::Mode::Calibration) {
      if (top_bar_view_->view_type == top_bar_view::ViewType::Pose) {
        view_controller->update(mouse_state::get_mouse_state(handle));
        float radius = view_controller->get_radius();
        glm::vec3 forward(0.f, 0.f, 1.f);
        glm::vec3 up(0.0f, 1.0f, 0.0f);
        glm::vec3 view_pos =
            glm::rotate(glm::inverse(view_controller->get_rotation_quaternion()), forward * radius);
        glm::mat4 view = glm::lookAt(view_pos, glm::vec3(0, 0, 0), up);

        context->view = view;

        pose_view_->cameras.clear();
        bind_extrinsic_calibration_pose_property();
        update_pose_from_properties();
      } else if (top_bar_view_->view_type == top_bar_view::ViewType::Contrail) {
        for (const auto& stream : contrail_tile_view_->streams) {
          upload_contrail_property_stream(stream);
        }
      }
    } else if (top_bar_view_->view_mode == top_bar_view::Mode::Reconstruction) {
      if (top_bar_view_->view_type == top_bar_view::ViewType::Pose) {
        view_controller->update(mouse_state::get_mouse_state(handle));
        float radius = view_controller->get_radius();
        glm::vec3 forward(0.f, 0.f, 1.f);
        glm::vec3 up(0.0f, 1.0f, 0.0f);
        glm::vec3 view_pos =
            glm::rotate(glm::inverse(view_controller->get_rotation_quaternion()), forward * radius);
        glm::mat4 view = glm::lookAt(view_pos, glm::vec3(0, 0, 0), up);

        context->view = view;

        pose_view_->cameras.clear();
        bind_pose_property();
        update_pose_from_properties();
      }
    }

    if (top_bar_view_->view_type == top_bar_view::ViewType::Image) {
      image_tile_view_->render(context.get());
    } else if (top_bar_view_->view_type == top_bar_view::ViewType::Contrail) {
      contrail_tile_view_->render(context.get());
    } else if (top_bar_view_->view_type == top_bar_view::ViewType::Point) {
      point_tile_view_->render(context.get());
    } else if (top_bar_view_->view_type == top_bar_view::ViewType::Pose) {
      if (gfx_ctx && pose_view_) {
        vk::CommandBuffer cmd = gfx_ctx->command_buffers[gfx_ctx->current_frame].get();
        pose_view_->render(context.get(), cmd, gfx_ctx->swapchain_extent);
      }
    }

    // Note: ImGui::Render() and ImGui_ImplVulkan_RenderDrawData() are called in
    // graphics_context::swap_buffer()
  }

  virtual void on_char(unsigned int codepoint) override {}
};  // class viewer_app

static void sigint_handler(int) { window_manager::get_instance()->exit(); }

int main() {
  signal(SIGINT, sigint_handler);

  const auto win_mgr = window_manager::get_instance();
  win_mgr->initialize();

  auto window = std::make_shared<viewer_app>();

  window->create();
  auto graphics_ctx = window->create_graphics_context();
  graphics_ctx.attach();
  window->set_graphics_context(&graphics_ctx);

  window->initialize();

  // Set graphics context again after initialization to update all stream textures
  window->set_graphics_context(&graphics_ctx);

  // Initialize ImGui Vulkan backend
  std::unique_ptr<imgui_context> imgui_ctx = std::make_unique<imgui_context>(&graphics_ctx);
  imgui_ctx->initialize();

  window->show();

  while (!win_mgr->should_close()) {
    win_mgr->handle_event();
    imgui_ctx->begin_frame();
    graphics_ctx.begin_frame();
    window->update();
    imgui_ctx->end_frame();
    graphics_ctx.end_frame();
  }

  window->finalize();

  imgui_ctx->cleanup();
  imgui_ctx.reset();

  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  graphics_ctx.detach();
  window->destroy();

  window.reset();

  win_mgr->terminate();

  return 0;
}
