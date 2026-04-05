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
#include <set>
#include <sstream>
#include <vulkan/vulkan.hpp>

#include "coalsack_math_conv.hpp"
#include "config.hpp"
#include "gui.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "parameters.hpp"
#include "pipeline.hpp"
#include "render3d.hpp"
#include "viewer.hpp"
#include "views.hpp"

using namespace stargazer;

const int SCREEN_WIDTH = 1680;
const int SCREEN_HEIGHT = 1050;

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
  std::unique_ptr<pipeline_panel_view> panel_view_;
  std::unique_ptr<image_tile_view> image_tile_view_;
  std::unique_ptr<image_tile_view> point_tile_view_;
  std::unique_ptr<image_tile_view> contrail_tile_view_;
  std::unique_ptr<pose_view> pose_view_;
  std::shared_ptr<azimuth_elevation> view_controller;

  std::shared_ptr<parameters_t> parameters;
  std::string config_path_;
  std::unique_ptr<stargazer::pipeline> pipeline_;
  std::unique_ptr<configuration> config_;
  bool pipeline_running_ = false;

  std::optional<coalsack::property_value> query_runtime_node_property(
      const stargazer::config_tree_ref& ref, const std::string& key) const {
    if (pipeline_) {
      return pipeline_->get_node_property(ref.node_name, key);
    }
    return std::nullopt;
  }

  void bind_pose_property() {
    if (!pose_view_ || !config_) return;
    pose_view_->camera_sources.clear();
    pose_view_->point_sources.clear();
    pose_view_->axis_source = {};

    const auto nodes = config_->get_nodes();
    for (const auto& node : nodes) {
      if (node.get_type() == stargazer::node_type::epipolar_reconstruction) {
        const stargazer::config_tree_ref ref{node.name};
        pose_view_->axis_source = {ref, "axis"};
        for (const auto& camera_node : nodes) {
          const auto camera_name = try_get_node_camera_name(camera_node);
          if (!camera_name.has_value()) {
            continue;
          }
          pose_view_->camera_sources[*camera_name] = {ref, "camera." + *camera_name};
        }
      }

      if (node.get_type() == stargazer::node_type::extrinsic_calibration) {
        const stargazer::config_tree_ref ref{node.name};
        for (const auto& [input_name, _input] : node.inputs) {
          const std::string prefix{"camera."};
          if (input_name.rfind(prefix, 0) != 0) {
            continue;
          }
          const auto camera_name = input_name.substr(prefix.size());
          pose_view_->camera_sources[camera_name] = {ref, "calibrated." + camera_name};
        }
      }

      if (node.get_type() == stargazer::node_type::marker_property) {
        const stargazer::config_tree_ref ref{node.name};
        pose_view_->point_sources.push_back({ref, "markers"});
      }
    }
  }

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

  static int get_node_dimension(const stargazer::node_def& node, const std::string& key) {
    if (!node.contains_param(key)) {
      return 0;
    }
    try {
      return static_cast<int>(node.get_param<std::int64_t>(key));
    } catch (...) {
      return static_cast<int>(std::round(node.get_param<float>(key)));
    }
  }

  static std::optional<std::string> try_get_node_camera_name(const stargazer::node_def& node) {
    if (!node.contains_param("camera_name")) {
      return std::nullopt;
    }
    const auto camera_name = node.get_param<std::string>("camera_name");
    if (camera_name.empty()) {
      return std::nullopt;
    }
    return camera_name;
  }

  static std::string get_stream_name(const stargazer::node_def& node) {
    const auto camera_name = try_get_node_camera_name(node);
    return camera_name.has_value() ? *camera_name : node.name;
  }

  void rebuild_panel_state() {
    if (!panel_view_ || !config_) {
      return;
    }

    panel_view_->has_gate = false;
    for (const auto& node : config_->get_nodes()) {
      panel_view_->has_gate =
          panel_view_->has_gate || node.get_type() == stargazer::node_type::gate;
    }

    panel_view_->tree = build_config_tree(*config_);
  }

  image_tile_view* target_tile_view(const std::string& target) const {
    if (target == "image") {
      return image_tile_view_.get();
    }
    if (target == "contrail") {
      return contrail_tile_view_.get();
    }
    if (target == "point") {
      return point_tile_view_.get();
    }
    return nullptr;
  }

  void add_streams_from_properties(const std::vector<stargazer::node_def>& nodes) {
    for (const auto& node : nodes) {
      bool has_stream_target = false;
      for (const auto& property : node.properties) {
        if (target_tile_view(property.target)) {
          has_stream_target = true;
          break;
        }
      }
      if (!has_stream_target) {
        continue;
      }

      const int width = get_node_dimension(node, "width");
      const int height = get_node_dimension(node, "height");
      const auto stream_name = get_stream_name(node);

      for (const auto& property : node.properties) {
        auto* tile_view = target_tile_view(property.target);
        if (!tile_view) {
          continue;
        }

        const auto stream = std::make_shared<image_tile_view::stream_info>(
            stream_name, float2{(float)width, (float)height}, gfx_ctx);
        stream->property_node_name = node.name;
        stream->property_key = property.source_key;
        stream->property_resource_kind = property.resource_kind;
        stream->property_selector = property.selector;
        tile_view->streams.push_back(stream);
      }
    }
  }

  void remove_streams_from_properties(const std::vector<stargazer::node_def>& nodes) {
    for (const auto& node : nodes) {
      const auto stream_name = get_stream_name(node);

      for (const auto& property : node.properties) {
        auto* tile_view = target_tile_view(property.target);
        if (!tile_view) {
          continue;
        }

        const auto stream_it = std::find_if(
            tile_view->streams.begin(), tile_view->streams.end(), [&](const auto& stream) {
              return stream->name == stream_name && stream->property_node_name == node.name &&
                     stream->property_key == property.source_key;
            });
        if (stream_it != tile_view->streams.end()) {
          tile_view->streams.erase(stream_it);
        }
      }
    }
  }

  bool upload_property_stream(const std::shared_ptr<image_tile_view::stream_info>& stream) const {
    if (!stream || stream->property_node_name.empty() || stream->property_key.empty()) {
      return false;
    }
    if (!stream->property_resource_kind.empty() && stream->property_resource_kind != "raw" &&
        stream->property_resource_kind != "feature") {
      return false;
    }

    if (!pipeline_ || !stream || stream->property_node_name.empty() ||
        stream->property_key.empty()) {
      return false;
    }

    const auto value =
        pipeline_->get_node_property(stream->property_node_name, stream->property_key);
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

  void init_panel() {
    panel_view_ = std::make_unique<pipeline_panel_view>();
    panel_view_->resolve_detail_value =
        [this](const stargazer::config_tree_item& item) -> std::optional<std::string> {
      return resolve_panel_detail_value(panel_view_.get(), item);
    };
    rebuild_panel_state();

    panel_view_->is_all_streaming_changed.push_back([this](bool is_streaming) {
      const auto nodes = config_->get_nodes();
      if (is_streaming) {
        if (pipeline_running_) {
          return false;
        }
        pipeline_running_ = true;
        pipeline_->start();
        add_streams_from_properties(nodes);
      } else {
        if (!pipeline_running_) {
          return true;
        }
        pipeline_running_ = false;
        pipeline_->pause();
        remove_streams_from_properties(nodes);
        panel_view_->is_marker_collecting = false;
      }
      return true;
    });

    panel_view_->is_marker_collecting_changed.push_back([this](bool is_marker_collecting) {
      std::unordered_set<std::string> seen_camera_names;
      for (const auto& node : config_->get_nodes()) {
        const auto camera_name = try_get_node_camera_name(node);
        if (!camera_name.has_value() || !seen_camera_names.insert(*camera_name).second) {
          continue;
        }
        if (is_marker_collecting) {
          pipeline_->enable_marker_collecting(*camera_name);
        } else {
          pipeline_->disable_marker_collecting(*camera_name);
        }
      }
      return true;
    });

    panel_view_->on_action.push_back([this](const std::string&, const std::string& action_id) {
      pipeline_->dispatch_action(action_id);
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
  explicit viewer_app(std::string config_path)
      : window_base("Stargazer", SCREEN_WIDTH, SCREEN_HEIGHT),
        gfx_ctx(nullptr),
        config_path_(std::move(config_path)) {}

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
    config_ = std::make_unique<configuration>(config_path_);

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

    init_panel();

    view_controller =
        std::make_shared<azimuth_elevation>(glm::u32vec2(0, 0), glm::u32vec2(width, height));

    pose_view_ = std::make_unique<pose_view>();

    // Initialize pose_view with Vulkan resources
    if (gfx_ctx) {
      pose_view_->initialize(gfx_ctx->device.get(), gfx_ctx->physical_device,
                             gfx_ctx->render_pass.get());
    }

    pipeline_ = std::make_unique<stargazer::pipeline>(parameters);
    pipeline_->run(config_->get_nodes());

    bind_pose_property();

    window_base::initialize();
  }

  virtual void finalize() override {
    // Cleanup pose_view Vulkan resources
    if (pose_view_) {
      pose_view_->cleanup();
    }

    if (pipeline_running_) {
      pipeline_->pause();
      pipeline_running_ = false;
    }
    if (pipeline_) {
      pipeline_->stop();
    }

    if (image_tile_view_) image_tile_view_->streams.clear();
    if (point_tile_view_) point_tile_view_->streams.clear();
    if (contrail_tile_view_) contrail_tile_view_->streams.clear();

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

    for (const auto& stream : image_tile_view_->streams) {
      upload_property_stream(stream);
    }
    for (const auto& stream : point_tile_view_->streams) {
      upload_property_stream(stream);
    }

    for (const auto& stream : contrail_tile_view_->streams) {
      upload_property_stream(stream);
    }

    panel_view_->render(context.get());

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

int main(int argc, char** argv) {
  signal(SIGINT, sigint_handler);

  if (argc != 2) {
    std::cerr << "Usage: stargazer_viewer <config.json>" << std::endl;
    return 1;
  }

  const auto win_mgr = window_manager::get_instance();
  win_mgr->initialize();

  auto window = std::make_shared<viewer_app>(argv[1]);

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
