#pragma once

#include <glad/glad.h>
#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <opencv2/core.hpp>

#include "imgui-fonts-fontawesome.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <spdlog/spdlog.h>

#include <array>
#include <cmath>
#include <functional>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/transform.hpp>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

#include "config.hpp"
#include "viewer.hpp"

static ImVec4 from_rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a, bool consistent_color = false) {
  auto res = ImVec4(r / (float)255, g / (float)255, b / (float)255, a / (float)255);
#ifdef FLIP_COLOR_SCHEME
  if (!consistent_color) return flip(res);
#endif
  return res;
}

static const ImVec4 light_blue = from_rgba(
    0, 174, 239, 255,
    true);  // Light blue color for selected elements such as play button glyph when paused
static const ImVec4 regular_blue =
    from_rgba(0, 115, 200, 255, true);  // Checkbox mark, slider grabber
static const ImVec4 light_grey = from_rgba(0xc3, 0xd5, 0xe5, 0xff, true);  // Text
static const ImVec4 dark_window_background = from_rgba(9, 11, 13, 255);
static const ImVec4 almost_white_bg = from_rgba(230, 230, 230, 255, true);
static const ImVec4 black = from_rgba(0, 0, 0, 255, true);
static const ImVec4 transparent = from_rgba(0, 0, 0, 0, true);
static const ImVec4 white = from_rgba(0xff, 0xff, 0xff, 0xff, true);
static const ImVec4 scrollbar_bg = from_rgba(14, 17, 20, 255);
static const ImVec4 scrollbar_grab = from_rgba(54, 66, 67, 255);
static const ImVec4 grey{0.5f, 0.5f, 0.5f, 1.f};
static const ImVec4 dark_grey = from_rgba(30, 30, 30, 255);
static const ImVec4 sensor_header_light_blue = from_rgba(80, 99, 115, 0xff);
static const ImVec4 sensor_bg = from_rgba(36, 44, 51, 0xff);
static const ImVec4 redish = from_rgba(255, 46, 54, 255, true);
static const ImVec4 light_red = from_rgba(255, 146, 154, 255, true);
static const ImVec4 dark_red = from_rgba(200, 46, 54, 255, true);
static const ImVec4 button_color = from_rgba(62, 77, 89, 0xff);
static const ImVec4 header_window_bg = from_rgba(36, 44, 54, 0xff);
static const ImVec4 header_color = from_rgba(62, 77, 89, 255);
static const ImVec4 title_color = from_rgba(27, 33, 38, 255);
static const ImVec4 node_info_color = from_rgba(33, 40, 46, 255);
static const ImVec4 yellow = from_rgba(229, 195, 101, 255, true);
static const ImVec4 yellowish = from_rgba(255, 253, 191, 255, true);
static const ImVec4 green = from_rgba(0x20, 0xe0, 0x20, 0xff, true);
static const ImVec4 dark_sensor_bg = from_rgba(0x1b, 0x21, 0x25, 170);
static const ImVec4 red = from_rgba(233, 0, 0, 255, true);
static const ImVec4 greenish = from_rgba(67, 163, 97, 255);
static const ImVec4 orange = from_rgba(255, 157, 0, 255, true);

inline ImVec4 operator+(const ImVec4& a, const ImVec4& b) {
  return ImVec4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline ImVec4 operator+(const ImVec4& c, float v) {
  return ImVec4(std::max(0.f, std::min(1.f, c.x + v)), std::max(0.f, std::min(1.f, c.y + v)),
                std::max(0.f, std::min(1.f, c.z + v)), std::max(0.f, std::min(1.f, c.w)));
}

template <typename T>
T normalize(const T& in_val, const T& min, const T& max) {
  if (min >= max) return 0;
  return ((in_val - min) / (max - min));
}

template <typename T>
T unnormalize(const T& in_val, const T& min, const T& max) {
  if (min == max) return min;
  return ((in_val * (max - min)) + min);
}

struct float2 {
  float x, y;
};

struct rect {
  float x, y;
  float w, h;
  float area() const { return w * h; }
  rect adjust_ratio(float2 size) const {
    auto H = static_cast<float>(h), W = static_cast<float>(h) * size.x / size.y;
    if (W > w) {
      auto scale = w / W;
      W *= scale;
      H *= scale;
    }

    return {float(floor(x + floor(w - W) / 2)), float(floor(y + floor(h - H) / 2)), W, H};
  }
  rect normalize(const rect& normalize_to) const {
    return rect{::normalize(x, normalize_to.x, normalize_to.x + normalize_to.w),
                ::normalize(y, normalize_to.y, normalize_to.y + normalize_to.h),
                ::normalize(w, 0.f, normalize_to.w), ::normalize(h, 0.f, normalize_to.h)};
  }

  rect unnormalize(const rect& unnormalize_to) const {
    return rect{::unnormalize(x, unnormalize_to.x, unnormalize_to.x + unnormalize_to.w),
                ::unnormalize(y, unnormalize_to.y, unnormalize_to.y + unnormalize_to.h),
                ::unnormalize(w, 0.f, unnormalize_to.w), ::unnormalize(h, 0.f, unnormalize_to.h)};
  }
};

class texture_buffer {
  GLuint texture;

 public:
  GLuint get_gl_handle() const { return texture; }

  texture_buffer() : texture() { glGenTextures(1, &texture); }

  ~texture_buffer() {
    if (texture) {
      glDeleteTextures(1, &texture);
      texture = 0;
    }
  }

  void upload_image(int w, int h, void* data, int format = GL_RGBA) {
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, format, w, h, 0, format, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glBindTexture(GL_TEXTURE_2D, 0);
  }

  void show(const rect& r, float alpha, const rect& normalized_zoom = rect{0, 0, 1, 1}) const {
    if (!texture) return;
    ImGui::Image((ImTextureID)get_gl_handle(), ImVec2{r.w, r.h});
  }
};

struct view_context {
 public:
  ImFont* default_font;
  ImFont* large_font;
  window_base* window;
  glm::mat4 view;

  ImVec2 get_window_size() const;
};

class top_bar_view {
 public:
  enum class Mode {
    Capture,
    Calibration,
    Reconstruction,
  };

  Mode view_mode = Mode::Capture;

  enum class ViewType {
    Image,
    Point,
    Contrail,
    Pose,
  };
  ViewType view_type = ViewType::Image;

  void render(view_context* context);
};

class capture_panel_view {
 public:
  struct node_info {
    std::string name;
    std::string address;
    std::unordered_map<std::string, stargazer::node_param_t> params;
    bool is_streaming = false;

    node_info(const std::string& name, const std::string& address,
              const std::unordered_map<std::string, stargazer::node_param_t>& params)
        : name(name), address(address), params(params) {}
  };
  std::vector<node_info> devices;

  bool is_streaming = false;
  std::vector<std::function<bool(const std::vector<node_info>&, bool)>> is_all_streaming_changed;
  std::vector<std::function<bool(const node_info&)>> is_streaming_changed;

 private:
  float draw_control_panel(view_context* context);

  void draw_controls(view_context* context, float panel_height);

  std::string ip_address;
  std::string gateway_address;
  std::string device_name;
  int node_type_index;

 public:
  std::vector<std::function<void(const std::string&, stargazer::node_type, const std::string&,
                                 const std::string&)>>
      on_add_device;
  std::vector<std::function<void(const std::string&)>> on_remove_device;

  void render(view_context* context);
};

class calibration_panel_view {
 public:
  struct node_info {
    std::string name;
    std::string address;
    std::unordered_map<std::string, stargazer::node_param_t> params;
    bool is_streaming = true;
    size_t num_points = 0;

    node_info(const std::string& name, const std::string& address,
              const std::unordered_map<std::string, stargazer::node_param_t>& params)
        : name(name), address(address), params(params) {}
  };
  std::vector<node_info> devices;
  bool is_marker_collecting = false;
  bool is_streaming = false;
  bool is_masking = false;

  std::vector<std::function<bool(const std::vector<node_info>&, bool)>>
      is_marker_collecting_changed;
  std::vector<std::function<bool(const std::vector<node_info>&, bool)>> is_streaming_changed;
  std::vector<std::function<bool(const std::vector<node_info>&, bool)>> is_masking_changed;
  std::vector<std::function<bool(const std::vector<node_info>&, bool)>> on_calibrate;
  std::vector<std::function<void(const node_info&)>> on_intrinsic_calibration_device_changed;

  int intrinsic_calibration_device_index = 0;
  int calibration_target_index = 0;

  float fx = 0;
  float fy = 0;
  float cx = 0;
  float cy = 0;
  float k0 = 0;
  float k1 = 0;
  float k2 = 0;
  float p0 = 0;
  float p1 = 0;
  float rms = 0;

 private:
  float draw_control_panel(view_context* context);

  void draw_extrinsic_calibration_control_panel(view_context* context);

  void draw_intrinsic_calibration_control_panel(view_context* context);

  void draw_controls(view_context* context, float panel_height);

 public:
  void render(view_context* context);
};

class reconstruction_panel_view {
 public:
  struct node_info {
    std::string name;
    std::string address;
    bool is_streaming = true;

    node_info(const std::string& name, const std::string& address) : name(name), address(address) {}
  };
  std::vector<node_info> devices;
  bool is_streaming = false;
  bool is_recording = false;
  int source = 0;

  std::vector<std::function<bool(const std::vector<node_info>&, bool)>> is_streaming_changed;
  std::vector<std::function<bool(const std::vector<node_info>&, bool)>> is_recording_changed;

 private:
  float draw_control_panel(view_context* context);

  void draw_controls(view_context* context, float panel_height);

 public:
  void render(view_context* context);
};

class image_tile_view {
 public:
  struct stream_info {
    std::string name;
    float2 size;
    texture_buffer texture;

    stream_info(std::string name, float2 size) : name(name), size(size) {}
  };
  std::vector<std::shared_ptr<stream_info>> streams;
  std::map<stream_info*, int> stream_index;

 private:
  std::map<int, rect> generate_layout(
      const rect& r, int top_bar_height, size_t factor,
      const std::vector<std::shared_ptr<stream_info>>& active_streams,
      std::map<stream_info*, int>& stream_index);

  float evaluate_layout(const std::map<int, rect>& l);

  std::map<int, rect> calc_layout(const rect& r);

  void draw_stream_header(view_context* context, const rect& stream_rect);

 public:
  void render(view_context* context);
};

class pose_view {
 public:
  struct camera_t {
    int width;  /**< Width of the image in pixels */
    int height; /**< Height of the image in pixels */
    float ppx;  /**< Horizontal coordinate of the principal point of the image, as a pixel offset
                   from the left edge */
    float ppy;  /**< Vertical coordinate of the principal point of the image, as a pixel offset from
                   the top edge */
    float fx;   /**< Focal length of the image plane, as a multiple of pixel width */
    float fy;   /**< Focal length of the image plane, as a multiple of pixel height */
    std::array<float, 5> coeffs;
    glm::mat4 pose;
  };
  std::map<std::string, camera_t> cameras;
  std::vector<glm::vec3> points;
  glm::mat4 axis;

  void render(view_context* context);
};

class azimuth_elevation {
 public:
  azimuth_elevation(glm::u32vec2 screen_offset, glm::u32vec2 screen_size)
      : screen_offset(screen_offset),
        screen_size(screen_size),
        start_position(0.0f, 0.0f, 0.0f),
        current_position(0.0f, 0.0f, 0.0f),
        drag_rotation(false),
        drag_transition(false) {
    reset();
  }

  glm::mat4 get_rotation_matrix() { return glm::toMat4(current_rotation); }

  glm::quat get_rotation_quaternion() { return current_rotation; }

  glm::mat4 get_translation_matrix() { return translation_matrix; }

  glm::mat4 get_translation_delta_matrix() { return translation_delta_matrix; }

  void set_radius_translation(float value) { radius_translation = value; }

  void set_radius(float value) { radius = value; }
  float get_radius() const { return radius; }

  float get_screen_w() const { return (float)screen_size.x; }
  float get_screen_h() const { return (float)screen_size.y; }
  float get_screen_x() const { return (float)screen_offset.x; }
  float get_screen_y() const { return (float)screen_offset.y; }

  glm::quat quat_from_screen(glm::vec3 from, glm::vec3 to) {
    const auto vector = (to - from) * 1.f /*radius*/;

    angle.x += vector.y / get_screen_w();
    angle.y += vector.x / get_screen_h();

    return glm::quat_cast(glm::rotate(angle.x, glm::vec3(1.f, 0.f, 0.f)) *
                          glm::rotate(angle.t, glm::vec3(0.f, 1.f, 0.f)));
  }
  glm::vec3 screen_to_vector(float sx, float sy) { return glm::vec3(sx, sy, 0); }

  bool on_target(int x, int y) const {
    x -= screen_offset.x;
    y -= screen_offset.y;

    return (x >= 0) && (y >= 0) && (x < static_cast<int>(screen_size.x)) &&
           (y < static_cast<int>(screen_size.y));
  }

  void begin_rotation(int x, int y) {
    if (on_target(x, y)) {
      drag_rotation = true;
      previous_rotation = current_rotation;
      start_position = screen_to_vector((float)x, (float)y);
    }
  }
  void update_rotation(float x, float y) {
    if (drag_rotation) {
      current_position = screen_to_vector(x, y);
      current_rotation = quat_from_screen(start_position, current_position);
      start_position = current_position;
    }
  }
  void end_rotation() { drag_rotation = false; }

  void begin_transition(int x, int y) {
    if (on_target(x, y)) {
      drag_transition = true;
      previsou_position.x = (float)x;
      previsou_position.y = (float)y;
    }
  }
  void update_transition(int x, int y, bool zoom) {
    if (drag_transition) {
      float delta_x = (previsou_position.x - (float)x) * radius_translation / get_screen_w();
      float delta_y = (previsou_position.y - (float)y) * radius_translation / get_screen_h();

      if (!zoom) {
        translation_delta_matrix = glm::translate(glm::vec3(-2 * delta_x, 2 * delta_y, 0.0f));
        translation_matrix = translation_delta_matrix * translation_matrix;
      } else {
        translation_delta_matrix = glm::translate(glm::vec3(0.0f, 0.0f, 5 * delta_y));
        translation_matrix = translation_delta_matrix * translation_matrix;
      }

      previsou_position.x = (float)x;
      previsou_position.y = (float)y;
    }
  }
  void end_transition() {
    translation_delta_matrix = glm::identity<glm::mat4>();
    drag_transition = false;
  }

  void reset() {
    angle = glm::vec2(0.f, 0.f);
    previous_rotation = glm::quat(1.f, 0.f, 0.f, 0.f);
    current_rotation = glm::angleAxis(glm::radians(30.f), glm::vec3(1.f, 0.f, 0.f));
    translation_matrix = glm::translate(glm::vec3(0.f, 1.f, 0.f));
    translation_delta_matrix = glm::identity<glm::mat4>();
    drag_rotation = false;
    radius_translation = 1.0f;
    radius = 5.0f;
  }

  void update(mouse_state mouse);

  void scroll(double x, double y) { radius -= (static_cast<float>(y) * 1.0f); }

 private:
  static glm::vec2 get_center(int width, int height) {
    return glm::vec2(width * 0.5f, height * 0.5f);
  }
  static glm::vec2 get_center(const glm::u32vec2& screen) {
    return glm::vec2(screen.x * 0.5f, screen.y * 0.5f);
  }
  glm::u32vec2 screen_offset;
  glm::u32vec2 screen_size;

  float radius;
  float radius_translation;

  glm::vec2 angle;
  glm::vec3 start_position;
  glm::vec3 previsou_position;
  glm::vec3 current_position;
  glm::quat previous_rotation;
  glm::quat current_rotation;

  glm::mat4 translation_matrix;
  glm::mat4 translation_delta_matrix;

  bool drag_rotation, drag_transition;

  mouse_state previous_state;
};
