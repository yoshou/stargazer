#include "views.hpp"

using namespace stargazer;

namespace {
struct float3 {
  float x, y, z;

  float length() const { return sqrt(x * x + y * y + z * z); }

  float3 normalize() const {
    return (length() > 0) ? float3{x / length(), y / length(), z / length()} : *this;
  }
};

inline float3 cross(const float3 &a, const float3 &b) {
  return {a.y * b.z - b.y * a.z, a.x * b.z - b.x * a.z, a.x * b.y - a.y * b.x};
}

inline float3 operator*(const float3 &a, float t) { return {a.x * t, a.y * t, a.z * t}; }

inline float3 operator/(const float3 &a, float t) { return {a.x / t, a.y / t, a.z / t}; }

inline float3 operator+(const float3 &a, const float3 &b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline float3 operator-(const float3 &a, const float3 &b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline float3 lerp(const float3 &a, const float3 &b, float t) { return b * t + a * (1 - t); }

inline float3 lerp(const std::array<float3, 4> &rect, const float2 &p) {
  auto v1 = lerp(rect[0], rect[1], p.x);
  auto v2 = lerp(rect[3], rect[2], p.x);
  return lerp(v1, v2, p.y);
}

inline float operator*(const float3 &a, const float3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
}  // namespace

struct matrix4 {
  float mat[4][4];

  static matrix4 identity() {
    matrix4 m;
    for (int i = 0; i < 4; i++) m.mat[i][i] = 1.f;
    return m;
  }

  operator float *() const { return (float *)&mat; }
};

inline matrix4 operator*(const matrix4 &a, const matrix4 &b) {
  matrix4 res;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      float sum = 0.0f;
      for (int k = 0; k < 4; k++) {
        sum += a.mat[i][k] * b.mat[k][j];
      }
      res.mat[i][j] = sum;
    }
  }
  return res;
}

namespace ImGui {
class ScopePushFont {
 public:
  ScopePushFont(ImFont *font) { PushFont(font); }
  ~ScopePushFont() { PopFont(); }
};
class ScopePushStyleColor {
 public:
  ScopePushStyleColor(ImGuiCol idx, const ImVec4 &col) { PushStyleColor(idx, col); }
  ~ScopePushStyleColor() { PopStyleColor(); }
};
class ScopePushStyleVar {
 public:
  ScopePushStyleVar(ImGuiStyleVar idx, float val) { PushStyleVar(idx, val); }
  ScopePushStyleVar(ImGuiStyleVar idx, const ImVec2 &val) { PushStyleVar(idx, val); }
  ~ScopePushStyleVar() { PopStyleVar(); }
};
}  // namespace ImGui

#define CONCAT_(x, y) x##y
#define CONCAT(x, y) CONCAT_(x, y)
#define ImGui_ScopePushFont(f) ImGui::ScopePushFont CONCAT(scope_push_font, __LINE__)(f)
#define ImGui_ScopePushStyleColor(idx, col) \
  ImGui::ScopePushStyleColor CONCAT(scope_push_style_color, __LINE__)(idx, col)
#define ImGui_ScopePushStyleVar(idx, val) \
  ImGui::ScopePushStyleVar CONCAT(scope_push_style_var, __LINE__)(idx, val)

static std::vector<const char *> get_string_pointers(const std::vector<std::string> &vec) {
  std::vector<const char *> res;
  for (auto &&s : vec) res.push_back(s.c_str());
  return res;
}

struct to_string {
  std::ostringstream ss;
  template <class T>
  to_string &operator<<(const T &val) {
    ss << val;
    return *this;
  }
  operator std::string() const { return ss.str(); }
};

struct textual_icon {
  explicit constexpr textual_icon(const char8_t (&unicode_icon)[4])
      : _icon{unicode_icon[0], unicode_icon[1], unicode_icon[2], unicode_icon[3]} {}
  operator const char *() const { return reinterpret_cast<const char *>(_icon.data()); }

 private:
  std::array<char8_t, 5> _icon;
};

namespace textual_icons {
// A note to a maintainer - preserve order when adding values to avoid duplicates
static const textual_icon file_movie{u8"\uf008"};
static const textual_icon times{u8"\uf00d"};
static const textual_icon download{u8"\uf019"};
static const textual_icon refresh{u8"\uf021"};
static const textual_icon lock{u8"\uf023"};
static const textual_icon camera{u8"\uf030"};
static const textual_icon video_camera{u8"\uf03d"};
static const textual_icon edit{u8"\uf044"};
static const textual_icon step_backward{u8"\uf048"};
static const textual_icon play{u8"\uf04b"};
static const textual_icon pause{u8"\uf04c"};
static const textual_icon stop{u8"\uf04d"};
static const textual_icon step_forward{u8"\uf051"};
static const textual_icon plus_circle{u8"\uf055"};
static const textual_icon question_mark{u8"\uf059"};
static const textual_icon info_circle{u8"\uf05a"};
static const textual_icon fix_up{u8"\uf062"};
static const textual_icon minus{u8"\uf068"};
static const textual_icon exclamation_triangle{u8"\uf071"};
static const textual_icon shopping_cart{u8"\uf07a"};
static const textual_icon bar_chart{u8"\uf080"};
static const textual_icon upload{u8"\uf093"};
static const textual_icon square_o{u8"\uf096"};
static const textual_icon unlock{u8"\uf09c"};
static const textual_icon floppy{u8"\uf0c7"};
static const textual_icon square{u8"\uf0c8"};
static const textual_icon bars{u8"\uf0c9"};
static const textual_icon caret_down{u8"\uf0d7"};
static const textual_icon repeat{u8"\uf0e2"};
static const textual_icon circle{u8"\uf111"};
static const textual_icon check_square_o{u8"\uf14a"};
static const textual_icon cubes{u8"\uf1b3"};
static const textual_icon toggle_off{u8"\uf204"};
static const textual_icon toggle_on{u8"\uf205"};
static const textual_icon connectdevelop{u8"\uf20e"};
static const textual_icon usb_type{u8"\uf287"};
static const textual_icon braille{u8"\uf2a1"};
static const textual_icon window_maximize{u8"\uf2d0"};
static const textual_icon window_restore{u8"\uf2d2"};
static const textual_icon grid{u8"\uf1cb"};
static const textual_icon exit{u8"\uf011"};
static const textual_icon see_less{u8"\uf070"};
static const textual_icon dotdotdot{u8"\uf141"};
static const textual_icon link{u8"\uf08e"};
static const textual_icon throphy{u8"\uF091"};
static const textual_icon metadata{u8"\uF0AE"};
static const textual_icon check{u8"\uF00C"};
static const textual_icon mail{u8"\uF01C"};
static const textual_icon cube{u8"\uf1b2"};
static const textual_icon measure{u8"\uf545"};
static const textual_icon wifi{u8"\uf1eb"};
}  // namespace textual_icons

void azimuth_elevation::update(mouse_state mouse) {
  auto mouse_x = static_cast<int>(mouse.x);
  auto mouse_y = static_cast<int>(mouse.y);

  if (!on_target(mouse_x, mouse_y)) {
    return;
  }

  if (mouse.right_button == GLFW_PRESS && on_target(mouse_x, mouse_y)) {
    if (previous_state.right_button == GLFW_RELEASE) {
      begin_rotation(mouse_x, mouse_y);
    } else {
      update_rotation(mouse_x, mouse_y);
    }
  } else if (mouse.right_button == GLFW_RELEASE) {
    end_rotation();
  }

  if (mouse.middle_button == GLFW_PRESS && on_target(mouse_x, mouse_y)) {
    if (previous_state.middle_button == GLFW_RELEASE) {
      begin_transition(mouse_x, mouse_y);
    } else {
      update_transition(mouse_x, mouse_y, false);
    }
  } else
  // else if (mouse.get_middle_button() == BUTTON_STATE::RELEASED && mouse.get_left_button() ==
  // BUTTON_STATE::RELEASED)
  {
    end_transition();
  }
  previous_state = mouse;
}

ImVec2 view_context::get_window_size() const {
  int width, height;
  window_manager::get_instance()->get_window_size(window, &width, &height);
  return ImVec2{static_cast<float>(width), static_cast<float>(height)};
}

void top_bar_view::render(view_context *context) {
  auto flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
               ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings;

  const auto top_bar_height = 50;
  const auto button_width = 150;

  const auto window_size = context->get_window_size();

  ImGui::SetNextWindowPos({0, 0});
  ImGui::SetNextWindowSize({window_size.x, top_bar_height});

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
  ImGui::PushStyleColor(ImGuiCol_WindowBg, button_color);
  ImGui::Begin("Toolbar Panel", nullptr, flags);

  ImGui::PushFont(context->large_font);
  ImGui::PushStyleColor(ImGuiCol_Border, black);

  {
    ImGui::SetCursorPosX(0);
    ImGui::PushStyleColor(ImGuiCol_Text, (view_mode != Mode::Capture) ? light_grey : light_blue);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg,
                          (view_mode != Mode::Capture) ? light_grey : light_blue);
    if (ImGui::Button("Capture", {button_width, top_bar_height})) {
      view_mode = Mode::Capture;
    }
    ImGui::PopStyleColor(2);
  }

  ImGui::SameLine();

  {
    ImGui::SetCursorPosX(button_width);
    auto pos1 = ImGui::GetCursorScreenPos();

    ImGui::PushStyleColor(ImGuiCol_Text,
                          (view_mode != Mode::Calibration) ? light_grey : light_blue);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg,
                          (view_mode != Mode::Calibration) ? light_grey : light_blue);
    if (ImGui::Button("Calibration", {button_width, top_bar_height})) {
      view_mode = Mode::Calibration;
    }

    ImGui::PopStyleColor(2);
  }

  ImGui::SameLine();

  {
    ImGui::SetCursorPosX(button_width * 2);
    auto pos1 = ImGui::GetCursorScreenPos();

    ImGui::PushStyleColor(ImGuiCol_Text,
                          (view_mode != Mode::Reconstruction) ? light_grey : light_blue);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg,
                          (view_mode != Mode::Reconstruction) ? light_grey : light_blue);
    if (ImGui::Button("Reconstruction", {button_width, top_bar_height})) {
      view_mode = Mode::Reconstruction;
    }

    ImGui::PopStyleColor(2);
  }

  ImGui::SameLine();

  {
    const auto buttons = 4;
    const auto button_width = 80;

    ImGui::SetCursorPosX(window_size.x - button_width * (buttons));
    ImGui::PushStyleColor(ImGuiCol_Text, (view_type != ViewType::Image) ? light_grey : light_blue);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg,
                          (view_type != ViewType::Image) ? light_grey : light_blue);
    if (ImGui::Button("Image", {button_width, top_bar_height})) {
      view_type = ViewType::Image;
    }
    ImGui::PopStyleColor(2);
    ImGui::SameLine();

    ImGui::SetCursorPosX(window_size.x - button_width * (buttons - 1));

    ImGui::PushStyleColor(ImGuiCol_Text, (view_type != ViewType::Point) ? light_grey : light_blue);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg,
                          (view_type != ViewType::Point) ? light_grey : light_blue);
    if (ImGui::Button("Point", {button_width, top_bar_height})) {
      view_type = ViewType::Point;
    }

    ImGui::PopStyleColor(2);
    ImGui::SameLine();

    ImGui::SetCursorPosX(window_size.x - button_width * (buttons - 2));

    ImGui::PushStyleColor(ImGuiCol_Text,
                          (view_type != ViewType::Contrail) ? light_grey : light_blue);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg,
                          (view_type != ViewType::Contrail) ? light_grey : light_blue);
    if (ImGui::Button("Contrail", {button_width, top_bar_height})) {
      view_type = ViewType::Contrail;
    }

    ImGui::PopStyleColor(2);
    ImGui::SameLine();

    ImGui::SetCursorPosX(window_size.x - button_width * (buttons - 3));

    ImGui::PushStyleColor(ImGuiCol_Text, (view_type != ViewType::Pose) ? light_grey : light_blue);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg,
                          (view_type != ViewType::Pose) ? light_grey : light_blue);
    if (ImGui::Button("Pose", {button_width, top_bar_height})) {
      view_type = ViewType::Pose;
    }

    ImGui::PopStyleColor(2);
  }

  ImGui::PopStyleColor();
  ImGui::PopFont();

  ImGui::End();
  ImGui::PopStyleColor();
  ImGui::PopStyleVar();
}

void capture_panel_view::draw_controls(view_context *context, float panel_height)

{
  std::vector<std::function<void()>> draw_later;

  auto panel_width = 350;
  auto header_h = panel_height;
  ImColor device_header_background_color = title_color;
  const float left_space = 3.f;
  const float upper_space = 3.f;

  // if (is_ip_device)
  header_h += 32;

  // draw controls
  {
    const auto pos = ImGui::GetCursorPos();
    const float vertical_space_before_device_control = 10.0f;
    const float horizontal_space_before_device_control = 3.0f;
    auto device_panel_pos = ImVec2{pos.x + horizontal_space_before_device_control,
                                   pos.y + vertical_space_before_device_control};
    ImGui::SetCursorPos(device_panel_pos);
    const float device_panel_height = draw_control_panel(context);
    ImGui::SetCursorPos({device_panel_pos.x, device_panel_pos.y + device_panel_height});
  }

  {
    const auto panel_y = 50;
    const auto panel_width = 350;

    ImGui::PushFont(context->large_font);
    ImGui::PushStyleColor(ImGuiCol_PopupBg, from_rgba(230, 230, 230, 255));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, from_rgba(0, 0xae, 0xff, 255));
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, from_rgba(255, 255, 255, 255));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(5, 5));

    std::string add_source_button_text = to_string() << " " << textual_icons::plus_circle
                                                     << "  Add Source\t\t\t\t\t\t\t\t\t\t\t";
    if (ImGui::Button(add_source_button_text.c_str(), {panel_width - 1, panel_y})) {
      node_type_index = 0;
      ip_address = "192.168.0.1";
      gateway_address = "192.168.0.254";
      device_name = "camera";
      ImGui::OpenPopup("Network Device");
    }

    ImGui::PopFont();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    ImGui::PopStyleColor();
    ImGui::PopStyleColor();
  }

  for (auto &device : devices) {
    ImVec2 initial_screen_pos = ImGui::GetCursorScreenPos();

    // Upper Space
    ImGui::GetWindowDrawList()->AddRectFilled(
        {initial_screen_pos.x, initial_screen_pos.y},
        {initial_screen_pos.x + panel_width, initial_screen_pos.y + upper_space}, ImColor(black));
    // if (draw_device_outline)
    {
      // Upper Line
      ImGui::GetWindowDrawList()->AddLine(
          {initial_screen_pos.x, initial_screen_pos.y + upper_space},
          {initial_screen_pos.x + panel_width, initial_screen_pos.y + upper_space},
          ImColor(header_color));
    }
    // Device Header area
    ImGui::GetWindowDrawList()->AddRectFilled(
        {initial_screen_pos.x + 1, initial_screen_pos.y + upper_space + 1},
        {initial_screen_pos.x + panel_width, initial_screen_pos.y + header_h + upper_space},
        device_header_background_color);

    auto pos = ImGui::GetCursorPos();
    ImGui::PushFont(context->large_font);
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)device_header_background_color);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)device_header_background_color);

    // draw name
    {
      const ImVec2 name_pos = {pos.x + 9, pos.y + 17};
      ImGui::SetCursorPos(name_pos);
      std::stringstream ss;
      // if (dev.supports(RS2_CAMERA_INFO_NAME))
      //     ss << dev.get_info(RS2_CAMERA_INFO_NAME);
      // if (is_ip_device)
      {
        ImGui::Text(" %s", device.name.c_str());

        ImGui::PushFont(context->large_font);
        ImGui::Text("\tNetwork Device at %s", device.address.c_str());
        ImGui::PopFont();
      }
    }

    ImGui::PopFont();

    // draw x button
    {
      bool _allow_remove = true;
      const auto id = device.name;

      ImGui::PushFont(context->large_font);
      ImGui::PushStyleColor(ImGuiCol_Text, light_grey);
      ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, light_grey);
      ImGui::PushStyleColor(ImGuiCol_PopupBg, almost_white_bg);
      ImGui::PushStyleColor(ImGuiCol_HeaderHovered, light_blue);
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(5, 5));
      if (_allow_remove) {
        ImGui::Columns(1);
        float horizontal_distance_from_right_side_of_panel = 47;
        ImGui::SetCursorPos({panel_width - horizontal_distance_from_right_side_of_panel,
                             pos.y + 9 + (header_h - panel_height) / 2});
        std::string remove_source_button_label = to_string() << textual_icons::times << "##" << id;
        if (ImGui::Button(remove_source_button_label.c_str(), {33, 35})) {
          for (auto &f : on_remove_device) {
            f(device.name);
          }
        }

        if (ImGui::IsItemHovered()) {
          ImGui::SetTooltip(
              "Remove selected device from current view\n(can be restored by clicking Add Source)");
          // window.link_hovered();
        }
      }
      ImGui::PopStyleColor(4);
      ImGui::PopStyleVar();
      ImGui::PopFont();
    }

    ImGui::SetCursorPos({0, pos.y + header_h});

    auto windows_width = ImGui::GetContentRegionMax().x;

    int info_control_panel_height = 0;
    pos = ImGui::GetCursorPos();

    ImGui::SetCursorPos({0, pos.y + info_control_panel_height});
    ImGui::PopStyleColor(2);

    auto sensor_top_y = ImGui::GetCursorPosY();
    // ImGui::SetContentRegionWidth(windows_width - 36);

    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, sensor_bg);
    ImGui::PushStyleColor(ImGuiCol_Text, light_grey);
    ImGui::PushFont(context->large_font);

    // draw streaming
    {
      draw_later.push_back([pos, windows_width, this, context, &device]() {
        const auto id = device.name;

        ImGui::SetCursorPos({windows_width - 35, pos.y + 3});
        ImGui_ScopePushFont(context->default_font);

        ImGui_ScopePushStyleColor(ImGuiCol_Button, sensor_bg);
        ImGui_ScopePushStyleColor(ImGuiCol_ButtonHovered, sensor_bg);
        ImGui_ScopePushStyleColor(ImGuiCol_ButtonActive, sensor_bg);

        if (!device.is_streaming) {
          std::string label = to_string()
                              << "  " << textual_icons::toggle_off << "\noff   ##" << id << ","
                              << "";

          ImGui_ScopePushStyleColor(ImGuiCol_Text, redish);
          ImGui_ScopePushStyleColor(ImGuiCol_TextSelectedBg, redish + 0.1f);

          if (ImGui::Button(label.c_str(), {30, 30})) {
            device.is_streaming = true;
            for (const auto &f : is_streaming_changed) {
              if (!f(device)) {
                device.is_streaming = false;
                break;
              }
            }
          }
        } else {
          std::string label = to_string()
                              << "  " << textual_icons::toggle_on << "\n    on##" << id << ","
                              << "";
          ImGui_ScopePushStyleColor(ImGuiCol_Text, light_blue);
          ImGui_ScopePushStyleColor(ImGuiCol_TextSelectedBg, light_blue + 0.1f);

          if (ImGui::Button(label.c_str(), {30, 30})) {
            device.is_streaming = false;
            for (const auto &f : is_streaming_changed) {
              if (!f(device)) {
                device.is_streaming = true;
                break;
              }
            }
          }
        }
      });

      const auto id = device.name;

      std::string label = to_string() << "Infrared Camera Module"
                                      << "##" << id;
      ImGui::PushStyleColor(ImGuiCol_Header, sensor_bg);
      ImGui::PushStyleColor(ImGuiCol_HeaderActive, sensor_bg);
      ImGui::PushStyleColor(ImGuiCol_HeaderHovered, sensor_bg);
      ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{10, 10});
      ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, {0, 0});
      ImGuiTreeNodeFlags flags{};
      // if (show_depth_only) flags = ImGuiTreeNodeFlags_DefaultOpen;
      if (ImGui::TreeNodeEx(label.c_str(), flags | ImGuiTreeNodeFlags_FramePadding)) {
        ImGui::PopStyleVar();
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {2, 2});
        ImGui::TreePop();
      }

      ImGui::PopStyleVar();
      ImGui::PopStyleVar();
      ImGui::PopStyleColor(3);

      ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 2);
    }

    ImGui::PopStyleColor(2);
    ImGui::PopFont();

    auto end_screen_pos = ImGui::GetCursorScreenPos();

    const auto draw_device_outline = true;
    if (draw_device_outline) {
      // Left space
      ImGui::GetWindowDrawList()->AddRectFilled({initial_screen_pos.x, initial_screen_pos.y},
                                                {end_screen_pos.x + left_space, end_screen_pos.y},
                                                ImColor(black));
      // Left line
      ImGui::GetWindowDrawList()->AddLine(
          {initial_screen_pos.x + left_space, initial_screen_pos.y + upper_space},
          {end_screen_pos.x + left_space, end_screen_pos.y}, ImColor(header_color));
      // Right line
      const float compenstaion_right = 17.f;
      ;
      ImGui::GetWindowDrawList()->AddLine(
          {initial_screen_pos.x + panel_width - compenstaion_right,
           initial_screen_pos.y + upper_space},
          {end_screen_pos.x + panel_width - compenstaion_right, end_screen_pos.y},
          ImColor(header_color));
      // Button line
      const float compenstaion_button = 1.0f;
      ImGui::GetWindowDrawList()->AddLine(
          {end_screen_pos.x + left_space, end_screen_pos.y - compenstaion_button},
          {end_screen_pos.x + left_space + panel_width, end_screen_pos.y - compenstaion_button},
          ImColor(header_color));
    }
  }

  for (const auto &func : draw_later) {
    func();
  }
}

void capture_panel_view::render(view_context *context) {
  const auto window_size = context->get_window_size();

  const auto top_bar_height = 50;

  ImGui::SetNextWindowPos({0, top_bar_height});
  ImGui::SetNextWindowSize({350, window_size.y - top_bar_height});

  auto flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
               ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings |
               ImGuiWindowFlags_NoBringToFrontOnFocus;

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
  ImGui::PushStyleColor(ImGuiCol_WindowBg, sensor_bg);
  ImGui::Begin("Control Panel", nullptr, flags | ImGuiWindowFlags_AlwaysVerticalScrollbar);

  draw_controls(context, 50);

  {
    float width = 320;
    float height = 200;
    float posx = window_size.x * 0.5f - width * 0.5f;
    float posy = window_size.y * 0.5f - height * 0.5f;
    ImGui::SetNextWindowPos({posx, posy});
    ImGui::SetNextWindowSize({width, height});
    ImGui::PushStyleColor(ImGuiCol_PopupBg, sensor_bg);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, white);
    ImGui::PushStyleColor(ImGuiCol_Text, light_grey);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);

    if (ImGui::BeginPopupModal("Network Device", nullptr,
                               ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove)) {
      ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 3);
      ImGui::SetCursorPosX(10);
      ImGui::Text("Connect to a Linux system running graph_proc_server");

      ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 5);

      {
        std::vector<std::string> node_type_names = {
            "raspi", "raspi_color", "depthai_color", "rs_d435", "rs_d435_color",
        };
        std::vector<const char *> node_type_names_chars = get_string_pointers(node_type_names);

        ImGui::SetCursorPosX(10);
        ImGui::Text("Device Type");
        ImGui::SameLine();
        ImGui::SetCursorPosX(80);
        ImGui::PushItemWidth(width - ImGui::GetCursorPosX() - 10);
        ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, light_blue);

        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 3);
        if (ImGui::Combo("##dev_type", &node_type_index, node_type_names_chars.data(),
                         static_cast<int>(node_type_names_chars.size()))) {
        }
        ImGui::PopStyleColor();

        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 6);

        ImGui::PopItemWidth();
      }
      static char ip_input[255];
      std::copy(ip_address.begin(), ip_address.end(), ip_input);
      ip_input[ip_address.size()] = '\0';
      {
        ImGui::SetCursorPosX(10);
        ImGui::Text("Device IP");
        ImGui::SameLine();
        ImGui::SetCursorPosX(80);
        ImGui::PushItemWidth(width - ImGui::GetCursorPosX() - 10);
        ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, light_blue);

        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 3);
        if (ImGui::InputText("##ip", ip_input, 255)) {
          ip_address = ip_input;
        }
        ImGui::PopStyleColor();

        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 6);

        ImGui::PopItemWidth();
      }
      static char gateway_input[255];
      std::copy(gateway_address.begin(), gateway_address.end(), gateway_input);
      gateway_input[gateway_address.size()] = '\0';
      {
        ImGui::SetCursorPosX(10);
        ImGui::Text("Gateway");
        ImGui::SameLine();
        ImGui::SetCursorPosX(80);
        ImGui::PushItemWidth(width - ImGui::GetCursorPosX() - 10);
        ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, light_blue);

        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 3);
        if (ImGui::InputText("##gateway", gateway_input, 255)) {
          device_name = gateway_input;
        }
        ImGui::PopStyleColor();

        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 6);

        ImGui::PopItemWidth();
      }
      static char dev_name_input[255];
      std::copy(device_name.begin(), device_name.end(), dev_name_input);
      dev_name_input[device_name.size()] = '\0';
      {
        ImGui::SetCursorPosX(10);
        ImGui::Text("Name");
        ImGui::SameLine();
        ImGui::SetCursorPosX(80);
        ImGui::PushItemWidth(width - ImGui::GetCursorPosX() - 10);
        ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, light_blue);

        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 3);
        if (ImGui::InputText("##name", dev_name_input, 255)) {
          device_name = dev_name_input;
        }
        ImGui::PopStyleColor();

        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 6);

        ImGui::PopItemWidth();
      }

      ImGui::SetCursorPosX(width / 2 - 105);

      if (ImGui::Button("OK", {100.f, 25.f}) || ImGui::IsKeyDown(ImGuiKey_Enter) ||
          ImGui::IsKeyDown(ImGuiKey_KeypadEnter)) {
        try {
          for (auto &f : on_add_device) {
            f(device_name, static_cast<node_type>(node_type_index), ip_address, gateway_address);
          }
        } catch (std::runtime_error e) {
          spdlog::error(e.what());
        }
        node_type_index = 0;
        device_name = "";
        ip_address = "";
        gateway_address = "";
        ImGui::CloseCurrentPopup();
      }
      ImGui::SameLine();
      ImGui::SetCursorPosX(width / 2 + 5);
      if (ImGui::Button("Cancel", {100.f, 25.f}) || ImGui::IsKeyDown(ImGuiKey_Escape)) {
        node_type_index = 0;
        device_name = "";
        ip_address = "";
        gateway_address = "";
        ImGui::CloseCurrentPopup();
      }
      ImGui::EndPopup();
    }
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(1);
  }

  ImGui::End();
  ImGui::PopStyleVar();
  ImGui::PopStyleColor();
}

float capture_panel_view::draw_control_panel(view_context *context) {
  const float device_panel_height = 60.0f;
  auto panel_pos = ImGui::GetCursorPos();

  ImGui::PushFont(context->large_font);
  ImGui::PushStyleColor(ImGuiCol_Button, sensor_bg);
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, sensor_bg);
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, sensor_bg);
  ImGui::PushStyleColor(ImGuiCol_Text, light_grey);
  ImGui::PushStyleColor(ImGuiCol_PopupBg, almost_white_bg);
  ImGui::PushStyleColor(ImGuiCol_HeaderHovered, light_blue);
  ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, light_grey);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(5, 5));

  const auto id = "";
  const float icons_width = 78.0f;
  const ImVec2 device_panel_icons_size{icons_width, 25};
  textual_icon button_icon = is_streaming ? textual_icons::stop : textual_icons::play;
  std::string play_button_name = to_string() << button_icon << "##" << id;
  auto play_button_color = is_streaming ? light_blue : light_grey;
  {
    ImGui::PushStyleColor(ImGuiCol_Text, play_button_color);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, play_button_color);
    if (ImGui::Button(play_button_name.c_str(), device_panel_icons_size)) {
      if (is_streaming) {
        is_streaming = false;
        for (const auto &f : is_all_streaming_changed) {
          if (!f(devices, is_streaming)) {
            is_streaming = true;
            break;
          }
        }
      } else {
        is_streaming = true;
        for (const auto &f : is_all_streaming_changed) {
          if (!f(devices, is_streaming)) {
            is_streaming = false;
            break;
          }
        }
      }
    }
    ImGui::PopStyleColor(2);
  }

  ImGui::SetCursorPos({panel_pos.x, ImGui::GetCursorPosY()});
  {
    // Using transparent-non-actionable buttons to have the same locations
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0, 0, 0, 0));
    const ImVec2 device_panel_icons_text_size = {icons_width, 5};

    ImGui::PushStyleColor(ImGuiCol_Text, play_button_color);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, play_button_color);
    ImGui::Button(is_streaming ? "Stop" : "Start", device_panel_icons_size);
    ImGui::PopStyleColor(2);
    ImGui::PopStyleColor(3);
  }

  ImGui::PopStyleVar();
  ImGui::PopStyleColor(7);
  ImGui::PopFont();

  return device_panel_height;
}

float calibration_panel_view::draw_control_panel(view_context *context) {
  const float device_panel_height = 60.0f;
  auto panel_pos = ImGui::GetCursorPos();

  ImGui::PushFont(context->large_font);
  ImGui::PushStyleColor(ImGuiCol_Button, sensor_bg);
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, sensor_bg);
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, sensor_bg);
  ImGui::PushStyleColor(ImGuiCol_Text, light_grey);
  ImGui::PushStyleColor(ImGuiCol_PopupBg, almost_white_bg);
  ImGui::PushStyleColor(ImGuiCol_HeaderHovered, light_blue);
  ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, light_grey);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(5, 5));

  const auto id = "";
  const float icons_width = 78.0f;
  const ImVec2 device_panel_icons_size{icons_width, 25};
  textual_icon button_icon = is_streaming ? textual_icons::stop : textual_icons::play;
  std::string play_button_name = to_string() << button_icon << "##" << id;
  auto play_button_color = is_streaming ? light_blue : light_grey;
  {
    ImGui::PushStyleColor(ImGuiCol_Text, play_button_color);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, play_button_color);
    if (ImGui::Button(play_button_name.c_str(), device_panel_icons_size)) {
      if (is_streaming) {
        is_streaming = false;
        for (const auto &f : is_streaming_changed) {
          if (!f(devices, is_streaming)) {
            is_streaming = true;
            break;
          }
        }
      } else {
        is_streaming = true;
        for (const auto &f : is_streaming_changed) {
          if (!f(devices, is_streaming)) {
            is_streaming = false;
            break;
          }
        }
      }
    }
    ImGui::PopStyleColor(2);
  }
  ImGui::SameLine();
  std::string mask_button_name = to_string() << textual_icons::edit << "##" << id;
  auto mask_button_color = is_masking ? light_blue : light_grey;
  {
    ImGui::PushStyleColor(ImGuiCol_Text, mask_button_color);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, mask_button_color);
    if (ImGui::Button(mask_button_name.c_str(), device_panel_icons_size)) {
      if (is_masking) {
        is_masking = false;
        for (const auto &f : is_masking_changed) {
          if (!f(devices, is_masking)) {
            is_masking = true;
            break;
          }
        }
      } else {
        is_masking = true;
        for (const auto &f : is_masking_changed) {
          if (!f(devices, is_masking)) {
            is_masking = false;
            break;
          }
        }
      }
    }
    ImGui::PopStyleColor(2);
  }
  ImGui::SameLine();
  std::string calibrate_button_name = to_string() << textual_icons::refresh << "##" << id;
  bool is_calibrateing = false;
  auto calibrate_button_color = is_calibrateing ? light_blue : light_grey;
  {
    ImGui::PushStyleColor(ImGuiCol_Text, calibrate_button_color);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, calibrate_button_color);
    if (ImGui::Button(calibrate_button_name.c_str(), device_panel_icons_size)) {
      for (const auto &f : on_calibrate) {
        f(devices, true);
      }
      if (is_calibrateing) {
        is_calibrateing = false;
      } else {
        is_calibrateing = true;
      }
    }
    ImGui::PopStyleColor(2);
  }

  {
    ImGui::SetCursorPos({panel_pos.x, ImGui::GetCursorPosY()});
    // Using transparent-non-actionable buttons to have the same locations
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0, 0, 0, 0));
    const ImVec2 device_panel_icons_text_size = {icons_width, 5};

    ImGui::PushStyleColor(ImGuiCol_Text, play_button_color);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, play_button_color);
    ImGui::Button(is_streaming ? "Stop" : "Start", device_panel_icons_size);
    ImGui::PopStyleColor(2);
    ImGui::PopStyleColor(3);
  }
  ImGui::SameLine();
  {
    ImGui::PushStyleColor(ImGuiCol_Text, mask_button_color);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, mask_button_color);
    ImGui::Button("Mask", device_panel_icons_size);
    ImGui::PopStyleColor(2);
  }
  ImGui::SameLine();
  {
    ImGui::PushStyleColor(ImGuiCol_Text, calibrate_button_color);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, calibrate_button_color);
    ImGui::Button("Calibrate", device_panel_icons_size);
    ImGui::PopStyleColor(2);
  }

  ImGui::PopStyleVar();
  ImGui::PopStyleColor(7);
  ImGui::PopFont();

  return device_panel_height;
}

void calibration_panel_view::draw_extrinsic_calibration_control_panel(view_context *context) {
  std::vector<std::function<void()>> draw_later;
  {
    auto pos = ImGui::GetCursorPos();
    auto windows_width = ImGui::GetContentRegionMax().x;

    auto sensor_top_y = ImGui::GetCursorPosY();
    // ImGui::SetContentRegionWidth(windows_width - 36);

    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, sensor_bg);
    ImGui::PushStyleColor(ImGuiCol_Text, light_grey);
    ImGui::PushFont(context->large_font);

    // draw streaming
    {
      draw_later.push_back([pos, windows_width, this, context]() {
        const auto id = "";

        ImGui::SetCursorPos({windows_width - 35, pos.y + 3});
        ImGui_ScopePushFont(context->default_font);

        ImGui_ScopePushStyleColor(ImGuiCol_Button, sensor_bg);
        ImGui_ScopePushStyleColor(ImGuiCol_ButtonHovered, sensor_bg);
        ImGui_ScopePushStyleColor(ImGuiCol_ButtonActive, sensor_bg);

        if (!is_marker_collecting) {
          std::string label = to_string()
                              << "  " << textual_icons::toggle_off << "\noff   ##" << id << ","
                              << "";

          ImGui_ScopePushStyleColor(ImGuiCol_Text, redish);
          ImGui_ScopePushStyleColor(ImGuiCol_TextSelectedBg, redish + 0.1f);

          if (ImGui::Button(label.c_str(), {30, 30})) {
            is_marker_collecting = true;
            for (const auto &f : is_marker_collecting_changed) {
              if (!f(devices, is_marker_collecting)) {
                is_marker_collecting = false;
                break;
              }
            }
          }
        } else {
          std::string label = to_string()
                              << "  " << textual_icons::toggle_on << "\n    on##" << id << ","
                              << "";
          ImGui_ScopePushStyleColor(ImGuiCol_Text, light_blue);
          ImGui_ScopePushStyleColor(ImGuiCol_TextSelectedBg, light_blue + 0.1f);

          if (ImGui::Button(label.c_str(), {30, 30})) {
            is_marker_collecting = false;
            for (const auto &f : is_marker_collecting_changed) {
              if (!f(devices, is_marker_collecting)) {
                is_marker_collecting = true;
                break;
              }
            }
          }
        }
      });

      const auto id = "";

      std::string label = to_string() << "Collect Markers"
                                      << "##" << id;
      ImGui::PushStyleColor(ImGuiCol_Header, sensor_bg);
      ImGui::PushStyleColor(ImGuiCol_HeaderActive, sensor_bg);
      ImGui::PushStyleColor(ImGuiCol_HeaderHovered, sensor_bg);
      ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{10, 10});
      ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, {0, 0});
      ImGuiTreeNodeFlags flags{};
      // if (show_depth_only) flags = ImGuiTreeNodeFlags_DefaultOpen;
      if (ImGui::TreeNodeEx(label.c_str(), flags | ImGuiTreeNodeFlags_FramePadding)) {
        for (auto &device : devices) {
          ImVec2 initial_screen_pos = ImGui::GetCursorScreenPos();

          auto pos = ImGui::GetCursorPos();
          auto windows_width = ImGui::GetContentRegionMax().x;

          auto sensor_top_y = ImGui::GetCursorPosY();
          // ImGui::SetContentRegionWidth(windows_width - 36);

          ImGui::PopStyleVar();
          ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{2, 2});

          // draw streaming
          {
            draw_later.push_back([pos, windows_width, this, context, &device]() {
              const auto id = device.name;

              ImGui::SetCursorPos({windows_width - 35, pos.y + 3});
              ImGui_ScopePushFont(context->default_font);

              ImGui_ScopePushStyleColor(ImGuiCol_Button, sensor_bg);
              ImGui_ScopePushStyleColor(ImGuiCol_ButtonHovered, sensor_bg);
              ImGui_ScopePushStyleColor(ImGuiCol_ButtonActive, sensor_bg);

              if (!device.is_streaming) {
                std::string label = to_string() << "  " << textual_icons::toggle_off << "\noff   ##"
                                                << id << ","
                                                << "";

                ImGui_ScopePushStyleColor(ImGuiCol_Text, redish);
                ImGui_ScopePushStyleColor(ImGuiCol_TextSelectedBg, redish + 0.1f);

                if (ImGui::Button(label.c_str(), {30, 30})) {
                  device.is_streaming = true;
                }
              } else {
                std::string label = to_string()
                                    << "  " << textual_icons::toggle_on << "\n    on##" << id << ","
                                    << "";
                ImGui_ScopePushStyleColor(ImGuiCol_Text, light_blue);
                ImGui_ScopePushStyleColor(ImGuiCol_TextSelectedBg, light_blue + 0.1f);

                if (ImGui::Button(label.c_str(), {30, 30})) {
                  device.is_streaming = false;
                }
              }
            });

            ImGui::Text(device.name.c_str());

            ImGui::SameLine();
            ImGui::SetCursorPosX(220);

            const auto screen_pos = ImGui::GetCursorScreenPos();
            auto c = ImGui::GetColorU32(ImGuiCol_FrameBg);

            ImGui::GetWindowDrawList()->AddRectFilled({200, screen_pos.y}, {300, screen_pos.y + 20},
                                                      c);

            ImGui::Text(std::to_string(device.num_points).c_str());

            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 2);
          }
        }

        ImGui::TreePop();
      }

      ImGui::PopStyleVar();
      ImGui::PopStyleVar();
      ImGui::PopStyleColor(3);
    }

    ImGui::PopStyleColor(2);
    ImGui::PopFont();
  }

  for (const auto &func : draw_later) {
    func();
  }
}

void calibration_panel_view::draw_intrinsic_calibration_control_panel(view_context *context) {
  const auto panel_width = 350;

  // draw selecting device
  {
    std::vector<std::string> intrinsic_calibration_devices;
    for (const auto &device : devices) {
      intrinsic_calibration_devices.push_back(device.name);
    }
    std::string id = "##intrinsic_calibration_device";
    std::vector<const char *> intrinsic_calibration_devices_chars =
        get_string_pointers(intrinsic_calibration_devices);

    const auto pos = ImGui::GetCursorPos();
    ImGui::PushItemWidth(panel_width - 40);
    ImGui::PushFont(context->large_font);
    ImGui::SetCursorPos({pos.x + 10, pos.y});
    if (ImGui::Combo(id.c_str(), &intrinsic_calibration_device_index,
                     intrinsic_calibration_devices_chars.data(),
                     static_cast<int>(intrinsic_calibration_devices.size()))) {
      for (auto &func : on_intrinsic_calibration_device_changed) {
        func(devices.at(intrinsic_calibration_device_index));
      }
    }
    ImGui::SetCursorPos({pos.x, ImGui::GetCursorPos().y});
    ImGui::PopFont();
    ImGui::PopItemWidth();
  }

  std::vector<std::function<void()>> draw_later;
  {
    auto &device = devices[intrinsic_calibration_device_index];

    auto pos = ImGui::GetCursorPos();
    auto windows_width = ImGui::GetContentRegionMax().x;

    auto sensor_top_y = ImGui::GetCursorPosY();

    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, sensor_bg);
    ImGui::PushStyleColor(ImGuiCol_Text, light_grey);
    ImGui::PushFont(context->large_font);

    // draw streaming
    {
      const auto id = "";

      std::string label = to_string() << "Collect Markers"
                                      << "##" << id;
      ImGui::PushStyleColor(ImGuiCol_Header, sensor_bg);
      ImGui::PushStyleColor(ImGuiCol_HeaderActive, sensor_bg);
      ImGui::PushStyleColor(ImGuiCol_HeaderHovered, sensor_bg);
      ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{10, 10});
      ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, {0, 0});

      {
        ImVec2 initial_screen_pos = ImGui::GetCursorScreenPos();

        auto pos = ImGui::GetCursorPos();
        auto windows_width = ImGui::GetContentRegionMax().x;

        auto sensor_top_y = ImGui::GetCursorPosY();
        // ImGui::SetContentRegionWidth(windows_width - 36);

        ImGui::PopStyleVar();
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{2, 2});

        // draw streaming
        {
          draw_later.push_back([pos, windows_width, this, context, &device]() {
            const auto id = device.name;

            ImGui::SetCursorPos({windows_width - 35, pos.y + 3});
            ImGui_ScopePushFont(context->default_font);

            ImGui_ScopePushStyleColor(ImGuiCol_Button, sensor_bg);
            ImGui_ScopePushStyleColor(ImGuiCol_ButtonHovered, sensor_bg);
            ImGui_ScopePushStyleColor(ImGuiCol_ButtonActive, sensor_bg);

            if (!is_marker_collecting) {
              std::string label = to_string()
                                  << "  " << textual_icons::toggle_off << "\noff   ##" << id << ","
                                  << "";

              ImGui_ScopePushStyleColor(ImGuiCol_Text, redish);
              ImGui_ScopePushStyleColor(ImGuiCol_TextSelectedBg, redish + 0.1f);

              if (ImGui::Button(label.c_str(), {30, 30})) {
                is_marker_collecting = true;
              }
            } else {
              std::string label = to_string()
                                  << "  " << textual_icons::toggle_on << "\n    on##" << id << ","
                                  << "";
              ImGui_ScopePushStyleColor(ImGuiCol_Text, light_blue);
              ImGui_ScopePushStyleColor(ImGuiCol_TextSelectedBg, light_blue + 0.1f);

              if (ImGui::Button(label.c_str(), {30, 30})) {
                is_marker_collecting = false;
              }
            }
          });

          {
            ImGui::SetCursorPosX(20);

            ImGui::Text("Num collected makers");

            ImGui::SameLine();
            ImGui::SetCursorPosX(220);

            const auto screen_pos = ImGui::GetCursorScreenPos();
            auto c = ImGui::GetColorU32(ImGuiCol_FrameBg);

            ImGui::GetWindowDrawList()->AddRectFilled({200, screen_pos.y}, {300, screen_pos.y + 20},
                                                      c);

            ImGui::Text(std::to_string(device.num_points).c_str());

            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 2);
          }

          const auto draw_param = [&](const std::string &name, float value) {
            ImGui::SetCursorPosX(20);

            ImGui::Text(name.c_str());

            ImGui::SameLine();
            ImGui::SetCursorPosX(220);

            const auto screen_pos = ImGui::GetCursorScreenPos();
            auto c = ImGui::GetColorU32(ImGuiCol_FrameBg);

            ImGui::GetWindowDrawList()->AddRectFilled({200, screen_pos.y}, {300, screen_pos.y + 20},
                                                      c);

            ImGui::Text(fmt::format("{:6.3f}", value).c_str());

            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 2);
          };

          draw_param("rms", rms);
          draw_param("fx", fx);
          draw_param("fy", fy);
          draw_param("cx", cx);
          draw_param("cy", cy);
          draw_param("k0", k0);
          draw_param("k1", k1);
          draw_param("k2", k2);
          draw_param("p0", p0);
          draw_param("p1", p1);
        }
      }

      ImGui::PopStyleVar();
      ImGui::PopStyleVar();
      ImGui::PopStyleColor(3);
    }

    ImGui::PopStyleColor(2);
    ImGui::PopFont();
  }

  for (const auto &func : draw_later) {
    func();
  }
}

void calibration_panel_view::draw_controls(view_context *context, float panel_height) {
  const auto panel_width = 350;
  auto header_h = panel_height;
  const ImColor device_header_background_color = title_color;
  const float left_space = 3.f;
  const float upper_space = 3.f;

  header_h += 32;

  // draw controls
  {
    const auto pos = ImGui::GetCursorPos();
    const float vertical_space_before_device_control = 10.0f;
    const float horizontal_space_before_device_control = 3.0f;
    auto device_panel_pos = ImVec2{pos.x + horizontal_space_before_device_control,
                                   pos.y + vertical_space_before_device_control};
    ImGui::SetCursorPos(device_panel_pos);
    const float device_panel_height = draw_control_panel(context);
    ImGui::SetCursorPos({device_panel_pos.x, device_panel_pos.y + device_panel_height});
  }

  // draw selecting calibration target
  {
    std::vector<std::string> calibration_targets = {
        "Extrinsic parameters",
        "Intrinsic parameters",
        "Scene parameters",
    };
    std::string id = "##calibration_target";
    std::vector<const char *> calibration_targets_chars = get_string_pointers(calibration_targets);

    const auto pos = ImGui::GetCursorPos();
    ImGui::PushItemWidth(panel_width - 40);
    ImGui::PushFont(context->large_font);
    ImGui::SetCursorPos({pos.x + 10, pos.y});
    if (ImGui::Combo(id.c_str(), &calibration_target_index, calibration_targets_chars.data(),
                     static_cast<int>(calibration_targets.size()))) {
    }
    ImGui::SetCursorPos({pos.x, ImGui::GetCursorPos().y});
    ImGui::PopFont();
    ImGui::PopItemWidth();
  }

  switch (calibration_target_index) {
    case 0:
      draw_extrinsic_calibration_control_panel(context);
      break;
    case 1:
      draw_intrinsic_calibration_control_panel(context);
      break;
    case 2:
      draw_extrinsic_calibration_control_panel(context);
      break;
    default:
      break;
  }
}

void calibration_panel_view::render(view_context *context) {
  const auto window_size = context->get_window_size();

  const auto top_bar_height = 50;

  ImGui::SetNextWindowPos({0, top_bar_height});
  ImGui::SetNextWindowSize({350, window_size.y - top_bar_height});

  auto flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
               ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings |
               ImGuiWindowFlags_NoBringToFrontOnFocus;

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
  ImGui::PushStyleColor(ImGuiCol_WindowBg, sensor_bg);
  ImGui::Begin("Control Panel", nullptr, flags | ImGuiWindowFlags_AlwaysVerticalScrollbar);

  draw_controls(context, 50);

  ImGui::End();
  ImGui::PopStyleVar();
  ImGui::PopStyleColor();
}

float reconstruction_panel_view::draw_control_panel(view_context *context) {
  const float device_panel_height = 60.0f;
  auto panel_pos = ImGui::GetCursorPos();

  ImGui::PushFont(context->large_font);
  ImGui::PushStyleColor(ImGuiCol_Button, sensor_bg);
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, sensor_bg);
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, sensor_bg);
  ImGui::PushStyleColor(ImGuiCol_Text, light_grey);
  ImGui::PushStyleColor(ImGuiCol_PopupBg, almost_white_bg);
  ImGui::PushStyleColor(ImGuiCol_HeaderHovered, light_blue);
  ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, light_grey);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(5, 5));

  const auto id = "";
  const float icons_width = 78.0f;
  const ImVec2 device_panel_icons_size{icons_width, 25};
  textual_icon button_icon = is_streaming ? textual_icons::stop : textual_icons::play;
  std::string play_button_name = to_string() << button_icon << "##" << id;
  auto play_button_color = is_streaming ? light_blue : light_grey;
  {
    ImGui::PushStyleColor(ImGuiCol_Text, play_button_color);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, play_button_color);
    if (ImGui::Button(play_button_name.c_str(), device_panel_icons_size)) {
      if (is_streaming) {
        is_streaming = false;
        for (const auto &f : is_streaming_changed) {
          if (!f(devices, is_streaming)) {
            is_streaming = true;
            break;
          }
        }
      } else {
        is_streaming = true;
        for (const auto &f : is_streaming_changed) {
          if (!f(devices, is_streaming)) {
            is_streaming = false;
            break;
          }
        }
      }
    }
    ImGui::PopStyleColor(2);
  }
  ImGui::SameLine();
  std::string record_button_name = to_string() << textual_icons::circle << "##" << id;
  auto record_button_color = is_recording ? light_blue : light_grey;
  {
    ImGui::PushStyleColor(ImGuiCol_Text, record_button_color);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, record_button_color);
    if (ImGui::Button(record_button_name.c_str(), device_panel_icons_size)) {
      if (is_recording) {
        is_recording = false;
        for (const auto &f : is_recording_changed) {
          if (!f(devices, is_recording)) {
            is_recording = true;
            break;
          }
        }
      } else {
        is_recording = true;
        for (const auto &f : is_recording_changed) {
          if (!f(devices, is_recording)) {
            is_recording = false;
            break;
          }
        }
      }
    }
    ImGui::PopStyleColor(2);
  }

  {
    ImGui::SetCursorPos({panel_pos.x, ImGui::GetCursorPosY()});
    // Using transparent-non-actionable buttons to have the same locations
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0, 0, 0, 0));
    const ImVec2 device_panel_icons_text_size = {icons_width, 5};

    ImGui::PushStyleColor(ImGuiCol_Text, play_button_color);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, play_button_color);
    ImGui::Button(is_streaming ? "Stop" : "Start", device_panel_icons_size);
    ImGui::PopStyleColor(2);
    ImGui::PopStyleColor(3);
  }
  ImGui::SameLine();
  {
    ImGui::PushStyleColor(ImGuiCol_Text, record_button_color);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, record_button_color);
    ImGui::Button("Record", device_panel_icons_size);
    ImGui::PopStyleColor(2);
  }

  ImGui::PopStyleVar();
  ImGui::PopStyleColor(7);
  ImGui::PopFont();

  return device_panel_height;
}

void reconstruction_panel_view::draw_controls(view_context *context, float panel_height) {
  std::vector<std::function<void()>> draw_later;

  auto panel_width = 350;
  auto header_h = panel_height;
  ImColor device_header_background_color = title_color;
  const float left_space = 3.f;
  const float upper_space = 3.f;

  // if (is_ip_device)
  header_h += 32;

  // draw controls
  {
    auto pos = ImGui::GetCursorPos();
    const float vertical_space_before_device_control = 10.0f;
    const float horizontal_space_before_device_control = 3.0f;
    auto device_panel_pos = ImVec2{pos.x + horizontal_space_before_device_control,
                                   pos.y + vertical_space_before_device_control};
    ImGui::SetCursorPos(device_panel_pos);
    const float device_panel_height = draw_control_panel(context);
    ImGui::SetCursorPos({device_panel_pos.x, device_panel_pos.y + device_panel_height});
  }

  {
    auto pos = ImGui::GetCursorPos();
    auto windows_width = ImGui::GetContentRegionMax().x;

    auto sensor_top_y = ImGui::GetCursorPosY();
    // ImGui::SetContentRegionWidth(windows_width - 36);

    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, sensor_bg);
    ImGui::PushStyleColor(ImGuiCol_Text, light_grey);
    ImGui::PushFont(context->large_font);

    ImGui::PopStyleColor(2);
    ImGui::PopFont();
  }

  // draw selecting calibration target
  {
    std::vector<std::string> sources = {
        "Marker",
        "Image",
    };
    std::string id = "##sources";
    std::vector<const char *> sources_chars = get_string_pointers(sources);

    const auto pos = ImGui::GetCursorPos();
    ImGui::PushItemWidth(panel_width - 40);
    ImGui::PushFont(context->large_font);
    ImGui::SetCursorPos({pos.x + 10, pos.y});
    if (ImGui::Combo(id.c_str(), &source, sources_chars.data(),
                     static_cast<int>(sources_chars.size()))) {
    }
    ImGui::SetCursorPos({pos.x, ImGui::GetCursorPos().y});
    ImGui::PopFont();
    ImGui::PopItemWidth();
  }

  for (const auto &func : draw_later) {
    func();
  }
}

void reconstruction_panel_view::render(view_context *context) {
  const auto window_size = context->get_window_size();

  const auto top_bar_height = 50;

  ImGui::SetNextWindowPos({0, top_bar_height});
  ImGui::SetNextWindowSize({350, window_size.y - top_bar_height});

  auto flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
               ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings |
               ImGuiWindowFlags_NoBringToFrontOnFocus;

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
  ImGui::PushStyleColor(ImGuiCol_WindowBg, sensor_bg);
  ImGui::Begin("Control Panel", nullptr, flags | ImGuiWindowFlags_AlwaysVerticalScrollbar);

  draw_controls(context, 50);

  ImGui::End();
  ImGui::PopStyleVar();
  ImGui::PopStyleColor();
}

static void deproject_pixel_to_point(float point[3], const pose_view::camera_t *intrin,
                                     const float pixel[2], float depth) {
  float x = (pixel[0] - intrin->ppx) / intrin->fx;
  float y = (pixel[1] - intrin->ppy) / intrin->fy;

  point[0] = depth * x;
  point[1] = depth * y;
  point[2] = depth;
}

static void draw_sphere(double r, int lats, int longs) {
  int i, j;
  for (i = 0; i <= lats; i++) {
    double lat0 = M_PI * (-0.5 + (double)(i - 1) / lats);
    double z0 = sin(lat0);
    double zr0 = cos(lat0);

    double lat1 = M_PI * (-0.5 + (double)i / lats);
    double z1 = sin(lat1);
    double zr1 = cos(lat1);

    glBegin(GL_QUAD_STRIP);
    for (j = 0; j <= longs; j++) {
      double lng = 2 * M_PI * (double)(j - 1) / longs;
      double x = cos(lng);
      double y = sin(lng);

      glNormal3f(x * zr0, y * zr0, z0);
      glVertex3f(r * x * zr0, r * y * zr0, r * z0);
      glNormal3f(x * zr1, y * zr1, z1);
      glVertex3f(r * x * zr1, r * y * zr1, r * z1);
    }
    glEnd();
  }
}

void pose_view::render(view_context *context) {
  int left, top, right, bottom;
  window_manager::get_instance()->get_window_frame_size(context->window, &left, &top, &right,
                                                        &bottom);

  const auto window_width = context->get_window_size().x;
  const auto window_height = context->get_window_size().y;

  const auto framebuf_width = window_width + left + right;
  const auto framebuf_height = window_height + top + bottom;

  auto output_height = 30;
  auto panel_width = 350;
  const auto top_bar_height = 50;

  const float x = panel_width;
  const float y = top_bar_height;
  const float width = window_width - panel_width;
  const float height = window_height - top_bar_height - output_height;

  auto viewer_rect = rect{x, y, width, height};

  rect window_size{0, 0, window_width, window_height};
  rect fb_size{0, 0, framebuf_width, framebuf_height};
  viewer_rect = viewer_rect.normalize(window_size).unnormalize(fb_size);

  glViewport(viewer_rect.x, viewer_rect.y, static_cast<GLsizei>(viewer_rect.w),
             static_cast<GLsizei>(viewer_rect.h));
  glClearColor(0, 0, 0, 1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glLoadIdentity();

  matrix4 perspective_mat;
  {
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    gluPerspective(45, viewer_rect.w / framebuf_height, 0.001f, 100.0f);
    glGetFloatv(GL_PROJECTION_MATRIX, (GLfloat *)&perspective_mat);
    glPopMatrix();
  }

  {
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadMatrixf((float *)perspective_mat.mat);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    matrix4 view_mat;
    memcpy(&view_mat, (float *)&context->view, sizeof(matrix4));
    glLoadMatrixf((float *)view_mat);

    glDisable(GL_TEXTURE_2D);

    glEnable(GL_DEPTH_TEST);

    {
      float tiles = 24;
      const auto metric_system = true;
      static const float FEET_TO_METER = 0.3048f;
      if (!metric_system) tiles *= 1.f / FEET_TO_METER;

      glTranslatef(0, 0, 0);
      glLineWidth(1);

      // Render "floor" grid
      glBegin(GL_LINES);

      auto T = tiles * 0.5f;

      if (!metric_system) T *= FEET_TO_METER;

      for (int i = 0; i <= ceil(tiles); i++) {
        float I = float(i);
        if (!metric_system) I *= FEET_TO_METER;

        if (i == tiles / 2)
          glColor4f(0.7f, 0.7f, 0.7f, 1.f);
        else
          glColor4f(0.4f, 0.4f, 0.4f, 1.f);

        glVertex3f(I - T, 0, -T);
        glVertex3f(I - T, 0, T);
        glVertex3f(-T, 0, I - T);
        glVertex3f(T, 0, I - T);
      }
      glEnd();
    }

    {
      glColor4f(1.f, 1.f, 1.f, 1.f);

      for (const auto &p : cameras) {
        const auto &camera = p.second;

        const auto camera_pose = axis * camera.pose;

        matrix4 r1;
        memcpy(&r1, (float *)&camera_pose, sizeof(matrix4));
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadMatrixf(r1 * view_mat);

        glTranslatef(0, 0, 0);
        glLineWidth(1.f);
        glBegin(GL_LINES);

        const auto _pc_selected = false;
        if (_pc_selected)
          glColor4f(light_blue.x, light_blue.y, light_blue.z, 0.5f);
        else
          glColor4f(sensor_bg.x, sensor_bg.y, sensor_bg.z, 0.5f);

        for (float d = 1; d < 6; d += 2) {
          auto get_point = [&](float x, float y) -> float3 {
            float point[3];
            float pixel[2]{x, y};
            deproject_pixel_to_point(point, &camera, pixel, d * 0.03f);
            glVertex3f(0.f, 0.f, 0.f);
            glVertex3fv(point);
            return {point[0], point[1], point[2]};
          };

          auto top_left = get_point(0, 0);
          auto top_right = get_point(static_cast<float>(camera.width), 0);
          auto bottom_right =
              get_point(static_cast<float>(camera.width), static_cast<float>(camera.height));
          auto bottom_left = get_point(0, static_cast<float>(camera.height));

          glVertex3fv(&top_left.x);
          glVertex3fv(&top_right.x);
          glVertex3fv(&top_right.x);
          glVertex3fv(&bottom_right.x);
          glVertex3fv(&bottom_right.x);
          glVertex3fv(&bottom_left.x);
          glVertex3fv(&bottom_left.x);
          glVertex3fv(&top_left.x);
        }

        glEnd();

        glPopMatrix();

        glColor4f(1.f, 1.f, 1.f, 1.f);
      }
    }

    {
      glColor4f(1.f, 1.f, 1.f, 1.f);

      for (const auto p : points) {
        glPushMatrix();
        glTranslatef(p.x, p.y, p.z);

        draw_sphere(0.01, 20, 20);

        glPopMatrix();
        glColor4f(1.f, 1.f, 1.f, 1.f);
      }
    }

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
  }
}

// Generate streams layout, creates a grid-like layout with factor amount of columns
std::map<int, rect> image_tile_view::generate_layout(
    const rect &r, int top_bar_height, size_t factor,
    const std::vector<std::shared_ptr<stream_info>> &active_streams,
    std::map<stream_info *, int> &stream_index) {
  std::map<int, rect> results;
  if (factor == 0) return results;

  // Calc the number of rows
  auto complement = static_cast<size_t>(std::ceil((float)active_streams.size() / factor));

  auto cell_width = static_cast<float>(r.w / factor);
  auto cell_height = static_cast<float>(r.h / complement);

  auto it = active_streams.begin();
  for (size_t x = 0; x < factor; x++) {
    for (size_t y = 0; y < complement; y++) {
      // There might be spare boxes at the end (3 streams in 2x2 array for example)
      if (it == active_streams.end()) break;

      rect rxy = {r.x + x * cell_width, r.y + y * cell_height + top_bar_height, cell_width,
                  cell_height - top_bar_height};
      // Generate box to display the stream in
      results[stream_index[(*it).get()]] = rxy.adjust_ratio((*it)->size);
      ++it;
    }
  }

  return results;
}

float image_tile_view::evaluate_layout(const std::map<int, rect> &l) {
  float res = 0.f;
  for (auto &&kvp : l) res += kvp.second.area();
  return res;
}

std::map<int, rect> image_tile_view::calc_layout(const rect &r) {
  std::map<int, rect> results;
  const auto top_bar_height = 50;
  for (size_t f = 1; f <= streams.size(); f++) {
    auto l = generate_layout(r, top_bar_height, f, streams, stream_index);

    // Keep the "best" layout in result
    if (evaluate_layout(l) > evaluate_layout(results)) results = l;
  }

  return results;
}

void image_tile_view::draw_stream_header(view_context *context, const rect &stream_rect) {
  const auto top_bar_height = 32.f;

  ImGui_ScopePushFont(context->large_font);
  ImGui::PushStyleColor(ImGuiCol_Text, light_grey);
  ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, white);

  ImGui::PushStyleColor(ImGuiCol_Button, header_window_bg);
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, header_window_bg);
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, header_window_bg);

  ImGui::GetWindowDrawList()->AddRectFilled({stream_rect.x, stream_rect.y - top_bar_height},
                                            {stream_rect.x + stream_rect.w, stream_rect.y},
                                            ImColor(sensor_bg));

  ImGui::PopStyleColor(5);
}

void image_tile_view::render(view_context *context) {
  auto flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
               ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings |
               ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {5, 5});
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);

  const auto window_size = context->get_window_size();
  const auto window_width = window_size.x;
  const auto window_height = window_size.y;

  auto output_height = 30;
  auto panel_width = 350;
  const auto top_bar_height = 50;

  const float x = panel_width;
  const float y = top_bar_height;
  const float width = window_width - panel_width;
  const float height = window_height - top_bar_height - output_height;

  ImGui::SetNextWindowPos({x, y});
  ImGui::SetNextWindowSize({width, height});

  ImGui::Begin("Viewport", nullptr, flags);

  stream_index.clear();
  for (size_t i = 0; i < streams.size(); i++) {
    stream_index[streams[i].get()] = i;
  }

  const auto r = rect{x, y, width, height};

  auto layout = calc_layout(r);

  for (auto &&kvp : layout) {
    auto &&view_rect = kvp.second;
    auto stream = kvp.first;
    auto &&stream_mv = streams[stream];
    auto &&stream_size = stream_mv->size;

    draw_stream_header(context, view_rect);

    ImGui::SetCursorPos(ImVec2{view_rect.x - r.x, view_rect.y - r.y});

    stream_mv->texture.show(view_rect, 1.f);
  }

  ImGui::End();
  ImGui::PopStyleVar(2);
}