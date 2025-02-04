#pragma once

#include <glad/glad.h>
#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>
#include <opencv2/core.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui-fonts-fontawesome.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/euler_angles.hpp>

#include <spdlog/spdlog.h>

#include <cmath>
#include <array>
#include <sstream>
#include <functional>
#include <map>
#include <vector>
#include <memory>

#include "viewer.hpp"
#include "node_info.hpp"

#include <iostream>

static std::vector<const char *> get_string_pointers(const std::vector<std::string> &vec)
{
    std::vector<const char *> res;
    for (auto &&s : vec)
        res.push_back(s.c_str());
    return res;
}

struct to_string
{
    std::ostringstream ss;
    template <class T>
    to_string &operator<<(const T &val)
    {
        ss << val;
        return *this;
    }
    operator std::string() const { return ss.str(); }
};

static ImVec4 from_rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a, bool consistent_color = false)
{
    auto res = ImVec4(r / (float)255, g / (float)255, b / (float)255, a / (float)255);
#ifdef FLIP_COLOR_SCHEME
    if (!consistent_color)
        return flip(res);
#endif
    return res;
}

static const ImVec4 light_blue = from_rgba(0, 174, 239, 255, true);       // Light blue color for selected elements such as play button glyph when paused
static const ImVec4 regular_blue = from_rgba(0, 115, 200, 255, true);     // Checkbox mark, slider grabber
static const ImVec4 light_grey = from_rgba(0xc3, 0xd5, 0xe5, 0xff, true); // Text
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

inline ImVec4 operator+(const ImVec4 &a, const ImVec4 &b)
{
    return ImVec4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline ImVec4 operator+(const ImVec4 &c, float v)
{
    return ImVec4(
        std::max(0.f, std::min(1.f, c.x + v)),
        std::max(0.f, std::min(1.f, c.y + v)),
        std::max(0.f, std::min(1.f, c.z + v)),
        std::max(0.f, std::min(1.f, c.w)));
}

struct textual_icon
{
    explicit constexpr textual_icon(const char (&unicode_icon)[4]) : _icon{unicode_icon[0], unicode_icon[1], unicode_icon[2], unicode_icon[3]}
    {
    }
    operator const char *() const
    {
        return _icon.data();
    }

private:
    std::array<char, 5> _icon;
};

namespace textual_icons
{
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
}

namespace ImGui
{
    class ScopePushFont
    {
    public:
        ScopePushFont(ImFont *font) { PushFont(font); }
        ~ScopePushFont() { PopFont(); }
    };
    class ScopePushStyleColor
    {
    public:
        ScopePushStyleColor(ImGuiCol idx, const ImVec4 &col) { PushStyleColor(idx, col); }
        ~ScopePushStyleColor() { PopStyleColor(); }
    };
    class ScopePushStyleVar
    {
    public:
        ScopePushStyleVar(ImGuiStyleVar idx, float val) { PushStyleVar(idx, val); }
        ScopePushStyleVar(ImGuiStyleVar idx, const ImVec2 &val) { PushStyleVar(idx, val); }
        ~ScopePushStyleVar() { PopStyleVar(); }
    };
}

#define CONCAT_(x, y) x##y
#define CONCAT(x, y) CONCAT_(x, y)
#define ImGui_ScopePushFont(f) ImGui::ScopePushFont CONCAT(scope_push_font, __LINE__)(f)
#define ImGui_ScopePushStyleColor(idx, col) ImGui::ScopePushStyleColor CONCAT(scope_push_style_color, __LINE__)(idx, col)
#define ImGui_ScopePushStyleVar(idx, val) ImGui::ScopePushStyleVar CONCAT(scope_push_style_var, __LINE__)(idx, val)

template <typename T>
T normalizeT(const T &in_val, const T &min, const T &max)
{
    if (min >= max)
        return 0;
    return ((in_val - min) / (max - min));
}

template <typename T>
T unnormalizeT(const T &in_val, const T &min, const T &max)
{
    if (min == max)
        return min;
    return ((in_val * (max - min)) + min);
}

struct float2
{
    float x, y;
};

struct rect
{
    float x, y;
    float w, h;
    float area() const { return w * h; }
    rect adjust_ratio(float2 size) const
    {
        auto H = static_cast<float>(h), W = static_cast<float>(h) * size.x / size.y;
        if (W > w)
        {
            auto scale = w / W;
            W *= scale;
            H *= scale;
        }

        return {float(floor(x + floor(w - W) / 2)), float(floor(y + floor(h - H) / 2)), W, H};
    }
    rect normalize(const rect &normalize_to) const
    {
        return rect{normalizeT(x, normalize_to.x, normalize_to.x + normalize_to.w),
                    normalizeT(y, normalize_to.y, normalize_to.y + normalize_to.h),
                    normalizeT(w, 0.f, normalize_to.w),
                    normalizeT(h, 0.f, normalize_to.h)};
    }

    rect unnormalize(const rect &unnormalize_to) const
    {
        return rect{unnormalizeT(x, unnormalize_to.x, unnormalize_to.x + unnormalize_to.w),
                    unnormalizeT(y, unnormalize_to.y, unnormalize_to.y + unnormalize_to.h),
                    unnormalizeT(w, 0.f, unnormalize_to.w),
                    unnormalizeT(h, 0.f, unnormalize_to.h)};
    }
};

struct matrix4
{
    float mat[4][4];

    static matrix4 identity()
    {
        matrix4 m;
        for (int i = 0; i < 4; i++)
            m.mat[i][i] = 1.f;
        return m;
    }

    operator float *() const
    {
        return (float *)&mat;
    }
};

inline matrix4 operator*(const matrix4 &a, const matrix4 &b)
{
    matrix4 res;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++)
            {
                sum += a.mat[i][k] * b.mat[k][j];
            }
            res.mat[i][j] = sum;
        }
    }
    return res;
}

struct texture_buffer
{
    GLuint texture;
    GLuint get_gl_handle() const
    {
        return texture;
    }

    texture_buffer() : texture() {}

    void upload_image(int w, int h, void *data, int format = GL_RGBA)
    {
        if (!texture)
            glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, format, w, h, 0, format, GL_UNSIGNED_BYTE, data);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void draw_texture(const rect &s, const rect &t) const
    {
        glBegin(GL_QUAD_STRIP);
        {
            glTexCoord2f(s.x, s.y + s.h);
            glVertex2f(t.x, t.y + t.h);
            glTexCoord2f(s.x, s.y);
            glVertex2f(t.x, t.y);
            glTexCoord2f(s.x + s.w, s.y + s.h);
            glVertex2f(t.x + t.w, t.y + t.h);
            glTexCoord2f(s.x + s.w, s.y);
            glVertex2f(t.x + t.w, t.y);
        }
        glEnd();
    }

    void show(const rect &r, float alpha, const rect &normalized_zoom = rect{0, 0, 1, 1}) const
    {
        if (!texture)
            return;
        // glEnable(GL_BLEND);

        // glBlendFunc(GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA);
        // glBegin(GL_QUADS);
        // glColor4f(1.0f, 1.0f, 1.0f, 1 - alpha);
        // glEnd();

        // glBindTexture(GL_TEXTURE_2D, get_gl_handle());

        // glEnable(GL_TEXTURE_2D);
        // draw_texture(normalized_zoom, r);

        // glDisable(GL_TEXTURE_2D);
        // glBindTexture(GL_TEXTURE_2D, 0);

        // glDisable(GL_BLEND);

        ImGui::Image((ImTextureID)get_gl_handle(), ImVec2{r.w, r.h});
    }
};

struct view_context
{
public:
    ImFont *default_font;
    ImFont *large_font;
    GLFWwindow *window;
    glm::mat4 view;

    ImVec2 get_window_size() const
    {
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        return ImVec2{static_cast<float>(width), static_cast<float>(height)};
    }
};

class top_bar_view
{
public:
    enum class Mode
    {
        Capture,
        Calibration,
        Reconstruction,
    };

    Mode view_mode = Mode::Capture;

    enum class ViewType
    {
        Image,
        Point,
        Contrail,
        Pose,
    };
    ViewType view_type = ViewType::Image;

    void render(view_context *context);
};

class capture_panel_view
{
public:
    struct node_info
    {
        std::string name;
        std::string address;
        std::unordered_map<std::string, std::variant<float, bool>> params;
        bool is_streaming = false;

        node_info(const std::string &name, const std::string &address, const std::unordered_map<std::string, std::variant<float, bool>> &params)
            : name(name), address(address), params(params)
        {
        }
    };
    std::vector<node_info> devices;

    bool is_streaming = false;
    std::vector<std::function<bool(const std::vector<node_info> &, bool)>> is_streaming_changed2;
    std::vector<std::function<bool(const node_info &)>> is_streaming_changed;

private:
    float draw_control_panel(view_context *context);

    void draw_controls(view_context *context, float panel_height);

    std::string ip_address;
    std::string gateway_address;
    std::string device_name;
    int node_type_index;

public:
    std::vector<std::function<void(const std::string&, node_type, const std::string&, const std::string&)>> on_add_device;
    std::vector<std::function<void(const std::string&)>> on_remove_device;

    void render(view_context *context);
};

class calibration_panel_view
{
public:
    struct node_info
    {
        std::string id;
        std::string name;
        std::string address;
        std::unordered_map<std::string, std::variant<float, bool>> params;
        bool is_streaming = true;
        size_t num_points = 0;

        node_info(const std::string &id, const std::string &name, const std::string &address, const std::unordered_map<std::string, std::variant<float, bool>> &params)
            : id(id), name(name), address(address), params(params)
        {
        }
    };
    std::vector<node_info> devices;
    bool is_marker_collecting = false;
    bool is_streaming = false;
    bool is_masking = false;

    std::vector<std::function<bool(const std::vector<node_info> &, bool)>> is_marker_collecting_changed;
    std::vector<std::function<bool(const std::vector<node_info> &, bool)>> is_streaming_changed;
    std::vector<std::function<bool(const std::vector<node_info> &, bool)>> is_masking_changed;
    std::vector<std::function<bool(const std::vector<node_info> &, bool)>> on_calibrate;
    std::vector<std::function<void(const node_info &)>> on_intrinsic_calibration_device_changed;

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
    float draw_control_panel(view_context *context);

    void draw_extrinsic_calibration_control_panel(view_context *context)
    {
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
                draw_later.push_back([pos, windows_width, this, context]()
                                     {
                const auto id = "";

                ImGui::SetCursorPos({windows_width - 35, pos.y + 3});
                ImGui_ScopePushFont(context->default_font);

                ImGui_ScopePushStyleColor(ImGuiCol_Button, sensor_bg);
                ImGui_ScopePushStyleColor(ImGuiCol_ButtonHovered, sensor_bg);
                ImGui_ScopePushStyleColor(ImGuiCol_ButtonActive, sensor_bg);

                if (!is_marker_collecting)
                {
                    std::string label = to_string() << "  " << textual_icons::toggle_off << "\noff   ##" << id << ","
                                                    << "";

                    ImGui_ScopePushStyleColor(ImGuiCol_Text, redish);
                    ImGui_ScopePushStyleColor(ImGuiCol_TextSelectedBg, redish + 0.1f);

                    if (ImGui::Button(label.c_str(), {30, 30}))
                    {
                        is_marker_collecting = true;
                        for (const auto &f : is_marker_collecting_changed)
                        {
                            if (!f(devices, is_marker_collecting))
                            {
                                is_marker_collecting = false;
                                break;
                            }
                        }
                    }
                }
                else
                {

                    std::string label = to_string() << "  " << textual_icons::toggle_on << "\n    on##" << id << ","
                                                    << "";
                    ImGui_ScopePushStyleColor(ImGuiCol_Text, light_blue);
                    ImGui_ScopePushStyleColor(ImGuiCol_TextSelectedBg, light_blue + 0.1f);

                    if (ImGui::Button(label.c_str(), {30, 30}))
                    {
                        is_marker_collecting = false;
                        for (const auto &f : is_marker_collecting_changed)
                        {
                            if (!f(devices, is_marker_collecting))
                            {
                                is_marker_collecting = true;
                                break;
                            }
                        }
                    }
                } });

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
                if (ImGui::TreeNodeEx(label.c_str(), flags | ImGuiTreeNodeFlags_FramePadding))
                {
                    for (auto &device : devices)
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
                            draw_later.push_back([pos, windows_width, this, context, &device]()
                                                 {
                                const auto id = device.name;

                                ImGui::SetCursorPos({windows_width - 35, pos.y + 3});
                                ImGui_ScopePushFont(context->default_font);

                                ImGui_ScopePushStyleColor(ImGuiCol_Button, sensor_bg);
                                ImGui_ScopePushStyleColor(ImGuiCol_ButtonHovered, sensor_bg);
                                ImGui_ScopePushStyleColor(ImGuiCol_ButtonActive, sensor_bg);

                                if (!device.is_streaming)
                                {
                                    std::string label = to_string() << "  " << textual_icons::toggle_off << "\noff   ##" << id << ","
                                                                    << "";

                                    ImGui_ScopePushStyleColor(ImGuiCol_Text, redish);
                                    ImGui_ScopePushStyleColor(ImGuiCol_TextSelectedBg, redish + 0.1f);

                                    if (ImGui::Button(label.c_str(), {30, 30}))
                                    {
                                        device.is_streaming = true;
                                    }
                                }
                                else
                                {

                                    std::string label = to_string() << "  " << textual_icons::toggle_on << "\n    on##" << id << ","
                                                                    << "";
                                    ImGui_ScopePushStyleColor(ImGuiCol_Text, light_blue);
                                    ImGui_ScopePushStyleColor(ImGuiCol_TextSelectedBg, light_blue + 0.1f);

                                    if (ImGui::Button(label.c_str(), {30, 30}))
                                    {
                                        device.is_streaming = false;
                                    }
                                } });

                            ImGui::Text(device.name.c_str());

                            ImGui::SameLine();
                            ImGui::SetCursorPosX(220);

                            const auto screen_pos = ImGui::GetCursorScreenPos();
                            auto c = ImGui::GetColorU32(ImGuiCol_FrameBg);

                            ImGui::GetWindowDrawList()->AddRectFilled({200, screen_pos.y}, {300, screen_pos.y + 20}, c);

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

        for (const auto &func : draw_later)
        {
            func();
        }
    }

    void draw_intrinsic_calibration_control_panel(view_context *context)
    {
        const auto panel_width = 350;

        // draw selecting device
        {
            std::vector<std::string> intrinsic_calibration_devices;
            for (const auto &device : devices)
            {
                intrinsic_calibration_devices.push_back(device.name);
            }
            std::string id = "##intrinsic_calibration_device";
            std::vector<const char *> intrinsic_calibration_devices_chars = get_string_pointers(intrinsic_calibration_devices);

            const auto pos = ImGui::GetCursorPos();
            ImGui::PushItemWidth(panel_width - 40);
            ImGui::PushFont(context->large_font);
            ImGui::SetCursorPos({pos.x + 10, pos.y});
            if (ImGui::Combo(id.c_str(), &intrinsic_calibration_device_index, intrinsic_calibration_devices_chars.data(), static_cast<int>(intrinsic_calibration_devices.size())))
            {
                for (auto &func : on_intrinsic_calibration_device_changed)
                {
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
                        draw_later.push_back([pos, windows_width, this, context, &device]()
                                             {
                            const auto id = device.name;

                            ImGui::SetCursorPos({windows_width - 35, pos.y + 3});
                            ImGui_ScopePushFont(context->default_font);

                            ImGui_ScopePushStyleColor(ImGuiCol_Button, sensor_bg);
                            ImGui_ScopePushStyleColor(ImGuiCol_ButtonHovered, sensor_bg);
                            ImGui_ScopePushStyleColor(ImGuiCol_ButtonActive, sensor_bg);

                            if (!is_marker_collecting)
                            {
                                std::string label = to_string() << "  " << textual_icons::toggle_off << "\noff   ##" << id << ","
                                                                << "";

                                ImGui_ScopePushStyleColor(ImGuiCol_Text, redish);
                                ImGui_ScopePushStyleColor(ImGuiCol_TextSelectedBg, redish + 0.1f);

                                if (ImGui::Button(label.c_str(), {30, 30}))
                                {
                                    is_marker_collecting = true;
                                }
                            }
                            else
                            {

                                std::string label = to_string() << "  " << textual_icons::toggle_on << "\n    on##" << id << ","
                                                                << "";
                                ImGui_ScopePushStyleColor(ImGuiCol_Text, light_blue);
                                ImGui_ScopePushStyleColor(ImGuiCol_TextSelectedBg, light_blue + 0.1f);

                                if (ImGui::Button(label.c_str(), {30, 30}))
                                {
                                    is_marker_collecting = false;
                                }
                            } });

                        {
                            ImGui::SetCursorPosX(20);

                            ImGui::Text("Num collected makers");

                            ImGui::SameLine();
                            ImGui::SetCursorPosX(220);

                            const auto screen_pos = ImGui::GetCursorScreenPos();
                            auto c = ImGui::GetColorU32(ImGuiCol_FrameBg);

                            ImGui::GetWindowDrawList()->AddRectFilled({200, screen_pos.y}, {300, screen_pos.y + 20}, c);

                            ImGui::Text(std::to_string(device.num_points).c_str());

                            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 2);
                        }

                        const auto draw_param = [&](const std::string &name, float value)
                        {
                            ImGui::SetCursorPosX(20);

                            ImGui::Text(name.c_str());

                            ImGui::SameLine();
                            ImGui::SetCursorPosX(220);

                            const auto screen_pos = ImGui::GetCursorScreenPos();
                            auto c = ImGui::GetColorU32(ImGuiCol_FrameBg);

                            ImGui::GetWindowDrawList()->AddRectFilled({200, screen_pos.y}, {300, screen_pos.y + 20}, c);

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

        for (const auto &func : draw_later)
        {
            func();
        }
    }

    void draw_controls(view_context *context, float panel_height)
    {
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
            auto device_panel_pos = ImVec2{pos.x + horizontal_space_before_device_control, pos.y + vertical_space_before_device_control};
            ImGui::SetCursorPos(device_panel_pos);
            const float device_panel_height = draw_control_panel(context);
            ImGui::SetCursorPos({device_panel_pos.x, device_panel_pos.y + device_panel_height});
        }

        // draw selecting calibration target
        {
            std::vector<std::string> calibration_targets = {
                "Extrinsic parameters",
                "Intrinsic parameters",
            };
            std::string id = "##calibration_target";
            std::vector<const char *> calibration_targets_chars = get_string_pointers(calibration_targets);

            const auto pos = ImGui::GetCursorPos();
            ImGui::PushItemWidth(panel_width - 40);
            ImGui::PushFont(context->large_font);
            ImGui::SetCursorPos({pos.x + 10, pos.y});
            if (ImGui::Combo(id.c_str(), &calibration_target_index, calibration_targets_chars.data(), static_cast<int>(calibration_targets.size())))
            {
            }
            ImGui::SetCursorPos({pos.x, ImGui::GetCursorPos().y});
            ImGui::PopFont();
            ImGui::PopItemWidth();
        }

        switch (calibration_target_index)
        {
        case 0:
            draw_extrinsic_calibration_control_panel(context);
            break;
        case 1:
            draw_intrinsic_calibration_control_panel(context);
            break;
        default:
            break;
        }
    }

public:
    void render(view_context *context)
    {
        const auto window_size = context->get_window_size();

        const auto top_bar_height = 50;

        ImGui::SetNextWindowPos({0, top_bar_height});
        ImGui::SetNextWindowSize({350, window_size.y - top_bar_height});

        auto flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar |
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, sensor_bg);
        ImGui::Begin("Control Panel", nullptr, flags | ImGuiWindowFlags_AlwaysVerticalScrollbar);

        draw_controls(context, 50);

        ImGui::End();
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
    }
};

class reconstruction_panel_view
{
public:
    struct node_info
    {
        std::string name;
        std::string address;
        bool is_streaming = true;

        node_info(const std::string &name, const std::string &address)
            : name(name), address(address)
        {
        }
    };
    std::vector<node_info> devices;
    bool is_reconstructing = false;
    bool is_streaming = false;
    int source = 0;

    std::vector<std::function<bool(const std::vector<node_info> &, bool)>> is_reconstructing_changed;
    std::vector<std::function<bool(const std::vector<node_info> &, bool)>> is_streaming_changed;
    std::vector<std::function<bool(const std::vector<node_info> &)>> set_axis_pressed;

private:
    float draw_control_panel(view_context *context)
    {
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
            if (ImGui::Button(play_button_name.c_str(), device_panel_icons_size))
            {
                if (is_streaming)
                {
                    is_streaming = false;
                    for (const auto &f : is_streaming_changed)
                    {
                        if (!f(devices, is_streaming))
                        {
                            is_streaming = true;
                            break;
                        }
                    }
                }
                else
                {
                    is_streaming = true;
                    for (const auto &f : is_streaming_changed)
                    {
                        if (!f(devices, is_streaming))
                        {
                            is_streaming = false;
                            break;
                        }
                    }
                }
            }
            ImGui::PopStyleColor(2);
        }
        ImGui::SameLine();
        std::string reconstructing_button_name = to_string() << textual_icons::edit << "##" << id;
        auto reconstructing_button_color = is_reconstructing ? light_blue : light_grey;
        {
            ImGui::PushStyleColor(ImGuiCol_Text, reconstructing_button_color);
            ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, reconstructing_button_color);
            if (ImGui::Button(reconstructing_button_name.c_str(), device_panel_icons_size))
            {
                if (is_reconstructing)
                {
                    is_reconstructing = false;
                    for (const auto &f : is_reconstructing_changed)
                    {
                        if (!f(devices, is_reconstructing))
                        {
                            is_reconstructing = true;
                            break;
                        }
                    }
                }
                else
                {
                    is_reconstructing = true;
                    for (const auto &f : is_reconstructing_changed)
                    {
                        if (!f(devices, is_reconstructing))
                        {
                            is_reconstructing = false;
                            break;
                        }
                    }
                }
            }
            ImGui::PopStyleColor(2);
        }
        ImGui::SameLine();
        std::string set_axis_button_name = to_string() << textual_icons::cube << "##" << id;
        auto set_axis_button_color = light_grey;
        {
            ImGui::PushStyleColor(ImGuiCol_Text, set_axis_button_color);
            ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, set_axis_button_color);
            if (ImGui::Button(set_axis_button_name.c_str(), device_panel_icons_size))
            {
                for (const auto &f : set_axis_pressed)
                {
                    if (!f(devices))
                    {
                        break;
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
            ImGui::PushStyleColor(ImGuiCol_Text, reconstructing_button_color);
            ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, reconstructing_button_color);
            ImGui::Button("reconstructing", device_panel_icons_size);
            ImGui::PopStyleColor(2);
        }
        ImGui::SameLine();
        {
            ImGui::PushStyleColor(ImGuiCol_Text, set_axis_button_color);
            ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, set_axis_button_color);
            ImGui::Button("set axis", device_panel_icons_size);
            ImGui::PopStyleColor(2);
        }

        ImGui::PopStyleVar();
        ImGui::PopStyleColor(7);
        ImGui::PopFont();

        return device_panel_height;
    }

    void draw_controls(view_context *context, float panel_height)
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
            auto pos = ImGui::GetCursorPos();
            const float vertical_space_before_device_control = 10.0f;
            const float horizontal_space_before_device_control = 3.0f;
            auto device_panel_pos = ImVec2{pos.x + horizontal_space_before_device_control, pos.y + vertical_space_before_device_control};
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
                "Playback Marker",
                "Marker",
                "Image",
            };
            std::string id = "##sources";
            std::vector<const char *> sources_chars = get_string_pointers(sources);

            const auto pos = ImGui::GetCursorPos();
            ImGui::PushItemWidth(panel_width - 40);
            ImGui::PushFont(context->large_font);
            ImGui::SetCursorPos({pos.x + 10, pos.y});
            if (ImGui::Combo(id.c_str(), &source, sources_chars.data(), static_cast<int>(sources_chars.size())))
            {
            }
            ImGui::SetCursorPos({pos.x, ImGui::GetCursorPos().y});
            ImGui::PopFont();
            ImGui::PopItemWidth();
        }

        for (const auto &func : draw_later)
        {
            func();
        }
    }

public:
    void render(view_context *context)
    {
        const auto window_size = context->get_window_size();

        const auto top_bar_height = 50;

        ImGui::SetNextWindowPos({0, top_bar_height});
        ImGui::SetNextWindowSize({350, window_size.y - top_bar_height});

        auto flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar |
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, sensor_bg);
        ImGui::Begin("Control Panel", nullptr, flags | ImGuiWindowFlags_AlwaysVerticalScrollbar);

        draw_controls(context, 50);

        ImGui::End();
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
    }
};

class image_tile_view
{
public:
    struct stream_info
    {
        std::string name;
        float2 size;
        texture_buffer texture;

        stream_info(std::string name, float2 size) : name(name), size(size) {}
    };
    std::vector<std::shared_ptr<stream_info>> streams;
    std::map<stream_info *, int> stream_index;

    // Generate streams layout, creates a grid-like layout with factor amount of columns
    std::map<int, rect> generate_layout(const rect &r,
                                        int top_bar_height, size_t factor,
                                        const std::vector<std::shared_ptr<stream_info>> &active_streams,
                                        std::map<stream_info *, int> &stream_index)
    {
        std::map<int, rect> results;
        if (factor == 0)
            return results;

        // Calc the number of rows
        auto complement = static_cast<size_t>(std::ceil((float)active_streams.size() / factor));

        auto cell_width = static_cast<float>(r.w / factor);
        auto cell_height = static_cast<float>(r.h / complement);

        auto it = active_streams.begin();
        for (size_t x = 0; x < factor; x++)
        {
            for (size_t y = 0; y < complement; y++)
            {
                // There might be spare boxes at the end (3 streams in 2x2 array for example)
                if (it == active_streams.end())
                    break;

                rect rxy = {r.x + x * cell_width, r.y + y * cell_height + top_bar_height,
                            cell_width, cell_height - top_bar_height};
                // Generate box to display the stream in
                results[stream_index[(*it).get()]] = rxy.adjust_ratio((*it)->size);
                ++it;
            }
        }

        return results;
    }
    float evaluate_layout(const std::map<int, rect> &l)
    {
        float res = 0.f;
        for (auto &&kvp : l)
            res += kvp.second.area();
        return res;
    }

    std::map<int, rect> calc_layout(const rect &r)
    {
        std::map<int, rect> results;
        const auto top_bar_height = 50;
        for (size_t f = 1; f <= streams.size(); f++)
        {
            auto l = generate_layout(r, top_bar_height, f,
                                     streams, stream_index);

            // Keep the "best" layout in result
            if (evaluate_layout(l) > evaluate_layout(results))
                results = l;
        }

        return results;
    }

    void draw_stream_header(view_context *context, const rect &stream_rect)
    {
        const auto top_bar_height = 32.f;

        ImGui_ScopePushFont(context->large_font);
        ImGui::PushStyleColor(ImGuiCol_Text, light_grey);
        ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, white);

        ImGui::PushStyleColor(ImGuiCol_Button, header_window_bg);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, header_window_bg);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, header_window_bg);

        ImGui::GetWindowDrawList()->AddRectFilled({stream_rect.x, stream_rect.y - top_bar_height},
                                                  {stream_rect.x + stream_rect.w, stream_rect.y}, ImColor(sensor_bg));

        ImGui::PopStyleColor(5);
    }

    void render(view_context *context);
};

class pose_view
{
public:
    struct camera_t
    {
        int width;  /**< Width of the image in pixels */
        int height; /**< Height of the image in pixels */
        float ppx;  /**< Horizontal coordinate of the principal point of the image, as a pixel offset from the left edge */
        float ppy;  /**< Vertical coordinate of the principal point of the image, as a pixel offset from the top edge */
        float fx;   /**< Focal length of the image plane, as a multiple of pixel width */
        float fy;   /**< Focal length of the image plane, as a multiple of pixel height */
        std::array<float, 5> coeffs;
        glm::mat4 pose;
    };
    std::map<std::string, camera_t> cameras;
    std::vector<glm::vec3> points;
    glm::mat4 axis;

    void render(view_context *context);
};

class azimuth_elevation
{
public:
    azimuth_elevation(glm::u32vec2 screen_offset, glm::u32vec2 screen_size)
        : screen_offset(screen_offset), screen_size(screen_size), start_position(0.0f, 0.0f, 0.0f), current_position(0.0f, 0.0f, 0.0f), drag_rotation(false), drag_transition(false)
    {
        reset();
    }

    glm::mat4 get_rotation_matrix()
    {
        return glm::toMat4(current_rotation);
    }

    glm::quat get_rotation_quaternion()
    {
        return current_rotation;
    }

    glm::mat4 get_translation_matrix()
    {
        return translation_matrix;
    }

    glm::mat4 get_translation_delta_matrix()
    {
        return translation_delta_matrix;
    }

    void set_radius_translation(float value)
    {
        radius_translation = value;
    }

    void set_radius(float value)
    {
        radius = value;
    }
    float get_radius() const
    {
        return radius;
    }

    float get_screen_w() const
    {
        return (float)screen_size.x;
    }
    float get_screen_h() const
    {
        return (float)screen_size.y;
    }
    float get_screen_x() const
    {
        return (float)screen_offset.x;
    }
    float get_screen_y() const
    {
        return (float)screen_offset.y;
    }

    glm::quat quat_from_screen(glm::vec3 from, glm::vec3 to)
    {
        const auto vector = (to - from) * 1.f /*radius*/;

        angle.x += vector.y / get_screen_w();
        angle.y += vector.x / get_screen_h();

        return glm::quat_cast(glm::rotate(angle.x, glm::vec3(1.f, 0.f, 0.f)) * glm::rotate(angle.t, glm::vec3(0.f, 1.f, 0.f)));
    }
    glm::vec3 screen_to_vector(float sx, float sy)
    {
        return glm::vec3(sx, sy, 0);
    }

    bool on_target(int x, int y) const
    {
        x -= screen_offset.x;
        y -= screen_offset.y;

        return (x >= 0) && (y >= 0) && (x < static_cast<int>(screen_size.x)) && (y < static_cast<int>(screen_size.y));
    }

    void begin_rotation(int x, int y)
    {
        if (on_target(x, y))
        {
            drag_rotation = true;
            previous_rotation = current_rotation;
            start_position = screen_to_vector((float)x, (float)y);
        }
    }
    void update_rotation(float x, float y)
    {
        if (drag_rotation)
        {
            current_position = screen_to_vector(x, y);
            current_rotation = quat_from_screen(start_position, current_position);
            start_position = current_position;
        }
    }
    void end_rotation()
    {
        drag_rotation = false;
    }

    void begin_transition(int x, int y)
    {
        if (on_target(x, y))
        {
            drag_transition = true;
            previsou_position.x = (float)x;
            previsou_position.y = (float)y;
        }
    }
    void update_transition(int x, int y, bool zoom)
    {
        if (drag_transition)
        {
            float delta_x = (previsou_position.x - (float)x) * radius_translation / get_screen_w();
            float delta_y = (previsou_position.y - (float)y) * radius_translation / get_screen_h();

            if (!zoom)
            {
                translation_delta_matrix = glm::translate(glm::vec3(-2 * delta_x, 2 * delta_y, 0.0f));
                translation_matrix = translation_delta_matrix * translation_matrix;
            }
            else
            {
                translation_delta_matrix = glm::translate(glm::vec3(0.0f, 0.0f, 5 * delta_y));
                translation_matrix = translation_delta_matrix * translation_matrix;
            }

            previsou_position.x = (float)x;
            previsou_position.y = (float)y;
        }
    }
    void end_transition()
    {
        translation_delta_matrix = glm::identity<glm::mat4>();
        drag_transition = false;
    }

    void reset()
    {
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

    void scroll(double x, double y)
    {
        radius -= (static_cast<float>(y) * 1.0f);
    }

private:
    static glm::vec2 get_center(int width, int height)
    {
        return glm::vec2(width * 0.5f, height * 0.5f);
    }
    static glm::vec2 get_center(const glm::u32vec2 &screen)
    {
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
