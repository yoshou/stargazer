#include "views.hpp"

#include <spdlog/spdlog.h>

#include "coalsack_math_conv.hpp"
#include "render3d.hpp"

using namespace stargazer;

// Texture buffer implementation for Vulkan
texture_buffer::texture_buffer()
    : gfx_ctx(nullptr),
      descriptor_set(VK_NULL_HANDLE),
      width(0),
      height(0),
      staging_buffer_size(0) {}

texture_buffer::~texture_buffer() {
  // Release ImGui descriptor set before Vulkan resources
  // Only if graphics context is still valid
  if (descriptor_set != VK_NULL_HANDLE && gfx_ctx && gfx_ctx->device) {
    ImGui_ImplVulkan_RemoveTexture(descriptor_set);
    descriptor_set = VK_NULL_HANDLE;
  }
  // Vulkan resources are automatically cleaned up by UniqueHandles
}

static uint32_t find_memory_type(vk::PhysicalDevice physical_device, uint32_t type_filter,
                                 vk::MemoryPropertyFlags properties) {
  vk::PhysicalDeviceMemoryProperties mem_properties = physical_device.getMemoryProperties();

  for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
    if ((type_filter & (1 << i)) &&
        (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }

  throw std::runtime_error("Failed to find suitable memory type");
}

void texture_buffer::upload_image(int w, int h, void* data, int format) {
  if (!gfx_ctx || !gfx_ctx->device) {
    return;
  }

  // Validate input parameters
  if (!data || w <= 0 || h <= 0) {
    return;
  }

  // Check if we need to recreate resources (size or format changed)
  bool need_recreate = (width != w || height != h);

  if (need_recreate) {
    // Release existing resources before creating new ones
    if (descriptor_set != VK_NULL_HANDLE) {
      ImGui_ImplVulkan_RemoveTexture(descriptor_set);
      descriptor_set = VK_NULL_HANDLE;
    }
    sampler.reset();
    image_view.reset();
    image_memory.reset();
    image.reset();

    width = w;
    height = h;
  }

  // Assume RGB 3-channel format (convert to RGBA)
  vk::DeviceSize image_size = w * h * 4;
  std::vector<uint8_t> rgba_data(image_size);

  // Convert RGB to RGBA
  const uint8_t* src = static_cast<const uint8_t*>(data);
  for (int i = 0; i < w * h; ++i) {
    rgba_data[i * 4 + 0] = src[i * 3 + 0];  // R
    rgba_data[i * 4 + 1] = src[i * 3 + 1];  // G
    rgba_data[i * 4 + 2] = src[i * 3 + 2];  // B
    rgba_data[i * 4 + 3] = 255;             // A
  }

  // Create or reuse staging buffer
  if (staging_buffer_size < image_size) {
    // Need to recreate staging buffer with larger size
    staging_buffer.reset();
    staging_memory.reset();

    vk::BufferCreateInfo buffer_info;
    buffer_info.size = image_size;
    buffer_info.usage = vk::BufferUsageFlagBits::eTransferSrc;
    buffer_info.sharingMode = vk::SharingMode::eExclusive;

    staging_buffer = gfx_ctx->device->createBufferUnique(buffer_info);

    vk::MemoryRequirements mem_requirements =
        gfx_ctx->device->getBufferMemoryRequirements(staging_buffer.get());

    vk::MemoryAllocateInfo alloc_info;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = find_memory_type(
        gfx_ctx->physical_device, mem_requirements.memoryTypeBits,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    staging_memory = gfx_ctx->device->allocateMemoryUnique(alloc_info);
    gfx_ctx->device->bindBufferMemory(staging_buffer.get(), staging_memory.get(), 0);

    staging_buffer_size = image_size;
  }

  // Copy RGBA data to staging buffer
  void* mapped = gfx_ctx->device->mapMemory(staging_memory.get(), 0, image_size);
  memcpy(mapped, rgba_data.data(), static_cast<size_t>(image_size));
  gfx_ctx->device->unmapMemory(staging_memory.get());

  // Create image resources only if needed
  if (need_recreate) {
    // Create image
    vk::ImageCreateInfo image_info;
    image_info.imageType = vk::ImageType::e2D;
    image_info.extent.width = w;
    image_info.extent.height = h;
    image_info.extent.depth = 1;
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.format = vk::Format::eR8G8B8A8Unorm;
    image_info.tiling = vk::ImageTiling::eOptimal;
    image_info.initialLayout = vk::ImageLayout::eUndefined;
    image_info.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
    image_info.samples = vk::SampleCountFlagBits::e1;
    image_info.sharingMode = vk::SharingMode::eExclusive;

    image = gfx_ctx->device->createImageUnique(image_info);

    vk::MemoryRequirements img_mem_requirements =
        gfx_ctx->device->getImageMemoryRequirements(image.get());

    vk::MemoryAllocateInfo img_alloc_info;
    img_alloc_info.allocationSize = img_mem_requirements.size;
    img_alloc_info.memoryTypeIndex =
        find_memory_type(gfx_ctx->physical_device, img_mem_requirements.memoryTypeBits,
                         vk::MemoryPropertyFlagBits::eDeviceLocal);

    image_memory = gfx_ctx->device->allocateMemoryUnique(img_alloc_info);
    gfx_ctx->device->bindImageMemory(image.get(), image_memory.get(), 0);
  }

  // Transition image layout and copy buffer to image
  vk::CommandBufferAllocateInfo cmd_alloc_info;
  cmd_alloc_info.commandPool = gfx_ctx->command_pool.get();
  cmd_alloc_info.level = vk::CommandBufferLevel::ePrimary;
  cmd_alloc_info.commandBufferCount = 1;

  auto command_buffers = gfx_ctx->device->allocateCommandBuffersUnique(cmd_alloc_info);
  auto& command_buffer = command_buffers[0];

  vk::CommandBufferBeginInfo begin_info;
  begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
  command_buffer->begin(begin_info);

  // Transition to transfer dst
  vk::ImageMemoryBarrier barrier;
  barrier.oldLayout =
      need_recreate ? vk::ImageLayout::eUndefined : vk::ImageLayout::eShaderReadOnlyOptimal;
  barrier.newLayout = vk::ImageLayout::eTransferDstOptimal;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image.get();
  barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  barrier.srcAccessMask = need_recreate ? vk::AccessFlags() : vk::AccessFlagBits::eShaderRead;
  barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

  command_buffer->pipelineBarrier(need_recreate ? vk::PipelineStageFlagBits::eTopOfPipe
                                                : vk::PipelineStageFlagBits::eFragmentShader,
                                  vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags(), 0,
                                  nullptr, 0, nullptr, 1, &barrier);

  // Copy buffer to image
  vk::BufferImageCopy region;
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageOffset = vk::Offset3D{0, 0, 0};
  region.imageExtent = vk::Extent3D{static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1};

  command_buffer->copyBufferToImage(staging_buffer.get(), image.get(),
                                    vk::ImageLayout::eTransferDstOptimal, 1, &region);

  // Transition to shader read
  barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
  barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
  barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
  barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

  command_buffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                  vk::PipelineStageFlagBits::eFragmentShader, vk::DependencyFlags(),
                                  0, nullptr, 0, nullptr, 1, &barrier);

  command_buffer->end();

  vk::SubmitInfo submit_info;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer.get();

  (void)gfx_ctx->graphics_queue.submit(1, &submit_info, nullptr);
  (void)gfx_ctx->graphics_queue.waitIdle();

  // Create image view and sampler only if needed
  if (need_recreate) {
    // Create image view
    vk::ImageViewCreateInfo view_info;
    view_info.image = image.get();
    view_info.viewType = vk::ImageViewType::e2D;
    view_info.format = vk::Format::eR8G8B8A8Unorm;
    view_info.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    image_view = gfx_ctx->device->createImageViewUnique(view_info);

    // Create sampler
    vk::SamplerCreateInfo sampler_info;
    sampler_info.magFilter = vk::Filter::eLinear;
    sampler_info.minFilter = vk::Filter::eLinear;
    sampler_info.addressModeU = vk::SamplerAddressMode::eRepeat;
    sampler_info.addressModeV = vk::SamplerAddressMode::eRepeat;
    sampler_info.addressModeW = vk::SamplerAddressMode::eRepeat;
    sampler_info.anisotropyEnable = VK_FALSE;
    sampler_info.maxAnisotropy = 1.0f;
    sampler_info.borderColor = vk::BorderColor::eIntOpaqueBlack;
    sampler_info.unnormalizedCoordinates = VK_FALSE;
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.compareOp = vk::CompareOp::eAlways;
    sampler_info.mipmapMode = vk::SamplerMipmapMode::eLinear;

    sampler = gfx_ctx->device->createSamplerUnique(sampler_info);

    // Add ImGui descriptor set
    descriptor_set = ImGui_ImplVulkan_AddTexture(sampler.get(), image_view.get(),
                                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }
}

void texture_buffer::show(const rect& r, float alpha, const rect& normalized_zoom) const {
  if (descriptor_set == VK_NULL_HANDLE) {
    return;
  }

  ImVec2 uv0(normalized_zoom.x, normalized_zoom.y);
  ImVec2 uv1(normalized_zoom.x + normalized_zoom.w, normalized_zoom.y + normalized_zoom.h);

  ImGui::GetWindowDrawList()->AddImage((ImTextureID)(intptr_t)descriptor_set, ImVec2(r.x, r.y),
                                       ImVec2(r.x + r.w, r.y + r.h), uv0, uv1,
                                       ImGui::ColorConvertFloat4ToU32(ImVec4(1, 1, 1, alpha)));
}

namespace {
struct float3 {
  float x, y, z;

  float length() const { return sqrt(x * x + y * y + z * z); }

  float3 normalize() const {
    return (length() > 0) ? float3{x / length(), y / length(), z / length()} : *this;
  }
};
}  // namespace

struct matrix4 {
  float mat[4][4];

  static matrix4 identity() {
    matrix4 m;
    for (int i = 0; i < 4; i++) m.mat[i][i] = 1.f;
    return m;
  }

  operator float*() const { return (float*)&mat; }
};

inline matrix4 operator*(const matrix4& a, const matrix4& b) {
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
  ScopePushFont(ImFont* font) { PushFont(font); }
  ~ScopePushFont() { PopFont(); }
};
class ScopePushStyleColor {
 public:
  ScopePushStyleColor(ImGuiCol idx, const ImVec4& col) { PushStyleColor(idx, col); }
  ~ScopePushStyleColor() { PopStyleColor(); }
};
class ScopePushStyleVar {
 public:
  ScopePushStyleVar(ImGuiStyleVar idx, float val) { PushStyleVar(idx, val); }
  ScopePushStyleVar(ImGuiStyleVar idx, const ImVec2& val) { PushStyleVar(idx, val); }
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

struct to_string {
  std::ostringstream ss;
  template <class T>
  to_string& operator<<(const T& val) {
    ss << val;
    return *this;
  }
  operator std::string() const { return ss.str(); }
};

struct textual_icon {
  explicit constexpr textual_icon(const char8_t (&unicode_icon)[4])
      : _icon{unicode_icon[0], unicode_icon[1], unicode_icon[2], unicode_icon[3]} {}
  operator const char*() const { return reinterpret_cast<const char*>(_icon.data()); }

 private:
  std::array<char8_t, 5> _icon;
};

namespace textual_icons {
// A note to a maintainer - preserve order when adding values to avoid duplicates
static const textual_icon refresh{u8"\uf021"};
static const textual_icon edit{u8"\uf044"};
static const textual_icon play{u8"\uf04b"};
static const textual_icon stop{u8"\uf04d"};
static const textual_icon eraser{u8"\uf12d"};
static const textual_icon circle{u8"\uf111"};
}  // namespace textual_icons

namespace {

textual_icon get_textual_icon(const std::string& name) {
  if (name == "refresh") return textual_icons::refresh;
  if (name == "play") return textual_icons::play;
  if (name == "stop") return textual_icons::stop;
  if (name == "edit") return textual_icons::edit;
  if (name == "eraser") return textual_icons::eraser;
  if (name == "circle") return textual_icons::circle;
  return textual_icons::refresh;
}

void draw_badges(const std::vector<std::string>& badges) {
  for (const auto& badge : badges) {
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, light_red);
    ImGui::Text("[%s]", badge.c_str());
    ImGui::PopStyleColor();
  }
}

void draw_detail_row(const stargazer::pipeline_item& item,
                     const std::optional<std::string>& value_override = std::nullopt) {
  const float panel_width = ImGui::GetContentRegionAvail().x;
  const float row_height = 24.0f;
  const float value_width = std::min(125.0f, panel_width * 0.42f);
  const float label_width = std::max(60.0f, panel_width - value_width - 24.0f);
  const auto start = ImGui::GetCursorScreenPos();
  auto* font = ImGui::GetFont();
  const auto font_size = ImGui::GetFontSize();

  ImGui::Dummy({0.0f, 1.0f});
  ImGui::Indent(12.0f);
  ImGui::PushStyleColor(ImGuiCol_Text, light_grey);
  ImGui::TextUnformatted(item.label.c_str());
  ImGui::PopStyleColor();

  const auto value_min = ImVec2(start.x + 24.0f + label_width, start.y + 1.0f);
  const auto value_max = ImVec2(start.x + 24.0f + label_width + value_width, start.y + row_height);
  ImGui::GetWindowDrawList()->AddRectFilled(value_min, value_max,
                                            ImGui::ColorConvertFloat4ToU32(node_info_color), 2.0f);
  const auto& value = value_override.has_value() ? value_override.value() : item.summary;
  ImGui::GetWindowDrawList()->AddText(font, font_size, ImVec2(value_min.x + 6.0f, start.y + 4.0f),
                                      ImGui::ColorConvertFloat4ToU32(yellowish), value.c_str());
  ImGui::Unindent(12.0f);
  ImGui::Dummy({0.0f, row_height - ImGui::GetTextLineHeight()});
}

ImVec4 get_tree_text_color(stargazer::pipeline_item_kind kind) {
  switch (kind) {
    case stargazer::pipeline_item_kind::pipeline:
      return light_blue;
    case stargazer::pipeline_item_kind::subgraph:
      return light_grey;
    case stargazer::pipeline_item_kind::node:
      return white;
    case stargazer::pipeline_item_kind::detail:
      return light_grey;
  }
  return light_grey;
}

void draw_pipeline_tree_item(pipeline_panel_view* panel, const stargazer::pipeline_item& item,
                             view_context* context) {
  (void)context;
  if (item.kind == stargazer::pipeline_item_kind::detail) {
    if (item.detail_kind == stargazer::pipeline_detail_kind::property &&
        panel->resolve_detail_value) {
      draw_detail_row(item, panel->resolve_detail_value(item));
    } else {
      draw_detail_row(item);
    }
    return;
  }

  ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_FramePadding;
  if (item.children.empty()) {
    flags |= ImGuiTreeNodeFlags_Leaf;
  }

  ImGui::PushStyleColor(ImGuiCol_Text, get_tree_text_color(item.kind));
  const bool open = ImGui::TreeNodeEx((item.label + "##" + item.stable_id).c_str(), flags);
  ImGui::PopStyleColor();

  if (!item.runtime_node_id.empty()) {
    auto runtime_it = panel->tree.runtime_nodes.find(item.runtime_node_id);
    if (runtime_it != panel->tree.runtime_nodes.end()) {
      auto& runtime_node = runtime_it->second;
      draw_badges(runtime_node.badges);
    }
  }

  if (open) {
    for (const auto& child : item.children) {
      draw_pipeline_tree_item(panel, child, context);
    }
    ImGui::TreePop();
  }
}

}  // namespace

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

void top_bar_view::render(view_context* context) {
  auto flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
               ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings |
               ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;

  const auto top_bar_height = 50;
  const auto window_size = context->get_window_size();
  const auto item_spacing = ImGui::GetStyle().ItemSpacing.x;
  const float viewport_button_width = 80.0f;
  const float viewport_buttons_width = viewport_button_width * 4.0f + item_spacing * 3.0f;
  const float right_button_start_x = std::max(0.0f, window_size.x - viewport_buttons_width);

  ImGui::SetNextWindowPos({0, 0});
  ImGui::SetNextWindowSize({window_size.x, top_bar_height});

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
  ImGui::PushStyleColor(ImGuiCol_WindowBg, button_color);
  ImGui::Begin("Toolbar Panel", nullptr, flags);

  ImGui::PushFont(context->large_font);
  ImGui::PushStyleColor(ImGuiCol_Border, black);

  {
    const auto draw_view_button = [&](float x, const char* label, ViewType type) {
      ImGui::SetCursorPos({x, 0.0f});
      ImGui::PushStyleColor(ImGuiCol_Text, (view_type != type) ? light_grey : light_blue);
      ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, (view_type != type) ? light_grey : light_blue);
      if (ImGui::Button(label, {viewport_button_width, top_bar_height})) {
        view_type = type;
      }
      ImGui::PopStyleColor(2);
    };

    draw_view_button(right_button_start_x, "Image", ViewType::Image);
    draw_view_button(right_button_start_x + viewport_button_width + item_spacing, "Point",
                     ViewType::Point);
    draw_view_button(right_button_start_x + (viewport_button_width + item_spacing) * 2.0f,
                     "Contrail", ViewType::Contrail);
    draw_view_button(right_button_start_x + (viewport_button_width + item_spacing) * 3.0f, "Pose",
                     ViewType::Pose);
  }

  ImGui::PopStyleColor();
  ImGui::PopFont();

  ImGui::End();
  ImGui::PopStyleColor();
  ImGui::PopStyleVar();
}

float pipeline_panel_view::draw_control_panel(view_context* context) {
  auto panel_pos = ImGui::GetCursorPos();

  // Collect unique action buttons
  std::vector<std::pair<std::string, runtime_node_action>> unique_actions;
  {
    std::unordered_set<std::string> seen;
    for (const auto& [node_id, runtime_node] : tree.runtime_nodes) {
      if (runtime_node.actions.empty()) continue;
      for (const auto& action : runtime_node.actions) {
        if (seen.insert(action.id).second) {
          unique_actions.emplace_back(node_id, action);
        }
      }
    }
  }

  ImGui::PushFont(context->large_font);
  ImGui::PushStyleColor(ImGuiCol_Button, sensor_bg);
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, sensor_bg);
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, sensor_bg);
  ImGui::PushStyleColor(ImGuiCol_Text, light_grey);
  ImGui::PushStyleColor(ImGuiCol_PopupBg, almost_white_bg);
  ImGui::PushStyleColor(ImGuiCol_HeaderHovered, light_blue);
  ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, light_grey);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(5, 5));

  const float icons_width = 78.0f;
  const ImVec2 node_panel_icons_size{icons_width, 25};

  const float panel_width = ImGui::GetContentRegionAvail().x;
  const int buttons_per_row = std::max(1, static_cast<int>(panel_width / icons_width));
  const int fixed_count = has_gate ? 2 : 1;
  struct ButtonSlot {
    int row;
    int col;
  };
  std::vector<ButtonSlot> action_slots;
  for (int i = 0; i < (int)unique_actions.size(); ++i) {
    int global_idx = fixed_count + i;
    action_slots.push_back({global_idx / buttons_per_row, global_idx % buttons_per_row});
  }
  int total_rows = unique_actions.empty() ? 1 : action_slots.back().row + 1;
  const float node_panel_height = total_rows * 60.0f;

  const auto draw_icon_button = [&](const std::string& button_name, textual_icon icon,
                                    const ImVec4& text_color,
                                    const std::function<void()>& on_click) {
    ImGui::PushStyleColor(ImGuiCol_Text, text_color);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, text_color);
    const std::string icon_button_name = to_string() << icon << button_name;
    if (ImGui::Button(icon_button_name.c_str(), node_panel_icons_size)) {
      on_click();
    }
    ImGui::PopStyleColor(2);
  };

  const auto draw_label_button = [&](const char* label, const ImVec4& text_color) {
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Text, text_color);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, text_color);
    ImGui::Button(label, node_panel_icons_size);
    ImGui::PopStyleColor(5);
  };

  const auto toggle_collect = [&]() {
    const bool next_state = !is_marker_collecting;
    bool accepted = true;
    for (const auto& f : is_marker_collecting_changed) {
      if (!f(next_state)) {
        accepted = false;
        break;
      }
    }
    if (accepted) is_marker_collecting = next_state;
  };

  const auto toggle_streaming = [&]() {
    if (is_streaming) {
      is_streaming = false;
      for (const auto& f : is_all_streaming_changed) {
        if (!f(is_streaming)) {
          is_streaming = true;
          break;
        }
      }
    } else {
      is_streaming = true;
      for (const auto& f : is_all_streaming_changed) {
        if (!f(is_streaming)) {
          is_streaming = false;
          break;
        }
      }
    }
  };

  const auto collect_button_color = is_marker_collecting ? light_blue : light_grey;
  const auto streaming_button_color = is_streaming ? light_blue : light_grey;

  draw_icon_button("##streaming", is_streaming ? textual_icons::stop : textual_icons::play,
                   streaming_button_color, toggle_streaming);
  if (has_gate) {
    ImGui::SameLine();
    draw_icon_button("##collect", is_marker_collecting ? textual_icons::stop : textual_icons::play,
                     collect_button_color, toggle_collect);
  }
  for (size_t i = 0; i < unique_actions.size(); ++i) {
    const auto& [action_node_id, action] = unique_actions[i];
    const auto& slot = action_slots[i];
    if (slot.col == 0) {
      ImGui::SetCursorPos({panel_pos.x, panel_pos.y + slot.row * 60.0f});
    } else {
      ImGui::SameLine();
    }
    draw_icon_button("##" + action.id, get_textual_icon(action.icon), light_grey, [&]() {
      for (const auto& f : on_action) {
        f(action_node_id, action.id);
      }
    });
  }

  ImGui::SetCursorPos({panel_pos.x, panel_pos.y + 30.0f});
  draw_label_button(is_streaming ? "Stop" : "Start", streaming_button_color);
  if (has_gate) {
    ImGui::SameLine();
    draw_label_button(is_marker_collecting ? "Stop Collect" : "Collect", collect_button_color);
  }
  for (size_t i = 0; i < unique_actions.size(); ++i) {
    const auto& slot = action_slots[i];
    if (slot.col == 0) {
      ImGui::SetCursorPos({panel_pos.x, panel_pos.y + slot.row * 60.0f + 30.0f});
    } else {
      ImGui::SameLine();
    }
    draw_label_button(unique_actions[i].second.label.c_str(), light_grey);
  }

  ImGui::SetCursorPos({panel_pos.x, panel_pos.y + node_panel_height});

  ImGui::PopStyleVar();
  ImGui::PopStyleColor(7);
  ImGui::PopFont();

  return node_panel_height;
}

void pipeline_panel_view::draw_controls(view_context* context, float panel_height) {
  {
    const auto pos = ImGui::GetCursorPos();
    const float vertical_space_before_node_control = 10.0f;
    const float horizontal_space_before_node_control = 3.0f;
    auto node_panel_pos = ImVec2{pos.x + horizontal_space_before_node_control,
                                 pos.y + vertical_space_before_node_control};
    ImGui::SetCursorPos(node_panel_pos);
    const float node_panel_height = draw_control_panel(context);
    ImGui::SetCursorPos({node_panel_pos.x, node_panel_pos.y + node_panel_height});
  }

  const float content_left_inset = 10.0f;
  ImGui::Separator();
  ImGui::Indent(content_left_inset - 2.0f);
  ImGui::PushFont(context->large_font);
  for (const auto& root : tree.roots) {
    draw_pipeline_tree_item(this, root, context);
  }
  ImGui::PopFont();
  ImGui::Unindent(content_left_inset - 2.0f);
}

void pipeline_panel_view::render(view_context* context) {
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

static void deproject_pixel_to_point(float point[3], const pose_view::camera_t* intrin,
                                     const float pixel[2], float depth) {
  float x = (pixel[0] - intrin->ppx) / intrin->fx;
  float y = (pixel[1] - intrin->ppy) / intrin->fy;

  point[0] = depth * x;
  point[1] = depth * y;
  point[2] = depth;
}

void pose_view::initialize(vk::Device node, vk::PhysicalDevice physical_device,
                           vk::RenderPass render_pass) {
  pipeline_ = std::make_unique<render3d_pipeline>();
  pipeline_->initialize(node, physical_device, render_pass);
}

void pose_view::cleanup() {
  if (pipeline_) {
    pipeline_->cleanup();
    pipeline_.reset();
  }
}

void pose_view::render(view_context* context, vk::CommandBuffer cmd, vk::Extent2D extent) {
  if (!pipeline_) {
    return;
  }

  auto window_size = context->get_window_size();

  // Layout constants (same as OpenGL version)
  constexpr float panel_width = 350.0f;
  constexpr float top_bar_height = 50.0f;
  constexpr float output_height = 30.0f;

  // Calculate pose view area (excluding panel and bars)
  const float view_x = panel_width;
  const float view_y = top_bar_height;
  const float view_width = window_size.x - panel_width;
  const float view_height = window_size.y - top_bar_height - output_height;

  // Use view matrix from context (set by view_controller in viewer_app)
  glm::mat4 view = context ? context->view : glm::mat4(1.0f);

  // Calculate aspect ratio based on pose view area, not full window
  float aspect =
      view_width > 0 && view_height > 0 ? view_width / view_height : window_size.x / window_size.y;

  // Standard perspective projection
  // Using 0.1f near clip for better depth precision with scaled scene
  glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
  projection[1][1] *= -1;  // Flip Y for Vulkan

  // Calculate viewport offset and extent for pose view area
  vk::Offset2D viewport_offset{static_cast<int32_t>(view_x), static_cast<int32_t>(view_y)};
  vk::Extent2D viewport_extent{static_cast<uint32_t>(view_width),
                               static_cast<uint32_t>(view_height)};

  pipeline_->begin_frame(cmd, projection, view, extent, viewport_offset, viewport_extent);

  // Apply scale to entire scene
  glm::mat4 scale_matrix =
      glm::scale(glm::mat4(1.0f), glm::vec3(POSE_VIEW_SCALE, POSE_VIEW_SCALE, POSE_VIEW_SCALE));
  glm::mat4 scaled_axis = scale_matrix * axis;

  // Draw floor grid (24 tiles with POSE_VIEW_SCALE spacing for better depth precision)
  // This creates a (24 * POSE_VIEW_SCALE) meter grid
  pipeline_->draw_grid(24, 1.0f * POSE_VIEW_SCALE,
                       glm::vec3(1.0f, 1.0f, 1.0f));  // White base color with 0.4/0.7 intensity

  // Draw camera frustums
  for (const auto& [name, camera] : cameras) {
    // Camera pose in world space (apply axis transformation and scale)
    glm::mat4 camera_pose = scaled_axis * stargazer::to_glm(camera.pose);

    // Draw frustum using deproject method
    for (float d = 1; d < 6; d += 2) {
      auto get_point = [&](float x, float y) -> glm::vec3 {
        float point[3];
        float pixel[2]{x, y};
        deproject_pixel_to_point(point, &camera, pixel, d * 0.03f);

        // Transform point from camera space to world space
        glm::vec4 camera_space_point(point[0], point[1], point[2], 1.0f);
        glm::vec4 world_point = camera_pose * camera_space_point;
        return glm::vec3(world_point);
      };

      glm::vec3 camera_origin = glm::vec3(camera_pose[3]);
      auto top_left = get_point(0, 0);
      auto top_right = get_point(static_cast<float>(camera.width), 0);
      auto bottom_right =
          get_point(static_cast<float>(camera.width), static_cast<float>(camera.height));
      auto bottom_left = get_point(0, static_cast<float>(camera.height));

      glm::vec3 color = glm::vec3(0.5f, 0.5f, 0.5f);  // Gray color

      // Lines from camera origin to corners
      pipeline_->draw_line(camera_origin, top_left, color);
      pipeline_->draw_line(camera_origin, top_right, color);
      pipeline_->draw_line(camera_origin, bottom_right, color);
      pipeline_->draw_line(camera_origin, bottom_left, color);

      // Rectangle at depth d
      pipeline_->draw_line(top_left, top_right, color);
      pipeline_->draw_line(top_right, bottom_right, color);
      pipeline_->draw_line(bottom_right, bottom_left, color);
      pipeline_->draw_line(bottom_left, top_left, color);
    }
  }

  // Draw 3D points as spheres
  for (const auto& point : points) {
    glm::vec3 scaled_point = glm::vec3(scale_matrix * glm::vec4(point, 1.0f));
    pipeline_->draw_sphere(scaled_point, 0.01f * POSE_VIEW_SCALE, glm::vec3(1.0f, 1.0f, 1.0f), 20,
                           20);  // White spheres
  }

  pipeline_->end_frame(cmd);
}

// Generate streams layout, creates a grid-like layout with factor amount of columns
std::map<int, rect> image_tile_view::generate_layout(
    const rect& r, int top_bar_height, size_t factor,
    const std::vector<std::shared_ptr<stream_info>>& active_streams,
    std::map<stream_info*, int>& stream_index) {
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

float image_tile_view::evaluate_layout(const std::map<int, rect>& l) {
  float res = 0.f;
  for (auto&& kvp : l) res += kvp.second.area();
  return res;
}

std::map<int, rect> image_tile_view::calc_layout(const rect& r) {
  std::map<int, rect> results;
  const auto top_bar_height = 50;
  for (size_t f = 1; f <= streams.size(); f++) {
    auto l = generate_layout(r, top_bar_height, f, streams, stream_index);

    // Keep the "best" layout in result
    if (evaluate_layout(l) > evaluate_layout(results)) results = l;
  }

  return results;
}

void image_tile_view::draw_stream_header(view_context* context, const rect& stream_rect,
                                         const stream_info& stream) {
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

  if (!stream.name.empty()) {
    const ImVec2 text_min{stream_rect.x + 10.0f, stream_rect.y - top_bar_height};
    const ImVec2 text_max{stream_rect.x + stream_rect.w - 10.0f, stream_rect.y};
    const float font_size =
        context->large_font ? context->large_font->FontSize : ImGui::GetFontSize();
    const float text_y = text_min.y + std::max(0.0f, (top_bar_height - font_size) * 0.5f - 1.0f);
    auto* draw_list = ImGui::GetWindowDrawList();
    draw_list->PushClipRect(text_min, text_max, true);
    draw_list->AddText(context->large_font, font_size, ImVec2{text_min.x, text_y},
                       ImGui::ColorConvertFloat4ToU32(light_grey), stream.name.c_str());
    draw_list->PopClipRect();
  }

  ImGui::PopStyleColor(5);
}

void image_tile_view::render(view_context* context) {
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

  for (auto&& kvp : layout) {
    auto&& view_rect = kvp.second;
    auto stream = kvp.first;
    auto&& stream_mv = streams[stream];

    draw_stream_header(context, view_rect, *stream_mv);

    ImGui::SetCursorPos(ImVec2{view_rect.x - r.x, view_rect.y - r.y});

    stream_mv->texture.show(view_rect, 1.f);
  }

  ImGui::End();
  ImGui::PopStyleVar(2);
}