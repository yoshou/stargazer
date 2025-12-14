#include "views.hpp"

#include <spdlog/spdlog.h>

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

static std::vector<const char*> get_string_pointers(const std::vector<std::string>& vec) {
  std::vector<const char*> res;
  for (auto&& s : vec) res.push_back(s.c_str());
  return res;
}

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
static const textual_icon times{u8"\uf00d"};
static const textual_icon refresh{u8"\uf021"};
static const textual_icon edit{u8"\uf044"};
static const textual_icon play{u8"\uf04b"};
static const textual_icon stop{u8"\uf04d"};
static const textual_icon plus_circle{u8"\uf055"};
static const textual_icon circle{u8"\uf111"};
static const textual_icon toggle_off{u8"\uf204"};
static const textual_icon toggle_on{u8"\uf205"};
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

void top_bar_view::render(view_context* context) {
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

void capture_panel_view::draw_controls(view_context* context, float panel_height)

{
  std::vector<std::function<void()>> draw_later;

  auto panel_width = 350;
  auto header_h = panel_height;
  ImColor node_header_background_color = title_color;
  const float left_space = 3.f;
  const float upper_space = 3.f;

  // if (is_ip_device)
  header_h += 32;

  // draw controls
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
      node_name = "camera";
      ImGui::OpenPopup("Node");
    }

    ImGui::PopFont();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    ImGui::PopStyleColor();
    ImGui::PopStyleColor();
  }

  for (auto& node : nodes) {
    ImVec2 initial_screen_pos = ImGui::GetCursorScreenPos();

    // Upper Space
    ImGui::GetWindowDrawList()->AddRectFilled(
        {initial_screen_pos.x, initial_screen_pos.y},
        {initial_screen_pos.x + panel_width, initial_screen_pos.y + upper_space}, ImColor(black));
    // if (draw_node_outline)
    {
      // Upper Line
      ImGui::GetWindowDrawList()->AddLine(
          {initial_screen_pos.x, initial_screen_pos.y + upper_space},
          {initial_screen_pos.x + panel_width, initial_screen_pos.y + upper_space},
          ImColor(header_color));
    }
    // Node Header area
    ImGui::GetWindowDrawList()->AddRectFilled(
        {initial_screen_pos.x + 1, initial_screen_pos.y + upper_space + 1},
        {initial_screen_pos.x + panel_width, initial_screen_pos.y + header_h + upper_space},
        node_header_background_color);

    auto pos = ImGui::GetCursorPos();
    ImGui::PushFont(context->large_font);
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)node_header_background_color);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)node_header_background_color);

    // draw name
    {
      const ImVec2 name_pos = {pos.x + 9, pos.y + 17};
      ImGui::SetCursorPos(name_pos);
      std::stringstream ss;
      // if (dev.supports(RS2_CAMERA_INFO_NAME))
      //     ss << dev.get_info(RS2_CAMERA_INFO_NAME);
      // if (is_ip_device)
      {
        ImGui::Text(" %s", node.name.c_str());

        ImGui::PushFont(context->large_font);
        ImGui::Text("\tNode at %s", node.address.c_str());
        ImGui::PopFont();
      }
    }

    ImGui::PopFont();

    // draw x button
    {
      bool _allow_remove = true;
      const auto id = node.name;

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
          for (auto& f : on_remove_node) {
            f(node.name);
          }
        }

        if (ImGui::IsItemHovered()) {
          ImGui::SetTooltip(
              "Remove selected node from current view\n(can be restored by clicking Add Source)");
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

    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, sensor_bg);
    ImGui::PushStyleColor(ImGuiCol_Text, light_grey);
    ImGui::PushFont(context->large_font);

    // draw streaming
    {
      draw_later.push_back([pos, windows_width, this, context, &node]() {
        const auto id = node.name;

        ImGui::SetCursorPos({windows_width - 35, pos.y + 3});
        ImGui_ScopePushFont(context->default_font);

        ImGui_ScopePushStyleColor(ImGuiCol_Button, sensor_bg);
        ImGui_ScopePushStyleColor(ImGuiCol_ButtonHovered, sensor_bg);
        ImGui_ScopePushStyleColor(ImGuiCol_ButtonActive, sensor_bg);

        if (!node.is_streaming) {
          std::string label = to_string() << "  " << textual_icons::toggle_off << "\noff   ##" << id
                                          << "," << "";

          ImGui_ScopePushStyleColor(ImGuiCol_Text, redish);
          ImGui_ScopePushStyleColor(ImGuiCol_TextSelectedBg, redish + 0.1f);

          if (ImGui::Button(label.c_str(), {30, 30})) {
            node.is_streaming = true;
            for (const auto& f : is_streaming_changed) {
              if (!f(node)) {
                node.is_streaming = false;
                break;
              }
            }
          }
        } else {
          std::string label = to_string() << "  " << textual_icons::toggle_on << "\n    on##" << id
                                          << "," << "";
          ImGui_ScopePushStyleColor(ImGuiCol_Text, light_blue);
          ImGui_ScopePushStyleColor(ImGuiCol_TextSelectedBg, light_blue + 0.1f);

          if (ImGui::Button(label.c_str(), {30, 30})) {
            node.is_streaming = false;
            for (const auto& f : is_streaming_changed) {
              if (!f(node)) {
                node.is_streaming = true;
                break;
              }
            }
          }
        }
      });

      const auto id = node.name;

      std::string label = to_string() << "Infrared Camera Module" << "##" << id;
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

    const auto draw_node_outline = true;
    if (draw_node_outline) {
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

  for (const auto& func : draw_later) {
    func();
  }
}

void capture_panel_view::render(view_context* context) {
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

    if (ImGui::BeginPopupModal("Node", nullptr,
                               ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove)) {
      ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 3);
      ImGui::SetCursorPosX(10);
      ImGui::Text("Connect to a Linux system running graph_proc_server");

      ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 5);

      {
        std::vector<std::string> node_type_names = {
            "raspi", "raspi_color", "depthai_color", "rs_d435", "rs_d435_color",
        };
        std::vector<const char*> node_type_names_chars = get_string_pointers(node_type_names);

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
          node_name = gateway_input;
        }
        ImGui::PopStyleColor();

        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 6);

        ImGui::PopItemWidth();
      }
      static char dev_name_input[255];
      std::copy(node_name.begin(), node_name.end(), dev_name_input);
      dev_name_input[node_name.size()] = '\0';
      {
        ImGui::SetCursorPosX(10);
        ImGui::Text("Name");
        ImGui::SameLine();
        ImGui::SetCursorPosX(80);
        ImGui::PushItemWidth(width - ImGui::GetCursorPosX() - 10);
        ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, light_blue);

        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 3);
        if (ImGui::InputText("##name", dev_name_input, 255)) {
          node_name = dev_name_input;
        }
        ImGui::PopStyleColor();

        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 6);

        ImGui::PopItemWidth();
      }

      ImGui::SetCursorPosX(width / 2 - 105);

      if (ImGui::Button("OK", {100.f, 25.f}) || ImGui::IsKeyDown(ImGuiKey_Enter) ||
          ImGui::IsKeyDown(ImGuiKey_KeypadEnter)) {
        try {
          for (auto& f : on_add_node) {
            f(node_name, static_cast<node_type>(node_type_index), ip_address, gateway_address);
          }
        } catch (std::runtime_error e) {
          spdlog::error(e.what());
        }
        node_type_index = 0;
        node_name = "";
        ip_address = "";
        gateway_address = "";
        ImGui::CloseCurrentPopup();
      }
      ImGui::SameLine();
      ImGui::SetCursorPosX(width / 2 + 5);
      if (ImGui::Button("Cancel", {100.f, 25.f}) || ImGui::IsKeyDown(ImGuiKey_Escape)) {
        node_type_index = 0;
        node_name = "";
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

float capture_panel_view::draw_control_panel(view_context* context) {
  const float node_panel_height = 60.0f;
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
  const ImVec2 node_panel_icons_size{icons_width, 25};
  textual_icon button_icon = is_streaming ? textual_icons::stop : textual_icons::play;
  std::string play_button_name = to_string() << button_icon << "##" << id;
  auto play_button_color = is_streaming ? light_blue : light_grey;
  {
    ImGui::PushStyleColor(ImGuiCol_Text, play_button_color);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, play_button_color);
    if (ImGui::Button(play_button_name.c_str(), node_panel_icons_size)) {
      if (is_streaming) {
        is_streaming = false;
        for (const auto& f : is_all_streaming_changed) {
          if (!f(nodes, is_streaming)) {
            is_streaming = true;
            break;
          }
        }
      } else {
        is_streaming = true;
        for (const auto& f : is_all_streaming_changed) {
          if (!f(nodes, is_streaming)) {
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

    ImGui::PushStyleColor(ImGuiCol_Text, play_button_color);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, play_button_color);
    ImGui::Button(is_streaming ? "Stop" : "Start", node_panel_icons_size);
    ImGui::PopStyleColor(2);
    ImGui::PopStyleColor(3);
  }

  ImGui::PopStyleVar();
  ImGui::PopStyleColor(7);
  ImGui::PopFont();

  return node_panel_height;
}

float calibration_panel_view::draw_control_panel(view_context* context) {
  const float node_panel_height = 60.0f;
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
  const ImVec2 node_panel_icons_size{icons_width, 25};
  textual_icon button_icon = is_streaming ? textual_icons::stop : textual_icons::play;
  std::string play_button_name = to_string() << button_icon << "##" << id;
  auto play_button_color = is_streaming ? light_blue : light_grey;
  {
    ImGui::PushStyleColor(ImGuiCol_Text, play_button_color);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, play_button_color);
    if (ImGui::Button(play_button_name.c_str(), node_panel_icons_size)) {
      if (is_streaming) {
        is_streaming = false;
        for (const auto& f : is_streaming_changed) {
          if (!f(nodes, is_streaming)) {
            is_streaming = true;
            break;
          }
        }
      } else {
        is_streaming = true;
        for (const auto& f : is_streaming_changed) {
          if (!f(nodes, is_streaming)) {
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
    if (ImGui::Button(mask_button_name.c_str(), node_panel_icons_size)) {
      if (is_masking) {
        is_masking = false;
        for (const auto& f : is_masking_changed) {
          if (!f(nodes, is_masking)) {
            is_masking = true;
            break;
          }
        }
      } else {
        is_masking = true;
        for (const auto& f : is_masking_changed) {
          if (!f(nodes, is_masking)) {
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
    if (ImGui::Button(calibrate_button_name.c_str(), node_panel_icons_size)) {
      for (const auto& f : on_calibrate) {
        f(nodes, true);
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

    ImGui::PushStyleColor(ImGuiCol_Text, play_button_color);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, play_button_color);
    ImGui::Button(is_streaming ? "Stop" : "Start", node_panel_icons_size);
    ImGui::PopStyleColor(2);
    ImGui::PopStyleColor(3);
  }
  ImGui::SameLine();
  {
    ImGui::PushStyleColor(ImGuiCol_Text, mask_button_color);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, mask_button_color);
    ImGui::Button("Mask", node_panel_icons_size);
    ImGui::PopStyleColor(2);
  }
  ImGui::SameLine();
  {
    ImGui::PushStyleColor(ImGuiCol_Text, calibrate_button_color);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, calibrate_button_color);
    ImGui::Button("Calibrate", node_panel_icons_size);
    ImGui::PopStyleColor(2);
  }

  ImGui::PopStyleVar();
  ImGui::PopStyleColor(7);
  ImGui::PopFont();

  return node_panel_height;
}

void calibration_panel_view::draw_extrinsic_calibration_control_panel(view_context* context) {
  std::vector<std::function<void()>> draw_later;
  {
    auto pos = ImGui::GetCursorPos();
    auto windows_width = ImGui::GetContentRegionMax().x;

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
          std::string label = to_string() << "  " << textual_icons::toggle_off << "\noff   ##" << id
                                          << "," << "";

          ImGui_ScopePushStyleColor(ImGuiCol_Text, redish);
          ImGui_ScopePushStyleColor(ImGuiCol_TextSelectedBg, redish + 0.1f);

          if (ImGui::Button(label.c_str(), {30, 30})) {
            is_marker_collecting = true;
            for (const auto& f : is_marker_collecting_changed) {
              if (!f(nodes, is_marker_collecting)) {
                is_marker_collecting = false;
                break;
              }
            }
          }
        } else {
          std::string label = to_string() << "  " << textual_icons::toggle_on << "\n    on##" << id
                                          << "," << "";
          ImGui_ScopePushStyleColor(ImGuiCol_Text, light_blue);
          ImGui_ScopePushStyleColor(ImGuiCol_TextSelectedBg, light_blue + 0.1f);

          if (ImGui::Button(label.c_str(), {30, 30})) {
            is_marker_collecting = false;
            for (const auto& f : is_marker_collecting_changed) {
              if (!f(nodes, is_marker_collecting)) {
                is_marker_collecting = true;
                break;
              }
            }
          }
        }
      });

      const auto id = "";

      std::string label = to_string() << "Collect Markers" << "##" << id;
      ImGui::PushStyleColor(ImGuiCol_Header, sensor_bg);
      ImGui::PushStyleColor(ImGuiCol_HeaderActive, sensor_bg);
      ImGui::PushStyleColor(ImGuiCol_HeaderHovered, sensor_bg);
      ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{10, 10});
      ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, {0, 0});
      ImGuiTreeNodeFlags flags{};
      // if (show_depth_only) flags = ImGuiTreeNodeFlags_DefaultOpen;
      if (ImGui::TreeNodeEx(label.c_str(), flags | ImGuiTreeNodeFlags_FramePadding)) {
        for (auto& node : nodes) {
          auto pos = ImGui::GetCursorPos();
          auto windows_width = ImGui::GetContentRegionMax().x;

          ImGui::PopStyleVar();
          ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{2, 2});

          // draw streaming
          {
            draw_later.push_back([pos, windows_width, context, &node]() {
              const auto id = node.name;

              ImGui::SetCursorPos({windows_width - 35, pos.y + 3});
              ImGui_ScopePushFont(context->default_font);

              ImGui_ScopePushStyleColor(ImGuiCol_Button, sensor_bg);
              ImGui_ScopePushStyleColor(ImGuiCol_ButtonHovered, sensor_bg);
              ImGui_ScopePushStyleColor(ImGuiCol_ButtonActive, sensor_bg);

              if (!node.is_streaming) {
                std::string label = to_string() << "  " << textual_icons::toggle_off << "\noff   ##"
                                                << id << "," << "";

                ImGui_ScopePushStyleColor(ImGuiCol_Text, redish);
                ImGui_ScopePushStyleColor(ImGuiCol_TextSelectedBg, redish + 0.1f);

                if (ImGui::Button(label.c_str(), {30, 30})) {
                  node.is_streaming = true;
                }
              } else {
                std::string label = to_string() << "  " << textual_icons::toggle_on << "\n    on##"
                                                << id << "," << "";
                ImGui_ScopePushStyleColor(ImGuiCol_Text, light_blue);
                ImGui_ScopePushStyleColor(ImGuiCol_TextSelectedBg, light_blue + 0.1f);

                if (ImGui::Button(label.c_str(), {30, 30})) {
                  node.is_streaming = false;
                }
              }
            });

            ImGui::Text("%s", node.name.c_str());

            ImGui::SameLine();
            ImGui::SetCursorPosX(220);

            const auto screen_pos = ImGui::GetCursorScreenPos();
            auto c = ImGui::GetColorU32(ImGuiCol_FrameBg);

            ImGui::GetWindowDrawList()->AddRectFilled({200, screen_pos.y}, {300, screen_pos.y + 20},
                                                      c);

            ImGui::Text("%s", std::to_string(node.num_points).c_str());

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

  for (const auto& func : draw_later) {
    func();
  }
}

void calibration_panel_view::draw_intrinsic_calibration_control_panel(view_context* context) {
  const auto panel_width = 350;

  // draw selecting node
  {
    std::vector<std::string> intrinsic_calibration_devices;
    for (const auto& node : nodes) {
      intrinsic_calibration_devices.push_back(node.name);
    }
    std::string id = "##intrinsic_calibration_device";
    std::vector<const char*> intrinsic_calibration_devices_chars =
        get_string_pointers(intrinsic_calibration_devices);

    const auto pos = ImGui::GetCursorPos();
    ImGui::PushItemWidth(panel_width - 40);
    ImGui::PushFont(context->large_font);
    ImGui::SetCursorPos({pos.x + 10, pos.y});
    if (ImGui::Combo(id.c_str(), &intrinsic_calibration_target_index,
                     intrinsic_calibration_devices_chars.data(),
                     static_cast<int>(intrinsic_calibration_devices.size()))) {
      for (auto& func : on_intrinsic_calibration_target_changed) {
        func(nodes.at(intrinsic_calibration_target_index));
      }
    }
    ImGui::SetCursorPos({pos.x, ImGui::GetCursorPos().y});
    ImGui::PopFont();
    ImGui::PopItemWidth();
  }

  std::vector<std::function<void()>> draw_later;
  {
    auto& node = nodes[intrinsic_calibration_target_index];

    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, sensor_bg);
    ImGui::PushStyleColor(ImGuiCol_Text, light_grey);
    ImGui::PushFont(context->large_font);

    // draw streaming
    {
      const auto id = "";

      std::string label = to_string() << "Collect Markers" << "##" << id;
      ImGui::PushStyleColor(ImGuiCol_Header, sensor_bg);
      ImGui::PushStyleColor(ImGuiCol_HeaderActive, sensor_bg);
      ImGui::PushStyleColor(ImGuiCol_HeaderHovered, sensor_bg);
      ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{10, 10});
      ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, {0, 0});

      {
        auto pos = ImGui::GetCursorPos();
        auto windows_width = ImGui::GetContentRegionMax().x;

        ImGui::PopStyleVar();
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{2, 2});

        // draw streaming
        {
          draw_later.push_back([pos, windows_width, this, context, &node]() {
            const auto id = node.name;

            ImGui::SetCursorPos({windows_width - 35, pos.y + 3});
            ImGui_ScopePushFont(context->default_font);

            ImGui_ScopePushStyleColor(ImGuiCol_Button, sensor_bg);
            ImGui_ScopePushStyleColor(ImGuiCol_ButtonHovered, sensor_bg);
            ImGui_ScopePushStyleColor(ImGuiCol_ButtonActive, sensor_bg);

            if (!is_marker_collecting) {
              std::string label = to_string() << "  " << textual_icons::toggle_off << "\noff   ##"
                                              << id << "," << "";

              ImGui_ScopePushStyleColor(ImGuiCol_Text, redish);
              ImGui_ScopePushStyleColor(ImGuiCol_TextSelectedBg, redish + 0.1f);

              if (ImGui::Button(label.c_str(), {30, 30})) {
                is_marker_collecting = true;
              }
            } else {
              std::string label = to_string() << "  " << textual_icons::toggle_on << "\n    on##"
                                              << id << "," << "";
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

            ImGui::Text("%s", std::to_string(node.num_points).c_str());

            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 2);
          }

          const auto draw_param = [&](const std::string& name, float value) {
            ImGui::SetCursorPosX(20);

            ImGui::Text("%s", name.c_str());

            ImGui::SameLine();
            ImGui::SetCursorPosX(220);

            const auto screen_pos = ImGui::GetCursorScreenPos();
            auto c = ImGui::GetColorU32(ImGuiCol_FrameBg);

            ImGui::GetWindowDrawList()->AddRectFilled({200, screen_pos.y}, {300, screen_pos.y + 20},
                                                      c);

            ImGui::Text("%s", fmt::format("{:6.3f}", value).c_str());

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

  for (const auto& func : draw_later) {
    func();
  }
}

void calibration_panel_view::draw_controls(view_context* context, float panel_height) {
  const auto panel_width = 350;

  // draw controls
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

  // draw selecting calibration target
  {
    std::vector<std::string> calibration_targets = {
        "Extrinsic parameters",
        "Intrinsic parameters",
        "Scene parameters",
    };
    std::string id = "##calibration_target";
    std::vector<const char*> calibration_targets_chars = get_string_pointers(calibration_targets);

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

void calibration_panel_view::render(view_context* context) {
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

float reconstruction_panel_view::draw_control_panel(view_context* context) {
  const float node_panel_height = 60.0f;
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
  const ImVec2 node_panel_icons_size{icons_width, 25};
  textual_icon button_icon = is_streaming ? textual_icons::stop : textual_icons::play;
  std::string play_button_name = to_string() << button_icon << "##" << id;
  auto play_button_color = is_streaming ? light_blue : light_grey;
  {
    ImGui::PushStyleColor(ImGuiCol_Text, play_button_color);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, play_button_color);
    if (ImGui::Button(play_button_name.c_str(), node_panel_icons_size)) {
      if (is_streaming) {
        is_streaming = false;
        for (const auto& f : is_streaming_changed) {
          if (!f(nodes, is_streaming)) {
            is_streaming = true;
            break;
          }
        }
      } else {
        is_streaming = true;
        for (const auto& f : is_streaming_changed) {
          if (!f(nodes, is_streaming)) {
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
    if (ImGui::Button(record_button_name.c_str(), node_panel_icons_size)) {
      if (is_recording) {
        is_recording = false;
        for (const auto& f : is_recording_changed) {
          if (!f(nodes, is_recording)) {
            is_recording = true;
            break;
          }
        }
      } else {
        is_recording = true;
        for (const auto& f : is_recording_changed) {
          if (!f(nodes, is_recording)) {
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

    ImGui::PushStyleColor(ImGuiCol_Text, play_button_color);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, play_button_color);
    ImGui::Button(is_streaming ? "Stop" : "Start", node_panel_icons_size);
    ImGui::PopStyleColor(2);
    ImGui::PopStyleColor(3);
  }
  ImGui::SameLine();
  {
    ImGui::PushStyleColor(ImGuiCol_Text, record_button_color);
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, record_button_color);
    ImGui::Button("Record", node_panel_icons_size);
    ImGui::PopStyleColor(2);
  }

  ImGui::PopStyleVar();
  ImGui::PopStyleColor(7);
  ImGui::PopFont();

  return node_panel_height;
}

void reconstruction_panel_view::draw_controls(view_context* context, float panel_height) {
  std::vector<std::function<void()>> draw_later;

  auto panel_width = 350;

  // draw controls
  {
    auto pos = ImGui::GetCursorPos();
    const float vertical_space_before_node_control = 10.0f;
    const float horizontal_space_before_node_control = 3.0f;
    auto node_panel_pos = ImVec2{pos.x + horizontal_space_before_node_control,
                                 pos.y + vertical_space_before_node_control};
    ImGui::SetCursorPos(node_panel_pos);
    const float node_panel_height = draw_control_panel(context);
    ImGui::SetCursorPos({node_panel_pos.x, node_panel_pos.y + node_panel_height});
  }

  {
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
    std::vector<const char*> sources_chars = get_string_pointers(sources);

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

  for (const auto& func : draw_later) {
    func();
  }
}

void reconstruction_panel_view::render(view_context* context) {
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
    glm::mat4 camera_pose = scaled_axis * camera.pose;

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

void image_tile_view::draw_stream_header(view_context* context, const rect& stream_rect) {
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

    draw_stream_header(context, view_rect);

    ImGui::SetCursorPos(ImVec2{view_rect.x - r.x, view_rect.y - r.y});

    stream_mv->texture.show(view_rect, 1.f);
  }

  ImGui::End();
  ImGui::PopStyleVar(2);
}