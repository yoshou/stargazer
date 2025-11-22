#pragma once

#include <vulkan/vulkan.hpp>

struct graphics_context;

// ImGui integration for Vulkan
namespace stargazer {
struct imgui_context {
  graphics_context* graphics_ctx;
  vk::UniqueDescriptorPool descriptor_pool;
  bool is_initialized = false;

  imgui_context(graphics_context* ctx) : graphics_ctx(ctx) {}

  void initialize();
  void cleanup();
  void begin_frame();
  void end_frame();

  ~imgui_context();
};
}  // namespace stargazer
