#include "gui.hpp"

#include "imgui.h"
#include "imgui_impl_vulkan.h"
#include "viewer.hpp"

namespace stargazer {

static void create_imgui_descriptor_pool(imgui_context* ctx) {
  vk::DescriptorPoolSize pool_sizes[] = {{vk::DescriptorType::eSampler, 1000},
                                         {vk::DescriptorType::eCombinedImageSampler, 1000},
                                         {vk::DescriptorType::eSampledImage, 1000},
                                         {vk::DescriptorType::eStorageImage, 1000},
                                         {vk::DescriptorType::eUniformTexelBuffer, 1000},
                                         {vk::DescriptorType::eStorageTexelBuffer, 1000},
                                         {vk::DescriptorType::eUniformBuffer, 1000},
                                         {vk::DescriptorType::eStorageBuffer, 1000},
                                         {vk::DescriptorType::eUniformBufferDynamic, 1000},
                                         {vk::DescriptorType::eStorageBufferDynamic, 1000},
                                         {vk::DescriptorType::eInputAttachment, 1000}};

  vk::DescriptorPoolCreateInfo pool_info;
  pool_info.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
  pool_info.maxSets = 1000;
  pool_info.poolSizeCount = std::size(pool_sizes);
  pool_info.pPoolSizes = pool_sizes;

  ctx->descriptor_pool = ctx->graphics_ctx->device->createDescriptorPoolUnique(pool_info);
}

void imgui_context::initialize() {
  if (!graphics_ctx || !graphics_ctx->device || !graphics_ctx->render_pass) {
    return;
  }

  create_imgui_descriptor_pool(this);

  QueueFamilyIndices indices =
      find_queue_families(graphics_ctx->physical_device, graphics_ctx->surface.get());

  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.Instance = graphics_ctx->instance.get();
  init_info.PhysicalDevice = graphics_ctx->physical_device;
  init_info.Device = graphics_ctx->device.get();
  init_info.QueueFamily = indices.graphics_family.value();
  init_info.Queue = graphics_ctx->graphics_queue;
  init_info.PipelineCache = VK_NULL_HANDLE;
  init_info.DescriptorPool = descriptor_pool.get();
  init_info.RenderPass = graphics_ctx->render_pass.get();
  init_info.Subpass = 0;
  init_info.MinImageCount = graphics_context::MAX_FRAMES_IN_FLIGHT;
  init_info.ImageCount = static_cast<uint32_t>(graphics_ctx->swapchain_images.size());
  init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
  init_info.Allocator = nullptr;
  init_info.CheckVkResultFn = nullptr;
  init_info.UseDynamicRendering = false;  // Use legacy render pass

  ImGui_ImplVulkan_Init(&init_info);

  // Upload fonts
  vk::CommandBufferAllocateInfo alloc_info;
  alloc_info.commandPool = graphics_ctx->command_pool.get();
  alloc_info.level = vk::CommandBufferLevel::ePrimary;
  alloc_info.commandBufferCount = 1;

  auto command_buffers_temp = graphics_ctx->device->allocateCommandBuffersUnique(alloc_info);
  auto& command_buffer = command_buffers_temp[0];

  vk::CommandBufferBeginInfo begin_info;
  begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

  command_buffer->begin(begin_info);
  ImGui_ImplVulkan_CreateFontsTexture();
  (void)command_buffer->end();

  vk::SubmitInfo submit_info;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer.get();

  (void)graphics_ctx->graphics_queue.submit(1, &submit_info, nullptr);
  (void)graphics_ctx->graphics_queue.waitIdle();

  ImGui_ImplVulkan_DestroyFontsTexture();

  is_initialized = true;
}

void imgui_context::cleanup() {
  if (!is_initialized) {
    return;
  }

  if (graphics_ctx && graphics_ctx->device) {
    graphics_ctx->device->waitIdle();
  }
  ImGui_ImplVulkan_Shutdown();

  is_initialized = false;
}

void imgui_context::begin_frame() { ImGui_ImplVulkan_NewFrame(); }

void imgui_context::end_frame() {
  if (!graphics_ctx || !graphics_ctx->device) {
    return;
  }

  ImGui::Render();
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(),
                                  graphics_ctx->command_buffers[graphics_ctx->current_frame].get());
}

imgui_context::~imgui_context() { cleanup(); }

}  // namespace stargazer
