#include "viewer.hpp"

#include <GLFW/glfw3.h>

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>
#include <vulkan/vulkan.hpp>

static constexpr int DEFAULT_SCREEN_WIDTH = 1680;
static constexpr int DEFAULT_SCREEN_HEIGHT = 1050;

bool window_manager::should_close() { return should_close_flag.load(); }

window_manager::window_manager() : should_close_flag(false) {}

void window_manager::handle_event() {
  if (std::this_thread::get_id() != thread_id) {
    throw std::runtime_error("Invalid call");
  }

  glfwPollEvents();

  std::vector<std::unique_ptr<action_func_base>> events;
  {
    std::lock_guard<std::mutex> lock(mtx);
    while (queue.size() > 0) {
      events.emplace_back(std::move(queue.front()));
      queue.pop_front();
    }
  }

  for (const auto& event : events) {
    event->invoke();
  }
}

static void mouse_scroll_callback(GLFWwindow* handle, double x, double y) {
  const auto callback = static_cast<window_base*>(glfwGetWindowUserPointer((GLFWwindow*)handle));
  if (callback) {
    callback->on_scroll(x, y);
  }
}
static void mouse_button_callback(GLFWwindow* handle, int button, int action, int mods) {
  const auto callback = static_cast<window_base*>(glfwGetWindowUserPointer((GLFWwindow*)handle));
  if (callback) {
    callback->on_mouse_click(button, action, mods);
  }
}
void mouse_cursor_callback(GLFWwindow* handle, double x, double y) {
  const auto callback = static_cast<window_base*>(glfwGetWindowUserPointer((GLFWwindow*)handle));
  if (callback) {
    callback->on_mouse(x, y);
  }
}
void mouse_cursor_enter_callback(GLFWwindow* handle, int entered) {
  const auto callback = static_cast<window_base*>(glfwGetWindowUserPointer((GLFWwindow*)handle));
  if (callback) {
    callback->on_enter(entered);
  }
}
void key_callback(GLFWwindow* handle, int key, int scancode, int action, int mods) {
  const auto callback = static_cast<window_base*>(glfwGetWindowUserPointer((GLFWwindow*)handle));
  if (callback) {
    callback->on_key(key, scancode, action, mods);
  }
}
void char_callback(GLFWwindow* handle, unsigned int codepoint) {
  const auto callback = static_cast<window_base*>(glfwGetWindowUserPointer((GLFWwindow*)handle));
  if (callback) {
    callback->on_char(codepoint);
  }
}
void window_size_callback(GLFWwindow* handle, int width, int height) {
  const auto callback = static_cast<window_base*>(glfwGetWindowUserPointer((GLFWwindow*)handle));
  if (callback) {
    callback->on_resize(width, height);
  }
}
void window_close_callback(GLFWwindow* handle) {
  const auto callback = static_cast<window_base*>(glfwGetWindowUserPointer((GLFWwindow*)handle));
  if (callback) {
    callback->on_close();
  }
}

void* window_manager::create_window_handle(std::string name, int width, int height,
                                           window_base* window) {
  std::promise<GLFWwindow*> p;
  auto f = p.get_future();
  auto func = [_p = std::move(p), name, width, height, window]() mutable {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    auto handle = glfwCreateWindow(width, height, name.c_str(), NULL, NULL);
    glfwSetWindowUserPointer(handle, window);
    glfwSetScrollCallback(handle, mouse_scroll_callback);
    glfwSetMouseButtonCallback(handle, mouse_button_callback);
    glfwSetCursorPosCallback(handle, mouse_cursor_callback);
    glfwSetCursorEnterCallback(handle, mouse_cursor_enter_callback);
    glfwSetKeyCallback(handle, key_callback);
    glfwSetCharCallback(handle, char_callback);
    glfwSetWindowSizeCallback(handle, window_size_callback);
    glfwSetWindowCloseCallback(handle, window_close_callback);
    _p.set_value(handle);
  };
  if (std::this_thread::get_id() == thread_id) {
    func();
  } else {
    {
      std::lock_guard<std::mutex> lock(mtx);
      queue.emplace_back(std::make_unique<action_func<decltype(func)>>(std::move(func)));
    }
    glfwPostEmptyEvent();
  }
  return f.get();
}

void window_manager::destroy_window_handle(window_base* window) {
  const auto handle = (GLFWwindow*)window->get_handle();
  auto func = [handle]() mutable { glfwDestroyWindow((GLFWwindow*)handle); };
  if (std::this_thread::get_id() == thread_id) {
    func();
  } else {
    {
      std::lock_guard<std::mutex> lock(mtx);
      queue.emplace_back(std::make_unique<action_func<decltype(func)>>(std::move(func)));
    }
    glfwPostEmptyEvent();
  }
}

void window_manager::show_window(window_base* window) {
  const auto handle = (GLFWwindow*)window->get_handle();
  auto func = [handle]() mutable { glfwShowWindow((GLFWwindow*)handle); };
  if (std::this_thread::get_id() == thread_id) {
    func();
  } else {
    {
      std::lock_guard<std::mutex> lock(mtx);
      queue.emplace_back(std::make_unique<action_func<decltype(func)>>(std::move(func)));
    }
    glfwPostEmptyEvent();
  }
}

void window_manager::hide_window(window_base* window) {
  const auto handle = (GLFWwindow*)window->get_handle();
  auto func = [handle]() mutable { glfwHideWindow((GLFWwindow*)handle); };
  if (std::this_thread::get_id() == thread_id) {
    func();
  } else {
    {
      std::lock_guard<std::mutex> lock(mtx);
      queue.emplace_back(std::make_unique<action_func<decltype(func)>>(std::move(func)));
    }
    glfwPostEmptyEvent();
  }
}

void window_manager::initialize() {
  thread_id = std::this_thread::get_id();

  if (glfwInit() == GL_FALSE) {
    throw std::runtime_error("Can't initialize GLFW");
  }
  if (!glfwVulkanSupported()) {
    throw std::runtime_error("Vulkan is not supported");
  }
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
}

void window_manager::exit() {
  should_close_flag.store(true);
  glfwPostEmptyEvent();
}

void window_manager::terminate() { glfwTerminate(); }

void window_manager::get_window_size(window_base* window, int* width, int* height) {
  std::promise<std::tuple<int, int>> p;
  auto f = p.get_future();
  const auto handle = (GLFWwindow*)window->get_handle();
  auto func = [_p = std::move(p), handle]() mutable {
    int w, h;
    glfwGetWindowSize(handle, &w, &h);
    _p.set_value(std::make_tuple(w, h));
  };
  if (std::this_thread::get_id() == thread_id) {
    func();
  } else {
    {
      std::lock_guard<std::mutex> lock(mtx);
      queue.emplace_back(std::make_unique<action_func<decltype(func)>>(std::move(func)));
    }
    glfwPostEmptyEvent();
  }
  const auto [w, h] = f.get();
  *width = w;
  *height = h;
}

void window_manager::get_window_frame_size(window_base* window, int* left, int* top, int* right,
                                           int* bottom) {
  std::promise<std::tuple<int, int, int, int>> p;
  auto f = p.get_future();
  const auto handle = (GLFWwindow*)window->get_handle();
  auto func = [_p = std::move(p), handle]() mutable {
    int l, t, r, b;
    glfwGetWindowFrameSize(handle, &l, &t, &r, &b);
    _p.set_value(std::make_tuple(l, t, r, b));
  };
  if (std::this_thread::get_id() == thread_id) {
    func();
  } else {
    {
      std::lock_guard<std::mutex> lock(mtx);
      queue.emplace_back(std::make_unique<action_func<decltype(func)>>(std::move(func)));
    }
    glfwPostEmptyEvent();
  }
  const auto [l, t, r, b] = f.get();
  *left = l;
  *top = t;
  *right = r;
  *bottom = b;
}

std::shared_ptr<window_manager> window_manager::get_instance() {
  static const auto win_mgr = std::make_shared<window_manager>();
  return win_mgr;
}

window_base::window_base(std::string name, std::size_t width, std::size_t height)
    : handle(nullptr), name(name), width(width), height(height) {}

void* window_base::get_handle() const { return handle; }

bool window_base::is_closed() const {
  if (handle == nullptr) {
    return true;
  }
  return glfwWindowShouldClose((GLFWwindow*)handle) == GL_TRUE;
}

void window_base::on_close() {}
void window_base::on_key(int key, int scancode, int action, int mods) {}
void window_base::on_char(unsigned int codepoint) {}
void window_base::on_scroll(double x, double y) {}
void window_base::on_mouse_click(int button, int action, int mods) {}
void window_base::on_mouse(double x, double y) {}
void window_base::on_enter(int entered) {}
void window_base::on_resize(int width, int height) {
  this->width = width;
  this->height = height;
}
void window_base::show() { window_manager::get_instance()->show_window(this); }
void window_base::create() {
  handle = (GLFWwindow*)window_manager::get_instance()->create_window_handle(
      name, DEFAULT_SCREEN_WIDTH, DEFAULT_SCREEN_HEIGHT, this);
  if (handle == nullptr) {
    std::cerr << "Can't create GLFW window." << std::endl;
    return;
  }
}
void window_base::initialize() {}
void window_base::finalize() {}
void window_base::destroy() {
  if (this->handle) {
    window_manager::get_instance()->destroy_window_handle(this);
  }
  this->handle = nullptr;
}
graphics_context window_base::create_graphics_context() { return graphics_context(this); }

void window_base::update() {}

mouse_state mouse_state::get_mouse_state(void* handle) {
  mouse_state mouse;
  glfwGetCursorPos((GLFWwindow*)handle, &mouse.x, &mouse.y);
  mouse.right_button = glfwGetMouseButton((GLFWwindow*)handle, GLFW_MOUSE_BUTTON_RIGHT);
  mouse.left_button = glfwGetMouseButton((GLFWwindow*)handle, GLFW_MOUSE_BUTTON_LEFT);
  mouse.middle_button = glfwGetMouseButton((GLFWwindow*)handle, GLFW_MOUSE_BUTTON_MIDDLE);
  return mouse;
}

static void create_vulkan_instance(graphics_context* ctx) {
  uint32_t required_extensions_count;
  const char** required_extensions = glfwGetRequiredInstanceExtensions(&required_extensions_count);

  vk::InstanceCreateInfo create_info;
  create_info.enabledExtensionCount = required_extensions_count;
  create_info.ppEnabledExtensionNames = required_extensions;

  ctx->instance = vk::createInstanceUnique(create_info);
}

static void create_surface(graphics_context* ctx) {
  VkSurfaceKHR c_surface;
  auto result = glfwCreateWindowSurface(ctx->instance.get(), (GLFWwindow*)ctx->window->get_handle(),
                                        nullptr, &c_surface);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create window surface");
  }
  ctx->surface = vk::UniqueSurfaceKHR(c_surface, ctx->instance.get());
}

QueueFamilyIndices find_queue_families(vk::PhysicalDevice device, vk::SurfaceKHR surface) {
  QueueFamilyIndices indices;
  auto queue_families = device.getQueueFamilyProperties();

  uint32_t i = 0;
  for (const auto& queue_family : queue_families) {
    if (queue_family.queueFlags & vk::QueueFlagBits::eGraphics) {
      indices.graphics_family = i;
    }

    if (device.getSurfaceSupportKHR(i, surface)) {
      indices.present_family = i;
    }

    if (indices.is_complete()) {
      break;
    }
    i++;
  }

  return indices;
}

static bool is_device_suitable(vk::PhysicalDevice device, vk::SurfaceKHR surface) {
  QueueFamilyIndices indices = find_queue_families(device, surface);

  // Check if device supports required extensions
  auto available_extensions = device.enumerateDeviceExtensionProperties();
  std::set<std::string> required_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  for (const auto& extension : available_extensions) {
    required_extensions.erase(extension.extensionName);
  }

  bool extensions_supported = required_extensions.empty();
  bool swap_chain_adequate = false;

  if (extensions_supported) {
    auto formats = device.getSurfaceFormatsKHR(surface);
    auto present_modes = device.getSurfacePresentModesKHR(surface);
    swap_chain_adequate = !formats.empty() && !present_modes.empty();
  }

  return indices.is_complete() && extensions_supported && swap_chain_adequate;
}

static void pick_physical_device(graphics_context* ctx) {
  auto devices = ctx->instance->enumeratePhysicalDevices();
  if (devices.empty()) {
    throw std::runtime_error("Failed to find GPUs with Vulkan support");
  }

  // Try to find a suitable discrete GPU first
  for (const auto& device : devices) {
    if (is_device_suitable(device, ctx->surface.get())) {
      auto properties = device.getProperties();
      if (properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
        ctx->physical_device = device;
        return;
      }
    }
  }

  // Fall back to any suitable device
  for (const auto& device : devices) {
    if (is_device_suitable(device, ctx->surface.get())) {
      ctx->physical_device = device;
      return;
    }
  }

  throw std::runtime_error("Failed to find a suitable GPU. DRI3 support may be required.");
}

static void create_logical_device(graphics_context* ctx) {
  QueueFamilyIndices indices = find_queue_families(ctx->physical_device, ctx->surface.get());

  if (!indices.is_complete()) {
    throw std::runtime_error(
        "Failed to find suitable queue families. Graphics or present queue not supported.");
  }

  std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;
  std::set<uint32_t> unique_queue_families = {indices.graphics_family.value(),
                                              indices.present_family.value()};

  float queue_priority = 1.0f;
  for (uint32_t queue_family : unique_queue_families) {
    vk::DeviceQueueCreateInfo queue_create_info;
    queue_create_info.queueFamilyIndex = queue_family;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;
    queue_create_infos.push_back(queue_create_info);
  }

  vk::PhysicalDeviceFeatures device_features{};

  vk::DeviceCreateInfo create_info;
  create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
  create_info.pQueueCreateInfos = queue_create_infos.data();
  create_info.pEnabledFeatures = &device_features;

  const std::vector<const char*> device_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  create_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
  create_info.ppEnabledExtensionNames = device_extensions.data();

  ctx->device = ctx->physical_device.createDeviceUnique(create_info);
  ctx->graphics_queue = ctx->device->getQueue(indices.graphics_family.value(), 0);
  ctx->present_queue = ctx->device->getQueue(indices.present_family.value(), 0);
}

struct SwapChainSupportDetails {
  vk::SurfaceCapabilitiesKHR capabilities;
  std::vector<vk::SurfaceFormatKHR> formats;
  std::vector<vk::PresentModeKHR> present_modes;
};

static SwapChainSupportDetails query_swap_chain_support(vk::PhysicalDevice device,
                                                        vk::SurfaceKHR surface) {
  SwapChainSupportDetails details;
  details.capabilities = device.getSurfaceCapabilitiesKHR(surface);
  details.formats = device.getSurfaceFormatsKHR(surface);
  details.present_modes = device.getSurfacePresentModesKHR(surface);
  return details;
}

static vk::SurfaceFormatKHR choose_swap_surface_format(
    const std::vector<vk::SurfaceFormatKHR>& available_formats) {
  // Prefer UNORM format for ImGui (no sRGB conversion)
  for (const auto& available_format : available_formats) {
    if (available_format.format == vk::Format::eB8G8R8A8Unorm &&
        available_format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
      return available_format;
    }
  }
  // Fallback to sRGB if UNORM not available
  for (const auto& available_format : available_formats) {
    if (available_format.format == vk::Format::eB8G8R8A8Srgb &&
        available_format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
      return available_format;
    }
  }
  return available_formats[0];
}

static vk::PresentModeKHR choose_swap_present_mode(
    const std::vector<vk::PresentModeKHR>& available_present_modes) {
  for (const auto& available_present_mode : available_present_modes) {
    if (available_present_mode == vk::PresentModeKHR::eMailbox) {
      return available_present_mode;
    }
  }
  return vk::PresentModeKHR::eFifo;
}

static vk::Extent2D choose_swap_extent(const vk::SurfaceCapabilitiesKHR& capabilities,
                                       GLFWwindow* window) {
  if (capabilities.currentExtent.width != UINT32_MAX) {
    return capabilities.currentExtent;
  } else {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    vk::Extent2D actual_extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

    actual_extent.width = std::clamp(actual_extent.width, capabilities.minImageExtent.width,
                                     capabilities.maxImageExtent.width);
    actual_extent.height = std::clamp(actual_extent.height, capabilities.minImageExtent.height,
                                      capabilities.maxImageExtent.height);

    return actual_extent;
  }
}

static void create_swap_chain(graphics_context* ctx) {
  SwapChainSupportDetails swap_chain_support =
      query_swap_chain_support(ctx->physical_device, ctx->surface.get());

  vk::SurfaceFormatKHR surface_format = choose_swap_surface_format(swap_chain_support.formats);
  vk::PresentModeKHR present_mode = choose_swap_present_mode(swap_chain_support.present_modes);
  vk::Extent2D extent =
      choose_swap_extent(swap_chain_support.capabilities, (GLFWwindow*)ctx->window->get_handle());

  uint32_t image_count = swap_chain_support.capabilities.minImageCount + 1;
  if (swap_chain_support.capabilities.maxImageCount > 0 &&
      image_count > swap_chain_support.capabilities.maxImageCount) {
    image_count = swap_chain_support.capabilities.maxImageCount;
  }

  vk::SwapchainCreateInfoKHR create_info;
  create_info.surface = ctx->surface.get();
  create_info.minImageCount = image_count;
  create_info.imageFormat = surface_format.format;
  create_info.imageColorSpace = surface_format.colorSpace;
  create_info.imageExtent = extent;
  create_info.imageArrayLayers = 1;
  create_info.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;

  QueueFamilyIndices indices = find_queue_families(ctx->physical_device, ctx->surface.get());
  uint32_t queue_family_indices[] = {indices.graphics_family.value(),
                                     indices.present_family.value()};

  if (indices.graphics_family != indices.present_family) {
    create_info.imageSharingMode = vk::SharingMode::eConcurrent;
    create_info.queueFamilyIndexCount = 2;
    create_info.pQueueFamilyIndices = queue_family_indices;
  } else {
    create_info.imageSharingMode = vk::SharingMode::eExclusive;
  }

  create_info.preTransform = swap_chain_support.capabilities.currentTransform;
  create_info.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
  create_info.presentMode = present_mode;
  create_info.clipped = VK_TRUE;

  ctx->swapchain = ctx->device->createSwapchainKHRUnique(create_info);
  ctx->swapchain_images = ctx->device->getSwapchainImagesKHR(ctx->swapchain.get());
  ctx->swapchain_image_format = surface_format.format;
  ctx->swapchain_extent = extent;
}

static void create_command_pool(graphics_context* ctx) {
  QueueFamilyIndices queue_family_indices =
      find_queue_families(ctx->physical_device, ctx->surface.get());

  vk::CommandPoolCreateInfo pool_info;
  pool_info.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
  pool_info.queueFamilyIndex = queue_family_indices.graphics_family.value();

  ctx->command_pool = ctx->device->createCommandPoolUnique(pool_info);
}

static void create_command_buffers(graphics_context* ctx) {
  vk::CommandBufferAllocateInfo alloc_info;
  alloc_info.commandPool = ctx->command_pool.get();
  alloc_info.level = vk::CommandBufferLevel::ePrimary;
  alloc_info.commandBufferCount = graphics_context::MAX_FRAMES_IN_FLIGHT;

  ctx->command_buffers = ctx->device->allocateCommandBuffersUnique(alloc_info);
}

static void create_sync_objects(graphics_context* ctx) {
  ctx->image_available_semaphores.resize(graphics_context::MAX_FRAMES_IN_FLIGHT);
  ctx->render_finished_semaphores.resize(graphics_context::MAX_FRAMES_IN_FLIGHT);
  ctx->in_flight_fences.resize(graphics_context::MAX_FRAMES_IN_FLIGHT);

  vk::SemaphoreCreateInfo semaphore_info;
  vk::FenceCreateInfo fence_info;
  fence_info.flags = vk::FenceCreateFlagBits::eSignaled;

  for (size_t i = 0; i < graphics_context::MAX_FRAMES_IN_FLIGHT; i++) {
    ctx->image_available_semaphores[i] = ctx->device->createSemaphoreUnique(semaphore_info);
    ctx->render_finished_semaphores[i] = ctx->device->createSemaphoreUnique(semaphore_info);
    ctx->in_flight_fences[i] = ctx->device->createFenceUnique(fence_info);
  }
}

static void create_render_pass(graphics_context* ctx) {
  std::array<vk::AttachmentDescription, 2> attachments{};

  // Color attachment
  attachments[0].format = ctx->swapchain_image_format;
  attachments[0].samples = vk::SampleCountFlagBits::e1;
  attachments[0].loadOp = vk::AttachmentLoadOp::eClear;
  attachments[0].storeOp = vk::AttachmentStoreOp::eStore;
  attachments[0].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
  attachments[0].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
  attachments[0].initialLayout = vk::ImageLayout::eUndefined;
  attachments[0].finalLayout = vk::ImageLayout::ePresentSrcKHR;

  // Depth attachment
  attachments[1].format = vk::Format::eD32Sfloat;
  attachments[1].samples = vk::SampleCountFlagBits::e1;
  attachments[1].loadOp = vk::AttachmentLoadOp::eClear;
  attachments[1].storeOp = vk::AttachmentStoreOp::eDontCare;
  attachments[1].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
  attachments[1].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
  attachments[1].initialLayout = vk::ImageLayout::eUndefined;
  attachments[1].finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

  vk::AttachmentReference color_attachment_ref;
  color_attachment_ref.attachment = 0;
  color_attachment_ref.layout = vk::ImageLayout::eColorAttachmentOptimal;

  vk::AttachmentReference depth_attachment_ref;
  depth_attachment_ref.attachment = 1;
  depth_attachment_ref.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

  vk::SubpassDescription subpass;
  subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &color_attachment_ref;
  subpass.pDepthStencilAttachment = &depth_attachment_ref;

  vk::SubpassDependency dependency;
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput |
                            vk::PipelineStageFlagBits::eEarlyFragmentTests;
  dependency.srcAccessMask = vk::AccessFlags();
  dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput |
                            vk::PipelineStageFlagBits::eEarlyFragmentTests;
  dependency.dstAccessMask =
      vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite;

  vk::RenderPassCreateInfo render_pass_info;
  render_pass_info.attachmentCount = static_cast<uint32_t>(attachments.size());
  render_pass_info.pAttachments = attachments.data();
  render_pass_info.subpassCount = 1;
  render_pass_info.pSubpasses = &subpass;
  render_pass_info.dependencyCount = 1;
  render_pass_info.pDependencies = &dependency;

  ctx->render_pass = ctx->device->createRenderPassUnique(render_pass_info);
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

  throw std::runtime_error("failed to find suitable memory type!");
}

static void create_depth_resources(graphics_context* ctx) {
  vk::Format depth_format = vk::Format::eD32Sfloat;

  // Create depth image
  vk::ImageCreateInfo image_info;
  image_info.imageType = vk::ImageType::e2D;
  image_info.extent.width = ctx->swapchain_extent.width;
  image_info.extent.height = ctx->swapchain_extent.height;
  image_info.extent.depth = 1;
  image_info.mipLevels = 1;
  image_info.arrayLayers = 1;
  image_info.format = depth_format;
  image_info.tiling = vk::ImageTiling::eOptimal;
  image_info.initialLayout = vk::ImageLayout::eUndefined;
  image_info.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;
  image_info.samples = vk::SampleCountFlagBits::e1;
  image_info.sharingMode = vk::SharingMode::eExclusive;

  ctx->depth_image = ctx->device->createImageUnique(image_info);

  // Allocate depth image memory
  vk::MemoryRequirements mem_requirements =
      ctx->device->getImageMemoryRequirements(ctx->depth_image.get());

  vk::MemoryAllocateInfo alloc_info;
  alloc_info.allocationSize = mem_requirements.size;
  alloc_info.memoryTypeIndex =
      find_memory_type(ctx->physical_device, mem_requirements.memoryTypeBits,
                       vk::MemoryPropertyFlagBits::eDeviceLocal);

  ctx->depth_image_memory = ctx->device->allocateMemoryUnique(alloc_info);
  ctx->device->bindImageMemory(ctx->depth_image.get(), ctx->depth_image_memory.get(), 0);

  // Create depth image view
  vk::ImageViewCreateInfo view_info;
  view_info.image = ctx->depth_image.get();
  view_info.viewType = vk::ImageViewType::e2D;
  view_info.format = depth_format;
  view_info.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
  view_info.subresourceRange.baseMipLevel = 0;
  view_info.subresourceRange.levelCount = 1;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount = 1;

  ctx->depth_image_view = ctx->device->createImageViewUnique(view_info);
}

static void create_image_views(graphics_context* ctx) {
  ctx->swapchain_image_views.resize(ctx->swapchain_images.size());

  for (size_t i = 0; i < ctx->swapchain_images.size(); i++) {
    vk::ImageViewCreateInfo create_info;
    create_info.image = ctx->swapchain_images[i];
    create_info.viewType = vk::ImageViewType::e2D;
    create_info.format = ctx->swapchain_image_format;
    create_info.components.r = vk::ComponentSwizzle::eIdentity;
    create_info.components.g = vk::ComponentSwizzle::eIdentity;
    create_info.components.b = vk::ComponentSwizzle::eIdentity;
    create_info.components.a = vk::ComponentSwizzle::eIdentity;
    create_info.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    create_info.subresourceRange.baseMipLevel = 0;
    create_info.subresourceRange.levelCount = 1;
    create_info.subresourceRange.baseArrayLayer = 0;
    create_info.subresourceRange.layerCount = 1;

    ctx->swapchain_image_views[i] = ctx->device->createImageViewUnique(create_info);
  }
}

static void create_framebuffers(graphics_context* ctx) {
  ctx->framebuffers.resize(ctx->swapchain_image_views.size());

  for (size_t i = 0; i < ctx->swapchain_image_views.size(); i++) {
    std::array<vk::ImageView, 2> attachments = {ctx->swapchain_image_views[i].get(),
                                                ctx->depth_image_view.get()};

    vk::FramebufferCreateInfo framebuffer_info;
    framebuffer_info.renderPass = ctx->render_pass.get();
    framebuffer_info.attachmentCount = static_cast<uint32_t>(attachments.size());
    framebuffer_info.pAttachments = attachments.data();
    framebuffer_info.width = ctx->swapchain_extent.width;
    framebuffer_info.height = ctx->swapchain_extent.height;
    framebuffer_info.layers = 1;

    ctx->framebuffers[i] = ctx->device->createFramebufferUnique(framebuffer_info);
  }
}

void graphics_context::attach() {
  create_vulkan_instance(this);
  create_surface(this);
  pick_physical_device(this);
  create_logical_device(this);
  create_swap_chain(this);
  create_image_views(this);
  create_render_pass(this);
  create_depth_resources(this);
  create_framebuffers(this);
  create_command_pool(this);
  create_command_buffers(this);
  create_sync_objects(this);
}

void graphics_context::detach() {
  if (device) {
    device->waitIdle();
  }
}

void graphics_context::begin_frame() {
  if (!device || !swapchain) {
    return;
  }

  device->waitForFences(1, &in_flight_fences[current_frame].get(), VK_TRUE, UINT64_MAX);

  auto result = device->acquireNextImageKHR(swapchain.get(), UINT64_MAX,
                                            image_available_semaphores[current_frame].get(),
                                            nullptr, &current_image_index);

  if (result == vk::Result::eErrorOutOfDateKHR) {
    return;
  } else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
    throw std::runtime_error("Failed to acquire swap chain image");
  }

  device->resetFences(1, &in_flight_fences[current_frame].get());

  command_buffers[current_frame]->reset();

  vk::CommandBufferBeginInfo begin_info;
  (void)command_buffers[current_frame]->begin(begin_info);

  // Begin render pass
  vk::RenderPassBeginInfo render_pass_info;
  render_pass_info.renderPass = render_pass.get();
  render_pass_info.framebuffer = framebuffers[current_image_index].get();
  render_pass_info.renderArea.offset = vk::Offset2D{0, 0};
  render_pass_info.renderArea.extent = swapchain_extent;

  std::array<vk::ClearValue, 2> clear_values{};
  clear_values[0].color = std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f};  // Black background
  clear_values[1].depthStencil = vk::ClearDepthStencilValue{1.0f, 0};    // Clear depth to 1.0 (far)

  render_pass_info.clearValueCount = static_cast<uint32_t>(clear_values.size());
  render_pass_info.pClearValues = clear_values.data();

  command_buffers[current_frame]->beginRenderPass(render_pass_info, vk::SubpassContents::eInline);
}

void graphics_context::end_frame() {
  if (!device || !swapchain) {
    return;
  }

  // End render pass
  command_buffers[current_frame]->endRenderPass();
  (void)command_buffers[current_frame]->end();

  vk::SubmitInfo submit_info;

  vk::Semaphore wait_semaphores[] = {image_available_semaphores[current_frame].get()};
  vk::PipelineStageFlags wait_stages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
  submit_info.waitSemaphoreCount = 1;
  submit_info.pWaitSemaphores = wait_semaphores;
  submit_info.pWaitDstStageMask = wait_stages;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffers[current_frame].get();

  vk::Semaphore signal_semaphores[] = {render_finished_semaphores[current_frame].get()};
  submit_info.signalSemaphoreCount = 1;
  submit_info.pSignalSemaphores = signal_semaphores;

  (void)graphics_queue.submit(1, &submit_info, in_flight_fences[current_frame].get());

  vk::PresentInfoKHR present_info;
  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores = signal_semaphores;

  vk::SwapchainKHR swapchains[] = {swapchain.get()};
  present_info.swapchainCount = 1;
  present_info.pSwapchains = swapchains;
  present_info.pImageIndices = &current_image_index;

  (void)present_queue.presentKHR(present_info);

  current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
}

graphics_context::~graphics_context() {}
