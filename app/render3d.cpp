#include "render3d.hpp"

#include <array>
#include <cmath>
#include <cstring>

#include "viewer.hpp"

// Vertex shader (SPIR-V embedded as array)
static const uint8_t vert_shader_code[] = {
#include "simple3d.vert.spv.inc"
};

// Fragment shader (SPIR-V embedded as array)
static const uint8_t frag_shader_code[] = {
#include "simple3d.frag.spv.inc"
};

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

render3d_pipeline::render3d_pipeline()
    : device_(nullptr),
      render_pass_(nullptr),
      current_cmd_buffer(nullptr),
      view_matrix(1.0f),
      proj_matrix(1.0f) {}

render3d_pipeline::~render3d_pipeline() { cleanup(); }

void render3d_pipeline::initialize(vk::Device device, vk::PhysicalDevice physical_device,
                                   vk::RenderPass render_pass) {
  device_ = device;
  physical_device_ = physical_device;
  render_pass_ = render_pass;
  current_extent_ = vk::Extent2D{1920, 1080};

  create_descriptor_set_layout();
  create_graphics_pipeline();
  create_uniform_buffers();
  create_descriptor_pool();
  create_descriptor_sets();
  create_vertex_buffer();
}

void render3d_pipeline::cleanup() {
  if (device_) {
    device_.waitIdle();

    // UniqueHandles will automatically clean up
    descriptor_sets.clear();
    uniform_buffers.clear();
    uniform_buffers_memory.clear();
    vertex_buffer.reset();
    vertex_buffer_memory.reset();
    descriptor_pool.reset();
    descriptor_set_layout.reset();
    graphics_pipeline.reset();
    pipeline_layout.reset();

    device_ = nullptr;
    render_pass_ = nullptr;
  }
}

void render3d_pipeline::create_descriptor_set_layout() {
  vk::DescriptorSetLayoutBinding ubo_layout_binding;
  ubo_layout_binding.binding = 0;
  ubo_layout_binding.descriptorType = vk::DescriptorType::eUniformBuffer;
  ubo_layout_binding.descriptorCount = 1;
  ubo_layout_binding.stageFlags = vk::ShaderStageFlagBits::eVertex;

  vk::DescriptorSetLayoutCreateInfo layout_info;
  layout_info.bindingCount = 1;
  layout_info.pBindings = &ubo_layout_binding;

  descriptor_set_layout = device_.createDescriptorSetLayoutUnique(layout_info);
}

void render3d_pipeline::create_graphics_pipeline() {
  // Create shader modules
  vk::ShaderModuleCreateInfo vert_shader_info;
  vert_shader_info.codeSize = sizeof(vert_shader_code);
  vert_shader_info.pCode = reinterpret_cast<const uint32_t*>(vert_shader_code);
  auto vert_shader_module = device_.createShaderModuleUnique(vert_shader_info);

  vk::ShaderModuleCreateInfo frag_shader_info;
  frag_shader_info.codeSize = sizeof(frag_shader_code);
  frag_shader_info.pCode = reinterpret_cast<const uint32_t*>(frag_shader_code);
  auto frag_shader_module = device_.createShaderModuleUnique(frag_shader_info);

  vk::PipelineShaderStageCreateInfo vert_stage_info;
  vert_stage_info.stage = vk::ShaderStageFlagBits::eVertex;
  vert_stage_info.module = vert_shader_module.get();
  vert_stage_info.pName = "main";

  vk::PipelineShaderStageCreateInfo frag_stage_info;
  frag_stage_info.stage = vk::ShaderStageFlagBits::eFragment;
  frag_stage_info.module = frag_shader_module.get();
  frag_stage_info.pName = "main";

  vk::PipelineShaderStageCreateInfo shader_stages[] = {vert_stage_info, frag_stage_info};

  // Vertex input
  vk::VertexInputBindingDescription binding_description;
  binding_description.binding = 0;
  binding_description.stride = sizeof(vertex);
  binding_description.inputRate = vk::VertexInputRate::eVertex;

  std::array<vk::VertexInputAttributeDescription, 2> attribute_descriptions;
  attribute_descriptions[0].binding = 0;
  attribute_descriptions[0].location = 0;
  attribute_descriptions[0].format = vk::Format::eR32G32B32Sfloat;
  attribute_descriptions[0].offset = offsetof(vertex, pos);

  attribute_descriptions[1].binding = 0;
  attribute_descriptions[1].location = 1;
  attribute_descriptions[1].format = vk::Format::eR32G32B32Sfloat;
  attribute_descriptions[1].offset = offsetof(vertex, color);

  vk::PipelineVertexInputStateCreateInfo vertex_input_info;
  vertex_input_info.vertexBindingDescriptionCount = 1;
  vertex_input_info.pVertexBindingDescriptions = &binding_description;
  vertex_input_info.vertexAttributeDescriptionCount =
      static_cast<uint32_t>(attribute_descriptions.size());
  vertex_input_info.pVertexAttributeDescriptions = attribute_descriptions.data();

  // Input assembly
  vk::PipelineInputAssemblyStateCreateInfo input_assembly;
  input_assembly.topology = vk::PrimitiveTopology::eLineList;
  input_assembly.primitiveRestartEnable = VK_FALSE;

  // Viewport and scissor (dynamic)
  vk::PipelineViewportStateCreateInfo viewport_state;
  viewport_state.viewportCount = 1;
  viewport_state.scissorCount = 1;

  // Rasterizer
  vk::PipelineRasterizationStateCreateInfo rasterizer;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = vk::PolygonMode::eFill;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = vk::CullModeFlagBits::eNone;
  rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
  rasterizer.depthBiasEnable = VK_FALSE;

  // Multisampling
  vk::PipelineMultisampleStateCreateInfo multisampling;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;

  // Depth and stencil state (enabled for proper 3D rendering)
  vk::PipelineDepthStencilStateCreateInfo depth_stencil;
  depth_stencil.depthTestEnable = VK_TRUE;
  depth_stencil.depthWriteEnable = VK_TRUE;
  depth_stencil.depthCompareOp = vk::CompareOp::eLess;
  depth_stencil.depthBoundsTestEnable = VK_FALSE;
  depth_stencil.stencilTestEnable = VK_FALSE;

  // Color blending
  vk::PipelineColorBlendAttachmentState color_blend_attachment;
  color_blend_attachment.colorWriteMask =
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
      vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
  color_blend_attachment.blendEnable = VK_FALSE;

  vk::PipelineColorBlendStateCreateInfo color_blending;
  color_blending.logicOpEnable = VK_FALSE;
  color_blending.attachmentCount = 1;
  color_blending.pAttachments = &color_blend_attachment;

  // Dynamic state
  std::vector<vk::DynamicState> dynamic_states = {vk::DynamicState::eViewport,
                                                  vk::DynamicState::eScissor};

  vk::PipelineDynamicStateCreateInfo dynamic_state;
  dynamic_state.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
  dynamic_state.pDynamicStates = dynamic_states.data();

  // Pipeline layout
  vk::PipelineLayoutCreateInfo pipeline_layout_info;
  pipeline_layout_info.setLayoutCount = 1;
  pipeline_layout_info.pSetLayouts = &descriptor_set_layout.get();

  pipeline_layout = device_.createPipelineLayoutUnique(pipeline_layout_info);

  // Graphics pipeline
  vk::GraphicsPipelineCreateInfo pipeline_info;
  pipeline_info.stageCount = 2;
  pipeline_info.pStages = shader_stages;
  pipeline_info.pVertexInputState = &vertex_input_info;
  pipeline_info.pInputAssemblyState = &input_assembly;
  pipeline_info.pViewportState = &viewport_state;
  pipeline_info.pRasterizationState = &rasterizer;
  pipeline_info.pMultisampleState = &multisampling;
  pipeline_info.pDepthStencilState = &depth_stencil;  // Add depth stencil state
  pipeline_info.pColorBlendState = &color_blending;
  pipeline_info.pDynamicState = &dynamic_state;
  pipeline_info.layout = pipeline_layout.get();
  pipeline_info.renderPass = render_pass_;
  pipeline_info.subpass = 0;

  auto result = device_.createGraphicsPipelineUnique(nullptr, pipeline_info);
  graphics_pipeline = std::move(result.value);
}

void render3d_pipeline::create_uniform_buffers() {
  vk::DeviceSize buffer_size = sizeof(uniform_buffer_object);

  // Create 2 uniform buffers for double buffering
  uniform_buffers.resize(2);
  uniform_buffers_memory.resize(2);

  for (size_t i = 0; i < 2; i++) {
    vk::BufferCreateInfo buffer_info;
    buffer_info.size = buffer_size;
    buffer_info.usage = vk::BufferUsageFlagBits::eUniformBuffer;
    buffer_info.sharingMode = vk::SharingMode::eExclusive;

    uniform_buffers[i] = device_.createBufferUnique(buffer_info);

    // Get physical device for memory allocation (need to pass from initialize())
    // For now, allocate with basic properties
    auto mem_requirements = device_.getBufferMemoryRequirements(uniform_buffers[i].get());

    vk::MemoryAllocateInfo alloc_info;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = find_memory_type(
        physical_device_, mem_requirements.memoryTypeBits,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    uniform_buffers_memory[i] = device_.allocateMemoryUnique(alloc_info);
    device_.bindBufferMemory(uniform_buffers[i].get(), uniform_buffers_memory[i].get(), 0);
  }
}

void render3d_pipeline::create_descriptor_pool() {
  vk::DescriptorPoolSize pool_size;
  pool_size.type = vk::DescriptorType::eUniformBuffer;
  pool_size.descriptorCount = 2;

  vk::DescriptorPoolCreateInfo pool_info;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes = &pool_size;
  pool_info.maxSets = 2;

  descriptor_pool = device_.createDescriptorPoolUnique(pool_info);
}

void render3d_pipeline::create_descriptor_sets() {
  std::vector<vk::DescriptorSetLayout> layouts(2, descriptor_set_layout.get());

  vk::DescriptorSetAllocateInfo alloc_info;
  alloc_info.descriptorPool = descriptor_pool.get();
  alloc_info.descriptorSetCount = 2;
  alloc_info.pSetLayouts = layouts.data();

  descriptor_sets = device_.allocateDescriptorSets(alloc_info);

  for (size_t i = 0; i < 2; i++) {
    vk::DescriptorBufferInfo buffer_info;
    buffer_info.buffer = uniform_buffers[i].get();
    buffer_info.offset = 0;
    buffer_info.range = sizeof(uniform_buffer_object);

    vk::WriteDescriptorSet descriptor_write;
    descriptor_write.dstSet = descriptor_sets[i];
    descriptor_write.dstBinding = 0;
    descriptor_write.dstArrayElement = 0;
    descriptor_write.descriptorType = vk::DescriptorType::eUniformBuffer;
    descriptor_write.descriptorCount = 1;
    descriptor_write.pBufferInfo = &buffer_info;

    device_.updateDescriptorSets(1, &descriptor_write, 0, nullptr);
  }
}

void render3d_pipeline::create_vertex_buffer() { ensure_vertex_buffer_size(sizeof(vertex) * 256); }

void render3d_pipeline::ensure_vertex_buffer_size(vk::DeviceSize required_size) {
  if (required_size == 0) {
    required_size = sizeof(vertex) * 2;
  }

  if (vertex_buffer && vertex_buffer_size_ >= required_size) {
    return;
  }

  // Avoid destroying a buffer that might still be in flight.
  if (vertex_buffer) {
    device_.waitIdle();
  }

  vk::BufferCreateInfo buffer_info;
  buffer_info.size = required_size;
  buffer_info.usage = vk::BufferUsageFlagBits::eVertexBuffer;
  buffer_info.sharingMode = vk::SharingMode::eExclusive;

  vertex_buffer = device_.createBufferUnique(buffer_info);

  auto mem_requirements = device_.getBufferMemoryRequirements(vertex_buffer.get());
  vk::MemoryAllocateInfo alloc_info;
  alloc_info.allocationSize = mem_requirements.size;
  alloc_info.memoryTypeIndex = find_memory_type(
      physical_device_, mem_requirements.memoryTypeBits,
      vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

  vertex_buffer_memory = device_.allocateMemoryUnique(alloc_info);
  device_.bindBufferMemory(vertex_buffer.get(), vertex_buffer_memory.get(), 0);
  vertex_buffer_size_ = required_size;
}

void render3d_pipeline::begin_frame(vk::CommandBuffer cmd, const glm::mat4& proj,
                                    const glm::mat4& view, vk::Extent2D extent,
                                    vk::Offset2D viewport_offset, vk::Extent2D viewport_extent) {
  current_cmd_buffer = cmd;
  proj_matrix = proj;
  view_matrix = view;
  current_extent_ = extent;

  // Use provided viewport extent, or default to full extent
  viewport_offset_ = viewport_offset;
  if (viewport_extent.width == 0 || viewport_extent.height == 0) {
    viewport_extent_ = extent;
  } else {
    viewport_extent_ = viewport_extent;
  }

  vertices.clear();
}

void render3d_pipeline::end_frame(vk::CommandBuffer cmd) {
  if (vertices.empty()) {
    return;
  }

  // Update uniform buffer
  uniform_buffer_object ubo;
  ubo.model = glm::mat4(1.0f);
  ubo.view = view_matrix;
  ubo.proj = proj_matrix;

  // Map to first uniform buffer (simplified - should use frame index)
  void* data = device_.mapMemory(uniform_buffers_memory[0].get(), 0, sizeof(ubo));
  std::memcpy(data, &ubo, sizeof(ubo));
  device_.unmapMemory(uniform_buffers_memory[0].get());

  vk::DeviceSize buffer_size = sizeof(vertex) * vertices.size();
  ensure_vertex_buffer_size(buffer_size);

  // Copy vertex data into persistent buffer
  void* vertex_data = device_.mapMemory(vertex_buffer_memory.get(), 0, buffer_size);
  std::memcpy(vertex_data, vertices.data(), buffer_size);

  device_.unmapMemory(vertex_buffer_memory.get());

  // Bind pipeline and draw
  cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, graphics_pipeline.get());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout.get(), 0, 1,
                         &descriptor_sets[0], 0, nullptr);

  vk::Buffer vertex_buffers[] = {vertex_buffer.get()};
  vk::DeviceSize offsets[] = {0};
  cmd.bindVertexBuffers(0, 1, vertex_buffers, offsets);

  // Set dynamic viewport and scissor (use stored viewport info)
  vk::Viewport viewport;
  viewport.x = static_cast<float>(viewport_offset_.x);
  viewport.y = static_cast<float>(viewport_offset_.y);
  viewport.width = static_cast<float>(viewport_extent_.width);
  viewport.height = static_cast<float>(viewport_extent_.height);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  cmd.setViewport(0, 1, &viewport);

  vk::Rect2D scissor;
  scissor.offset = viewport_offset_;
  scissor.extent = viewport_extent_;
  cmd.setScissor(0, 1, &scissor);

  cmd.draw(static_cast<uint32_t>(vertices.size()), 1, 0, 0);
}

void render3d_pipeline::draw_line(const glm::vec3& start, const glm::vec3& end,
                                  const glm::vec3& color) {
  vertices.push_back({start, color});
  vertices.push_back({end, color});
}

void render3d_pipeline::draw_grid(int tiles, float spacing, const glm::vec3& color) {
  float half_size = tiles * spacing / 2.0f;

  for (int i = 0; i <= tiles; i++) {
    float pos = -half_size + i * spacing;

    // Color scheme: 0.7 for center lines, 0.4 for others
    float intensity = (i == tiles / 2) ? 0.7f : 0.4f;
    glm::vec3 line_color = color * intensity;

    // Lines along X axis
    draw_line(glm::vec3(pos, 0.0f, -half_size), glm::vec3(pos, 0.0f, half_size), line_color);

    // Lines along Z axis
    draw_line(glm::vec3(-half_size, 0.0f, pos), glm::vec3(half_size, 0.0f, pos), line_color);
  }
}

void render3d_pipeline::draw_frustum(const glm::mat4& proj_matrix, const glm::mat4& view_matrix,
                                     const glm::vec3& color) {
  // This function is no longer used - camera frustums are drawn using deproject method in views.cpp
  (void)proj_matrix;
  (void)view_matrix;
  (void)color;
}

void render3d_pipeline::draw_point(const glm::vec3& pos, float size, const glm::vec3& color) {
  // Draw a 3D cross at the point position
  float half_size = size * 0.5f;

  // X axis line
  draw_line(glm::vec3(pos.x - half_size, pos.y, pos.z), glm::vec3(pos.x + half_size, pos.y, pos.z),
            color);

  // Y axis line
  draw_line(glm::vec3(pos.x, pos.y - half_size, pos.z), glm::vec3(pos.x, pos.y + half_size, pos.z),
            color);

  // Z axis line
  draw_line(glm::vec3(pos.x, pos.y, pos.z - half_size), glm::vec3(pos.x, pos.y, pos.z + half_size),
            color);
}

void render3d_pipeline::draw_sphere(const glm::vec3& center, float radius, const glm::vec3& color,
                                    int lats, int longs) {
  // Draw sphere using latitude and longitude lines

  const float pi = 3.14159265359f;

  // Draw latitude circles
  for (int i = 0; i <= lats; i++) {
    float lat0 = pi * (-0.5f + (float)(i - 1) / lats);
    float lat1 = pi * (-0.5f + (float)i / lats);
    float z0 = radius * std::sin(lat0);
    float z1 = radius * std::sin(lat1);
    float r0 = radius * std::cos(lat0);
    float r1 = radius * std::cos(lat1);

    // Draw longitude lines
    for (int j = 0; j < longs; j++) {
      float lng0 = 2 * pi * (float)j / longs;
      float lng1 = 2 * pi * (float)(j + 1) / longs;

      float x00 = r0 * std::cos(lng0);
      float y00 = r0 * std::sin(lng0);
      float x01 = r0 * std::cos(lng1);
      float y01 = r0 * std::sin(lng1);

      float x10 = r1 * std::cos(lng0);
      float y10 = r1 * std::sin(lng0);
      float x11 = r1 * std::cos(lng1);
      float y11 = r1 * std::sin(lng1);

      if (i > 0) {
        // Horizontal edge
        draw_line(center + glm::vec3(x00, y00, z0), center + glm::vec3(x01, y01, z0), color);
        // Vertical edge
        draw_line(center + glm::vec3(x00, y00, z0), center + glm::vec3(x10, y10, z1), color);
      }
      if (i == lats) {
        // Last horizontal edge
        draw_line(center + glm::vec3(x10, y10, z1), center + glm::vec3(x11, y11, z1), color);
      }
    }
  }
}
