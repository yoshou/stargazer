#pragma once

#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <vulkan/vulkan.hpp>

struct graphics_context;

// 3D rendering pipeline for Vulkan
class render3d_pipeline {
 public:
  struct vertex {
    glm::vec3 pos;
    glm::vec3 color;
  };

  struct uniform_buffer_object {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
  };

  render3d_pipeline();
  ~render3d_pipeline();

  void initialize(vk::Device device, vk::PhysicalDevice physical_device,
                  vk::RenderPass render_pass);
  void cleanup();

  void begin_frame(vk::CommandBuffer cmd, const glm::mat4& proj, const glm::mat4& view,
                   vk::Extent2D extent, vk::Offset2D viewport_offset = {0, 0},
                   vk::Extent2D viewport_extent = {0, 0});
  void end_frame(vk::CommandBuffer cmd);

  void draw_line(const glm::vec3& start, const glm::vec3& end, const glm::vec3& color);
  void draw_grid(int size, float spacing, const glm::vec3& color);
  void draw_frustum(const glm::mat4& proj_matrix, const glm::mat4& view_matrix,
                    const glm::vec3& color);
  void draw_point(const glm::vec3& pos, float size, const glm::vec3& color);
  void draw_sphere(const glm::vec3& center, float radius, const glm::vec3& color, int lats = 10,
                   int longs = 10);

 private:
  vk::Device device_;
  vk::PhysicalDevice physical_device_;
  vk::RenderPass render_pass_;
  vk::Extent2D current_extent_;
  vk::Offset2D viewport_offset_;
  vk::Extent2D viewport_extent_;

  vk::UniquePipelineLayout pipeline_layout;
  vk::UniquePipeline graphics_pipeline;
  vk::UniqueDescriptorSetLayout descriptor_set_layout;
  vk::UniqueDescriptorPool descriptor_pool;
  std::vector<vk::DescriptorSet> descriptor_sets;

  std::vector<vk::UniqueBuffer> uniform_buffers;
  std::vector<vk::UniqueDeviceMemory> uniform_buffers_memory;

  vk::UniqueBuffer vertex_buffer;
  vk::UniqueDeviceMemory vertex_buffer_memory;
  vk::DeviceSize vertex_buffer_size_ = 0;

  std::vector<vertex> vertices;
  vk::CommandBuffer current_cmd_buffer;

  glm::mat4 view_matrix;
  glm::mat4 proj_matrix;

  void create_descriptor_set_layout();
  void create_render_pass();
  void create_graphics_pipeline();
  void create_uniform_buffers();
  void create_descriptor_pool();
  void create_descriptor_sets();
  void update_uniform_buffer(uint32_t current_image);
  void create_vertex_buffer();
  void ensure_vertex_buffer_size(vk::DeviceSize required_size);
  void update_vertex_buffer();
};
