#pragma once

#include <array>
#include <glm/glm.hpp>
#include <memory>
#include <opencv2/imgproc/imgproc.hpp>

namespace stargazer::voxelpose {
struct camera_data {
  double fx;
  double fy;
  double cx;
  double cy;
  std::array<double, 3> k;
  std::array<double, 2> p;
  std::array<std::array<double, 3>, 3> rotation;
  std::array<double, 3> translation;
};

struct roi_data {
  std::array<double, 2> scale;
  double rotation;
  std::array<double, 2> center;
};

class voxel_projector;
class dnn_inference;
class dnn_inference_heatmap;
class get_proposal;
class joint_extractor;
class proposal_extractor;

class voxelpose {
  std::unique_ptr<dnn_inference_heatmap> inference_heatmap;
  std::unique_ptr<dnn_inference> inference_proposal;
  std::unique_ptr<dnn_inference> inference_pose;
  std::unique_ptr<voxel_projector> global_proj;
  std::unique_ptr<voxel_projector> local_proj;
#if defined(USE_HIP) || defined(USE_MIGRAPHX)
  std::unique_ptr<proposal_extractor> prop_gpu;
#else
  std::unique_ptr<get_proposal> prop;
#endif
  std::unique_ptr<joint_extractor> joint_extract;

  std::array<float, 3> grid_center;
  std::array<float, 3> grid_size;

 public:
  voxelpose();
  ~voxelpose();

  std::vector<glm::vec3> inference(const std::vector<cv::Mat>& images_list,
                                   const std::vector<camera_data>& cameras_list);

  uint32_t get_heatmap_width() const;
  uint32_t get_heatmap_height() const;
  uint32_t get_num_joints() const;

  std::array<int32_t, 3> get_cube_size() const;
  std::array<float, 3> get_grid_size() const;
  std::array<float, 3> get_grid_center() const;
  void set_grid_size(const std::array<float, 3>& value);
  void set_grid_center(const std::array<float, 3>& value);

  const float* get_heatmaps() const;
  void copy_heatmap_to(size_t num_views, float* data) const;
};
}  // namespace stargazer::voxelpose
