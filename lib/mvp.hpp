#pragma once

#include <array>
#include <glm/glm.hpp>
#include <memory>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

namespace stargazer::mvp {

struct camera_data {
  double fx;
  double fy;
  double cx;
  double cy;
  std::array<double, 5> dist_coeff;  // k1, k2, p1, p2, k3
  std::array<std::array<double, 3>, 3> rotation;
  std::array<double, 3> translation;
};

class dnn_inference_backbone;
class dnn_inference_head;

class mvp {
  std::unique_ptr<dnn_inference_backbone> inference_backbone;
  std::unique_ptr<dnn_inference_head> inference_head;

  std::array<float, 3> grid_center;
  std::array<float, 3> grid_size;

 public:
  mvp();
  ~mvp();

  std::vector<glm::vec3> inference(const std::vector<cv::Mat>& images_list,
                                   const std::vector<camera_data>& cameras_list);

  static const uint32_t num_joints = 15;
  static const uint32_t num_instances = 10;

  std::array<float, 3> get_grid_size() const;
  std::array<float, 3> get_grid_center() const;
  void set_grid_size(const std::array<float, 3>& value);
  void set_grid_center(const std::array<float, 3>& value);
};

}  // namespace stargazer::mvp
