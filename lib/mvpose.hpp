#pragma once

#include <array>
#include <glm/glm.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "parameters.hpp"

namespace stargazer::mvpose {
class dnn_inference_det;
class dnn_inference_det_trt;
class dnn_inference_pose;
class dnn_inference_pose_trt;

struct roi_data {
  std::array<double, 2> scale;
  double rotation;
  std::array<double, 2> center;
};

class mvpose_matcher;

class mvpose {
  std::unique_ptr<dnn_inference_det_trt> inference_det;
  std::unique_ptr<dnn_inference_pose_trt> inference_pose;
  std::unique_ptr<mvpose_matcher> matcher;

 public:
  mvpose();
  ~mvpose();
  std::vector<glm::vec3> inference(const std::vector<cv::Mat> &images_list,
                                   const std::vector<stargazer::camera_t> &cameras_list);
};
}  // namespace stargazer::mvpose
