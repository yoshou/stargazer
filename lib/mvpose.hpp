#pragma once

#include <array>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>
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

struct mvpose_pose2d {
  cv::Rect2f bbox;
  float bbox_score = -1.0f;
  std::vector<cv::Point2f> joints;
  std::vector<float> scores;
};

struct mvpose_view_result {
  std::vector<mvpose_pose2d> poses;
};

using mvpose_pose_id_t = std::pair<size_t, size_t>;  // (view_index, pose_index_in_view)
using mvpose_match_t = std::vector<mvpose_pose_id_t>;

struct mvpose_inference_result {
  std::vector<glm::vec3> points3d;
  std::vector<mvpose_view_result> views;
  std::vector<mvpose_match_t> matched_list;
  size_t num_joints = 0;
};

class mvpose {
  std::unique_ptr<dnn_inference_det_trt> inference_det;
  std::unique_ptr<dnn_inference_pose_trt> inference_pose;
  std::unique_ptr<mvpose_matcher> matcher;

 public:
  mvpose();
  ~mvpose();
  std::vector<glm::vec3> inference(const std::vector<cv::Mat>& images_list,
                                   const std::vector<stargazer::camera_t>& cameras_list);

  mvpose_inference_result inference_with_matches(
      const std::vector<cv::Mat>& images_list,
      const std::vector<stargazer::camera_t>& cameras_list);
};
}  // namespace stargazer::mvpose
