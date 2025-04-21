#pragma once

#include <array>
#include <glm/glm.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "parameters.hpp"

namespace stargazer_mvpose {
class dnn_inference_det;
class dnn_inference_det_trt;
class dnn_inference_pose;
class dnn_inference_pose_trt;

struct roi_data {
  std::array<double, 2> scale;
  double rotation;
  std::array<double, 2> center;
};

inline cv::Mat get_transform(const cv::Point2f &center, const cv::Size2f &scale,
                             const cv::Size2f &output_size) {
  const auto get_tri_3rd_point = [](const cv::Point2f &a, const cv::Point2f &b) {
    const auto direct = a - b;
    return b + cv::Point2f(-direct.y, direct.x);
  };

  const auto get_affine_transform = [&](const cv::Point2f &center, const cv::Size2f &scale,
                                        const cv::Size2f &output_size) {
    const auto src_w = scale.width * 200.0;
    const auto src_h = scale.height * 200.0;
    const auto dst_w = output_size.width;
    const auto dst_h = output_size.height;

    cv::Point2f src_dir, dst_dir;
    if (src_w >= src_h) {
      src_dir = cv::Point2f(0, src_w * -0.5);
      dst_dir = cv::Point2f(0, dst_w * -0.5);
    } else {
      src_dir = cv::Point2f(src_h * -0.5, 0);
      dst_dir = cv::Point2f(dst_h * -0.5, 0);
    }

    const auto src_tri_a = center;
    const auto src_tri_b = center + src_dir;
    const auto src_tri_c = get_tri_3rd_point(src_tri_a, src_tri_b);
    cv::Point2f src_tri[3] = {src_tri_a, src_tri_b, src_tri_c};

    const auto dst_tri_a = cv::Point2f(dst_w * 0.5, dst_h * 0.5);
    const auto dst_tri_b = dst_tri_a + dst_dir;
    const auto dst_tri_c = get_tri_3rd_point(dst_tri_a, dst_tri_b);

    cv::Point2f dst_tri[3] = {dst_tri_a, dst_tri_b, dst_tri_c};

    return cv::getAffineTransform(src_tri, dst_tri);
  };

  return get_affine_transform(center, scale, output_size);
}

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
}  // namespace stargazer_mvpose
