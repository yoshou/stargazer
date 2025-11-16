#pragma once

#include <array>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace stargazer {
inline cv::Mat get_transform(const cv::Point2f& center, const cv::Size2f& scale,
                             const cv::Size2f& output_size) {
  const auto get_tri_3rd_point = [](const cv::Point2f& a, const cv::Point2f& b) {
    const auto direct = a - b;
    return b + cv::Point2f(-direct.y, direct.x);
  };

  const auto get_affine_transform = [&](const cv::Point2f& center, const cv::Size2f& scale,
                                        const cv::Size2f& output_size) {
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

void preprocess_cuda(const uint8_t* src_data, int src_width, int src_height, int src_step,
                     float* dst_data, int dst_width, int dst_height, int dst_step, cv::Mat trans,
                     const std::array<float, 3>& mean, const std::array<float, 3>& std);
void preprocess_cuda(const uint8_t* src_data, int src_width, int src_height, int src_step,
                     float* dst_data, int dst_width, int dst_height, int dst_step,
                     const std::array<float, 3>& mean, const std::array<float, 3>& std);
}  // namespace stargazer
