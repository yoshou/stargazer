#include <gtest/gtest.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

#include "dust3r.hpp"

#include "parameters.hpp"

namespace {

stargazer::camera_intrin_t make_intrin(float fx, float fy, float cx, float cy,
                                       const std::array<float, 5>& coeffs = {}) {
  stargazer::camera_intrin_t intrin{};
  intrin.fx = fx;
  intrin.fy = fy;
  intrin.cx = cx;
  intrin.cy = cy;
  std::copy(coeffs.begin(), coeffs.end(), intrin.coeffs.begin());
  return intrin;
}

cv::Mat make_pattern_image(int width, int height) {
  cv::Mat bgr(height, width, CV_8UC3);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      bgr.at<cv::Vec3b>(y, x) = cv::Vec3b(static_cast<uint8_t>((x * 17 + y * 13) % 256),
                                          static_cast<uint8_t>((x * 29 + y * 7) % 256),
                                          static_cast<uint8_t>((x * 11 + y * 19) % 256));
    }
  }
  return bgr;
}

float mean_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  EXPECT_EQ(a.size(), b.size());
  float sum = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    sum += std::abs(a[i] - b[i]);
  }
  return sum / static_cast<float>(a.size());
}

}  // namespace

TEST(DustR3Inference, TC1_PreprocessShape) {
  cv::Mat bgr = make_pattern_image(37, 23);
  auto intrin = make_intrin(420.0f, 415.0f, 18.0f, 11.0f);
  auto nchw = stargazer::dust3r::preprocess_image(bgr, intrin);

  ASSERT_EQ(nchw.size(), 3u * 288u * 512u);

  const float min_val = *std::min_element(nchw.begin(), nchw.end());
  const float max_val = *std::max_element(nchw.begin(), nchw.end());
  EXPECT_GE(min_val, -1.1f);
  EXPECT_LE(max_val, 1.1f);

  std::cout << "[TC1] nchw size=" << nchw.size() << "  min=" << min_val << "  max=" << max_val
            << "\n";
}

TEST(DustR3Inference, TC2_PreprocessChannelOrderAndNormalization) {
  cv::Mat bgr(9, 7, CV_8UC3, cv::Scalar(0, 127, 255));
  auto intrin = make_intrin(300.0f, 300.0f, 3.0f, 4.0f);

  auto nchw = stargazer::dust3r::preprocess_image(bgr, intrin);
  const size_t pixels = static_cast<size_t>(stargazer::dust3r::ONNX_H * stargazer::dust3r::ONNX_W);

  const float expected_r = 1.0f;
  const float expected_g = (127.0f / 255.0f - 0.5f) / 0.5f;
  const float expected_b = -1.0f;

  const std::array<size_t, 3> sample_indices = {0u, pixels / 2u, pixels - 1u};
  for (size_t index : sample_indices) {
    EXPECT_NEAR(nchw[index], expected_r, 1e-5f);
    EXPECT_NEAR(nchw[pixels + index], expected_g, 1e-5f);
    EXPECT_NEAR(nchw[2u * pixels + index], expected_b, 1e-5f);
  }
}

TEST(DustR3Inference, TC3_PreprocessUsesIntrinsicsAndDistortion) {
  cv::Mat bgr = make_pattern_image(64, 48);

  const auto intrin_identity = make_intrin(500.0f, 500.0f, 31.5f, 23.5f);
  const auto intrin_distorted =
      make_intrin(500.0f, 500.0f, 31.5f, 23.5f, {0.7f, -0.3f, 0.02f, -0.02f, 0.05f});

  const auto base = stargazer::dust3r::preprocess_image(bgr, intrin_identity);
  const auto warped = stargazer::dust3r::preprocess_image(bgr, intrin_distorted);

  const float diff = mean_abs_diff(base, warped);
  std::cout << "[TC3] mean abs diff = " << diff << "\n";
  EXPECT_GT(diff, 0.005f) << "Preprocess output should react to distortion parameters";
}
