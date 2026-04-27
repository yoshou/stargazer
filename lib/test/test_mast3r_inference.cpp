#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdlib>

#include "mast3r.hpp"

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

}  // namespace

TEST(MASt3RInference, PreprocessShapeAndRange) {
  const cv::Mat bgr = make_pattern_image(37, 23);
  const auto intrin = make_intrin(420.0f, 415.0f, 18.0f, 11.0f);
  const auto nchw = stargazer::mast3r::preprocess_image(bgr, intrin);

  ASSERT_EQ(nchw.size(), 3u * 288u * 512u);
  const float min_val = *std::min_element(nchw.begin(), nchw.end());
  const float max_val = *std::max_element(nchw.begin(), nchw.end());
  EXPECT_GE(min_val, -1.1f);
  EXPECT_LE(max_val, 1.1f);
}

TEST(MASt3RInference, PreprocessChannelOrderAndNormalization) {
  const cv::Mat bgr(9, 7, CV_8UC3, cv::Scalar(0, 127, 255));
  const auto intrin = make_intrin(300.0f, 300.0f, 3.0f, 4.0f);

  const auto nchw = stargazer::mast3r::preprocess_image(bgr, intrin);
  const size_t pixels = static_cast<size_t>(stargazer::mast3r::ONNX_H * stargazer::mast3r::ONNX_W);

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

TEST(MASt3RInference, OptionalOnnxAdapterShapeCheck) {
  const char* model_path = std::getenv("MAST3R_TEST_ONNX");
  if (!model_path || std::string(model_path).empty()) {
    GTEST_SKIP() << "Set MAST3R_TEST_ONNX to run MASt3R ONNX adapter shape checks";
  }

  stargazer::mast3r::mast3r_inference engine(model_path);
  const cv::Mat img1 = make_pattern_image(64, 48);
  const cv::Mat img2 = make_pattern_image(64, 48);
  const auto res = engine.inference(stargazer::mast3r::preprocess_image(img1),
                                    stargazer::mast3r::preprocess_image(img2));

  EXPECT_EQ(res.pts3d_1.size(), 288u * 512u * 3u);
  EXPECT_EQ(res.pts3d_2.size(), 288u * 512u * 3u);
  EXPECT_EQ(res.conf_1.size(), 288u * 512u);
  EXPECT_EQ(res.conf_2.size(), 288u * 512u);
}
