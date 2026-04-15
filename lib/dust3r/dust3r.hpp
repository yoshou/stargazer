#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "parameters.hpp"

namespace stargazer::dust3r {

static constexpr int ONNX_H = 288;
static constexpr int ONNX_W = 512;

struct inference_result {
  std::vector<float> pts3d_1;  // [H*W*3]
  std::vector<float> conf_1;   // [H*W]
  std::vector<float> pts3d_2;  // [H*W*3]
  std::vector<float> conf_2;   // [H*W]
};

std::vector<float> preprocess_image(const cv::Mat& bgr, const camera_intrin_t& intrin);

// Preprocess without undistortion (for use when intrinsics are unknown)
std::vector<float> preprocess_image(const cv::Mat& bgr);

class dust3r_inference {
  struct impl;
  std::unique_ptr<impl> pimpl_;

 public:
  explicit dust3r_inference(const std::string& model_path);
  ~dust3r_inference();

  inference_result inference(const std::vector<float>& img1_nchw,
                             const std::vector<float>& img2_nchw);
};

}  // namespace stargazer::dust3r
