#pragma once

#include <array>
#include <memory>

#include "mvpose.hpp"

namespace stargazer_mvpose
{
    void preprocess_cuda(const uint8_t *src_data, int src_width, int src_height, int src_step, float *dst_data, int dst_width, int dst_height, int dst_step, cv::Mat trans, const std::array<float, 3> &mean, const std::array<float, 3> &std);
    void preprocess_cuda(const uint8_t *src_data, int src_width, int src_height, int src_step, float *dst_data, int dst_width, int dst_height, int dst_step, const std::array<float, 3> &mean, const std::array<float, 3> &std);
}
