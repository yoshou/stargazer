#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "preprocess.hpp"

namespace stargazer {
enum {
  INTER_BITS = 5,
  INTER_TAB_SIZE = (1 << INTER_BITS),
};

#define CUDA_SAFE_CALL(func)                                                                       \
  do {                                                                                             \
    cudaError_t err = (func);                                                                      \
    if (err != cudaSuccess) {                                                                      \
      fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, \
              __FILE__, __LINE__);                                                                 \
      exit(err);                                                                                   \
    }                                                                                              \
  } while (0)

template <typename _Tp>
__host__ __device__ inline _Tp saturate_cast(float v);

template <typename _Tp>
__host__ __device__ inline _Tp saturate_cast(int v);

template <>
inline int saturate_cast<int>(float v) {
  return lrintf(v);
}

template <>
inline short saturate_cast<short>(int v) {
  return (short)((unsigned)(v - SHRT_MIN) <= (unsigned)USHRT_MAX ? v : v > 0 ? SHRT_MAX : SHRT_MIN);
}

template <>
inline uchar saturate_cast<uchar>(int v) {
  return (uchar)((unsigned)v <= UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0);
}
template <>
inline short saturate_cast<short>(float v) {
  int iv = lrintf(v);
  return saturate_cast<short>(iv);
}

__global__ void preprocess_kernel(const uint8_t *src_data, int src_width, int src_height,
                                  int src_step, float *dst_data, int dst_width, int dst_height,
                                  int dst_step, float3 m0, float3 m1, float3 mean, float3 std) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= dst_width || y >= dst_height) {
    return;
  }

  constexpr int INTER_REMAP_COEF_BITS = 15;
  constexpr int INTER_REMAP_COEF_SCALE = 1 << INTER_REMAP_COEF_BITS;

  const int AB_BITS = max(10, (int)INTER_BITS);
  const int AB_SCALE = 1 << AB_BITS;
  const int round_delta = AB_SCALE / INTER_TAB_SIZE / 2;

  const int adelta = saturate_cast<int>(m0.x * x * AB_SCALE);
  const int bdelta = saturate_cast<int>(m1.x * x * AB_SCALE);

  const int temp_u = saturate_cast<int>((m0.y * y + m0.z) * AB_SCALE) + round_delta;
  const int temp_v = saturate_cast<int>((m1.y * y + m1.z) * AB_SCALE) + round_delta;

  const int u = (temp_u + adelta) >> (AB_BITS - INTER_BITS);
  const int v = (temp_v + bdelta) >> (AB_BITS - INTER_BITS);

  const int iu = saturate_cast<int16_t>(u >> INTER_BITS);
  const int iv = saturate_cast<int16_t>(v >> INTER_BITS);

  const int au = (u & (INTER_TAB_SIZE - 1));
  const int av = (v & (INTER_TAB_SIZE - 1));

  short4 sv_r = make_short4(0, 0, 0, 0);
  short4 sv_g = make_short4(0, 0, 0, 0);
  short4 sv_b = make_short4(0, 0, 0, 0);

  if (iu < src_width && iv < src_height && iu >= 0 && iv >= 0) {
    sv_b.x = src_data[iv * src_step + iu * 3 + 0];
    sv_g.x = src_data[iv * src_step + iu * 3 + 1];
    sv_r.x = src_data[iv * src_step + iu * 3 + 2];
  }
  if ((iu + 1) < src_width && iv < src_height && (iu + 1) >= 0 && iv >= 0) {
    sv_b.y = src_data[iv * src_step + (iu + 1) * 3 + 0];
    sv_g.y = src_data[iv * src_step + (iu + 1) * 3 + 1];
    sv_r.y = src_data[iv * src_step + (iu + 1) * 3 + 2];
  }
  if (iu < src_width && (iv + 1) < src_height && iu >= 0 && (iv + 1) >= 0) {
    sv_b.z = src_data[(iv + 1) * src_step + iu * 3 + 0];
    sv_g.z = src_data[(iv + 1) * src_step + iu * 3 + 1];
    sv_r.z = src_data[(iv + 1) * src_step + iu * 3 + 2];
  }
  if ((iu + 1) < src_width && (iv + 1) < src_height && (iu + 1) >= 0 && (iv + 1) >= 0) {
    sv_b.w = src_data[(iv + 1) * src_step + (iu + 1) * 3 + 0];
    sv_g.w = src_data[(iv + 1) * src_step + (iu + 1) * 3 + 1];
    sv_r.w = src_data[(iv + 1) * src_step + (iu + 1) * 3 + 2];
  }

  const float table_scale = 1.f / INTER_TAB_SIZE;

  short4 w;
  w.x = saturate_cast<int16_t>((1.0f - au * table_scale) * (1.0f - av * table_scale) *
                               INTER_REMAP_COEF_SCALE);
  w.y = saturate_cast<int16_t>((au * table_scale) * (1.0f - av * table_scale) *
                               INTER_REMAP_COEF_SCALE);
  w.z = saturate_cast<int16_t>((1.0f - au * table_scale) * (av * table_scale) *
                               INTER_REMAP_COEF_SCALE);
  w.w = saturate_cast<int16_t>((au * table_scale) * (av * table_scale) * INTER_REMAP_COEF_SCALE);

  constexpr auto bits = INTER_REMAP_COEF_BITS;
  constexpr auto shift = bits;
  constexpr auto delta = 1 << (bits - 1);

  const int dst_r = saturate_cast<uint8_t>(
      (sv_r.x * w.x + sv_r.y * w.y + sv_r.z * w.z + sv_r.w * w.w + delta) >> shift);
  const int dst_g = saturate_cast<uint8_t>(
      (sv_g.x * w.x + sv_g.y * w.y + sv_g.z * w.z + sv_g.w * w.w + delta) >> shift);
  const int dst_b = saturate_cast<uint8_t>(
      (sv_b.x * w.x + sv_b.y * w.y + sv_b.z * w.z + sv_b.w * w.w + delta) >> shift);

  float *dst_data_r = dst_data;
  float *dst_data_g = dst_data + dst_step * dst_height;
  float *dst_data_b = dst_data + dst_step * dst_height * 2;

  dst_data_r[y * dst_step + x] = (dst_r / 255.0f - mean.x) / std.x;
  dst_data_g[y * dst_step + x] = (dst_g / 255.0f - mean.y) / std.y;
  dst_data_b[y * dst_step + x] = (dst_b / 255.0f - mean.z) / std.z;
}

void preprocess_cuda(const uint8_t *src_data, int src_width, int src_height, int src_step,
                     float *dst_data, int dst_width, int dst_height, int dst_step, cv::Mat trans,
                     const std::array<float, 3> &mean, const std::array<float, 3> &std) {
  double M[6] = {0};
  cv::Mat matM(2, 3, CV_64F, M);
  trans.convertTo(matM, matM.type());
  {
    double D = M[0] * M[4] - M[1] * M[3];
    D = D != 0 ? 1. / D : 0;
    double A11 = M[4] * D, A22 = M[0] * D;
    M[0] = A11;
    M[1] *= -D;
    M[3] *= -D;
    M[4] = A22;
    double b1 = -M[0] * M[2] - M[1] * M[5];
    double b2 = -M[3] * M[2] - M[4] * M[5];
    M[2] = b1;
    M[5] = b2;
  }

  {
    dim3 block(64, 1024 / 64, 1);
    dim3 grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y, 1);

    preprocess_kernel<<<grid, block, 0, 0>>>(
        src_data, src_width, src_height, src_step, dst_data, dst_width, dst_height, dst_step,
        make_float3(M[0], M[1], M[2]), make_float3(M[3], M[4], M[5]),
        make_float3(mean[0], mean[1], mean[2]), make_float3(std[0], std[1], std[2]));
  }
}

void preprocess_cuda(const uint8_t *src_data, int src_width, int src_height, int src_step,
                     float *dst_data, int dst_width, int dst_height, int dst_step,
                     const std::array<float, 3> &mean, const std::array<float, 3> &std) {
  const auto &&image_size = cv::Size2f(src_width, src_height);
  const auto &&resized_size = cv::Size2f(dst_width, dst_height);

  const auto get_scale = [](const cv::Size2f &image_size, const cv::Size2f &resized_size) {
    float w_pad, h_pad;
    if (image_size.width / resized_size.width < image_size.height / resized_size.height) {
      w_pad = image_size.height / resized_size.height * resized_size.width;
      h_pad = image_size.height;
    } else {
      w_pad = image_size.width;
      h_pad = image_size.width / resized_size.width * resized_size.height;
    }

    return cv::Size2f(w_pad / 200.0, h_pad / 200.0);
  };

  const auto scale = get_scale(image_size, resized_size);
  const auto center = cv::Point2f(image_size.width / 2.0f, image_size.height / 2.0f);
  const auto rotation = 0.0;

  const auto trans = get_transform(center, scale, resized_size);

  preprocess_cuda(src_data, src_width, src_height, src_step, dst_data, dst_width, dst_height,
                  dst_step, trans, mean, std);
}
}  // namespace stargazer
