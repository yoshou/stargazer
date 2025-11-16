#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <algorithm>
#include <iostream>
#include <numeric>

#include "device_launch_parameters.h"
#include "preprocess.hpp"
#include "voxelpose_internal.hpp"

#define CUDA_SAFE_CALL(func)                                                                       \
  do {                                                                                             \
    cudaError_t err = (func);                                                                      \
    if (err != cudaSuccess) {                                                                      \
      fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, \
              __FILE__, __LINE__);                                                                 \
      exit(err);                                                                                   \
    }                                                                                              \
  } while (0)

#define CUDNN_SAFE_CALL(func)                                                                  \
  do {                                                                                         \
    cudnnStatus_t err = (func);                                                                \
    if (err != CUDNN_STATUS_SUCCESS) {                                                         \
      fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudnnGetErrorString(err), \
              err, __FILE__, __LINE__);                                                        \
      exit(err);                                                                               \
    }                                                                                          \
  } while (0)

#define CUBLAS_SAFE_CALL(func)                                                                   \
  do {                                                                                           \
    cublasStatus_t err = (func);                                                                 \
    if (err != CUBLAS_STATUS_SUCCESS) {                                                          \
      fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cublasGetStatusString(err), \
              err, __FILE__, __LINE__);                                                          \
      exit(err);                                                                                 \
    }                                                                                            \
  } while (0)

namespace stargazer::voxelpose {
enum {
  INTER_BITS = 5,
  INTER_TAB_SIZE = (1 << INTER_BITS),
};

static inline void interpolate_linear(float x, float* coeffs) {
  coeffs[0] = 1.f - x;
  coeffs[1] = x;
}

static void init_interpolation_table_line_bilinear(float* table, int table_size) {
  float scale = 1.f / table_size;
  for (int i = 0; i < table_size; i++, table += 2) interpolate_linear(i * scale, table);
}

static const float* init_interpolation_table_bilinear() {
  alignas(32) static float BilinearTab_f[INTER_TAB_SIZE * INTER_TAB_SIZE][2][2];

  float* tab = BilinearTab_f[0][0];

  static bool created = false;
  if (created) {
    return tab;
  }
  created = true;

  std::vector<float> _tab(8 * INTER_TAB_SIZE);
  init_interpolation_table_line_bilinear(_tab.data(), INTER_TAB_SIZE);
  for (int i = 0; i < INTER_TAB_SIZE; i++) {
    for (int j = 0; j < INTER_TAB_SIZE; j++) {
      for (int k1 = 0; k1 < 2; k1++) {
        const float vy = _tab[i * 2 + k1];
        for (int k2 = 0; k2 < 2; k2++) {
          const float v = vy * _tab[j * 2 + k2];
          tab[(i * INTER_TAB_SIZE + j) * 4 + k1 * 2 + k2] = v;
        }
      }
    }
  }

  return tab;
}

struct proj_camera {
  float fx;
  float fy;
  float cx;
  float cy;
  float k[3];
  float p[2];
  float rotation[3][3];
  float translation[3];
  float trans[2][3];
  float camera_width;
  float camera_height;
  float image_width;
  float image_height;
  float heatmap_width;
  float heatmap_height;
};

struct voxel_projector::cuda_data {
  cudaStream_t stream;
  float* bilinear_wtab;
  float* cubes_g;
  proj_camera* cameras_g;

  cuda_data() : bilinear_wtab(nullptr), cubes_g(nullptr), cameras_g(nullptr) {
    cudaStreamCreate(&stream);

    const float* wtab = init_interpolation_table_bilinear();

    cudaMalloc(&bilinear_wtab, INTER_TAB_SIZE * INTER_TAB_SIZE * 2 * 2 * sizeof(float));
    cudaMemcpy(bilinear_wtab, wtab, INTER_TAB_SIZE * INTER_TAB_SIZE * 2 * 2 * sizeof(float),
               cudaMemcpyHostToDevice);
  }

  ~cuda_data() {
    cudaStreamDestroy(stream);
    stream = nullptr;
  }
};

voxel_projector::voxel_projector() : cuda_data_(std::make_unique<cuda_data>()) {}

voxel_projector::~voxel_projector() {}

__device__ void project_point(const float p_x, const float p_y, const float p_z,
                              const proj_camera& camera, float& u, float& v) {
  using acc_t = float;

  const auto fx = static_cast<acc_t>(camera.fx);
  const auto fy = static_cast<acc_t>(camera.fy);
  const auto cx = static_cast<acc_t>(camera.cx);
  const auto cy = static_cast<acc_t>(camera.cy);
  const auto k1 = static_cast<acc_t>(camera.k[0]);
  const auto k2 = static_cast<acc_t>(camera.k[1]);
  const auto k3 = static_cast<acc_t>(camera.k[2]);
  const auto p1 = static_cast<acc_t>(camera.p[0]);
  const auto p2 = static_cast<acc_t>(camera.p[1]);

  const auto pt_x = static_cast<acc_t>(p_x) - static_cast<acc_t>(camera.translation[0]);
  const auto pt_y = static_cast<acc_t>(p_y) - static_cast<acc_t>(camera.translation[1]);
  const auto pt_z = static_cast<acc_t>(p_z) - static_cast<acc_t>(camera.translation[2]);
  const auto x = pt_x * static_cast<acc_t>(camera.rotation[0][0]) +
                 pt_y * static_cast<acc_t>(camera.rotation[0][1]) +
                 pt_z * static_cast<acc_t>(camera.rotation[0][2]);
  const auto y = pt_x * static_cast<acc_t>(camera.rotation[1][0]) +
                 pt_y * static_cast<acc_t>(camera.rotation[1][1]) +
                 pt_z * static_cast<acc_t>(camera.rotation[1][2]);
  const auto z = pt_x * static_cast<acc_t>(camera.rotation[2][0]) +
                 pt_y * static_cast<acc_t>(camera.rotation[2][1]) +
                 pt_z * static_cast<acc_t>(camera.rotation[2][2]);
  const auto x1 = x / (z + acc_t(1e-5));
  const auto y1 = y / (z + acc_t(1e-5));
  const auto r2 = x1 * x1 + y1 * y1;
  const auto x2 = x1 * (acc_t(1.0) + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) +
                  acc_t(2.0) * p1 * x1 * y1 + p2 * (r2 + acc_t(2.0) * x1 * x1);
  const auto y2 = y1 * (acc_t(1.0) + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) +
                  p1 * (r2 + acc_t(2.0) * y1 * y1) + acc_t(2.0) * p2 * x1 * y1;

  u = static_cast<float>(fx * x2 + cx);
  v = static_cast<float>(fy * y2 + cy);
}

__global__ void proj_kernel(const float* heatmaps, int heatmap_width, int heatmap_height,
                            int heatmap_channel, int grid_size_x, int grid_size_y, int grid_size_z,
                            float area_size_x, float area_size_y, float area_size_z,
                            float grid_center_x, float grid_center_y, float grid_center_z,
                            const proj_camera* cameras, int num_views, float* grid,
                            const float* wtab) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  const int grid_x = x % grid_size_x;
  const int grid_y = x / grid_size_y;
  const int grid_z = y % grid_size_z;
  const int channel = y / grid_size_z;

  if (grid_x < grid_size_x && grid_y < grid_size_y && grid_z < grid_size_z &&
      channel < heatmap_channel) {
    const int sample_idx = grid_z + grid_y * grid_size_z + grid_x * grid_size_z * grid_size_y;

    const auto gridx = -area_size_x / 2 + area_size_x * grid_x / (grid_size_x - 1) + grid_center_x;
    const auto gridy = -area_size_y / 2 + area_size_y * grid_y / (grid_size_y - 1) + grid_center_y;
    const auto gridz = -area_size_z / 2 + area_size_z * grid_z / (grid_size_z - 1) + grid_center_z;

    float tmp = 0.0f;
    float bounding_count = 0.0f;
    for (int i = 0; i < num_views; i++) {
      const float* heatmap = heatmaps + heatmap_width * heatmap_height * heatmap_channel * i +
                             heatmap_width * heatmap_height * channel;

      float u, v;
      project_point(gridx, gridy, gridz, cameras[i], u, v);

      float grid_sample_x, grid_sample_y;
      {
        const auto x0 = u;
        const auto y0 = v;

        const auto x1 = min(max(x0, -1.0f), max(cameras[i].camera_width, cameras[i].camera_height));
        const auto y1 = min(max(y0, -1.0f), max(cameras[i].camera_width, cameras[i].camera_height));

        const auto x2 =
            x1 * cameras[i].trans[0][0] + y1 * cameras[i].trans[0][1] + cameras[i].trans[0][2];
        const auto y2 =
            x1 * cameras[i].trans[1][0] + y1 * cameras[i].trans[1][1] + cameras[i].trans[1][2];

        const auto x3 = x2 * cameras[i].heatmap_width / cameras[i].image_width;
        const auto y3 = y2 * cameras[i].heatmap_height / cameras[i].image_height;

        const auto x4 = x3 / (cameras[i].heatmap_width - 1) * 2.0f - 1.0f;
        const auto y4 = y3 / (cameras[i].heatmap_height - 1) * 2.0f - 1.0f;

        const auto x5 = min(max(x4, -1.1f), 1.1f);
        const auto y5 = min(max(y4, -1.1f), 1.1f);

        grid_sample_x = x5;
        grid_sample_y = y5;
      }

      const bool bound =
          (u >= 0 && u < cameras[i].camera_width && v >= 0 && v < cameras[i].camera_height);

      const float x = ((grid_sample_x + 1) / 2) * (heatmap_width - 1);
      const float y = ((grid_sample_y + 1) / 2) * (heatmap_height - 1);

      const auto sx = lrint(x * INTER_TAB_SIZE);
      const auto sy = lrint(y * INTER_TAB_SIZE);
      const auto mx = static_cast<int16_t>(sx >> INTER_BITS);
      const auto my = static_cast<int16_t>(sy >> INTER_BITS);
      const auto ma = static_cast<uint16_t>((sy & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE +
                                            (sx & (INTER_TAB_SIZE - 1)));

      const auto get_value = [src_ptr = heatmap, sx = mx, sy = my, src_step = heatmap_width,
                              src_width = heatmap_width,
                              src_height = heatmap_height](int x, int y) {
        if ((sx + x) >= 0 && (sy + y) >= 0 && (sx + x) < src_width && (sy + y) < src_height) {
          return src_ptr[(sy + y) * src_step + (sx + x)];
        } else {
          return static_cast<float>(0);
        }
      };
      const float* w = wtab + ma * 4;
      const auto sample_value = get_value(0, 0) * w[0] + get_value(1, 0) * w[1] +
                                get_value(0, 1) * w[2] + get_value(1, 1) * w[3];

      tmp += sample_value * bound;
      bounding_count += bound;
    }
    tmp /= (bounding_count + 1e-6f);
    tmp = max(tmp, 0.0f);
    tmp = min(tmp, 1.0f);
    grid[sample_idx + grid_size_z * grid_size_y * grid_size_x * channel] = tmp;
  }
}

void voxel_projector::get_voxel(const float* heatmaps, int num_cameras, int heatmap_width,
                                int heatmap_height, const std::vector<camera_data>& cameras,
                                const std::vector<roi_data>& rois,
                                const std::array<float, 3>& grid_center) {
  const auto num_bins = static_cast<uint32_t>(
      std::accumulate(cube_size.begin(), cube_size.end(), 1, std::multiplies<int32_t>()));
  const auto num_joints = 15;
  const auto w = heatmap_width;
  const auto h = heatmap_height;

  std::vector<proj_camera> proj_cameras(num_cameras);

  for (uint32_t c = 0; c < num_cameras; c++) {
    const auto& roi = rois.at(c);
    const auto&& image_size = cv::Size2f(960, 512);
    const auto center = cv::Point2f(roi.center[0], roi.center[1]);
    const auto scale = cv::Size2f(roi.scale[0], roi.scale[1]);
    const auto width = center.x * 2;
    const auto height = center.y * 2;

    const auto trans = stargazer::get_transform(center, scale, image_size);
    cv::Mat transf;
    trans.convertTo(transf, cv::DataType<float>::type);

    proj_cameras[c].fx = cameras[c].fx;
    proj_cameras[c].fy = cameras[c].fy;
    proj_cameras[c].cx = cameras[c].cx;
    proj_cameras[c].cy = cameras[c].cy;
    proj_cameras[c].k[0] = cameras[c].k[0];
    proj_cameras[c].k[1] = cameras[c].k[1];
    proj_cameras[c].k[2] = cameras[c].k[2];
    proj_cameras[c].p[0] = cameras[c].p[0];
    proj_cameras[c].p[1] = cameras[c].p[1];
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        proj_cameras[c].rotation[i][j] = cameras[c].rotation[i][j];
      }
      proj_cameras[c].translation[i] = cameras[c].translation[i];
    }
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        proj_cameras[c].trans[i][j] = transf.at<float>(i, j);
      }
    }
    proj_cameras[c].camera_width = width;
    proj_cameras[c].camera_height = height;
    proj_cameras[c].image_width = image_size.width;
    proj_cameras[c].image_height = image_size.height;
    proj_cameras[c].heatmap_width = w;
    proj_cameras[c].heatmap_height = h;
  }

  const auto max_num_cameras = 5;

  if (!cuda_data_->cameras_g) {
    CUDA_SAFE_CALL(cudaMalloc(&cuda_data_->cameras_g, max_num_cameras * sizeof(proj_camera)));
  }
  if (!cuda_data_->cubes_g) {
    CUDA_SAFE_CALL(
        cudaMalloc(&cuda_data_->cubes_g, num_bins * num_joints * max_num_cameras * sizeof(float)));
  }

  CUDA_SAFE_CALL(cudaMemcpyAsync(cuda_data_->cameras_g, proj_cameras.data(),
                                 num_cameras * sizeof(proj_camera), cudaMemcpyHostToDevice,
                                 cuda_data_->stream));

  {
    dim3 block(256, 1, 1);
    dim3 grid(cube_size[0] * cube_size[1] / block.x, cube_size[2] * num_joints / block.y, 1);

    proj_kernel<<<grid, block, 0, cuda_data_->stream>>>(
        heatmaps, heatmap_width, heatmap_height, num_joints, cube_size[0], cube_size[1],
        cube_size[2], grid_size[0], grid_size[1], grid_size[2], grid_center[0], grid_center[1],
        grid_center[2], cuda_data_->cameras_g, num_cameras, cuda_data_->cubes_g,
        cuda_data_->bilinear_wtab);
  }

  CUDA_SAFE_CALL(cudaStreamSynchronize(cuda_data_->stream));
}

const float* voxel_projector::get_cubes() const { return cuda_data_->cubes_g; }

__global__ void soft_argmax_pre_kernel(const float* src, float* dst, float* grid, int grid_size_x,
                                       int grid_size_y, int grid_size_z, int num_channel,
                                       float area_size_x, float area_size_y, float area_size_z,
                                       float grid_center_x, float grid_center_y,
                                       float grid_center_z, float beta) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;

  const auto grid_x = x % grid_size_x;
  const auto grid_y = (x / grid_size_x) % grid_size_y;
  const auto grid_z = (x / (grid_size_x * grid_size_y)) % grid_size_z;

  const auto grid_size = grid_size_x * grid_size_y * grid_size_z;
  const auto grid_idx = grid_z + grid_y * grid_size_z + grid_x * grid_size_z * grid_size_y;

  if (grid_x < grid_size_x && grid_y < grid_size_y && grid_z < grid_size_z) {
    const auto gridx = -area_size_x / 2 + area_size_x * grid_x / (grid_size_x - 1) + grid_center_x;
    const auto gridy = -area_size_y / 2 + area_size_y * grid_y / (grid_size_y - 1) + grid_center_y;
    const auto gridz = -area_size_z / 2 + area_size_z * grid_z / (grid_size_z - 1) + grid_center_z;

    grid[grid_idx + grid_size * 0] = gridx;
    grid[grid_idx + grid_size * 1] = gridy;
    grid[grid_idx + grid_size * 2] = gridz;

    for (int channel = 0; channel < num_channel; channel++) {
      const auto value = src[grid_size * channel + grid_idx];
      dst[grid_idx + grid_size * channel] = value * beta;
    }
  }
}

struct joint_extractor::cuda_data {
  cudaStream_t stream;
  cudnnHandle_t cudnn_handle;
  cublasHandle_t cublas_handle;
  cudnnTensorDescriptor_t scaled_value_tensor, softmax_value_tensor;
  float* scaled_value;
  float* grid;
  float* softmax_value;
  float* joints;

  cuda_data() : scaled_value(nullptr), grid(nullptr), softmax_value(nullptr), joints(nullptr) {
    cudaStreamCreate(&stream);

    cudnnCreate(&cudnn_handle);
    cudnnSetStream(cudnn_handle, stream);

    cudnnCreateTensorDescriptor(&scaled_value_tensor);
    cudnnSetTensor4dDescriptor(scaled_value_tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 15, 1, 1,
                               64 * 64 * 64);

    cudnnCreateTensorDescriptor(&softmax_value_tensor);
    cudnnSetTensor4dDescriptor(softmax_value_tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 15, 1, 1,
                               64 * 64 * 64);

    cublasCreate(&cublas_handle);
    cublasSetStream(cublas_handle, stream);
  }

  ~cuda_data() {
    cudaStreamDestroy(stream);
    stream = nullptr;

    cudnnDestroyTensorDescriptor(scaled_value_tensor);
    scaled_value_tensor = nullptr;

    cudnnDestroyTensorDescriptor(softmax_value_tensor);
    softmax_value_tensor = nullptr;

    cudnnDestroy(cudnn_handle);
    cudnn_handle = nullptr;

    cublasDestroy(cublas_handle);
    cublas_handle = nullptr;

    if (scaled_value) {
      cudaFree(scaled_value);
      scaled_value = nullptr;
    }
    if (grid) {
      cudaFree(grid);
      grid = nullptr;
    }
    if (softmax_value) {
      cudaFree(softmax_value);
      softmax_value = nullptr;
    }
    if (joints) {
      cudaFree(joints);
      joints = nullptr;
    }
  }
};

joint_extractor::joint_extractor(int num_joints)
    : cuda_data_(std::make_unique<cuda_data>()), num_joints(num_joints) {}

joint_extractor::~joint_extractor() {}

void joint_extractor::soft_argmax(const float* src_data, float beta,
                                  const std::array<float, 3>& grid_size,
                                  const std::array<int32_t, 3>& cube_size,
                                  const std::array<float, 3>& grid_center) {
  const auto num_bins = static_cast<uint32_t>(
      std::accumulate(cube_size.begin(), cube_size.end(), 1, std::multiplies<int32_t>()));

  if (!cuda_data_->scaled_value) {
    CUDA_SAFE_CALL(cudaMalloc(&cuda_data_->scaled_value, num_bins * num_joints * sizeof(float)));
  }

  if (!cuda_data_->grid) {
    CUDA_SAFE_CALL(cudaMalloc(&cuda_data_->grid, num_bins * 3 * sizeof(float)));
  }

  if (!cuda_data_->softmax_value) {
    CUDA_SAFE_CALL(cudaMalloc(&cuda_data_->softmax_value, num_bins * num_joints * sizeof(float)));
  }

  if (!cuda_data_->joints) {
    CUDA_SAFE_CALL(cudaMalloc(&cuda_data_->joints, num_joints * 3 * sizeof(float)));
  }

  {
    dim3 block(256, 1, 1);
    dim3 grid((num_bins + block.x - 1) / block.x, 1, 1);

    soft_argmax_pre_kernel<<<grid, block, 0, cuda_data_->stream>>>(
        src_data, cuda_data_->scaled_value, cuda_data_->grid, cube_size[0], cube_size[1],
        cube_size[2], num_joints, grid_size[0], grid_size[1], grid_size[2], grid_center[0],
        grid_center[1], grid_center[2], beta);
  }

  {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUDNN_SAFE_CALL(cudnnSoftmaxForward(
        cuda_data_->cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha,
        cuda_data_->scaled_value_tensor, cuda_data_->scaled_value, &beta,
        cuda_data_->softmax_value_tensor, cuda_data_->softmax_value));
  }

  {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_SAFE_CALL(cublasSgemm(cuda_data_->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, 3, num_joints,
                                 num_bins, &alpha, cuda_data_->grid, num_bins,
                                 cuda_data_->softmax_value, num_bins, &beta, cuda_data_->joints,
                                 3));
  }

  CUDA_SAFE_CALL(cudaStreamSynchronize(cuda_data_->stream));
}

const float* joint_extractor::get_joints() const { return cuda_data_->joints; }
}  // namespace stargazer::voxelpose
