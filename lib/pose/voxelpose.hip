#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "preprocess.hpp"
#include "voxelpose_internal.hpp"

#define CUDA_SAFE_CALL(func)                                                                      \
  do {                                                                                            \
    hipError_t err = (func);                                                                      \
    if (err != hipSuccess) {                                                                      \
      fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", hipGetErrorString(err), err, \
              __FILE__, __LINE__);                                                                \
      exit(err);                                                                                  \
    }                                                                                             \
  } while (0)

#define HIPBLAS_SAFE_CALL(error)                                                                  \
  do {                                                                                            \
    if (error != HIPBLAS_STATUS_SUCCESS) {                                                        \
      fprintf(stderr, "hipBLAS error: ");                                                         \
      if (error == HIPBLAS_STATUS_NOT_INITIALIZED)                                                \
        fprintf(stderr, "HIPBLAS_STATUS_NOT_INITIALIZED");                                        \
      if (error == HIPBLAS_STATUS_ALLOC_FAILED) fprintf(stderr, "HIPBLAS_STATUS_ALLOC_FAILED");   \
      if (error == HIPBLAS_STATUS_INVALID_VALUE) fprintf(stderr, "HIPBLAS_STATUS_INVALID_VALUE"); \
      if (error == HIPBLAS_STATUS_MAPPING_ERROR) fprintf(stderr, "HIPBLAS_STATUS_MAPPING_ERROR"); \
      if (error == HIPBLAS_STATUS_EXECUTION_FAILED)                                               \
        fprintf(stderr, "HIPBLAS_STATUS_EXECUTION_FAILED");                                       \
      if (error == HIPBLAS_STATUS_INTERNAL_ERROR)                                                 \
        fprintf(stderr, "HIPBLAS_STATUS_INTERNAL_ERROR");                                         \
      if (error == HIPBLAS_STATUS_NOT_SUPPORTED) fprintf(stderr, "HIPBLAS_STATUS_NOT_SUPPORTED"); \
      if (error == HIPBLAS_STATUS_INVALID_ENUM) fprintf(stderr, "HIPBLAS_STATUS_INVALID_ENUM");   \
      if (error == HIPBLAS_STATUS_UNKNOWN) fprintf(stderr, "HIPBLAS_STATUS_UNKNOWN");             \
      fprintf(stderr, "\n");                                                                      \
      exit(EXIT_FAILURE);                                                                         \
    }                                                                                             \
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
  hipStream_t stream;
  float* bilinear_wtab;
  float* cubes_g;
  proj_camera* cameras_g;

  cuda_data() : bilinear_wtab(nullptr), cubes_g(nullptr), cameras_g(nullptr) {
    hipStreamCreate(&stream);

    const float* wtab = init_interpolation_table_bilinear();

    hipMalloc(&bilinear_wtab, INTER_TAB_SIZE * INTER_TAB_SIZE * 2 * 2 * sizeof(float));
    hipMemcpy(bilinear_wtab, wtab, INTER_TAB_SIZE * INTER_TAB_SIZE * 2 * 2 * sizeof(float),
              hipMemcpyHostToDevice);
  }

  ~cuda_data() {
    hipStreamDestroy(stream);
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
    CUDA_SAFE_CALL(hipMalloc(&cuda_data_->cameras_g, max_num_cameras * sizeof(proj_camera)));
  }
  if (!cuda_data_->cubes_g) {
    CUDA_SAFE_CALL(
        hipMalloc(&cuda_data_->cubes_g, num_bins * num_joints * max_num_cameras * sizeof(float)));
  }

  CUDA_SAFE_CALL(hipMemcpyAsync(cuda_data_->cameras_g, proj_cameras.data(),
                                num_cameras * sizeof(proj_camera), hipMemcpyHostToDevice,
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

  CUDA_SAFE_CALL(hipStreamSynchronize(cuda_data_->stream));
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

// Custom softmax kernel to replace MIOpen
__global__ void softmax_kernel(const float* input, float* output, int num_bins, int num_joints) {
  const int joint_idx = blockIdx.x;
  if (joint_idx >= num_joints) return;

  const float* input_joint = input + joint_idx * num_bins;
  float* output_joint = output + joint_idx * num_bins;

  // Find max value for numerical stability
  float max_val = -1e30f;
  for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
    max_val = fmaxf(max_val, input_joint[i]);
  }

  // Warp-level reduction for max
  for (int offset = 16; offset > 0; offset /= 2) {
    max_val = fmaxf(max_val, __shfl_down(max_val, offset));
  }

  __shared__ float block_max;
  if (threadIdx.x == 0) {
    block_max = -1e30f;
  }
  __syncthreads();

  if (threadIdx.x % 32 == 0) {
    atomicMax(reinterpret_cast<int*>(&block_max), __float_as_int(max_val));
  }
  __syncthreads();

  // Compute exp and sum
  float sum_exp = 0.0f;
  for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
    const float exp_val = expf(input_joint[i] - block_max);
    output_joint[i] = exp_val;
    sum_exp += exp_val;
  }

  // Warp-level reduction for sum
  for (int offset = 16; offset > 0; offset /= 2) {
    sum_exp += __shfl_down(sum_exp, offset);
  }

  __shared__ float block_sum;
  if (threadIdx.x == 0) {
    block_sum = 0.0f;
  }
  __syncthreads();

  if (threadIdx.x % 32 == 0) {
    atomicAdd(&block_sum, sum_exp);
  }
  __syncthreads();

  // Normalize
  const float inv_sum = 1.0f / (block_sum + 1e-8f);
  for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
    output_joint[i] *= inv_sum;
  }
}

__global__ void soft_argmax_gemm_kernel(const float* grid, const float* softmax_value,
                                        float* joints, int num_bins, int num_joints) {
  const int joint_idx = blockIdx.x;
  const int xyz_idx = blockIdx.y;

  if (joint_idx >= num_joints || xyz_idx >= 3) return;

  const float* grid_ptr = grid + xyz_idx * num_bins;
  const float* softmax_ptr = softmax_value + joint_idx * num_bins;

  float sum = 0.0f;
  for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
    sum += grid_ptr[i] * softmax_ptr[i];
  }

  // Warp-level reduction
  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down(sum, offset);
  }

  __shared__ float block_sum[32];  // One per warp
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;

  if (lane_id == 0) {
    block_sum[warp_id] = sum;
  }
  __syncthreads();

  // Final reduction across warps
  if (threadIdx.x == 0) {
    float total = 0.0f;
    const int num_warps = (blockDim.x + 31) / 32;
    for (int i = 0; i < num_warps; i++) {
      total += block_sum[i];
    }
    joints[joint_idx * 3 + xyz_idx] = total;
  }
}

struct joint_extractor::cuda_data {
  hipStream_t stream;
  float* scaled_value;
  float* grid;
  float* softmax_value;
  float* joints;

  cuda_data() : scaled_value(nullptr), grid(nullptr), softmax_value(nullptr), joints(nullptr) {
    hipStreamCreate(&stream);
  }

  ~cuda_data() {
    hipStreamDestroy(stream);
    stream = nullptr;

    if (scaled_value) {
      hipFree(scaled_value);
      scaled_value = nullptr;
    }
    if (grid) {
      hipFree(grid);
      grid = nullptr;
    }
    if (softmax_value) {
      hipFree(softmax_value);
      softmax_value = nullptr;
    }
    if (joints) {
      hipFree(joints);
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
    CUDA_SAFE_CALL(hipMalloc(&cuda_data_->scaled_value, num_bins * num_joints * sizeof(float)));
  }
  if (!cuda_data_->grid) {
    CUDA_SAFE_CALL(hipMalloc(&cuda_data_->grid, num_bins * 3 * sizeof(float)));
  }
  if (!cuda_data_->softmax_value) {
    CUDA_SAFE_CALL(hipMalloc(&cuda_data_->softmax_value, num_bins * num_joints * sizeof(float)));
  }
  if (!cuda_data_->joints) {
    CUDA_SAFE_CALL(hipMalloc(&cuda_data_->joints, num_joints * 3 * sizeof(float)));
  }

  // Step 1: Generate grid coordinates and scale input
  {
    dim3 block(256, 1, 1);
    dim3 grid((num_bins + block.x - 1) / block.x, 1, 1);
    soft_argmax_pre_kernel<<<grid, block, 0, cuda_data_->stream>>>(
        src_data, cuda_data_->scaled_value, cuda_data_->grid, cube_size[0], cube_size[1],
        cube_size[2], num_joints, grid_size[0], grid_size[1], grid_size[2], grid_center[0],
        grid_center[1], grid_center[2], beta);
  }

  // Step 2: Softmax
  {
    dim3 block(256, 1, 1);
    dim3 grid(num_joints, 1, 1);
    softmax_kernel<<<grid, block, 0, cuda_data_->stream>>>(
        cuda_data_->scaled_value, cuda_data_->softmax_value, num_bins, num_joints);
  }

  // Step 3: Weighted sum (GEMM)
  {
    dim3 block(256, 1, 1);
    dim3 grid(num_joints, 3, 1);
    soft_argmax_gemm_kernel<<<grid, block, 0, cuda_data_->stream>>>(
        cuda_data_->grid, cuda_data_->softmax_value, cuda_data_->joints, num_bins, num_joints);
  }

  CUDA_SAFE_CALL(hipStreamSynchronize(cuda_data_->stream));
}

const float* joint_extractor::get_joints() const { return cuda_data_->joints; }

namespace {
constexpr int kMaxTopK = 128;
}

__global__ void max_pool3d_kernel(const float* input, float* output, int size_x, int size_y,
                                  int size_z, int kernel_size, int padding) {
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int iy = blockIdx.y * blockDim.y + threadIdx.y;
  const int iz = blockIdx.z * blockDim.z + threadIdx.z;

  if (ix >= size_x || iy >= size_y || iz >= size_z) {
    return;
  }

  const int stride_yz = size_y * size_z;
  const int idx = (ix * stride_yz) + (iy * size_z) + iz;
  const float center_val = input[idx];

  float max_val = center_val;
  for (int kx = -padding; kx <= padding; ++kx) {
    for (int ky = -padding; ky <= padding; ++ky) {
      for (int kz = -padding; kz <= padding; ++kz) {
        const int nx = ix + kx;
        const int ny = iy + ky;
        const int nz = iz + kz;
        if (nx >= 0 && nx < size_x && ny >= 0 && ny < size_y && nz >= 0 && nz < size_z) {
          const int nidx = (nx * stride_yz) + (ny * size_z) + nz;
          max_val = fmaxf(max_val, input[nidx]);
        }
      }
    }
  }

  output[idx] = (center_val == max_val) ? center_val : 0.0f;
}

__device__ inline void bitonic_sort_descending(float* vals, int* idxs, int n) {
  for (int size = 2; size <= n; size *= 2) {
    for (int stride = size / 2; stride > 0; stride /= 2) {
      for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int ixj = i ^ stride;
        if (ixj > i) {
          if ((i & size) == 0) {
            if (vals[i] < vals[ixj]) {
              const float tmp_val = vals[i];
              const int tmp_idx = idxs[i];
              vals[i] = vals[ixj];
              idxs[i] = idxs[ixj];
              vals[ixj] = tmp_val;
              idxs[ixj] = tmp_idx;
            }
          } else {
            if (vals[i] > vals[ixj]) {
              const float tmp_val = vals[i];
              const int tmp_idx = idxs[i];
              vals[i] = vals[ixj];
              idxs[i] = idxs[ixj];
              vals[ixj] = tmp_val;
              idxs[ixj] = tmp_idx;
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

__global__ void topk_select_kernel(const float* input, float* topk_values, int* topk_indices,
                                   int input_size, int k) {
  const int valid_k = min(k, kMaxTopK);

  extern __shared__ unsigned char shared_buffer[];
  float* shared_vals = reinterpret_cast<float*>(shared_buffer);
  int* shared_idxs = reinterpret_cast<int*>(shared_buffer + blockDim.x * valid_k * sizeof(float));

  float local_vals[kMaxTopK];
  int local_idxs[kMaxTopK];
  for (int i = 0; i < valid_k; ++i) {
    local_vals[i] = -FLT_MAX;
    local_idxs[i] = -1;
  }

  for (int idx = threadIdx.x; idx < input_size; idx += blockDim.x) {
    const float val = input[idx];
    int min_pos = 0;
    float min_val = local_vals[0];
    for (int i = 1; i < valid_k; ++i) {
      if (local_vals[i] < min_val) {
        min_val = local_vals[i];
        min_pos = i;
      }
    }
    if (val > min_val) {
      local_vals[min_pos] = val;
      local_idxs[min_pos] = idx;
    }
  }

  for (int i = 0; i < valid_k; ++i) {
    shared_vals[threadIdx.x * valid_k + i] = local_vals[i];
    shared_idxs[threadIdx.x * valid_k + i] = local_idxs[i];
  }
  __syncthreads();

  // Parallel reduction across threads
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      const int base1 = threadIdx.x * valid_k;
      const int base2 = (threadIdx.x + stride) * valid_k;
      
      for (int i = 0; i < valid_k; ++i) {
        float local_min_val = shared_vals[base1 + 0];
        int local_min_pos = 0;
        for (int j = 1; j < valid_k; ++j) {
          if (shared_vals[base1 + j] < local_min_val) {
            local_min_val = shared_vals[base1 + j];
            local_min_pos = j;
          }
        }
        
        const float val2 = shared_vals[base2 + i];
        const int idx2 = shared_idxs[base2 + i];
        if (val2 > local_min_val && idx2 >= 0) {
          shared_vals[base1 + local_min_pos] = val2;
          shared_idxs[base1 + local_min_pos] = idx2;
        }
      }
    }
    __syncthreads();
  }

  // Bitonic sort in parallel for final k elements
  int padded_k = 1;
  while (padded_k < valid_k) padded_k *= 2;
  
  if (threadIdx.x < padded_k) {
    const int idx = threadIdx.x;
    if (idx < valid_k) {
      shared_vals[idx] = shared_vals[idx];
      shared_idxs[idx] = shared_idxs[idx];
    } else {
      shared_vals[idx] = -FLT_MAX;
      shared_idxs[idx] = -1;
    }
  }
  __syncthreads();

  bitonic_sort_descending(shared_vals, shared_idxs, padded_k);

  if (threadIdx.x < valid_k) {
    topk_values[threadIdx.x] = shared_vals[threadIdx.x];
    topk_indices[threadIdx.x] = shared_idxs[threadIdx.x];
  } else if (threadIdx.x < k) {
    topk_values[threadIdx.x] = -FLT_MAX;
    topk_indices[threadIdx.x] = -1;
  }
}

__global__ void unravel_and_convert_kernel(const int* indices, const float* values, float* centers,
                                           int max_num, int size_x, int size_y, int size_z,
                                           float grid_size_x, float grid_size_y, float grid_size_z,
                                           float grid_center_x, float grid_center_y,
                                           float grid_center_z, float threshold) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= max_num) {
    return;
  }

  const int linear_idx = indices[idx];
  if (linear_idx < 0) {
    centers[0 * max_num + idx] = 0.0f;
    centers[1 * max_num + idx] = 0.0f;
    centers[2 * max_num + idx] = 0.0f;
    centers[3 * max_num + idx] = 0.0f;
    centers[4 * max_num + idx] = 0.0f;
    return;
  }

  const int stride_yz = size_y * size_z;
  const int ix = linear_idx / stride_yz;
  const int iy = (linear_idx % stride_yz) / size_z;
  const int iz = linear_idx % size_z;

  const float x = -grid_size_x / 2.0f +
                  grid_size_x * static_cast<float>(ix) / static_cast<float>(size_x - 1) +
                  grid_center_x;
  const float y = -grid_size_y / 2.0f +
                  grid_size_y * static_cast<float>(iy) / static_cast<float>(size_y - 1) +
                  grid_center_y;
  const float z = -grid_size_z / 2.0f +
                  grid_size_z * static_cast<float>(iz) / static_cast<float>(size_z - 1) +
                  grid_center_z;

  centers[0 * max_num + idx] = x;
  centers[1 * max_num + idx] = y;
  centers[2 * max_num + idx] = z;
  centers[3 * max_num + idx] = (values[idx] > threshold ? 1.0f : 0.0f) - 1.0f;
  centers[4 * max_num + idx] = values[idx];
}

struct proposal_extractor::cuda_data {
  hipStream_t stream;
  float* nms_output;
  float* topk_values;
  int* topk_indices;
  float* centers;
  int allocated_max_num;
  int allocated_grid_size;

  cuda_data(int max_num, int grid_size)
      : stream(nullptr),
        nms_output(nullptr),
        topk_values(nullptr),
        topk_indices(nullptr),
        centers(nullptr),
        allocated_max_num(max_num),
        allocated_grid_size(grid_size) {
    CUDA_SAFE_CALL(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    CUDA_SAFE_CALL(hipMalloc(&nms_output, allocated_grid_size * sizeof(float)));
    CUDA_SAFE_CALL(hipMalloc(&topk_values, allocated_max_num * sizeof(float)));
    CUDA_SAFE_CALL(hipMalloc(&topk_indices, allocated_max_num * sizeof(int)));
    CUDA_SAFE_CALL(hipMalloc(&centers, 5 * allocated_max_num * sizeof(float)));
  }

  ~cuda_data() {
    if (nms_output) hipFree(nms_output);
    if (topk_values) hipFree(topk_values);
    if (topk_indices) hipFree(topk_indices);
    if (centers) hipFree(centers);
    if (stream) CUDA_SAFE_CALL(hipStreamDestroy(stream));
  }
};

proposal_extractor::proposal_extractor()
    : max_num(10),
      threshold(0.3f),
      grid_size{0.0f, 0.0f, 0.0f},
      grid_center{0.0f, 0.0f, 0.0f},
      cube_size{0, 0, 0},
      cuda_data_(nullptr) {}

proposal_extractor::~proposal_extractor() = default;

void proposal_extractor::get_centers(const float* proposal_data, float* centers_data) {
  if (!proposal_data || !centers_data) {
    return;
  }

  const int size_x = cube_size[0];
  const int size_y = cube_size[1];
  const int size_z = cube_size[2];
  const int total_size = size_x * size_y * size_z;

  if (!cuda_data_ || cuda_data_->allocated_max_num < static_cast<int>(max_num) ||
      cuda_data_->allocated_grid_size < total_size) {
    cuda_data_.reset();
    cuda_data_ = std::make_unique<cuda_data>(max_num, total_size);
  }

  // Step 1: 3D NMS via max pooling
  {
    dim3 block(8, 8, 4);
    dim3 grid((size_x + block.x - 1) / block.x, (size_y + block.y - 1) / block.y,
              (size_z + block.z - 1) / block.z);
    const int kernel_size = 3;
    const int padding = (kernel_size - 1) / 2;
    max_pool3d_kernel<<<grid, block, 0, cuda_data_->stream>>>(
        proposal_data, cuda_data_->nms_output, size_x, size_y, size_z, kernel_size, padding);
  }

  // Step 2: Top-k selection
  {
    dim3 block(256, 1, 1);
    const int k = static_cast<int>(std::min<uint32_t>(max_num, kMaxTopK));
    const int shared_mem_size = block.x * k * (sizeof(float) + sizeof(int));
    topk_select_kernel<<<1, block, shared_mem_size, cuda_data_->stream>>>(
        cuda_data_->nms_output, cuda_data_->topk_values, cuda_data_->topk_indices, total_size,
        static_cast<int>(max_num));
  }

  // Step 3: Convert to world coordinates
  {
    dim3 block(256, 1, 1);
    dim3 grid((max_num + block.x - 1) / block.x, 1, 1);
    unravel_and_convert_kernel<<<grid, block, 0, cuda_data_->stream>>>(
        cuda_data_->topk_indices, cuda_data_->topk_values, cuda_data_->centers,
        static_cast<int>(max_num), size_x, size_y, size_z, grid_size[0], grid_size[1], grid_size[2],
        grid_center[0], grid_center[1], grid_center[2], threshold);
  }

  CUDA_SAFE_CALL(hipMemcpyAsync(centers_data, cuda_data_->centers, 5 * max_num * sizeof(float),
                                hipMemcpyDeviceToHost, cuda_data_->stream));
  CUDA_SAFE_CALL(hipStreamSynchronize(cuda_data_->stream));
}
}  // namespace stargazer::voxelpose
