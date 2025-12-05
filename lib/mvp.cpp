#include "mvp.hpp"

#include <dlfcn.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>
#include <glm/ext.hpp>
#include <iostream>
#include <numeric>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include "preprocess.hpp"

using namespace stargazer;

#ifdef USE_CUDA

#include <NvInfer.h>
#include <cuda_runtime.h>

#define CUDA_SAFE_CALL(func)                                                                       \
  do {                                                                                             \
    cudaError_t err = (func);                                                                      \
    if (err != cudaSuccess) {                                                                      \
      fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, \
              __FILE__, __LINE__);                                                                 \
      exit(err);                                                                                   \
    }                                                                                              \
  } while (0)

namespace stargazer::mvp {

class logger : public nvinfer1::ILogger {
 public:
  void log(ILogger::Severity severity, const char* msg) noexcept override {
    if (severity == nvinfer1::ILogger::Severity::kINFO) {
      spdlog::info(msg);
    } else if (severity == nvinfer1::ILogger::Severity::kERROR) {
      spdlog::error(msg);
    }
  }
};

static std::vector<uint8_t> read_binary_file(const std::string& filename) {
  std::ifstream ifs(filename.c_str(), std::ios::binary);
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(ifs),
                              std::istreambuf_iterator<char>());
}

// Backbone inference: takes images and outputs multi-scale feature maps
class dnn_inference_backbone {
  logger logger;
  nvinfer1::IRuntime* infer;
  nvinfer1::ICudaEngine* engine;
  nvinfer1::IExecutionContext* context;

  cudaStream_t stream;

  // Input image buffer
  float* input_data = nullptr;
  uint8_t* input_image_data = nullptr;

  // Output feature map buffers
  // feat_0: (views, 256, 128, 240) - 1/4 scale
  // feat_1: (views, 256, 64, 120)  - 1/8 scale
  // feat_2: (views, 256, 32, 60)   - 1/16 scale
  float* feat_0_data = nullptr;
  float* feat_1_data = nullptr;
  float* feat_2_data = nullptr;

  static const int max_input_image_width = 1920;
  static const int max_input_image_height = 1080;

  int image_width = 960;
  int image_height = 512;

 public:
  dnn_inference_backbone(const std::vector<uint8_t>& model_data, size_t max_views) {
    infer = nvinfer1::createInferRuntime(logger);
    engine = infer->deserializeCudaEngine(model_data.data(), model_data.size());
    context = engine->createExecutionContext();

    // Allocate input buffers
    const auto input_size = image_width * image_height * 3 * max_views;
    CUDA_SAFE_CALL(cudaMalloc(&input_data, input_size * sizeof(float)));

    CUDA_SAFE_CALL(cudaMalloc(&input_image_data,
                              max_input_image_width * max_input_image_height * 3 * max_views));

    // Allocate output feature buffers
    const size_t feat_0_size = 256 * 128 * 240 * max_views;
    const size_t feat_1_size = 256 * 64 * 120 * max_views;
    const size_t feat_2_size = 256 * 32 * 60 * max_views;

    CUDA_SAFE_CALL(cudaMalloc(&feat_0_data, feat_0_size * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&feat_1_data, feat_1_size * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&feat_2_data, feat_2_size * sizeof(float)));

    CUDA_SAFE_CALL(cudaStreamCreate(&stream));
  }

  ~dnn_inference_backbone() {
    delete context;
    delete engine;
    delete infer;

    CUDA_SAFE_CALL(cudaStreamDestroy(stream));

    CUDA_SAFE_CALL(cudaFree(input_data));
    CUDA_SAFE_CALL(cudaFree(input_image_data));
    CUDA_SAFE_CALL(cudaFree(feat_0_data));
    CUDA_SAFE_CALL(cudaFree(feat_1_data));
    CUDA_SAFE_CALL(cudaFree(feat_2_data));
  }

  void process(const std::vector<cv::Mat>& images) {
    for (size_t i = 0; i < images.size(); i++) {
      const auto& data = images.at(i);

      const auto input_image_width = data.size().width;
      const auto input_image_height = data.size().height;

      assert(input_image_width <= max_input_image_width);
      assert(input_image_height <= max_input_image_height);

      const std::array<float, 3> mean = {0.485, 0.456, 0.406};
      const std::array<float, 3> std = {0.229, 0.224, 0.225};

      CUDA_SAFE_CALL(cudaMemcpy2D(input_image_data + i * input_image_width * 3 * input_image_height,
                                  input_image_width * 3, data.data, data.step, data.cols * 3,
                                  data.rows, cudaMemcpyHostToDevice));

      preprocess_cuda(input_image_data + i * input_image_width * 3 * input_image_height,
                      input_image_width, input_image_height, input_image_width * 3,
                      input_data + i * image_width * image_height * 3, image_width, image_height,
                      image_width, mean, std);
    }

    CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }

  void inference(size_t num_views) {
    // Process each view through backbone separately and collect features
    for (size_t view_idx = 0; view_idx < num_views; view_idx++) {
      const auto num_input_outputs = engine->getNbIOTensors();
      for (int i = 0; i < num_input_outputs; i++) {
        const auto name = engine->getIOTensorName(i);

        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
          nvinfer1::Dims4 input_dims = {1, 3, image_height, image_width};
          context->setInputShape(name, input_dims);
        }
      }

      if (context->inferShapes(0, nullptr) != 0) {
        spdlog::error("Failed to infer shapes");
      }

      if (!context->allInputDimensionsSpecified()) {
        spdlog::error("Failed to specify all input dimensions");
      }

      // Set tensor addresses for this view
      const size_t input_offset = view_idx * image_width * image_height * 3;
      const size_t feat_0_offset = view_idx * 256 * 128 * 240;
      const size_t feat_1_offset = view_idx * 256 * 64 * 120;
      const size_t feat_2_offset = view_idx * 256 * 32 * 60;

      context->setTensorAddress("images", input_data + input_offset);
      context->setTensorAddress("feat_0", feat_0_data + feat_0_offset);
      context->setTensorAddress("feat_1", feat_1_data + feat_1_offset);
      context->setTensorAddress("feat_2", feat_2_data + feat_2_offset);

      if (!context->enqueueV3(stream)) {
        spdlog::error("Failed to enqueue backbone");
      }
    }

    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
  }

  const float* get_feat_0() const { return feat_0_data; }
  const float* get_feat_1() const { return feat_1_data; }
  const float* get_feat_2() const { return feat_2_data; }

  int get_image_width() const { return image_width; }
  int get_image_height() const { return image_height; }
};

// Head inference: takes feature maps and camera parameters, outputs poses
class dnn_inference_head {
  logger logger;
  nvinfer1::IRuntime* infer;
  nvinfer1::ICudaEngine* engine;
  nvinfer1::IExecutionContext* context;

  cudaStream_t stream;

  // Input feature buffers (shared with backbone outputs)
  float* feat_0_data = nullptr;
  float* feat_1_data = nullptr;
  float* feat_2_data = nullptr;

  // Camera parameter buffers
  float* cam_K_data = nullptr;         // (1, views, 3, 3)
  float* cam_R_data = nullptr;         // (1, views, 3, 3)
  float* cam_T_data = nullptr;         // (1, views, 3)
  float* affine_trans_data = nullptr;  // (1, views, 3, 3)
  float* cam_T_proj_data = nullptr;    // (1, views, 3, 1)
  float* dist_coeff_data = nullptr;    // (1, views, 5)
  float* center_data = nullptr;        // (1, views, 2)

  // Output buffers
  float* pred_poses_data = nullptr;   // (1, n_instance * n_kps, 3)
  float* pred_logits_data = nullptr;  // (1, n_instance, 2)

  static const size_t n_instance = 10;
  static const size_t n_kps = 15;

 public:
  dnn_inference_head(const std::vector<uint8_t>& model_data, size_t max_views) {
    infer = nvinfer1::createInferRuntime(logger);
    engine = infer->deserializeCudaEngine(model_data.data(), model_data.size());
    context = engine->createExecutionContext();

    // Allocate feature input buffers
    const size_t feat_0_size = 256 * 128 * 240 * max_views;
    const size_t feat_1_size = 256 * 64 * 120 * max_views;
    const size_t feat_2_size = 256 * 32 * 60 * max_views;

    CUDA_SAFE_CALL(cudaMalloc(&feat_0_data, feat_0_size * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&feat_1_data, feat_1_size * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&feat_2_data, feat_2_size * sizeof(float)));

    // Allocate camera parameter buffers
    CUDA_SAFE_CALL(cudaMalloc(&cam_K_data, max_views * 3 * 3 * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&cam_R_data, max_views * 3 * 3 * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&cam_T_data, max_views * 3 * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&affine_trans_data, max_views * 3 * 3 * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&cam_T_proj_data, max_views * 3 * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&dist_coeff_data, max_views * 5 * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&center_data, max_views * 2 * sizeof(float)));

    // Allocate output buffers
    CUDA_SAFE_CALL(cudaMalloc(&pred_poses_data, n_instance * n_kps * 3 * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&pred_logits_data, n_instance * 2 * sizeof(float)));

    CUDA_SAFE_CALL(cudaStreamCreate(&stream));
  }

  ~dnn_inference_head() {
    delete context;
    delete engine;
    delete infer;

    CUDA_SAFE_CALL(cudaStreamDestroy(stream));

    CUDA_SAFE_CALL(cudaFree(feat_0_data));
    CUDA_SAFE_CALL(cudaFree(feat_1_data));
    CUDA_SAFE_CALL(cudaFree(feat_2_data));
    CUDA_SAFE_CALL(cudaFree(cam_K_data));
    CUDA_SAFE_CALL(cudaFree(cam_R_data));
    CUDA_SAFE_CALL(cudaFree(cam_T_data));
    CUDA_SAFE_CALL(cudaFree(affine_trans_data));
    CUDA_SAFE_CALL(cudaFree(cam_T_proj_data));
    CUDA_SAFE_CALL(cudaFree(dist_coeff_data));
    CUDA_SAFE_CALL(cudaFree(center_data));
    CUDA_SAFE_CALL(cudaFree(pred_poses_data));
    CUDA_SAFE_CALL(cudaFree(pred_logits_data));
  }

  void set_features(const float* feat_0, const float* feat_1, const float* feat_2,
                    size_t num_views) {
    const size_t feat_0_size = 256 * 128 * 240 * num_views;
    const size_t feat_1_size = 256 * 64 * 120 * num_views;
    const size_t feat_2_size = 256 * 32 * 60 * num_views;

    CUDA_SAFE_CALL(
        cudaMemcpy(feat_0_data, feat_0, feat_0_size * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(
        cudaMemcpy(feat_1_data, feat_1, feat_1_size * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(
        cudaMemcpy(feat_2_data, feat_2, feat_2_size * sizeof(float), cudaMemcpyDeviceToDevice));
  }

  void set_camera_params(const std::vector<camera_data>& cameras, int input_image_width,
                         int input_image_height, int model_image_width, int model_image_height) {
    const size_t num_views = cameras.size();

    std::vector<float> cam_K_host(num_views * 9);
    std::vector<float> cam_R_host(num_views * 9);
    std::vector<float> cam_T_host(num_views * 3);
    std::vector<float> affine_trans_host(num_views * 9);
    std::vector<float> cam_T_proj_host(num_views * 3);
    std::vector<float> dist_coeff_host(num_views * 5);
    std::vector<float> center_host(num_views * 2);

    // Calculate scale for affine transform (same as xrmocap get_scale)
    float w = static_cast<float>(input_image_width);
    float h = static_cast<float>(input_image_height);
    float w_resized = static_cast<float>(model_image_width);
    float h_resized = static_cast<float>(model_image_height);

    float w_pad, h_pad;
    if (w / w_resized < h / h_resized) {
      w_pad = h / h_resized * w_resized;
      h_pad = h;
    } else {
      w_pad = w;
      h_pad = w / w_resized * h_resized;
    }
    float scale_x = w_pad / 200.0f;
    float scale_y = h_pad / 200.0f;

    // Calculate affine transform matrix (simplified version of get_affine_transform)
    // src: center of original image, dst: center of output image
    float src_cx = w / 2.0f;
    float src_cy = h / 2.0f;
    float dst_cx = w_resized / 2.0f;
    float dst_cy = h_resized / 2.0f;

    // Scale from src to dst
    float sx = w_resized / (scale_x * 200.0f);
    float sy = h_resized / (scale_y * 200.0f);

    // Affine transform: translate to origin, scale, translate to dst center
    // [sx,  0, dst_cx - sx*src_cx]
    // [0,  sy, dst_cy - sy*src_cy]
    // [0,   0,                  1]
    float affine_00 = sx;
    float affine_01 = 0.0f;
    float affine_02 = dst_cx - sx * src_cx;
    float affine_10 = 0.0f;
    float affine_11 = sy;
    float affine_12 = dst_cy - sy * src_cy;

    // trans_ground for Panoptic dataset (row-major): [[1,0,0],[0,0,-1],[0,1,0]]
    // This transforms world coordinates: (x, y, z) -> (x, -z, y)
    // r_trans = R @ trans_ground
    float trans_ground[9] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f};

    for (size_t v = 0; v < num_views; v++) {
      const auto& cam = cameras[v];

      // K matrix (row-major: K[row][col])
      cam_K_host[v * 9 + 0] = static_cast<float>(cam.fx);
      cam_K_host[v * 9 + 1] = 0.0f;
      cam_K_host[v * 9 + 2] = static_cast<float>(cam.cx);
      cam_K_host[v * 9 + 3] = 0.0f;
      cam_K_host[v * 9 + 4] = static_cast<float>(cam.fy);
      cam_K_host[v * 9 + 5] = static_cast<float>(cam.cy);
      cam_K_host[v * 9 + 6] = 0.0f;
      cam_K_host[v * 9 + 7] = 0.0f;
      cam_K_host[v * 9 + 8] = 1.0f;

      // R matrix: cam.rotation is row-major in original Panoptic coordinate system
      // Apply trans_ground: R_trans = R @ trans_ground
      float R[9];
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          R[i * 3 + j] = static_cast<float>(cam.rotation[i][j]);
        }
      }

      // Matrix multiplication: R_trans = R @ trans_ground
      float R_trans[9];
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          R_trans[i * 3 + j] = 0.0f;
          for (int k = 0; k < 3; k++) {
            R_trans[i * 3 + j] += R[i * 3 + k] * trans_ground[k * 3 + j];
          }
        }
      }

      for (int i = 0; i < 9; i++) {
        cam_R_host[v * 9 + i] = R_trans[i];
      }

      // cam_T (camera_standard_T) - original translation (world-to-camera)
      float T[3];
      for (int i = 0; i < 3; i++) {
        T[i] = static_cast<float>(cam.translation[i]);
        cam_T_host[v * 3 + i] = T[i];
      }

      // Affine transform (3x3 matrix, row-major)
      affine_trans_host[v * 9 + 0] = affine_00;
      affine_trans_host[v * 9 + 1] = affine_01;
      affine_trans_host[v * 9 + 2] = affine_02;
      affine_trans_host[v * 9 + 3] = affine_10;
      affine_trans_host[v * 9 + 4] = affine_11;
      affine_trans_host[v * 9 + 5] = affine_12;
      affine_trans_host[v * 9 + 6] = 0.0f;
      affine_trans_host[v * 9 + 7] = 0.0f;
      affine_trans_host[v * 9 + 8] = 1.0f;

      // cam_T_proj = -R_trans^T @ T (camera position in transformed world coordinates)
      // R_trans^T[i][j] = R_trans[j][i]
      for (int i = 0; i < 3; i++) {
        cam_T_proj_host[v * 3 + i] = 0.0f;
        for (int j = 0; j < 3; j++) {
          cam_T_proj_host[v * 3 + i] -= R_trans[j * 3 + i] * T[j];
        }
      }

      // Distortion coefficients
      for (int i = 0; i < 5; i++) {
        dist_coeff_host[v * 5 + i] = static_cast<float>(cam.dist_coeff[i]);
      }

      // Image center (of original image)
      center_host[v * 2 + 0] = src_cx;
      center_host[v * 2 + 1] = src_cy;
    }

    CUDA_SAFE_CALL(cudaMemcpy(cam_K_data, cam_K_host.data(), num_views * 9 * sizeof(float),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(cam_R_data, cam_R_host.data(), num_views * 9 * sizeof(float),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(cam_T_data, cam_T_host.data(), num_views * 3 * sizeof(float),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(affine_trans_data, affine_trans_host.data(),
                              num_views * 9 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(cam_T_proj_data, cam_T_proj_host.data(),
                              num_views * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dist_coeff_data, dist_coeff_host.data(),
                              num_views * 5 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(center_data, center_host.data(), num_views * 2 * sizeof(float),
                              cudaMemcpyHostToDevice));
  }

  void inference(size_t num_views) {
    // Set input shapes
    const auto num_input_outputs = engine->getNbIOTensors();
    for (int i = 0; i < num_input_outputs; i++) {
      const auto name = engine->getIOTensorName(i);

      if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
        std::string name_str(name);
        if (name_str == "feat_0") {
          nvinfer1::Dims4 dims = {static_cast<int32_t>(num_views), 256, 128, 240};
          context->setInputShape(name, dims);
        } else if (name_str == "feat_1") {
          nvinfer1::Dims4 dims = {static_cast<int32_t>(num_views), 256, 64, 120};
          context->setInputShape(name, dims);
        } else if (name_str == "feat_2") {
          nvinfer1::Dims4 dims = {static_cast<int32_t>(num_views), 256, 32, 60};
          context->setInputShape(name, dims);
        } else if (name_str == "cam_K" || name_str == "cam_R" || name_str == "affine_trans") {
          nvinfer1::Dims4 dims = {1, static_cast<int32_t>(num_views), 3, 3};
          context->setInputShape(name, dims);
        } else if (name_str == "cam_T") {
          nvinfer1::Dims3 dims = {1, static_cast<int32_t>(num_views), 3};
          context->setInputShape(name, dims);
        } else if (name_str == "cam_T_proj") {
          nvinfer1::Dims4 dims = {1, static_cast<int32_t>(num_views), 3, 1};
          context->setInputShape(name, dims);
        } else if (name_str == "dist_coeff") {
          nvinfer1::Dims3 dims = {1, static_cast<int32_t>(num_views), 5};
          context->setInputShape(name, dims);
        } else if (name_str == "center") {
          nvinfer1::Dims3 dims = {1, static_cast<int32_t>(num_views), 2};
          context->setInputShape(name, dims);
        }
      }
    }

    if (context->inferShapes(0, nullptr) != 0) {
      spdlog::error("Failed to infer shapes for head");
    }

    if (!context->allInputDimensionsSpecified()) {
      spdlog::error("Failed to specify all input dimensions for head");
    }

    // Set tensor addresses
    context->setTensorAddress("feat_0", feat_0_data);
    context->setTensorAddress("feat_1", feat_1_data);
    context->setTensorAddress("feat_2", feat_2_data);
    context->setTensorAddress("cam_K", cam_K_data);
    context->setTensorAddress("cam_R", cam_R_data);
    context->setTensorAddress("cam_T", cam_T_data);
    context->setTensorAddress("affine_trans", affine_trans_data);
    context->setTensorAddress("cam_T_proj", cam_T_proj_data);
    context->setTensorAddress("dist_coeff", dist_coeff_data);
    context->setTensorAddress("center", center_data);
    context->setTensorAddress("pred_poses", pred_poses_data);
    context->setTensorAddress("pred_logits", pred_logits_data);

    if (!context->enqueueV3(stream)) {
      spdlog::error("Failed to enqueue head");
    }

    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
  }

  void get_outputs(std::vector<float>& pred_poses, std::vector<float>& pred_logits) const {
    pred_poses.resize(n_instance * n_kps * 3);
    pred_logits.resize(n_instance * 2);

    CUDA_SAFE_CALL(cudaMemcpy(pred_poses.data(), pred_poses_data,
                              n_instance * n_kps * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(pred_logits.data(), pred_logits_data, n_instance * 2 * sizeof(float),
                              cudaMemcpyDeviceToHost));
  }

  static constexpr size_t get_n_instance() { return n_instance; }
  static constexpr size_t get_n_kps() { return n_kps; }
};

mvp::mvp() {
  namespace fs = std::filesystem;

  const auto backbone_path = "../data/mvp/mvp_backbone_int8.trt";
  const auto head_path = "../data/mvp/mvp_head_fp32.trt";

  if (!fs::exists(backbone_path)) {
    throw std::runtime_error("MVP backbone model not found: " + std::string(backbone_path));
  }
  if (!fs::exists(head_path)) {
    throw std::runtime_error("MVP head model not found: " + std::string(head_path));
  }

  const auto backbone_data = read_binary_file(backbone_path);
  const auto head_data = read_binary_file(head_path);

  const size_t max_views = 5;

  inference_backbone = std::make_unique<dnn_inference_backbone>(backbone_data, max_views);
  inference_head = std::make_unique<dnn_inference_head>(head_data, max_views);

  // Default grid settings (for Panoptic dataset)
  grid_center = {0.0f, -500.0f, 800.0f};
  grid_size = {8000.0f, 8000.0f, 2000.0f};
}

mvp::~mvp() = default;

// Convert normalized coordinates to absolute coordinates
static glm::vec3 norm2absolute(const glm::vec3& norm_coords, const std::array<float, 3>& grid_size,
                               const std::array<float, 3>& grid_center) {
  return glm::vec3(norm_coords.x * grid_size[0] + grid_center[0] - grid_size[0] / 2.0f,
                   norm_coords.y * grid_size[1] + grid_center[1] - grid_size[1] / 2.0f,
                   norm_coords.z * grid_size[2] + grid_center[2] - grid_size[2] / 2.0f);
}

std::vector<glm::vec3> mvp::inference(const std::vector<cv::Mat>& images_list,
                                      const std::vector<camera_data>& cameras_list) {
  if (images_list.empty() || cameras_list.empty()) {
    return {};
  }

  if (images_list.size() != cameras_list.size()) {
    spdlog::error("Number of images and cameras must match");
    return {};
  }

  const size_t num_views = images_list.size();

  // Get input image size from first image
  const int input_image_width = images_list[0].cols;
  const int input_image_height = images_list[0].rows;

  // Preprocess images and run backbone
  inference_backbone->process(images_list);
  inference_backbone->inference(num_views);

  // Set features and camera parameters for head
  inference_head->set_features(inference_backbone->get_feat_0(), inference_backbone->get_feat_1(),
                               inference_backbone->get_feat_2(), num_views);
  inference_head->set_camera_params(cameras_list, input_image_width, input_image_height,
                                    inference_backbone->get_image_width(),
                                    inference_backbone->get_image_height());

  // Run head inference
  inference_head->inference(num_views);

  // Get outputs
  std::vector<float> pred_poses, pred_logits;
  inference_head->get_outputs(pred_poses, pred_logits);

  // Post-process: extract valid poses
  std::vector<glm::vec3> result;

  const size_t n_instance = dnn_inference_head::get_n_instance();
  const size_t n_kps = dnn_inference_head::get_n_kps();

  // Collect all valid detections with scores
  struct Detection {
    size_t instance_id;
    float score;
    glm::vec3 root_pos;  // Root joint (index 0) position for NMS
  };
  std::vector<Detection> detections;

  for (size_t inst = 0; inst < n_instance; inst++) {
    const float logit = pred_logits[inst * 2 + 1];
    const float score = 1.0f / (1.0f + std::exp(-logit));

    if (score > 0.3f) {  // Lower threshold, will filter with NMS
      // Get root joint (keypoint 0) position for NMS
      const size_t root_idx = inst * n_kps;
      glm::vec3 norm_pos(pred_poses[root_idx * 3 + 0], pred_poses[root_idx * 3 + 1],
                         pred_poses[root_idx * 3 + 2]);
      glm::vec3 abs_pos = norm2absolute(norm_pos, grid_size, grid_center);

      detections.push_back({inst, score, abs_pos});
    }
  }

  // Sort by score (descending)
  std::sort(detections.begin(), detections.end(),
            [](const Detection& a, const Detection& b) { return a.score > b.score; });

  // NMS: suppress detections that are too close to a higher-scoring detection
  const float nms_distance_threshold = 500.0f;  // 500mm = 50cm
  std::vector<bool> suppressed(detections.size(), false);

  for (size_t i = 0; i < detections.size(); i++) {
    if (suppressed[i]) continue;

    for (size_t j = i + 1; j < detections.size(); j++) {
      if (suppressed[j]) continue;

      float dist = glm::length(detections[i].root_pos - detections[j].root_pos);
      if (dist < nms_distance_threshold) {
        suppressed[j] = true;  // Suppress lower-scoring detection
      }
    }
  }

  // Extract keypoints from non-suppressed detections with score > 0.5
  for (size_t i = 0; i < detections.size(); i++) {
    if (suppressed[i] || detections[i].score < 0.5f) continue;

    const size_t inst = detections[i].instance_id;
    for (size_t kp = 0; kp < n_kps; kp++) {
      const size_t idx = inst * n_kps + kp;
      glm::vec3 norm_pos(pred_poses[idx * 3 + 0], pred_poses[idx * 3 + 1], pred_poses[idx * 3 + 2]);

      // Convert from normalized to absolute coordinates
      glm::vec3 abs_pos = norm2absolute(norm_pos, grid_size, grid_center);

      // Apply inverse of trans_ground to convert back to original Panoptic coordinate system
      // trans_ground: (x, y, z) -> (x, -z, y)
      // inverse:      (x, y, z) -> (x, z, -y)
      glm::vec3 world_pos(abs_pos.x, abs_pos.z, -abs_pos.y);

      // Convert from mm to meters
      result.push_back(world_pos / 1000.0f);
    }
  }

  return result;
}

std::array<float, 3> mvp::get_grid_size() const { return grid_size; }

std::array<float, 3> mvp::get_grid_center() const { return grid_center; }

void mvp::set_grid_size(const std::array<float, 3>& value) { grid_size = value; }

void mvp::set_grid_center(const std::array<float, 3>& value) { grid_center = value; }

}  // namespace stargazer::mvp

#else

// Stub implementation when CUDA is not available
namespace stargazer::mvp {

class dnn_inference_backbone {};
class dnn_inference_head {};

mvp::mvp() { throw std::runtime_error("MVP requires CUDA support"); }
mvp::~mvp() = default;

std::vector<glm::vec3> mvp::inference(const std::vector<cv::Mat>& images_list,
                                      const std::vector<camera_data>& cameras_list) {
  return {};
}

std::array<float, 3> mvp::get_grid_size() const { return {}; }
std::array<float, 3> mvp::get_grid_center() const { return {}; }
void mvp::set_grid_size(const std::array<float, 3>& value) {}
void mvp::set_grid_center(const std::array<float, 3>& value) {}

}  // namespace stargazer::mvp

#endif
