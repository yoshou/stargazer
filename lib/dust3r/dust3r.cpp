#include "dust3r.hpp"

#include <onnxruntime_cxx_api.h>
#include <spdlog/spdlog.h>

#include <array>
#include <cassert>
#include <numeric>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#if defined(USE_CUDA)
#include <cuda_runtime.h>
#define GPU_SAFE_CALL(func)                                                                        \
  do {                                                                                             \
    cudaError_t err = (func);                                                                      \
    if (err != cudaSuccess) {                                                                      \
      fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, \
              __FILE__, __LINE__);                                                                 \
      exit(err);                                                                                   \
    }                                                                                              \
  } while (0)
#else
#include <hip/hip_runtime.h>
#define GPU_SAFE_CALL(func)                                                                       \
  do {                                                                                            \
    hipError_t err = (func);                                                                      \
    if (err != hipSuccess) {                                                                      \
      fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", hipGetErrorString(err), err, \
              __FILE__, __LINE__);                                                                \
      exit(err);                                                                                  \
    }                                                                                             \
  } while (0)
#endif

namespace {

template <typename T>
static void gpu_malloc(T** ptr, size_t count) {
#if defined(USE_CUDA)
  GPU_SAFE_CALL(cudaMalloc(ptr, count * sizeof(T)));
#else
  GPU_SAFE_CALL(hipMalloc(ptr, count * sizeof(T)));
#endif
}

static void gpu_free(void* ptr) {
#if defined(USE_CUDA)
  GPU_SAFE_CALL(cudaFree(ptr));
#else
  GPU_SAFE_CALL(hipFree(ptr));
#endif
}

static void htod(void* dst, const void* src, size_t bytes) {
#if defined(USE_CUDA)
  GPU_SAFE_CALL(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
#else
  GPU_SAFE_CALL(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
#endif
}

static void dtoh(void* dst, const void* src, size_t bytes) {
#if defined(USE_CUDA)
  GPU_SAFE_CALL(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
#else
  GPU_SAFE_CALL(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost));
#endif
}

}  // namespace

namespace stargazer::dust3r {

std::vector<float> preprocess_image(const cv::Mat& bgr, const camera_intrin_t& intrin) {
  cv::Mat K = (cv::Mat_<double>(3, 3) << intrin.fx, 0.0, intrin.cx, 0.0, intrin.fy, intrin.cy, 0.0,
               0.0, 1.0);
  cv::Mat dist_coeffs(1, 5, CV_64F);
  for (int i = 0; i < 5; ++i) dist_coeffs.at<double>(0, i) = static_cast<double>(intrin.coeffs[i]);

  cv::Mat new_K = cv::getOptimalNewCameraMatrix(K, dist_coeffs, bgr.size(), 0.0);

  cv::Mat undistorted;
  cv::undistort(bgr, undistorted, K, dist_coeffs, new_K);

  cv::Mat resized;
  cv::resize(undistorted, resized, cv::Size(ONNX_W, ONNX_H), 0, 0, cv::INTER_AREA);

  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

  if (!rgb.isContinuous()) rgb = rgb.clone();

  const int pixels = ONNX_H * ONNX_W;
  std::vector<float> nchw(3 * pixels);
  const uint8_t* src = rgb.data;
  for (int c = 0; c < 3; ++c) {
    float* dst = nchw.data() + c * pixels;
    for (int i = 0; i < pixels; ++i) {
      dst[i] = (static_cast<float>(src[i * 3 + c]) / 255.0f - 0.5f) / 0.5f;
    }
  }
  return nchw;
}

std::vector<float> preprocess_image(const cv::Mat& bgr) {
  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(ONNX_W, ONNX_H), 0, 0, cv::INTER_AREA);

  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

  if (!rgb.isContinuous()) rgb = rgb.clone();

  const int pixels = ONNX_H * ONNX_W;
  std::vector<float> nchw(3 * pixels);
  const uint8_t* src = rgb.data;
  for (int c = 0; c < 3; ++c) {
    float* dst = nchw.data() + c * pixels;
    for (int i = 0; i < pixels; ++i) {
      dst[i] = (static_cast<float>(src[i * 3 + c]) / 255.0f - 0.5f) / 0.5f;
    }
  }
  return nchw;
}

struct dust3r_inference::impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING};
  Ort::Session session{nullptr};
  Ort::IoBinding io_binding{nullptr};

#if defined(USE_CUDA)
  Ort::MemoryInfo device_mem_info{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};
#else
  Ort::MemoryInfo device_mem_info{"Hip", OrtDeviceAllocator, 0, OrtMemTypeDefault};
#endif

  static constexpr int64_t batch = 1;
  static constexpr int64_t ch = 3;
  static constexpr int64_t H = ONNX_H;
  static constexpr int64_t W = ONNX_W;

  static constexpr size_t img_elems = batch * ch * H * W;
  static constexpr size_t pts3d_elems = batch * H * W * 3;
  static constexpr size_t conf_elems = batch * H * W;

  float* d_img1 = nullptr;
  float* d_img2 = nullptr;
  float* d_pts3d_1 = nullptr;
  float* d_conf_1 = nullptr;
  float* d_pts3d_2 = nullptr;
  float* d_conf_2 = nullptr;

  impl(const std::string& model_path) {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#if defined(USE_CUDA)
    try {
      OrtCUDAProviderOptions cuda_options{};
      cuda_options.device_id = 0;
      cuda_options.arena_extend_strategy = 1;
      cuda_options.gpu_mem_limit = 4ULL * 1024 * 1024 * 1024;
      cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
      cuda_options.do_copy_in_default_stream = 1;
      session_options.AppendExecutionProvider_CUDA(cuda_options);
    } catch (const Ort::Exception& e) {
      spdlog::warn("DUSt3R: CUDA EP not available: {}", e.what());
    }
#else
    try {
      OrtMIGraphXProviderOptions mgx{};
      mgx.device_id = 0;
      mgx.migraphx_arena_extend_strategy = 1;
      mgx.migraphx_mem_limit = 4ULL * 1024 * 1024 * 1024;
      session_options.AppendExecutionProvider_MIGraphX(mgx);
    } catch (const Ort::Exception& e) {
      spdlog::warn("DUSt3R: MIGraphX EP not available: {}", e.what());
    }
#endif

    session = Ort::Session(env, model_path.c_str(), session_options);
    io_binding = Ort::IoBinding(session);

    gpu_malloc(&d_img1, img_elems);
    gpu_malloc(&d_img2, img_elems);
    gpu_malloc(&d_pts3d_1, pts3d_elems);
    gpu_malloc(&d_conf_1, conf_elems);
    gpu_malloc(&d_pts3d_2, pts3d_elems);
    gpu_malloc(&d_conf_2, conf_elems);
  }

  ~impl() {
    gpu_free(d_img1);
    gpu_free(d_img2);
    gpu_free(d_pts3d_1);
    gpu_free(d_conf_1);
    gpu_free(d_pts3d_2);
    gpu_free(d_conf_2);
  }

  inference_result run(const std::vector<float>& img1_nchw, const std::vector<float>& img2_nchw) {
    assert(img1_nchw.size() == img_elems);
    assert(img2_nchw.size() == img_elems);

    htod(d_img1, img1_nchw.data(), img_elems * sizeof(float));
    htod(d_img2, img2_nchw.data(), img_elems * sizeof(float));

    io_binding.ClearBoundInputs();
    io_binding.ClearBoundOutputs();

    const std::array<int64_t, 4> img_dims{batch, ch, H, W};
    const std::array<int64_t, 4> pts3d_dims{batch, H, W, 3};
    const std::array<int64_t, 3> conf_dims{batch, H, W};

    {
      auto t1 = Ort::Value::CreateTensor<float>(device_mem_info, d_img1, img_elems, img_dims.data(),
                                                img_dims.size());
      io_binding.BindInput("img1", t1);

      auto t2 = Ort::Value::CreateTensor<float>(device_mem_info, d_img2, img_elems, img_dims.data(),
                                                img_dims.size());
      io_binding.BindInput("img2", t2);
    }

    {
      auto p1 = Ort::Value::CreateTensor<float>(device_mem_info, d_pts3d_1, pts3d_elems,
                                                pts3d_dims.data(), pts3d_dims.size());
      io_binding.BindOutput("pts3d_1", p1);

      auto c1 = Ort::Value::CreateTensor<float>(device_mem_info, d_conf_1, conf_elems,
                                                conf_dims.data(), conf_dims.size());
      io_binding.BindOutput("conf_1", c1);

      auto p2 = Ort::Value::CreateTensor<float>(device_mem_info, d_pts3d_2, pts3d_elems,
                                                pts3d_dims.data(), pts3d_dims.size());
      io_binding.BindOutput("pts3d_2", p2);

      auto c2 = Ort::Value::CreateTensor<float>(device_mem_info, d_conf_2, conf_elems,
                                                conf_dims.data(), conf_dims.size());
      io_binding.BindOutput("conf_2", c2);
    }

    io_binding.SynchronizeInputs();
    session.Run(Ort::RunOptions{nullptr}, io_binding);
    io_binding.SynchronizeOutputs();

    inference_result result;
    result.pts3d_1.resize(pts3d_elems);
    result.conf_1.resize(conf_elems);
    result.pts3d_2.resize(pts3d_elems);
    result.conf_2.resize(conf_elems);

    dtoh(result.pts3d_1.data(), d_pts3d_1, pts3d_elems * sizeof(float));
    dtoh(result.conf_1.data(), d_conf_1, conf_elems * sizeof(float));
    dtoh(result.pts3d_2.data(), d_pts3d_2, pts3d_elems * sizeof(float));
    dtoh(result.conf_2.data(), d_conf_2, conf_elems * sizeof(float));

    return result;
  }
};
dust3r_inference::dust3r_inference(const std::string& model_path)
    : pimpl_(std::make_unique<impl>(model_path)) {}

dust3r_inference::~dust3r_inference() = default;

inference_result dust3r_inference::inference(const std::vector<float>& img1_nchw,
                                             const std::vector<float>& img2_nchw) {
  return pimpl_->run(img1_nchw, img2_nchw);
}

}  // namespace stargazer::dust3r
