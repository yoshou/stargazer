#include "voxelpose.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <vector>

#include "graph_proc.h"
#include "graph_proc_cv.h"
#include "graph_proc_img.h"
#include "graph_proc_tensor.h"
#include "preprocess.hpp"
#include "voxelpose_internal.hpp"

using namespace stargazer;

namespace stargazer::voxelpose {
class dnn_inference {
 public:
  virtual ~dnn_inference() = default;
  virtual void inference(const float* input) = 0;
  virtual const float* get_output_data() const = 0;
};

class dnn_inference_heatmap {
 public:
  virtual ~dnn_inference_heatmap() = default;
  virtual void process(const std::vector<cv::Mat>& images, std::vector<roi_data>& rois) = 0;
  virtual void inference(size_t num_views) = 0;
  virtual const float* get_heatmaps() const = 0;
  virtual int get_heatmap_width() const = 0;
  virtual int get_heatmap_height() const = 0;
};
}  // namespace stargazer::voxelpose

#ifdef USE_CUDA

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

template <typename T>
static void malloc_device(T** device_ptr, size_t size) {
  CUDA_SAFE_CALL(cudaMalloc(device_ptr, size * sizeof(T)));
}
static void free_device(void* device_ptr) { CUDA_SAFE_CALL(cudaFree(device_ptr)); }
static void memcpy_dtoh(void* dst, const void* src, size_t size) {
  CUDA_SAFE_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}
static void memcpy_htod(void* dst, const void* src, size_t size) {
  CUDA_SAFE_CALL(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}
static void memcpy_dtod(void* dst, const void* src, size_t size) {
  CUDA_SAFE_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
}
static void memcpy2d_htod(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                          size_t height) {
  CUDA_SAFE_CALL(cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyHostToDevice));
}
static void synchronize_device() { CUDA_SAFE_CALL(cudaDeviceSynchronize()); }
#else

#include <hip/hip_runtime.h>

#define HIP_SAFE_CALL(func)                                                                       \
  do {                                                                                            \
    hipError_t err = (func);                                                                      \
    if (err != hipSuccess) {                                                                      \
      fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", hipGetErrorString(err), err, \
              __FILE__, __LINE__);                                                                \
      exit(err);                                                                                  \
    }                                                                                             \
  } while (0)

template <typename T>
static void malloc_device(T** device_ptr, size_t size) {
  HIP_SAFE_CALL(hipMalloc(device_ptr, size * sizeof(T)));
}
static void free_device(void* device_ptr) { HIP_SAFE_CALL(hipFree(device_ptr)); }
static void memcpy_dtoh(void* dst, const void* src, size_t size) {
  HIP_SAFE_CALL(hipMemcpy(dst, src, size, hipMemcpyDeviceToHost));
}
static void memcpy_htod(void* dst, const void* src, size_t size) {
  HIP_SAFE_CALL(hipMemcpy(dst, src, size, hipMemcpyHostToDevice));
}
static void memcpy_dtod(void* dst, const void* src, size_t size) {
  HIP_SAFE_CALL(hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice));
}
static void memcpy2d_htod(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                          size_t height) {
  HIP_SAFE_CALL(hipMemcpy2D(dst, dpitch, src, spitch, width, height, hipMemcpyHostToDevice));
}
static void synchronize_device() { HIP_SAFE_CALL(hipDeviceSynchronize()); }
#endif

#define ENABLE_ONNXRUNTIME

#ifdef ENABLE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>

namespace stargazer::voxelpose {
class ort_dnn_inference : public dnn_inference {
  std::vector<uint8_t> model_data;

  Ort::Env env{ORT_LOGGING_LEVEL_WARNING};
  Ort::Session session;
  Ort::IoBinding io_binding;
#if USE_CUDA
  Ort::MemoryInfo device_mem_info{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};
#else
  Ort::MemoryInfo device_mem_info{"Hip", OrtDeviceAllocator, 0, OrtMemTypeDefault};
#endif

  float* input_data = nullptr;
  float* output_data = nullptr;

  std::vector<std::string> input_node_names;
  std::vector<std::string> output_node_names;

  std::unordered_map<std::string, std::vector<int64_t>> input_node_dims;
  std::unordered_map<std::string, std::vector<int64_t>> output_node_dims;

 public:
  ort_dnn_inference(const std::vector<uint8_t>& model_data)
      : session(nullptr), io_binding(nullptr) {
    namespace fs = std::filesystem;

    // Create session
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#if USE_CUDA
    try {
      OrtCUDAProviderOptions cuda_options{};

      cuda_options.device_id = 0;
      cuda_options.arena_extend_strategy = 1;
      cuda_options.gpu_mem_limit = 4ULL * 1024 * 1024 * 1024;
      cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
      cuda_options.do_copy_in_default_stream = 1;

      session_options.AppendExecutionProvider_CUDA(cuda_options);
    } catch (const Ort::Exception& e) {
      spdlog::info(e.what());
    }
#else
    try {
      OrtMIGraphXProviderOptions migraphx_options{};

      migraphx_options.device_id = 0;

      session_options.AppendExecutionProvider_MIGraphX(migraphx_options);
    } catch (const Ort::Exception& e) {
      spdlog::info(e.what());
    }
#endif

    session = Ort::Session(env, model_data.data(), model_data.size(), session_options);
    io_binding = Ort::IoBinding(session);

    Ort::AllocatorWithDefaultOptions allocator;

    // Iterate over all input nodes
    const size_t num_input_nodes = session.GetInputCount();
    for (size_t i = 0; i < num_input_nodes; i++) {
      const auto input_name = session.GetInputNameAllocated(i, allocator);
      input_node_names.push_back(input_name.get());

      const auto type_info = session.GetInputTypeInfo(i);
      const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      const auto input_shape = tensor_info.GetShape();
      input_node_dims[input_name.get()] = input_shape;
    }

    // Iterate over all output nodes
    const size_t num_output_nodes = session.GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; i++) {
      const auto output_name = session.GetOutputNameAllocated(i, allocator);
      output_node_names.push_back(output_name.get());

      const auto type_info = session.GetOutputTypeInfo(i);
      const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      const auto output_shape = tensor_info.GetShape();
      output_node_dims[output_name.get()] = output_shape;
    }
    assert(input_node_names.size() == 1);
    assert(input_node_names[0] == "input");

    assert(output_node_names.size() == 1);
    assert(output_node_names[0] == "output");

    {
      const auto dims = input_node_dims.at(input_node_names[0]);
      const auto input_size =
          std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

      malloc_device(&input_data, input_size);
    }

    {
      const auto dims = output_node_dims.at(output_node_names[0]);
      const auto output_size =
          std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

      malloc_device(&output_data, output_size);
    }
  }

  void inference(const float* input) {
    assert(input_node_names.size() == 1);
    assert(input_node_names[0] == "input");

    assert(output_node_names.size() == 1);
    assert(output_node_names[0] == "output");

    std::vector<const char*> input_node_names;
    {
      input_node_names.push_back(this->input_node_names[0].c_str());
    }

    std::vector<const char*> output_node_names;
    {
      output_node_names.push_back(this->output_node_names[0].c_str());
    }

    io_binding.ClearBoundInputs();
    io_binding.ClearBoundOutputs();

    std::vector<Ort::Value> input_tensors;
    {
      const auto dims = input_node_dims.at(input_node_names[0]);
      const auto input_size =
          std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

      memcpy_dtod(input_data, input, input_size * sizeof(float));

      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
          device_mem_info, input_data, input_size, dims.data(), dims.size());

      io_binding.BindInput(input_node_names[0], input_tensor);

      input_tensors.emplace_back(std::move(input_tensor));
    }

    std::vector<Ort::Value> output_tensors;
    {
      const auto dims = output_node_dims.at(output_node_names[0]);
      const auto output_size =
          std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

      Ort::Value output_tensor = Ort::Value::CreateTensor(device_mem_info, output_data, output_size,
                                                          dims.data(), dims.size());

      io_binding.BindOutput(output_node_names[0], output_tensor);

      output_tensors.emplace_back(std::move(output_tensor));
    }

    io_binding.SynchronizeInputs();

    session.Run(Ort::RunOptions{nullptr}, io_binding);

    io_binding.SynchronizeOutputs();
  }

  const float* get_output_data() const { return output_data; }
};

class ort_dnn_inference_heatmap : public dnn_inference_heatmap {
  std::vector<uint8_t> model_data;

  Ort::Env env{ORT_LOGGING_LEVEL_WARNING};
  Ort::Session session;
  Ort::IoBinding io_binding;
#if USE_CUDA
  Ort::MemoryInfo device_mem_info{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};
#else
  Ort::MemoryInfo device_mem_info{"Hip", OrtDeviceAllocator, 0, OrtMemTypeDefault};
#endif

  float* input_data = nullptr;
  float* output_data = nullptr;

  std::vector<std::string> input_node_names;
  std::vector<std::string> output_node_names;

  std::unordered_map<std::string, std::vector<int64_t>> input_node_dims;
  std::unordered_map<std::string, std::vector<int64_t>> output_node_dims;

  static const int max_input_image_width = 1920;
  static const int max_input_image_height = 1080;

  int image_width = 960;
  int image_height = 512;

  uint8_t* input_image_data = nullptr;

 public:
  ort_dnn_inference_heatmap(const std::vector<uint8_t>& model_data, size_t max_views)
      : session(nullptr), io_binding(nullptr) {
    namespace fs = std::filesystem;

    // Create session
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#if USE_CUDA
    try {
      OrtCUDAProviderOptions cuda_options{};

      cuda_options.device_id = 0;
      cuda_options.arena_extend_strategy = 1;
      cuda_options.gpu_mem_limit = 4ULL * 1024 * 1024 * 1024;
      cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
      cuda_options.do_copy_in_default_stream = 1;

      session_options.AppendExecutionProvider_CUDA(cuda_options);
    } catch (const Ort::Exception& e) {
      spdlog::info(e.what());
    }
#else
    try {
      OrtMIGraphXProviderOptions migraphx_options{};

      migraphx_options.device_id = 0;
      migraphx_options.migraphx_arena_extend_strategy = 1;
      migraphx_options.migraphx_mem_limit = 2ULL * 1024 * 1024 * 1024;

      session_options.AppendExecutionProvider_MIGraphX(migraphx_options);
    } catch (const Ort::Exception& e) {
      spdlog::info(e.what());
    }
#endif
    try {
      session = Ort::Session(env, model_data.data(), model_data.size(), session_options);
    } catch (const Ort::Exception& e) {
      spdlog::info(e.what());
    }
    io_binding = Ort::IoBinding(session);

    Ort::AllocatorWithDefaultOptions allocator;

    // Iterate over all input nodes
    const size_t num_input_nodes = session.GetInputCount();
    for (size_t i = 0; i < num_input_nodes; i++) {
      const auto input_name = session.GetInputNameAllocated(i, allocator);
      input_node_names.push_back(input_name.get());

      const auto type_info = session.GetInputTypeInfo(i);
      const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      const auto input_shape = tensor_info.GetShape();
      input_node_dims[input_name.get()] = input_shape;
    }

    // Iterate over all output nodes
    const size_t num_output_nodes = session.GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; i++) {
      const auto output_name = session.GetOutputNameAllocated(i, allocator);
      output_node_names.push_back(output_name.get());

      const auto type_info = session.GetOutputTypeInfo(i);
      const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      const auto output_shape = tensor_info.GetShape();
      output_node_dims[output_name.get()] = output_shape;
    }
    assert(input_node_names.size() == 1);
    assert(input_node_names[0] == "input");

    assert(output_node_names.size() == 1);
    assert(output_node_names[0] == "output");

    const auto input_size = 960 * 512 * 3 * max_views;
    malloc_device(&input_data, input_size);

    const auto output_size = 240 * 128 * 15 * max_views;
    malloc_device(&output_data, output_size);

    malloc_device(&input_image_data,
                  max_input_image_width * max_input_image_height * 3 * max_views);
  }

  ~ort_dnn_inference_heatmap() {
    free_device(input_data);
    free_device(output_data);
    free_device(input_image_data);
  }

  void process(const std::vector<cv::Mat>& images, std::vector<roi_data>& rois) {
    const auto&& image_size = cv::Size(image_width, image_height);

    for (size_t i = 0; i < images.size(); i++) {
      const auto& data = images.at(i);

      const auto get_scale = [](const cv::Size2f& image_size, const cv::Size2f& resized_size) {
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

      const auto input_image_width = data.size().width;
      const auto input_image_height = data.size().height;

      assert(input_image_width <= max_input_image_width);
      assert(input_image_height <= max_input_image_height);

      const auto scale = get_scale(data.size(), image_size);
      const auto center = cv::Point2f(data.size().width / 2.0, data.size().height / 2.0);
      const auto rotation = 0.0;

      roi_data roi = {{scale.width, scale.height}, rotation, {center.x, center.y}};
      rois.push_back(roi);

      const std::array<float, 3> mean = {0.485, 0.456, 0.406};
      const std::array<float, 3> std = {0.229, 0.224, 0.225};

      memcpy2d_htod(input_image_data + i * input_image_width * 3 * input_image_height,
                    input_image_width * 3, data.data, data.step, data.cols * 3, data.rows);

      preprocess_cuda(input_image_data + i * input_image_width * 3 * input_image_height,
                      input_image_width, input_image_height, input_image_width * 3,
                      input_data + i * 960 * 512 * 3, 960, 512, 960, mean, std);
    }

    synchronize_device();

    inference(images.size());
  }

  void inference(size_t num_views) {
    assert(input_node_names.size() == 1);
    assert(input_node_names[0] == "input");

    assert(output_node_names.size() == 1);
    assert(output_node_names[0] == "output");

    std::vector<const char*> input_node_names;
    {
      input_node_names.push_back(this->input_node_names[0].c_str());
    }

    std::vector<const char*> output_node_names;
    {
      output_node_names.push_back(this->output_node_names[0].c_str());
    }

    io_binding.ClearBoundInputs();
    io_binding.ClearBoundOutputs();

    std::vector<Ort::Value> input_tensors;
    {
      auto dims = input_node_dims.at(input_node_names[0]);
      dims[0] = num_views;

      const auto input_size =
          std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
          device_mem_info, input_data, input_size, dims.data(), dims.size());

      io_binding.BindInput(input_node_names[0], input_tensor);

      input_tensors.emplace_back(std::move(input_tensor));
    }

    std::vector<Ort::Value> output_tensors;
    {
      auto dims = output_node_dims.at(output_node_names[0]);
      dims[0] = num_views;
      const auto output_size =
          std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

      Ort::Value output_tensor = Ort::Value::CreateTensor(device_mem_info, output_data, output_size,
                                                          dims.data(), dims.size());

      io_binding.BindOutput(output_node_names[0], output_tensor);

      output_tensors.emplace_back(std::move(output_tensor));
    }

    io_binding.SynchronizeInputs();

    session.Run(Ort::RunOptions{nullptr}, io_binding);

    io_binding.SynchronizeOutputs();
  }

  const float* get_heatmaps() const { return output_data; }

  int get_heatmap_width() const { return 240; }

  int get_heatmap_height() const { return 128; }
};
}  // namespace stargazer::voxelpose
#endif

// #define USE_TENSORRT
// #define USE_MIGRAPHX

#ifdef USE_TENSORRT

#include <NvInfer.h>
#include <dlfcn.h>

namespace stargazer::voxelpose {
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

class trt_dnn_inference : public dnn_inference {
  std::vector<uint8_t> model_data;

  void* plugin_handler;
  logger logger;
  nvinfer1::IRuntime* infer;
  nvinfer1::ICudaEngine* engine;
  nvinfer1::IExecutionContext* context;

  cudaStream_t stream;

  float* input_data = nullptr;
  float* output_data = nullptr;

  std::vector<std::string> input_node_names;
  std::vector<std::string> output_node_names;

  std::unordered_map<std::string, std::vector<int64_t>> input_node_dims;
  std::unordered_map<std::string, std::vector<int64_t>> output_node_dims;

 public:
  static const auto num_joints = 15;

  size_t max_batch_size = 4;
  size_t max_num_people;

  trt_dnn_inference(const std::vector<uint8_t>& model_data, size_t max_num_people)
      : max_num_people(max_num_people) {
    plugin_handler = dlopen("../data/mvpose/libmmdeploy_tensorrt_ops.so", RTLD_NOW);
    if (!plugin_handler) {
      throw std::runtime_error(dlerror());
    }

    infer = nvinfer1::createInferRuntime(logger);
    engine = infer->deserializeCudaEngine(model_data.data(), model_data.size());
    context = engine->createExecutionContext();

    const auto num_io_tensors = engine->getNbIOTensors();
    for (int i = 0; i < num_io_tensors; i++) {
      const auto name = engine->getIOTensorName(i);
      const auto shape = engine->getTensorShape(name);
      if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
        input_node_names.push_back(name);

        std::vector<int64_t> dims;
        for (int j = 0; j < shape.nbDims; j++) {
          dims.push_back(shape.d[j]);
        }
        input_node_dims[name] = dims;
      } else {
        output_node_names.push_back(name);

        std::vector<int64_t> dims;
        for (int j = 0; j < shape.nbDims; j++) {
          dims.push_back(shape.d[j]);
        }
        output_node_dims[name] = dims;
      }
    }

    assert(input_node_names.size() == 1);
    assert(input_node_names[0] == "input");

    assert(output_node_names.size() == 1);
    assert(output_node_names[0] == "output");

    {
      auto dims = input_node_dims.at(input_node_names[0]);
      dims[0] = max_batch_size;
      const auto input_size =
          std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

      malloc_device(&input_data, input_size);
    }

    {
      auto dims = output_node_dims.at(output_node_names[0]);
      dims[0] = max_batch_size;
      const auto output_size =
          std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

      malloc_device(&output_data, output_size);
    }

    CUDA_SAFE_CALL(cudaStreamCreate(&stream));
  }

  ~trt_dnn_inference() {
    delete context;
    delete engine;
    delete infer;

    dlclose(plugin_handler);

    CUDA_SAFE_CALL(cudaStreamDestroy(stream));

    CUDA_SAFE_CALL(cudaFree(input_data));
    CUDA_SAFE_CALL(cudaFree(output_data));
  }

  void inference(const float* input) {
    size_t num_batch = 1;

    {
      auto dims = input_node_dims.at(input_node_names[0]);
      dims[0] = num_batch;
      const auto input_size =
          std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

      memcpy_dtod(input_data, input, input_size * sizeof(float));
    }

    for (size_t k = 0; k < num_batch; k += max_batch_size) {
      const auto num_input_outputs = engine->getNbIOTensors();
      for (int i = 0; i < num_input_outputs; i++) {
        const auto name = engine->getIOTensorName(i);
        const auto tensor = engine->getTensorShape(name);

        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
          nvinfer1::Dims input_dims = tensor;

          if (std::string(name) == "input") {
            input_dims.d[0] = std::min(num_batch - k, max_batch_size);
          }

          context->setInputShape(name, input_dims);
        }
      }

      if (context->inferShapes(0, nullptr) != 0) {
        spdlog::error("Failed to infer shapes");
      }

      if (!context->allInputDimensionsSpecified()) {
        spdlog::error("Failed to specify all input dimensions");
      }

      std::unordered_map<std::string, void*> buffers;
      buffers["input"] = input_data;
      buffers["output"] = output_data;

      for (const auto& [name, buffer] : buffers) {
        if (!context->setTensorAddress(name.c_str(), buffer)) {
          spdlog::error("Failed to set tensor address");
        }
      }

      if (!context->enqueueV3(stream)) {
        spdlog::error("Failed to enqueue");
      }
    }

    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
  }

  const float* get_output_data() const { return output_data; }
};

class trt_dnn_inference_heatmap : public dnn_inference_heatmap {
  std::vector<uint8_t> model_data;

  void* plugin_handler;
  logger logger;
  nvinfer1::IRuntime* infer;
  nvinfer1::ICudaEngine* engine;
  nvinfer1::IExecutionContext* context;

  cudaStream_t stream;

  float* input_data = nullptr;
  float* output_data = nullptr;

  static const int max_input_image_width = 1920;
  static const int max_input_image_height = 1080;

  int image_width = 960;
  int image_height = 512;

  uint8_t* input_image_data = nullptr;

 public:
  trt_dnn_inference_heatmap(const std::vector<uint8_t>& model_data, size_t max_views) {
    plugin_handler = dlopen("../data/mvpose/libmmdeploy_tensorrt_ops.so", RTLD_NOW);
    if (!plugin_handler) {
      throw std::runtime_error(dlerror());
    }

    infer = nvinfer1::createInferRuntime(logger);
    engine = infer->deserializeCudaEngine(model_data.data(), model_data.size());
    context = engine->createExecutionContext();

    const auto&& image_size = cv::Size(image_width, image_height);

    const auto input_size = image_size.width * image_size.height * 3 * max_views;
    CUDA_SAFE_CALL(cudaMalloc(&input_data, input_size * sizeof(float)));

    const auto output_size = 15 * 128 * 240 * max_views;
    CUDA_SAFE_CALL(cudaMalloc(&output_data, output_size * sizeof(float)));

    CUDA_SAFE_CALL(cudaMalloc(&input_image_data,
                              max_input_image_width * max_input_image_height * 3 * max_views));

    std::unordered_map<std::string, void*> buffers;
    buffers["input"] = input_data;
    buffers["output"] = output_data;

    for (const auto& [name, buffer] : buffers) {
      if (!context->setTensorAddress(name.c_str(), buffer)) {
        spdlog::error("Failed to set tensor address");
      }
    }

    CUDA_SAFE_CALL(cudaStreamCreate(&stream));
  }

  ~trt_dnn_inference_heatmap() {
    delete context;
    delete engine;
    delete infer;

    dlclose(plugin_handler);

    CUDA_SAFE_CALL(cudaStreamDestroy(stream));

    CUDA_SAFE_CALL(cudaFree(input_data));
    CUDA_SAFE_CALL(cudaFree(input_image_data));
    CUDA_SAFE_CALL(cudaFree(output_data));
  }

  void process(const std::vector<cv::Mat>& images, std::vector<roi_data>& rois) {
    const auto&& image_size = cv::Size(image_width, image_height);

    for (size_t i = 0; i < images.size(); i++) {
      const auto& data = images.at(i);

      const auto get_scale = [](const cv::Size2f& image_size, const cv::Size2f& resized_size) {
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

      const auto input_image_width = data.size().width;
      const auto input_image_height = data.size().height;

      assert(input_image_width <= max_input_image_width);
      assert(input_image_height <= max_input_image_height);

      const auto scale = get_scale(data.size(), image_size);
      const auto center = cv::Point2f(data.size().width / 2.0, data.size().height / 2.0);
      const auto rotation = 0.0;

      const roi_data roi = {{scale.width, scale.height}, rotation, {center.x, center.y}};
      rois.push_back(roi);

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

    inference(images.size());
  }

  void inference(size_t num_views) {
    const auto num_input_outputs = engine->getNbIOTensors();
    for (int i = 0; i < num_input_outputs; i++) {
      const auto name = engine->getIOTensorName(i);
      const auto tensor = engine->getTensorShape(name);

      if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
        nvinfer1::Dims4 input_dims = {tensor.d[0], tensor.d[1], tensor.d[2], tensor.d[3]};

        if (std::string(name) == "input") {
          input_dims.d[0] = num_views;
          input_dims.d[2] = image_height;
          input_dims.d[3] = image_width;
        }

        context->setInputShape(name, input_dims);
      }
    }

    if (context->inferShapes(0, nullptr) != 0) {
      spdlog::error("Failed to infer shapes");
    }

    if (!context->allInputDimensionsSpecified()) {
      spdlog::error("Failed to specify all input dimensions");
    }

    if (!context->enqueueV3(stream)) {
      spdlog::error("Failed to enqueue");
    }

    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
  }

  const float* get_heatmaps() const { return output_data; }

  int get_heatmap_width() const { return 240; }

  int get_heatmap_height() const { return 128; }
};
}  // namespace stargazer::voxelpose
#endif

#ifdef USE_MIGRAPHX

#include <migraphx/migraphx.h>

#include <migraphx/migraphx.hpp>

namespace stargazer::voxelpose {

class mgx_dnn_inference : public dnn_inference {
  std::vector<uint8_t> model_data;

  migraphx::program prog;

  hipStream_t stream;

  float* input_data = nullptr;
  float* output_data = nullptr;

  std::vector<std::string> input_node_names;
  std::vector<std::string> output_node_names;
  std::vector<std::string> output_param_names;

  std::unordered_map<std::string, std::vector<int64_t>> input_node_dims;
  std::unordered_map<std::string, std::vector<int64_t>> output_node_dims;

 public:
  mgx_dnn_inference(const std::string& model_path) {
    spdlog::info("Loading MIGraphX compiled model: {}", model_path);
    prog = migraphx::load(model_path.c_str());

    // Get input parameter shapes
    auto param_shapes = prog.get_parameter_shapes();
    auto param_names = param_shapes.names();
    for (const auto& name : param_names) {
      std::string name_str(name);
      if (name_str.find("#output") != std::string::npos) {
        const auto& shape = param_shapes[name];
        std::vector<int64_t> dims;
        if (!shape.dynamic()) {
          auto lengths = shape.lengths();
          for (size_t i = 0; i < lengths.size(); i++) {
            dims.push_back(lengths[i]);
          }
        }
        output_param_names.push_back(name_str);
        output_node_dims[name_str] = dims;
        continue;
      }

      const auto& shape = param_shapes[name];

      std::vector<int64_t> dims;

      // Dynamic shapes not expected for these models, but check anyway
      if (!shape.dynamic()) {
        auto lengths = shape.lengths();
        for (size_t i = 0; i < lengths.size(); i++) {
          dims.push_back(lengths[i]);
        }
      } else {
        throw std::runtime_error("Dynamic shapes not supported for this model");
      }

      input_node_names.push_back(name_str);
      input_node_dims[name_str] = dims;
    }

    // Get output shapes
    auto output_shapes = prog.get_output_shapes();
    for (size_t i = 0; i < output_shapes.size(); i++) {
      const auto& shape = output_shapes[i];

      std::vector<int64_t> dims;

      // Dynamic shapes not expected for these models, but check anyway
      if (!shape.dynamic()) {
        auto lengths = shape.lengths();
        for (size_t j = 0; j < lengths.size(); j++) {
          dims.push_back(lengths[j]);
        }
      } else {
        throw std::runtime_error("Dynamic shapes not supported for this model");
      }

      // MIGraphX doesn't name outputs, use index
      std::string name = "output";
      output_node_names.push_back(name);
      output_node_dims[name] = dims;
    }

    if (input_node_names.size() != 1) {
      spdlog::error("Expected 1 input, got {}", input_node_names.size());
      throw std::runtime_error("Invalid number of inputs");
    }
    if (output_node_names.size() != 1) {
      spdlog::error("Expected 1 output, got {}", output_node_names.size());
      throw std::runtime_error("Invalid number of outputs");
    }

    HIP_SAFE_CALL(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    {
      const auto dims = input_node_dims.at(input_node_names[0]);
      const auto input_size =
          std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

      malloc_device(&input_data, input_size);
    }

    {
      const auto dims = output_node_dims.at(output_node_names[0]);
      const auto output_size =
          std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

      malloc_device(&output_data, output_size);
    }
  }

  ~mgx_dnn_inference() {
    HIP_SAFE_CALL(hipStreamDestroy(stream));
    free_device(input_data);
    free_device(output_data);
  }

  void inference(const float* input) {
    if (input_node_names.empty() || output_node_names.empty()) {
      throw std::runtime_error("Model not properly initialized");
    }

    const auto dims = input_node_dims.at(input_node_names[0]);
    const auto input_size =
        std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

    migraphx::program_parameters params;
    std::vector<size_t> shape_dims;
    for (auto d : dims) shape_dims.push_back(d);
    migraphx::shape input_shape{migraphx_shape_float_type, shape_dims};

    migraphx::argument input_arg(input_shape, const_cast<float*>(input));
    params.add(input_node_names[0].c_str(), input_arg);

    if (!output_param_names.empty()) {
      const auto& output_dims = output_node_dims.at(output_param_names[0]);
      std::vector<size_t> output_shape_dims;
      for (auto d : output_dims) output_shape_dims.push_back(d);
      migraphx::shape output_shape{migraphx_shape_float_type, output_shape_dims};
      migraphx::argument output_arg(output_shape, output_data);
      params.add(output_param_names[0].c_str(), output_arg);
    }

    prog.run_async(params, stream);
    HIP_SAFE_CALL(hipStreamSynchronize(stream));
  }

  const float* get_output_data() const { return output_data; }
};

class mgx_dnn_inference_heatmap : public dnn_inference_heatmap {
  std::unordered_map<size_t, migraphx::program> prog_map;  // batch_size -> program
  std::unordered_map<size_t, std::unordered_map<std::string, std::vector<int64_t>>> input_dims_map;
  std::unordered_map<size_t, std::unordered_map<std::string, std::vector<int64_t>>> output_dims_map;

  hipStream_t stream;

  float* input_data = nullptr;
  float* output_data = nullptr;

  std::vector<std::string> input_node_names;
  std::vector<std::string> output_node_names;
  std::vector<std::string> output_param_names;

  static const int max_input_image_width = 1920;
  static const int max_input_image_height = 1080;

  int image_width = 960;
  int image_height = 512;

  uint8_t* input_image_data = nullptr;

  size_t max_batch_size;

 public:
  mgx_dnn_inference_heatmap(const std::string& model_path_template, size_t max_views)
      : max_batch_size(max_views) {
    spdlog::info("Loading MIGraphX compiled heatmap models for batch sizes 1-{}", max_views);

    // Load models for each batch size
    for (size_t batch = 1; batch <= max_views; batch++) {
      std::string model_path = model_path_template;
      // Replace {batch} with actual batch number
      size_t pos = model_path.find("{batch}");
      if (pos != std::string::npos) {
        model_path.replace(pos, 7, std::to_string(batch));
      }

      spdlog::info("  Loading batch {} model: {}", batch, model_path);
      prog_map[batch] = migraphx::load(model_path.c_str());

      // Get input/output shapes for this batch size
      auto param_shapes = prog_map[batch].get_parameter_shapes();
      auto param_names = param_shapes.names();
      for (const auto& name : param_names) {
        std::string name_str(name);
        if (name_str.find("#output") != std::string::npos) {
          const auto& shape = param_shapes[name];
          std::vector<int64_t> dims;
          auto lengths = shape.lengths();
          for (size_t i = 0; i < lengths.size(); i++) {
            dims.push_back(lengths[i]);
          }
          output_dims_map[batch][name_str] = dims;
          if (output_param_names.empty()) {
            output_param_names.push_back(name_str);
          }
          continue;
        }

        if (input_node_names.empty()) {
          input_node_names.push_back(name_str);
        }
        const auto& shape = param_shapes[name];
        std::vector<int64_t> dims;
        auto lengths = shape.lengths();
        for (size_t i = 0; i < lengths.size(); i++) {
          dims.push_back(lengths[i]);
        }
        input_dims_map[batch][name_str] = dims;
      }

      auto output_shapes = prog_map[batch].get_output_shapes();
      for (size_t i = 0; i < output_shapes.size(); i++) {
        if (output_node_names.empty()) {
          output_node_names.push_back("output");
        }
        const auto& shape = output_shapes[i];
        std::vector<int64_t> dims;
        auto lengths = shape.lengths();
        for (size_t j = 0; j < lengths.size(); j++) {
          dims.push_back(lengths[j]);
        }
        output_dims_map[batch]["output"] = dims;
      }
    }

    if (input_node_names.size() != 1) {
      spdlog::error("Expected 1 input, got {}", input_node_names.size());
      throw std::runtime_error("Invalid number of inputs");
    }
    if (output_node_names.size() != 1) {
      spdlog::error("Expected 1 output, got {}", output_node_names.size());
      throw std::runtime_error("Invalid number of outputs");
    }

    HIP_SAFE_CALL(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    // Allocate buffers for max batch size
    const auto input_size = 960 * 512 * 3 * max_views;
    malloc_device(&input_data, input_size);

    const auto output_size = 240 * 128 * 15 * max_views;
    malloc_device(&output_data, output_size);

    malloc_device(&input_image_data,
                  max_input_image_width * max_input_image_height * 3 * max_views);
  }

  ~mgx_dnn_inference_heatmap() {
    HIP_SAFE_CALL(hipStreamDestroy(stream));
    free_device(input_data);
    free_device(output_data);
    free_device(input_image_data);
  }

  void process(const std::vector<cv::Mat>& images, std::vector<roi_data>& rois) {
    const auto&& image_size = cv::Size(image_width, image_height);

    for (size_t i = 0; i < images.size(); i++) {
      const auto& data = images.at(i);

      const auto get_scale = [](const cv::Size2f& image_size, const cv::Size2f& resized_size) {
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

      const auto input_image_width = data.size().width;
      const auto input_image_height = data.size().height;

      assert(input_image_width <= max_input_image_width);
      assert(input_image_height <= max_input_image_height);

      const auto scale = get_scale(data.size(), image_size);
      const auto center = cv::Point2f(data.size().width / 2.0, data.size().height / 2.0);
      const auto rotation = 0.0;

      roi_data roi = {{scale.width, scale.height}, rotation, {center.x, center.y}};
      rois.push_back(roi);

      const std::array<float, 3> mean = {0.485, 0.456, 0.406};
      const std::array<float, 3> std = {0.229, 0.224, 0.225};

      HIP_SAFE_CALL(hipMemcpy2DAsync(
          input_image_data + i * input_image_width * 3 * input_image_height, input_image_width * 3,
          data.data, data.step, data.cols * 3, data.rows, hipMemcpyHostToDevice, stream));

      preprocess_cuda(input_image_data + i * input_image_width * 3 * input_image_height,
                      input_image_width, input_image_height, input_image_width * 3,
                      input_data + i * 960 * 512 * 3, 960, 512, 960, mean, std);
    }

    HIP_SAFE_CALL(hipStreamSynchronize(stream));

    inference(images.size());
  }

  void inference(size_t num_views) {
    if (input_node_names.empty() || output_node_names.empty()) {
      throw std::runtime_error("Model not properly initialized");
    }

    // Clamp num_views to available models
    size_t batch_size = std::min(num_views, max_batch_size);
    if (batch_size == 0) batch_size = 1;

    // Get the appropriate program for this batch size
    auto& prog = prog_map.at(batch_size);
    auto dims = input_dims_map.at(batch_size).at(input_node_names[0]);

    migraphx::program_parameters params;
    std::vector<size_t> shape_dims;
    for (auto d : dims) shape_dims.push_back(d);
    migraphx::shape input_shape{migraphx_shape_float_type, shape_dims};

    const auto input_size =
        std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

    migraphx::argument input_arg(input_shape, input_data);
    params.add(input_node_names[0].c_str(), input_arg);

    if (!output_param_names.empty()) {
      const auto& batch_output_dims = output_dims_map.at(batch_size);
      if (batch_output_dims.find(output_param_names[0]) != batch_output_dims.end()) {
        const auto& output_dims = batch_output_dims.at(output_param_names[0]);
        std::vector<size_t> output_shape_dims;
        for (auto d : output_dims) output_shape_dims.push_back(d);
        migraphx::shape output_shape{migraphx_shape_float_type, output_shape_dims};
        migraphx::argument output_arg(output_shape, output_data);
        params.add(output_param_names[0].c_str(), output_arg);
      } else {
        spdlog::error("Output parameter {} not found in batch {} dimensions", output_param_names[0],
                      batch_size);
      }
    } else {
      spdlog::info("No output parameters (offload_copy might be True)");
    }

    prog.run_async(params, stream);
    HIP_SAFE_CALL(hipStreamSynchronize(stream));
  }

  const float* get_heatmaps() const { return output_data; }

  int get_heatmap_width() const { return 240; }

  int get_heatmap_height() const { return 128; }
};
}  // namespace stargazer::voxelpose
#endif

namespace stargazer::voxelpose {
class cv_dnn_inference : public dnn_inference {
  cv::dnn::Net net;

  std::vector<std::string> input_node_names;
  std::vector<std::string> output_node_names;

  std::unordered_map<std::string, std::vector<int64_t>> input_node_dims;
  std::unordered_map<std::string, std::vector<int64_t>> output_node_dims;

  std::vector<float> input_data;
  float* output_data = nullptr;

 public:
  cv_dnn_inference(const std::vector<uint8_t>& model_data) {
    const auto backend = cv::dnn::getAvailableBackends();
    net = cv::dnn::readNetFromONNX(model_data);

    std::vector<cv::dnn::MatShape> input_layer_shapes;
    std::vector<cv::dnn::MatShape> output_layer_shapes;
    net.getLayerShapes(cv::dnn::MatShape(), 0, input_layer_shapes, output_layer_shapes);

    assert(input_layer_shapes.size() == 1);
    assert(output_layer_shapes.size() == 1);

    const auto input_size = std::accumulate(
        input_layer_shapes[0].begin(), input_layer_shapes[0].end(), 1, std::multiplies<int64_t>());
    input_data.resize(input_size);

    const auto output_size =
        std::accumulate(output_layer_shapes[0].begin(), output_layer_shapes[0].end(), 1,
                        std::multiplies<int64_t>());
    malloc_device(&output_data, output_size);
  }

  void inference(const float* input) {
    std::vector<cv::dnn::MatShape> input_layer_shapes;
    std::vector<cv::dnn::MatShape> output_layer_shapes;
    net.getLayerShapes(cv::dnn::MatShape(), 0, input_layer_shapes, output_layer_shapes);

    assert(input_layer_shapes.size() == 1);
    assert(output_layer_shapes.size() == 1);

    memcpy_dtoh(input_data.data(), input, input_data.size() * sizeof(float));

    cv::Mat input_mat(input_layer_shapes[0], CV_32FC1, (void*)input_data.data());
    net.setInput(input_mat);
    const auto output_mat = net.forward();

    memcpy_htod(output_data, output_mat.data, output_mat.total() * sizeof(float));
  }

  const float* get_output_data() const { return output_data; }
};

class cv_dnn_inference_heatmap : public dnn_inference_heatmap {
  cv::dnn::Net net;

  std::vector<std::string> input_node_names;
  std::vector<std::string> output_node_names;

  std::unordered_map<std::string, std::vector<int64_t>> input_node_dims;
  std::unordered_map<std::string, std::vector<int64_t>> output_node_dims;

  float* input_data = nullptr;
  float* output_data = nullptr;
  std::vector<float> input_data_cpu;

  static const int max_input_image_width = 1920;
  static const int max_input_image_height = 1080;

  int image_width = 960;
  int image_height = 512;

  uint8_t* input_image_data = nullptr;

 public:
  cv_dnn_inference_heatmap(const std::vector<uint8_t>& model_data, size_t max_views) {
    const auto backends = cv::dnn::getAvailableBackends();
    net = cv::dnn::readNetFromONNX(model_data);

    const auto input_size = image_width * image_height * 3 * max_views;
    malloc_device(&input_data, input_size);
    input_data_cpu.resize(input_size);

    const auto output_size = 240 * 128 * 15 * max_views;
    malloc_device(&output_data, output_size);

    malloc_device(&input_image_data,
                  max_input_image_width * max_input_image_height * 3 * max_views);
  }

  ~cv_dnn_inference_heatmap() {}

  void process(const std::vector<cv::Mat>& images, std::vector<roi_data>& rois) {
    const auto&& image_size = cv::Size(image_width, image_height);

    for (size_t i = 0; i < images.size(); i++) {
      const auto& data = images.at(i);

      const auto get_scale = [](const cv::Size2f& image_size, const cv::Size2f& resized_size) {
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

      const auto input_image_width = data.size().width;
      const auto input_image_height = data.size().height;

      assert(input_image_width <= max_input_image_width);
      assert(input_image_height <= max_input_image_height);

      const auto scale = get_scale(data.size(), image_size);
      const auto center = cv::Point2f(data.size().width / 2.0, data.size().height / 2.0);
      const auto rotation = 0.0;

      roi_data roi = {{scale.width, scale.height}, rotation, {center.x, center.y}};
      rois.push_back(roi);

      const std::array<float, 3> mean = {0.485, 0.456, 0.406};
      const std::array<float, 3> std = {0.229, 0.224, 0.225};

      memcpy2d_htod(input_image_data + i * input_image_width * 3 * input_image_height,
                    input_image_width * 3, data.data, data.step, data.cols * 3, data.rows);

      preprocess_cuda(input_image_data + i * input_image_width * 3 * input_image_height,
                      input_image_width, input_image_height, input_image_width * 3,
                      input_data + i * image_width * image_height * 3, image_width, image_height,
                      image_width, mean, std);
    }

    synchronize_device();

    inference(images.size());
  }

  void inference(size_t num_views) {
    memcpy_dtoh(input_data_cpu.data(), input_data,
                num_views * image_width * image_height * 3 * sizeof(float));

    const cv::dnn::MatShape input_shape = {static_cast<int>(num_views), 3, image_height,
                                           image_width};

    cv::Mat input_mat(input_shape, CV_32FC1, (void*)input_data_cpu.data());
    net.setInput(input_mat);
    const auto output_mat = net.forward();

    memcpy_htod(output_data, output_mat.data, output_mat.total() * sizeof(float));
  }

  const float* get_heatmaps() const { return output_data; }

  int get_heatmap_width() const { return 240; }

  int get_heatmap_height() const { return 128; }
};
}  // namespace stargazer::voxelpose

namespace stargazer::voxelpose {
class get_proposal {
  uint32_t max_num;
  float threshold;
  std::array<float, 3> grid_size;
  std::array<float, 3> grid_center;
  std::array<int32_t, 3> cube_size;

 public:
  get_proposal() {}

  void set_max_num(uint32_t value) { max_num = value; }
  void set_threshold(float value) { threshold = value; }
  std::array<float, 3> get_grid_size() const { return grid_size; }
  void set_grid_size(const std::array<float, 3>& value) { grid_size = value; }
  std::array<float, 3> get_grid_center() const { return grid_center; }
  void set_grid_center(const std::array<float, 3>& value) { grid_center = value; }
  std::array<int32_t, 3> get_cube_size() const { return cube_size; }
  void set_cube_size(const std::array<int32_t, 3>& value) { cube_size = value; }

  static coalsack::tensor<float, 4> max_pool(const coalsack::tensor<float, 4>& inputs,
                                             size_t kernel = 3) {
    const auto padding = (kernel - 1) / 2;
    const auto max = inputs.max_pool3d(kernel, 1, padding, 1);
    const auto keep = inputs.transform(max, [](const float value1, const float value2, auto...) {
      return value1 == value2 ? value1 : 0.f;
    });
    return keep;
  }

  static coalsack::tensor<uint64_t, 2> get_index(const coalsack::tensor<uint64_t, 1>& indices,
                                                 const std::array<uint64_t, 3>& shape) {
    const auto result = indices.transform_expand<1>({3}, [shape](const uint64_t value, auto...) {
      const auto index_x = value / (shape[1] * shape[0]);
      const auto index_y = value % (shape[1] * shape[0]) / shape[0];
      const auto index_z = value % shape[0];
      return std::array<uint64_t, 3>{index_x, index_y, index_z};
    });
    return result;
  }

  coalsack::tensor<float, 2> get_real_loc(const coalsack::tensor<uint64_t, 2>& index) {
    const auto loc =
        index.cast<float>().transform([this](const float value, const size_t i, auto...) {
          return value / (cube_size[i] - 1) * grid_size[i] + grid_center[i] - grid_size[i] / 2.0f;
        });
    return loc;
  }

  coalsack::tensor<float, 2> get_centers(const coalsack::tensor<float, 5>& src) {
    const auto root_cubes =
        src.view<4>({src.shape[0], src.shape[1], src.shape[2], src.shape[3]}).contiguous();
    const auto root_cubes_nms = max_pool(root_cubes);
    const auto [topk_values, topk_index] =
        root_cubes_nms.view<1>({src.shape[0] * src.shape[1] * src.shape[2]}).topk(max_num);

    const auto topk_unravel_index =
        get_index(topk_index, {src.shape[0], src.shape[1], src.shape[2]});
    const auto topk_loc = get_real_loc(topk_unravel_index);

    auto grid_centers = coalsack::tensor<float, 2>::zeros({5, max_num});
    grid_centers.view({3, grid_centers.shape[1]}, {0, 0})
        .assign(topk_loc.view(), [](auto, const float value, auto...) { return value; });
    grid_centers.view<1>({0, grid_centers.shape[1]}, {4, 0})
        .assign(topk_values.view(), [](auto, const float value, auto...) { return value; });
    grid_centers.view<1>({0, grid_centers.shape[1]}, {3, 0})
        .assign(topk_values.view(), [this](auto, const float value, auto...) {
          return (value > threshold ? 1.f : 0.f) - 1.f;
        });

    return grid_centers;
  }
};

#if !defined(USE_MIGRAPHX)
static void load_model(std::string model_path, std::vector<uint8_t>& data) {
  std::ifstream ifs;
  ifs.open(model_path, std::ios_base::in | std::ios_base::binary);
  if (ifs.fail()) {
    std::cerr << "File open error: " << model_path << "\n";
    std::quick_exit(0);
  }

  ifs.seekg(0, std::ios::end);
  const auto length = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  data.resize(length);

  ifs.read((char*)data.data(), length);
  if (ifs.fail()) {
    std::cerr << "File read error: " << model_path << "\n";
    std::quick_exit(0);
  }
}
#endif

voxelpose::voxelpose()
    : inference_heatmap(),
      inference_proposal(),
      inference_pose(),
      global_proj(new voxel_projector()),
      local_proj(new voxel_projector()),
      prop(new get_proposal()),
      joint_extract(new joint_extractor()),
      grid_center({0.0, 0.0, 0.0}),
      grid_size({8000.0, 8000.0, 2000.0}) {
#ifdef USE_MIGRAPHX
  inference_heatmap.reset(
      new mgx_dnn_inference_heatmap("../data/voxelpose/backbone_fp16_b{batch}.mxr", 5));
#elif defined(USE_TENSORRT)
  std::vector<uint8_t> backbone_model_data;
  {
    const auto model_path = "../data/voxelpose/backbone-fp16.trt";
    std::vector<uint8_t> data;
    load_model(model_path, data);

    backbone_model_data = std::move(data);
  }

  inference_heatmap.reset(new trt_dnn_inference_heatmap(backbone_model_data, 5));
#elif defined(ENABLE_ONNXRUNTIME)
  std::vector<uint8_t> backbone_model_data;
  {
    const auto model_path = "../data/voxelpose/backbone.onnx";
    std::vector<uint8_t> data;
    load_model(model_path, data);

    backbone_model_data = std::move(data);
  }

  inference_heatmap.reset(new ort_dnn_inference_heatmap(backbone_model_data, 5));
#else
  std::vector<uint8_t> backbone_model_data;
  {
    const auto model_path = "../data/voxelpose/backbone.onnx";
    std::vector<uint8_t> data;
    load_model(model_path, data);

    backbone_model_data = std::move(data);
  }

  inference_heatmap.reset(new cv_dnn_inference_heatmap(backbone_model_data, 5));
#endif

#ifdef USE_MIGRAPHX
  inference_proposal.reset(new mgx_dnn_inference("../data/voxelpose/proposal_v2v_net_fp16.mxr"));
#elif defined(USE_TENSORRT)
  std::vector<uint8_t> proposal_v2v_net_model_data;
  {
    const auto model_path = "../data/voxelpose/proposal_v2v_net-fp16.trt";
    std::vector<uint8_t> data;
    load_model(model_path, data);

    proposal_v2v_net_model_data = std::move(data);
  }

  inference_proposal.reset(new trt_dnn_inference(proposal_v2v_net_model_data, 1));
#elif defined(ENABLE_ONNXRUNTIME)
  std::vector<uint8_t> proposal_v2v_net_model_data;
  {
    const auto model_path = "../data/voxelpose/proposal_v2v_net.onnx";
    std::vector<uint8_t> data;
    load_model(model_path, data);

    proposal_v2v_net_model_data = std::move(data);
  }

  inference_proposal.reset(new ort_dnn_inference(proposal_v2v_net_model_data));
#else
  std::vector<uint8_t> proposal_v2v_net_model_data;
  {
    const auto model_path = "../data/voxelpose/proposal_v2v_net.onnx";
    std::vector<uint8_t> data;
    load_model(model_path, data);

    proposal_v2v_net_model_data = std::move(data);
  }

  inference_proposal.reset(new cv_dnn_inference(proposal_v2v_net_model_data));
#endif

#ifdef USE_MIGRAPHX
  inference_pose.reset(new mgx_dnn_inference("../data/voxelpose/pose_v2v_net_fp16.mxr"));
#elif defined(USE_TENSORRT)
  std::vector<uint8_t> pose_v2v_net_model_data;
  {
    const auto model_path = "../data/voxelpose/pose_v2v_net-fp16.trt";
    std::vector<uint8_t> data;
    load_model(model_path, data);

    pose_v2v_net_model_data = std::move(data);
  }

  inference_pose.reset(new trt_dnn_inference(pose_v2v_net_model_data, 1));
#elif defined(ENABLE_ONNXRUNTIME)
  std::vector<uint8_t> pose_v2v_net_model_data;
  {
    const auto model_path = "../data/voxelpose/pose_v2v_net.onnx";
    std::vector<uint8_t> data;
    load_model(model_path, data);

    pose_v2v_net_model_data = std::move(data);
  }

  inference_pose.reset(new ort_dnn_inference(pose_v2v_net_model_data));
#else
  std::vector<uint8_t> pose_v2v_net_model_data;
  {
    const auto model_path = "../data/voxelpose/pose_v2v_net.onnx";
    std::vector<uint8_t> data;
    load_model(model_path, data);

    pose_v2v_net_model_data = std::move(data);
  }

  inference_pose.reset(new cv_dnn_inference(pose_v2v_net_model_data));
#endif
}

voxelpose::~voxelpose() = default;

std::array<int32_t, 3> voxelpose::get_cube_size() const {
  std::array<int32_t, 3> cube_size = {80, 80, 20};
  return cube_size;
}
std::array<float, 3> voxelpose::get_grid_size() const { return grid_size; }
std::array<float, 3> voxelpose::get_grid_center() const { return grid_center; }
void voxelpose::set_grid_size(const std::array<float, 3>& value) { grid_size = value; }
void voxelpose::set_grid_center(const std::array<float, 3>& value) { grid_center = value; }

std::vector<glm::vec3> voxelpose::inference(const std::vector<cv::Mat>& images_list,
                                            const std::vector<camera_data>& cameras_list) {
  const auto cube_size = get_cube_size();

  std::vector<roi_data> rois_list;
  inference_heatmap->process(images_list, rois_list);

  global_proj->set_grid_size(grid_size);
  global_proj->set_cube_size(cube_size);

  global_proj->get_voxel(
      inference_heatmap->get_heatmaps(), images_list.size(), inference_heatmap->get_heatmap_width(),
      inference_heatmap->get_heatmap_height(), cameras_list, rois_list, grid_center);

  inference_proposal->inference(global_proj->get_cubes());

  coalsack::tensor<float, 5> proposal({20, 80, 80, 1, 1});
#if defined(USE_CUDA)
  CUDA_SAFE_CALL(cudaMemcpy(proposal.get_data(), inference_proposal->get_output_data(),
                            proposal.get_size() * sizeof(float), cudaMemcpyDeviceToHost));
#elif defined(USE_HIP) || defined(USE_MIGRAPHX)
  HIP_SAFE_CALL(hipMemcpy(proposal.get_data(), inference_proposal->get_output_data(),
                          proposal.get_size() * sizeof(float), hipMemcpyDeviceToHost));
#endif

  prop->set_max_num(10);
  prop->set_threshold(0.3f);
  prop->set_grid_size(grid_size);
  prop->set_grid_center(grid_center);
  prop->set_cube_size(cube_size);

  const auto centers = prop->get_centers(proposal);

  std::vector<glm::vec3> points;

  for (uint32_t i = 0; i < centers.shape[1]; i++) {
    const auto score = centers.get({4, i});
    if (score > 0.3f) {
      const std::array<float, 3> center = {centers.get({0, i}), centers.get({1, i}),
                                           centers.get({2, i})};

      std::array<int32_t, 3> cube_size = {64, 64, 64};
      std::array<float, 3> grid_size = {2000.0, 2000.0, 2000.0};

      local_proj->set_grid_size(grid_size);
      local_proj->set_cube_size(cube_size);

      local_proj->get_voxel(inference_heatmap->get_heatmaps(), images_list.size(),
                            inference_heatmap->get_heatmap_width(),
                            inference_heatmap->get_heatmap_height(), cameras_list, rois_list,
                            center);

      inference_pose->inference(local_proj->get_cubes());

      joint_extract->soft_argmax(inference_pose->get_output_data(), 100, grid_size, cube_size,
                                 center);

      std::vector<glm::vec3> joints(15);

#if defined(USE_CUDA)
      CUDA_SAFE_CALL(cudaMemcpy(&joints[0][0], joint_extract->get_joints(), 3 * 15 * sizeof(float),
                                cudaMemcpyDeviceToHost));
#elif defined(USE_HIP) || defined(USE_MIGRAPHX)
      HIP_SAFE_CALL(hipMemcpy(&joints[0][0], joint_extract->get_joints(), 3 * 15 * sizeof(float),
                              hipMemcpyDeviceToHost));
#endif

      glm::mat4 basis(1.f);
      basis[0] = glm::vec4(1.f, 0.f, 0.f, 0.f);
      basis[1] = glm::vec4(0.f, 0.f, -1.f, 0.f);
      basis[2] = glm::vec4(0.f, 1.f, 0.f, 0.f);

      for (const auto& joint : joints) {
        points.push_back(basis * glm::vec4(joint / 1000.0f, 1.0f));
      }
    }
  }

  return points;
}

const float* voxelpose::get_heatmaps() const { return inference_heatmap->get_heatmaps(); }

void voxelpose::copy_heatmap_to(size_t num_views, float* data) const {
#if defined(USE_CUDA)
  const auto heatmap_size =
      get_heatmap_width() * get_heatmap_height() * get_num_joints() * num_views;
  CUDA_SAFE_CALL(
      cudaMemcpy(data, get_heatmaps(), heatmap_size * sizeof(float), cudaMemcpyDeviceToHost));
#elif defined(USE_HIP) || defined(USE_MIGRAPHX)
  const auto heatmap_size =
      get_heatmap_width() * get_heatmap_height() * get_num_joints() * num_views;
  HIP_SAFE_CALL(
      hipMemcpy(data, get_heatmaps(), heatmap_size * sizeof(float), hipMemcpyDeviceToHost));
#endif
}
uint32_t voxelpose::get_heatmap_width() const { return inference_heatmap->get_heatmap_width(); }
uint32_t voxelpose::get_heatmap_height() const { return inference_heatmap->get_heatmap_height(); }
uint32_t voxelpose::get_num_joints() const { return 15; }
}  // namespace stargazer::voxelpose