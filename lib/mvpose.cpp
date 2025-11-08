#include "mvpose.hpp"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>
#include <glm/ext.hpp>
#include <iostream>
#include <numeric>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <dlfcn.h>

#include "preprocess.hpp"

using namespace stargazer;

#ifdef USE_CUDA

#include <NvInfer.h>
#include <cuda_runtime.h>

class Logger : public nvinfer1::ILogger {
 public:
  void log(ILogger::Severity severity, const char *msg) noexcept override {
    if (severity == nvinfer1::ILogger::Severity::kINFO) {
      spdlog::info(msg);
    } else if (severity == nvinfer1::ILogger::Severity::kERROR) {
      spdlog::error(msg);
    }
  }
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

#define ENABLE_ONNXRUNTIME

#ifdef ENABLE_ONNXRUNTIME
#include <cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h>

namespace stargazer::mvpose {
class dnn_inference_pose {
  std::vector<uint8_t> model_data;

  Ort::Env env{ORT_LOGGING_LEVEL_WARNING};
  Ort::Session session;
  Ort::IoBinding io_binding;
  Ort::MemoryInfo info_cuda{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};

  float *input_data = nullptr;
  float *simcc_x_data = nullptr;
  float *simcc_y_data = nullptr;

  std::vector<std::string> input_node_names;
  std::vector<std::string> output_node_names;

  std::unordered_map<std::string, std::vector<int64_t>> input_node_dims;
  std::unordered_map<std::string, std::vector<int64_t>> output_node_dims;

  uint8_t *input_image_data = nullptr;

 public:
  static const int max_input_image_width = 1920;
  static const int max_input_image_height = 1080;

  static const int image_width = 288;
  static const int image_height = 384;

  static const auto num_joints = 133;

  dnn_inference_pose(const std::vector<uint8_t> &model_data, size_t max_batch_size)
      : session(nullptr), io_binding(nullptr) {
    // Create session
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    try {
      OrtCUDAProviderOptions cuda_options{};

      cuda_options.device_id = 0;
      cuda_options.arena_extend_strategy = 1;
      cuda_options.gpu_mem_limit = 2ULL * 1024 * 1024 * 1024;
      cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
      cuda_options.do_copy_in_default_stream = 1;

      session_options.AppendExecutionProvider_CUDA(cuda_options);
    } catch (const Ort::Exception &e) {
      spdlog::info(e.what());
    }

    OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 0);

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

    assert(output_node_names.size() == 2);
    assert(output_node_names[0] == "simcc_x");
    assert(output_node_names[1] == "simcc_y");

    const auto &&image_size = cv::Size(image_width, image_height);

    const auto input_size = image_size.width * image_size.height * 3 * max_batch_size;
    CUDA_SAFE_CALL(cudaMalloc(&input_data, input_size * sizeof(float)));

    const auto simcc_x_size = image_size.width * 2 * num_joints * max_batch_size;
    CUDA_SAFE_CALL(cudaMalloc(&simcc_x_data, simcc_x_size * sizeof(float)));

    const auto simcc_y_size = image_size.height * 2 * num_joints * max_batch_size;
    CUDA_SAFE_CALL(cudaMalloc(&simcc_y_data, simcc_y_size * sizeof(float)));

    CUDA_SAFE_CALL(cudaMalloc(&input_image_data,
                              max_input_image_width * max_input_image_height * 3 * max_batch_size));
  }

  ~dnn_inference_pose() {
    CUDA_SAFE_CALL(cudaFree(input_data));
    CUDA_SAFE_CALL(cudaFree(input_image_data));
    CUDA_SAFE_CALL(cudaFree(simcc_x_data));
    CUDA_SAFE_CALL(cudaFree(simcc_y_data));
  }

  void process(const cv::Mat &image, const cv::Rect2f &rect, roi_data &roi) {
    const auto &&image_size = cv::Size(image_width, image_height);

    {
      const auto &data = image;

      const auto get_scale = [](const cv::Size2f &image_size, const cv::Size2f &resized_size) {
        float w_pad, h_pad;
        if (image_size.width / resized_size.width < image_size.height / resized_size.height) {
          w_pad = image_size.height / resized_size.height * resized_size.width;
          h_pad = image_size.height;
        } else {
          w_pad = image_size.width;
          h_pad = image_size.width / resized_size.width * resized_size.height;
        }

        return cv::Size2f(w_pad * 1.2 / 200.0, h_pad * 1.2 / 200.0);
      };

      const auto input_image_width = data.size().width;
      const auto input_image_height = data.size().height;

      assert(input_image_width <= max_input_image_width);
      assert(input_image_height <= max_input_image_height);

      const auto scale = get_scale(rect.size(), image_size);
      const auto center =
          cv::Point2f(rect.x + rect.size().width / 2.0, rect.y + rect.size().height / 2.0);
      const auto rotation = 0.0;

      roi = {{scale.width, scale.height}, rotation, {center.x, center.y}};

      const auto trans = get_transform(center, scale, image_size);

      const std::array<float, 3> mean = {0.485, 0.456, 0.406};
      const std::array<float, 3> std = {0.229, 0.224, 0.225};

      CUDA_SAFE_CALL(cudaMemcpy2D(input_image_data, input_image_width * 3, data.data, data.step,
                                  data.cols * 3, data.rows, cudaMemcpyHostToDevice));

      preprocess_cuda(input_image_data, input_image_width, input_image_height,
                      input_image_width * 3, input_data, image_size.width, image_size.height,
                      image_size.width, trans, mean, std);
    }

    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    inference();
  }

  void inference() {
    assert(input_node_names.size() == 1);
    assert(input_node_names[0] == "input");

    assert(output_node_names.size() == 2);
    assert(output_node_names[0] == "simcc_x");
    assert(output_node_names[1] == "simcc_y");

    std::vector<const char *> input_node_names;
    {
      input_node_names.push_back(this->input_node_names[0].c_str());
    }

    std::vector<const char *> output_node_names;
    {
      output_node_names.push_back(this->output_node_names[0].c_str());
      output_node_names.push_back(this->output_node_names[1].c_str());
    }

    io_binding.ClearBoundInputs();
    io_binding.ClearBoundOutputs();

    std::vector<Ort::Value> input_tensors;
    {
      auto dims = input_node_dims.at(input_node_names[0]);
      dims[0] = 1;

      const auto input_size =
          std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(info_cuda, input_data, input_size,
                                                                dims.data(), dims.size());

      io_binding.BindInput(input_node_names[0], input_tensor);

      input_tensors.emplace_back(std::move(input_tensor));
    }

    std::vector<Ort::Value> output_tensors;
    {
      auto dims = output_node_dims.at(output_node_names[0]);
      dims[0] = 1;
      dims[1] = num_joints;
      const auto simcc_x_size =
          std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

      Ort::Value simcc_x_tensor = Ort::Value::CreateTensor<float>(
          info_cuda, simcc_x_data, simcc_x_size, dims.data(), dims.size());

      io_binding.BindOutput(output_node_names[0], simcc_x_tensor);

      output_tensors.emplace_back(std::move(simcc_x_tensor));
    }

    {
      auto dims = output_node_dims.at(output_node_names[1]);
      dims[0] = 1;
      dims[1] = num_joints;
      const auto simcc_y_size =
          std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

      Ort::Value simcc_y_tensor = Ort::Value::CreateTensor<float>(
          info_cuda, simcc_y_data, simcc_y_size, dims.data(), dims.size());

      io_binding.BindOutput(output_node_names[1], simcc_y_tensor);

      output_tensors.emplace_back(std::move(simcc_y_tensor));
    }

    io_binding.SynchronizeInputs();

    session.Run(Ort::RunOptions{nullptr}, io_binding);

    io_binding.SynchronizeOutputs();
  }

  void copy_simcc_x_to_cpu(float *simcc_x) const {
    CUDA_SAFE_CALL(cudaMemcpy(simcc_x, simcc_x_data, image_width * 2 * num_joints * sizeof(float),
                              cudaMemcpyDeviceToHost));
  }

  void copy_simcc_y_to_cpu(float *simcc_y) const {
    CUDA_SAFE_CALL(cudaMemcpy(simcc_y, simcc_y_data, image_height * 2 * num_joints * sizeof(float),
                              cudaMemcpyDeviceToHost));
  }

  const float *get_simcc_x() const { return simcc_x_data; }

  const float *get_simcc_y() const { return simcc_y_data; }
};

class dnn_inference_pose_trt {
  std::vector<uint8_t> model_data;

  void *plugin_handler;
  Logger logger;
  nvinfer1::IRuntime *infer;
  nvinfer1::ICudaEngine *engine;
  nvinfer1::IExecutionContext *context;

  cudaStream_t stream;

  float *input_data = nullptr;
  float *simcc_x_data = nullptr;
  float *simcc_y_data = nullptr;

  uint8_t *input_image_data = nullptr;

 public:
  static const int max_input_image_width = 1920;
  static const int max_input_image_height = 1080;

  static const int image_width = 288;
  static const int image_height = 384;

  static const auto num_joints = 133;

  size_t max_batch_size = 4;
  size_t max_num_people;

  dnn_inference_pose_trt(const std::vector<uint8_t> &model_data, size_t max_num_people)
      : max_num_people(max_num_people) {
    plugin_handler = dlopen("libmmdeploy_tensorrt_ops.so", RTLD_NOW);
    if (!plugin_handler) {
      throw std::runtime_error(dlerror());
    }

    infer = nvinfer1::createInferRuntime(logger);
    engine = infer->deserializeCudaEngine(model_data.data(), model_data.size());
    context = engine->createExecutionContext();

    const auto &&image_size = cv::Size(image_width, image_height);

    const auto input_size = image_size.width * image_size.height * 3 * max_num_people;
    CUDA_SAFE_CALL(cudaMalloc(&input_data, input_size * sizeof(float)));

    const auto simcc_x_size = image_size.width * 2 * num_joints * max_num_people;
    CUDA_SAFE_CALL(cudaMalloc(&simcc_x_data, simcc_x_size * sizeof(float)));

    const auto simcc_y_size = image_size.height * 2 * num_joints * max_num_people;
    CUDA_SAFE_CALL(cudaMalloc(&simcc_y_data, simcc_y_size * sizeof(float)));

    CUDA_SAFE_CALL(cudaMalloc(&input_image_data,
                              max_input_image_width * max_input_image_height * 3 * max_num_people));

    CUDA_SAFE_CALL(cudaStreamCreate(&stream));
  }

  ~dnn_inference_pose_trt() {
    delete context;
    delete engine;
    delete infer;

    dlclose(plugin_handler);

    CUDA_SAFE_CALL(cudaStreamDestroy(stream));

    CUDA_SAFE_CALL(cudaFree(input_data));
    CUDA_SAFE_CALL(cudaFree(input_image_data));
    CUDA_SAFE_CALL(cudaFree(simcc_x_data));
    CUDA_SAFE_CALL(cudaFree(simcc_y_data));
  }

  void process(const std::vector<cv::Mat> &images,
               const std::vector<std::vector<cv::Rect2f>> &rects,
               std::vector<std::vector<roi_data>> &rois) {
    const auto &&image_size = cv::Size(image_width, image_height);

    rois.resize(images.size());

    size_t k = 0;

    for (size_t i = 0; i < images.size(); i++) {
      if (k >= max_num_people) {
        spdlog::error("Batch size is too large");
        break;
      }

      const auto &data = images.at(i);

      rois.at(i).resize(rects.at(i).size());

      const auto input_image_width = data.size().width;
      const auto input_image_height = data.size().height;

      assert(input_image_width <= max_input_image_width);
      assert(input_image_height <= max_input_image_height);

      CUDA_SAFE_CALL(cudaMemcpy2D(input_image_data + i * input_image_width * 3 * input_image_height,
                                  input_image_width * 3, data.data, data.step, data.cols * 3,
                                  data.rows, cudaMemcpyHostToDevice));

      for (size_t j = 0; j < rects.at(i).size(); j++, k++) {
        if (k >= max_num_people) {
          spdlog::error("Batch size is too large");
          break;
        }
        const auto &rect = rects.at(i).at(j);

        const auto get_scale = [](const cv::Size2f &image_size, const cv::Size2f &resized_size) {
          float w_pad, h_pad;
          if (image_size.width / resized_size.width < image_size.height / resized_size.height) {
            w_pad = image_size.height / resized_size.height * resized_size.width;
            h_pad = image_size.height;
          } else {
            w_pad = image_size.width;
            h_pad = image_size.width / resized_size.width * resized_size.height;
          }

          return cv::Size2f(w_pad * 1.2 / 200.0, h_pad * 1.2 / 200.0);
        };

        const auto scale = get_scale(rect.size(), image_size);
        const auto center =
            cv::Point2f(rect.x + rect.size().width / 2.0, rect.y + rect.size().height / 2.0);
        const auto rotation = 0.0;

        const roi_data roi = {{scale.width, scale.height}, rotation, {center.x, center.y}};
        rois.at(i).at(j) = roi;

        const auto trans = get_transform(center, scale, image_size);

        const std::array<float, 3> mean = {0.485, 0.456, 0.406};
        const std::array<float, 3> std = {0.229, 0.224, 0.225};

        preprocess_cuda(input_image_data + i * input_image_width * 3 * input_image_height,
                        input_image_width, input_image_height, input_image_width * 3,
                        input_data + k * image_width * image_height * 3, image_size.width,
                        image_size.height, image_size.width, trans, mean, std);
      }
    }

    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    inference(k);
  }

  void inference(size_t num_batch) {
    for (size_t k = 0; k < num_batch; k += max_batch_size) {
      const auto num_input_outputs = engine->getNbIOTensors();
      for (int i = 0; i < num_input_outputs; i++) {
        const auto name = engine->getIOTensorName(i);
        const auto tensor = engine->getTensorShape(name);

        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
          nvinfer1::Dims4 input_dims = {tensor.d[0], tensor.d[1], tensor.d[2], tensor.d[3]};

          if (std::string(name) == "input") {
            input_dims.d[0] = std::min(num_batch - k, max_batch_size);
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

      std::unordered_map<std::string, void *> buffers;
      buffers["input"] = input_data + k * image_width * image_height * 3;
      buffers["simcc_x"] = simcc_x_data + k * image_width * 2 * num_joints;
      buffers["simcc_y"] = simcc_y_data + k * image_height * 2 * num_joints;

      for (const auto &[name, buffer] : buffers) {
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

  void copy_simcc_x_to_cpu(float *simcc_x, size_t i) const {
    if (i >= max_num_people) {
      spdlog::error("Batch size is too large");
      return;
    }
    CUDA_SAFE_CALL(cudaMemcpy(simcc_x, simcc_x_data + i * image_width * 2 * num_joints,
                              image_width * 2 * num_joints * sizeof(float),
                              cudaMemcpyDeviceToHost));
  }

  void copy_simcc_y_to_cpu(float *simcc_y, size_t i) const {
    if (i >= max_num_people) {
      spdlog::error("Batch size is too large");
      return;
    }
    CUDA_SAFE_CALL(cudaMemcpy(simcc_y, simcc_y_data + i * image_height * 2 * num_joints,
                              image_height * 2 * num_joints * sizeof(float),
                              cudaMemcpyDeviceToHost));
  }
};

class dnn_inference_det {
  std::vector<uint8_t> model_data;

  Ort::Env env{ORT_LOGGING_LEVEL_WARNING};
  Ort::Session session;
  Ort::IoBinding io_binding;
  Ort::MemoryInfo info_cuda{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};

  float *input_data = nullptr;
  float *dets_data = nullptr;
  int64_t *labels_data = nullptr;

  std::vector<std::string> input_node_names;
  std::vector<std::string> output_node_names;

  std::unordered_map<std::string, std::vector<int64_t>> input_node_dims;
  std::unordered_map<std::string, std::vector<int64_t>> output_node_dims;

  uint8_t *input_image_data = nullptr;

 public:
  static const int max_input_image_width = 1920;
  static const int max_input_image_height = 1080;

  static const int image_width = 640;
  static const int image_height = 640;

  static constexpr auto num_people = 100;

  dnn_inference_det(const std::vector<uint8_t> &model_data)
      : session(nullptr), io_binding(nullptr) {
    // Create session
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    try {
      OrtCUDAProviderOptions cuda_options{};

      cuda_options.device_id = 0;
      cuda_options.arena_extend_strategy = 1;
      cuda_options.gpu_mem_limit = 2ULL * 1024 * 1024 * 1024;
      cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
      cuda_options.do_copy_in_default_stream = 1;

      session_options.AppendExecutionProvider_CUDA(cuda_options);
    } catch (const Ort::Exception &e) {
      spdlog::info(e.what());
    }

    OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 0);

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

    assert(output_node_names.size() == 2);
    assert(output_node_names[0] == "dets");
    assert(output_node_names[1] == "labels");

    const auto &&image_size = cv::Size(image_width, image_height);

    const auto input_size = image_size.width * image_size.height * 3;
    CUDA_SAFE_CALL(cudaMalloc(&input_data, input_size * sizeof(float)));

    const auto dets_size = num_people * 5;
    CUDA_SAFE_CALL(cudaMalloc(&dets_data, dets_size * sizeof(float)));

    const auto labels_size = num_people;
    CUDA_SAFE_CALL(cudaMalloc(&labels_data, labels_size * sizeof(int64_t)));

    CUDA_SAFE_CALL(
        cudaMalloc(&input_image_data, max_input_image_width * max_input_image_height * 3));
  }

  ~dnn_inference_det() {
    CUDA_SAFE_CALL(cudaFree(input_data));
    CUDA_SAFE_CALL(cudaFree(input_image_data));
    CUDA_SAFE_CALL(cudaFree(dets_data));
    CUDA_SAFE_CALL(cudaFree(labels_data));
  }

  void process(const cv::Mat &image, roi_data &roi) {
    const auto &&image_size = cv::Size(image_width, image_height);

    {
      const auto &data = image;

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

      const auto input_image_width = data.size().width;
      const auto input_image_height = data.size().height;

      assert(input_image_width <= max_input_image_width);
      assert(input_image_height <= max_input_image_height);

      const auto scale = get_scale(data.size(), image_size);
      const auto center = cv::Point2f(data.size().width / 2.0, data.size().height / 2.0);
      const auto rotation = 0.0;

      roi = {{scale.width, scale.height}, rotation, {center.x, center.y}};

      const std::array<float, 3> mean = {0.485, 0.456, 0.406};
      const std::array<float, 3> std = {0.229, 0.224, 0.225};

      CUDA_SAFE_CALL(cudaMemcpy2D(input_image_data, input_image_width * 3, data.data, data.step,
                                  data.cols * 3, data.rows, cudaMemcpyHostToDevice));

      preprocess_cuda(input_image_data, input_image_width, input_image_height,
                      input_image_width * 3, input_data, image_size.width, image_size.height,
                      image_size.width, mean, std);
    }

    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    inference();
  }

  void inference() {
    assert(input_node_names.size() == 1);
    assert(input_node_names[0] == "input");

    assert(output_node_names.size() == 2);
    assert(output_node_names[0] == "dets");
    assert(output_node_names[1] == "labels");

    std::vector<const char *> input_node_names;
    {
      input_node_names.push_back(this->input_node_names[0].c_str());
    }

    std::vector<const char *> output_node_names;
    {
      output_node_names.push_back(this->output_node_names[0].c_str());
      output_node_names.push_back(this->output_node_names[1].c_str());
    }

    io_binding.ClearBoundInputs();
    io_binding.ClearBoundOutputs();

    std::vector<Ort::Value> input_tensors;
    {
      auto dims = input_node_dims.at(input_node_names[0]);

      const auto input_size =
          std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(info_cuda, input_data, input_size,
                                                                dims.data(), dims.size());

      io_binding.BindInput(input_node_names[0], input_tensor);

      input_tensors.emplace_back(std::move(input_tensor));
    }

    {
      io_binding.BindOutput(output_node_names[0], info_cuda);
    }

    {
      io_binding.BindOutput(output_node_names[1], info_cuda);
    }

    io_binding.SynchronizeInputs();

    session.Run(Ort::RunOptions{nullptr}, io_binding);

    io_binding.SynchronizeOutputs();

    const auto output_tensors = io_binding.GetOutputValues();

    CUDA_SAFE_CALL(cudaMemset(dets_data, 0, 5 * num_people * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(labels_data, 0, num_people * sizeof(int64_t)));

    CUDA_SAFE_CALL(cudaMemcpy(
        dets_data, output_tensors[0].GetTensorData<float>(),
        5 *
            std::min(static_cast<int>(output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[1]),
                     num_people) *
            sizeof(float),
        cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(
        labels_data, output_tensors[1].GetTensorData<int64_t>(),
        std::min(static_cast<int>(output_tensors[1].GetTensorTypeAndShapeInfo().GetShape()[1]),
                 num_people) *
            sizeof(int64_t),
        cudaMemcpyDeviceToDevice));
  }

  void copy_labels_to_cpu(int64_t *labels) const {
    CUDA_SAFE_CALL(
        cudaMemcpy(labels, labels_data, num_people * sizeof(int64_t), cudaMemcpyDeviceToHost));
  }

  void copy_dets_to_cpu(float *dets) const {
    CUDA_SAFE_CALL(
        cudaMemcpy(dets, dets_data, 5 * num_people * sizeof(float), cudaMemcpyDeviceToHost));
  }

  const int64_t *get_labels() const { return labels_data; }

  const float *get_dets() const { return dets_data; }
};

class dnn_inference_det_trt {
  std::vector<uint8_t> model_data;

  void *plugin_handler;
  Logger logger;
  nvinfer1::IRuntime *infer;
  nvinfer1::ICudaEngine *engine;
  nvinfer1::IExecutionContext *context;

  cudaStream_t stream;

  float *input_data = nullptr;
  float *dets_data = nullptr;
  int64_t *labels_data = nullptr;

  uint8_t *input_image_data = nullptr;

 public:
  static const int max_input_image_width = 1920;
  static const int max_input_image_height = 1080;

  static const int image_width = 640;
  static const int image_height = 640;

  static constexpr auto num_people = 100;

  dnn_inference_det_trt(const std::vector<uint8_t> &model_data, size_t max_views) {
    plugin_handler = dlopen("libmmdeploy_tensorrt_ops.so", RTLD_NOW);
    if (!plugin_handler) {
      throw std::runtime_error(dlerror());
    }

    infer = nvinfer1::createInferRuntime(logger);
    engine = infer->deserializeCudaEngine(model_data.data(), model_data.size());
    context = engine->createExecutionContext();

    const auto &&image_size = cv::Size(image_width, image_height);

    const auto input_size = image_size.width * image_size.height * 3 * max_views;
    CUDA_SAFE_CALL(cudaMalloc(&input_data, input_size * sizeof(float)));

    const auto dets_size = num_people * 5 * max_views;
    CUDA_SAFE_CALL(cudaMalloc(&dets_data, dets_size * sizeof(float)));

    const auto labels_size = num_people * max_views;
    CUDA_SAFE_CALL(cudaMalloc(&labels_data, labels_size * sizeof(int64_t)));

    CUDA_SAFE_CALL(cudaMalloc(&input_image_data,
                              max_input_image_width * max_input_image_height * 3 * max_views));

    std::unordered_map<std::string, void *> buffers;
    buffers["input"] = input_data;
    buffers["dets"] = dets_data;
    buffers["labels"] = labels_data;

    for (const auto &[name, buffer] : buffers) {
      if (!context->setTensorAddress(name.c_str(), buffer)) {
        spdlog::error("Failed to set tensor address");
      }
    }

    CUDA_SAFE_CALL(cudaStreamCreate(&stream));
  }

  ~dnn_inference_det_trt() {
    delete context;
    delete engine;
    delete infer;

    dlclose(plugin_handler);

    CUDA_SAFE_CALL(cudaStreamDestroy(stream));

    CUDA_SAFE_CALL(cudaFree(input_data));
    CUDA_SAFE_CALL(cudaFree(input_image_data));
    CUDA_SAFE_CALL(cudaFree(dets_data));
    CUDA_SAFE_CALL(cudaFree(labels_data));
  }

  void process(const std::vector<cv::Mat> &images, std::vector<roi_data> &rois) {
    const auto &&image_size = cv::Size(image_width, image_height);

    for (size_t i = 0; i < images.size(); i++) {
      const auto &data = images.at(i);

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

  void copy_labels_to_cpu(int64_t *labels, size_t i) const {
    CUDA_SAFE_CALL(cudaMemcpy(labels, labels_data + num_people * i, num_people * sizeof(int64_t),
                              cudaMemcpyDeviceToHost));
  }

  void copy_dets_to_cpu(float *dets, size_t i) const {
    CUDA_SAFE_CALL(cudaMemcpy(dets, dets_data + 5 * num_people * i, 5 * num_people * sizeof(float),
                              cudaMemcpyDeviceToHost));
  }
};
}  // namespace stargazer::mvpose
#else
#endif

#else

namespace stargazer::mvpose {
class dnn_inference_pose {
 public:
  static const auto num_joints = 133;

  dnn_inference_pose(const std::vector<uint8_t>& model_data, size_t max_batch_size) {}

  ~dnn_inference_pose() {}

  void process(const cv::Mat& image, const cv::Rect2f& rect, roi_data& roi) {}

  void inference() {}

  void copy_simcc_x_to_cpu(float* simcc_x) const {}

  void copy_simcc_y_to_cpu(float* simcc_y) const {}

  const float* get_simcc_x() const { return nullptr; }

  const float* get_simcc_y() const { return nullptr; }
};

class dnn_inference_pose_trt {
 public:
  dnn_inference_pose_trt(const std::vector<uint8_t>& model_data, size_t max_num_people) {}

  ~dnn_inference_pose_trt() {}

  void process(const std::vector<cv::Mat>& images,
               const std::vector<std::vector<cv::Rect2f>>& rects,
               std::vector<std::vector<roi_data>>& rois) {}

  void inference(size_t num_batch) {}

  void copy_simcc_x_to_cpu(float* simcc_x, size_t i) const {}

  void copy_simcc_y_to_cpu(float* simcc_y, size_t i) const {}
};

class dnn_inference_det {
 public:
  static constexpr auto num_people = 100;

  dnn_inference_det(const std::vector<uint8_t>& model_data) {}

  ~dnn_inference_det() {}

  void process(const cv::Mat& image, roi_data& roi) {}

  void inference() {}

  void copy_labels_to_cpu(int64_t* labels) const {}

  void copy_dets_to_cpu(float* dets) const {}

  const int64_t* get_labels() const { return nullptr; }

  const float* get_dets() const { return nullptr; }
};

class dnn_inference_det_trt {
 public:
  dnn_inference_det_trt(const std::vector<uint8_t>& model_data, size_t max_views) {}

  ~dnn_inference_det_trt() {}

  void process(const std::vector<cv::Mat>& images, std::vector<roi_data>& rois) {}

  void inference(size_t num_views) {}

  void copy_labels_to_cpu(int64_t* labels, size_t i) const {}

  void copy_dets_to_cpu(float* dets, size_t i) const {}
};
}  // namespace stargazer::mvpose
#endif

namespace stargazer::mvpose {
static cv::Point2f operator*(cv::Mat M, const cv::Point2f &p) {
  cv::Mat_<double> src(3, 1);

  src(0, 0) = p.x;
  src(1, 0) = p.y;
  src(2, 0) = 1.0;

  cv::Mat_<double> dst = M * src;
  return cv::Point2f(dst(0, 0), dst(1, 0));
}
using pose_joints_t = std::vector<std::tuple<cv::Point2f, float>>;

class mvpose_matcher {
 public:
  float factor;
  mvpose_matcher(float factor = 5.0f) : factor(factor) {}

  static glm::mat3 calculate_fundametal_matrix(const glm::mat3 &camera_mat1,
                                               const glm::mat3 &camera_mat2,
                                               const glm::mat4 &camera_pose1,
                                               const glm::mat4 &camera_pose2) {
    const auto pose = camera_pose2 * glm::inverse(camera_pose1);

    const auto R = glm::mat3(pose);
    const auto t = glm::vec3(pose[3]);

    const auto Tx = glm::mat3(0, -t[2], t[1], t[2], 0, -t[0], -t[1], t[0], 0);

    const auto E = Tx * R;

    const auto F = glm::inverse(glm::transpose(camera_mat2)) * E * glm::inverse(camera_mat1);

    return F;
  }

  static glm::mat3 calculate_fundametal_matrix(const stargazer::camera_t &camera1,
                                               const stargazer::camera_t &camera2) {
    const auto camera_mat1 = camera1.intrin.get_matrix();
    const auto camera_mat2 = camera2.intrin.get_matrix();

    return calculate_fundametal_matrix(camera_mat1, camera_mat2, camera1.extrin.transform_matrix(),
                                       camera2.extrin.transform_matrix());
  }

  static glm::vec3 normalize_line(const glm::vec3 &v) {
    const auto c = std::sqrt(v.x * v.x + v.y * v.y);
    return v / c;
  }

  static glm::vec3 compute_correspond_epiline(const glm::mat3 &F, const glm::vec2 &p) {
    const auto l = F * glm::vec3(p, 1.f);
    return normalize_line(l);
    // return l;
  }

  static glm::vec3 compute_correspond_epiline(const stargazer::camera_t &camera1,
                                              const stargazer::camera_t &camera2,
                                              const glm::vec2 &p) {
    const auto F = calculate_fundametal_matrix(camera1, camera2);
    return compute_correspond_epiline(F, p);
  }

  static float distance_sq_line_point(const glm::vec3 &line, const glm::vec2 &point) {
    const auto a = line.x;
    const auto b = line.y;
    const auto c = line.z;

    const auto num = a * point.x + b * point.y + c;
    const auto distsq = num * num / (a * a + b * b);

    return distsq;
  }

  static cv::Mat glm2cv_mat3(const glm::mat4 &m) {
    cv::Mat ret(3, 3, CV_32F);
    for (std::size_t i = 0; i < 3; i++) {
      for (std::size_t j = 0; j < 3; j++) {
        ret.at<float>(i, j) = m[j][i];
      }
    }
    return ret;
  }

  static glm::vec2 undistort(const glm::vec2 &pt, const stargazer::camera_t &camera) {
    auto pts = std::vector<cv::Point2f>{cv::Point2f(pt.x, pt.y)};
    cv::Mat m = glm2cv_mat3(camera.intrin.get_matrix());
    cv::Mat coeffs(5, 1, CV_32F);
    for (int i = 0; i < 5; i++) {
      coeffs.at<float>(i) = camera.intrin.coeffs[i];
    }

    std::vector<cv::Point2f> norm_pts;
    cv::undistortPoints(pts, norm_pts, m, coeffs);

    return glm::vec2(norm_pts[0].x, norm_pts[0].y);
  }

  static glm::vec2 project_undistorted(const glm::vec2 &pt, const stargazer::camera_t &camera) {
    const auto p = camera.intrin.get_matrix() * glm::vec3(pt.x, pt.y, 1.0f);
    return glm::vec2(p.x / p.z, p.y / p.z);
  }

  template <typename T, int m, int n>
  static inline glm::mat<m, n, float, glm::precision::highp> eigen2glm(
      const Eigen::Matrix<T, m, n> &src) {
    glm::mat<m, n, float, glm::precision::highp> dst;
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        dst[j][i] = src(i, j);
      }
    }
    return dst;
  }

  static float projected_distance(const pose_joints_t &points1, const pose_joints_t &points2,
                                  const Eigen::Matrix<float, 3, 3> &F) {
    const auto num_points = 17;
    auto result = 0.0f;
    {
      for (size_t i = 0; i < num_points; i++) {
        const auto pt1 = glm::vec2(std::get<0>(points1[i]).x, std::get<0>(points1[i]).y);
        const auto pt2 = glm::vec2(std::get<0>(points2[i]).x, std::get<0>(points2[i]).y);
        const auto line = compute_correspond_epiline(eigen2glm(F), pt1);
        const auto dist = std::abs(glm::dot(line, glm::vec3(pt2, 1.f)));
        result += dist;
      }
    }
    return result / num_points;
  }

  static float projected_distance(const pose_joints_t &points1, const pose_joints_t &points2,
                                  const stargazer::camera_t &camera1,
                                  const stargazer::camera_t &camera2) {
    const auto num_points = 17;
    auto result = 0.0f;
    {
      const auto F = calculate_fundametal_matrix(camera1, camera2);
      for (size_t i = 0; i < num_points; i++) {
        const auto pt1 = project_undistorted(
            undistort(glm::vec2(std::get<0>(points1[i]).x, std::get<0>(points1[i]).y), camera1),
            camera1);
        const auto pt2 = project_undistorted(
            undistort(glm::vec2(std::get<0>(points2[i]).x, std::get<0>(points2[i]).y), camera2),
            camera2);
        const auto line = compute_correspond_epiline(F, pt1);
        const auto dist = distance_sq_line_point(line, pt2);
        result += dist;
      }
    }
    {
      const auto F = calculate_fundametal_matrix(camera2, camera1);
      for (size_t i = 0; i < num_points; i++) {
        const auto pt1 = project_undistorted(
            undistort(glm::vec2(std::get<0>(points1[i]).x, std::get<0>(points1[i]).y), camera1),
            camera1);
        const auto pt2 = project_undistorted(
            undistort(glm::vec2(std::get<0>(points2[i]).x, std::get<0>(points2[i]).y), camera2),
            camera2);
        const auto line = compute_correspond_epiline(F, pt2);
        const auto dist = distance_sq_line_point(line, pt1);
        result += dist;
      }
    }
    return result / (num_points * 2);
  }

  static Eigen::MatrixXf compute_geometry_affinity(
      const std::vector<pose_joints_t> &points_set, const std::vector<size_t> &dim_group,
      const std::vector<std::vector<Eigen::Matrix<float, 3, 3>>> &cameras_list, float factor) {
    const auto M = points_set.size();
    Eigen::MatrixXf dist = Eigen::MatrixXf::Ones(M, M) * (factor * factor);
    dist.diagonal() = Eigen::VectorXf::Zero(M);

    for (size_t i = 0; i < dim_group.size() - 1; i++) {
      if (dim_group[i] == dim_group[i + 1]) {
        continue;
      }
      for (size_t j = i + 1; j < dim_group.size() - 1; j++) {
        if (dim_group[j] == dim_group[j + 1]) {
          continue;
        }

        for (size_t m = dim_group[i]; m < dim_group[i + 1]; m++) {
          for (size_t n = dim_group[j]; n < dim_group[j + 1]; n++) {
            const auto d1 = projected_distance(points_set[m], points_set[n], cameras_list[i][j]);
            const auto d2 = projected_distance(points_set[n], points_set[m], cameras_list[j][i]);
            dist(n, m) = dist(m, n) = (d1 + d2) / 2;
          }
        }
      }
    }

    const auto calc_std_dev = [](const auto &value) {
      return std::sqrt((value.array() - value.array().mean()).square().sum() /
                       (value.array().size() - 1));
    };

    {
      const auto std_dev = calc_std_dev(dist);
      if (std_dev < factor) {
        for (size_t i = 0; i < M; i++) {
          dist(i, i) = dist.array().mean();
        }
      }
    }

    Eigen::MatrixXf affinity_matrix =
        -(dist.array() - dist.array().mean()) / (calc_std_dev(dist) + 1e-12);
    affinity_matrix = 1 / (1 + (-factor * affinity_matrix.array()).exp());

    return affinity_matrix;
  }

  static Eigen::MatrixXf compute_geometry_affinity(
      const std::vector<pose_joints_t> &points_set, const std::vector<size_t> &dim_group,
      const std::vector<stargazer::camera_t> &cameras_list, float factor) {
    const auto M = points_set.size();
    Eigen::MatrixXf dist = Eigen::MatrixXf::Ones(M, M) * (factor * factor);
    dist.diagonal() = Eigen::VectorXf::Zero(M);

    for (size_t i = 0; i < dim_group.size() - 1; i++) {
      if (dim_group[i] == dim_group[i + 1]) {
        continue;
      }
      for (size_t j = i + 1; j < dim_group.size() - 1; j++) {
        if (dim_group[j] == dim_group[j + 1]) {
          continue;
        }

        for (size_t m = dim_group[i]; m < dim_group[i + 1]; m++) {
          for (size_t n = dim_group[j]; n < dim_group[j + 1]; n++) {
            const auto d =
                projected_distance(points_set[m], points_set[n], cameras_list[i], cameras_list[j]);
            dist(m, n) = d;
            dist(n, m) = d;
          }
        }
      }
    }

    const auto calc_std_dev = [](const auto &value) {
      return std::sqrt((value.array() - value.array().mean()).square().sum() /
                       (value.array().size() - 1));
    };

    {
      const auto std_dev = calc_std_dev(dist);
      if (std_dev < factor) {
        for (size_t i = 0; i < M; i++) {
          dist(i, i) = dist.array().mean();
        }
      }
    }

    Eigen::MatrixXf affinity_matrix =
        -(dist.array() - dist.array().mean()) / (calc_std_dev(dist) + 1e-12);
    affinity_matrix = 1 / (1 + (-factor * affinity_matrix.array()).exp());

    return affinity_matrix;
  }

  static Eigen::MatrixXi transform_closure(const Eigen::MatrixXi &X_binary) {
    // Convert binary relation matrix to permutation matrix.
    Eigen::MatrixXi temp = Eigen::MatrixXi::Zero(X_binary.rows(), X_binary.cols());
    const int N = X_binary.rows();
    for (int k = 0; k < N; k++) {
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          temp(i, j) = X_binary(i, j) | (X_binary(i, k) & X_binary(k, j));
        }
      }
    }
    Eigen::VectorXi vis = Eigen::VectorXi::Zero(N);
    Eigen::MatrixXi match = Eigen::MatrixXi::Zero(X_binary.rows(), X_binary.cols());
    for (int i = 0; i < N; i++) {
      if (vis(i)) {
        continue;
      }
      for (int j = 0; j < N; j++) {
        if (temp(i, j)) {
          vis(j) = 1;
          match(j, i) = 1;
        }
      }
    }
    return match;
  }

  /*
      Wang, Qianqian, Xiaowei Zhou, and Kostas Daniilidis. "Multi-Image Semantic
      Matching by Mining Consistent Features." arXiv preprint arXiv:1711.07641(2017).
  */

  static Eigen::VectorXf proj2pav(Eigen::VectorXf y) {
    y = y.array().max(0.0f);
    Eigen::VectorXf x = Eigen::VectorXf::Zero(y.size());
    if (y.array().sum() < 1) {
      x.array() += y.array();
    } else {
      Eigen::VectorXf u = y;
      std::sort(u.begin(), u.end(), std::greater{});

      Eigen::VectorXf sv = Eigen::VectorXf::Zero(y.size());
      std::partial_sum(u.begin(), u.end(), sv.begin());

      int rho = -1;
      for (int i = 0; i < y.size(); i++) {
        if (u(i) > (sv(i) - 1) / (i + 1)) {
          rho = i;
        }
      }
      assert(rho >= 0);
      const auto theta = std::max(0.0f, (sv(rho) - 1) / (rho + 1));
      x.array() += (y.array() - theta).max(0.0f);
    }
    return x;
  }

  static Eigen::MatrixXf project_row(Eigen::MatrixXf X) {
    for (int i = 0; i < X.rows(); i++) {
      X.row(i) = proj2pav(X.row(i));
    }
    return X;
  }
  static Eigen::MatrixXf project_col(Eigen::MatrixXf X) {
    for (int j = 0; j < X.cols(); j++) {
      X.col(j) = proj2pav(X.col(j));
    }
    return X;
  }

  static Eigen::MatrixXf proj2dpam(const Eigen::MatrixXf &Y, const float tol = 1e-4f) {
    Eigen::MatrixXf X0 = Y;
    Eigen::MatrixXf X = Y;
    Eigen::MatrixXf I2 = Eigen::MatrixXf::Zero(Y.rows(), Y.cols());

    for (size_t iter = 0; iter < 10; iter++) {
      Eigen::MatrixXf X1 = project_row(X0.array() + I2.array());
      Eigen::MatrixXf I1 = X1.array() - (X0.array() + I2.array());
      Eigen::MatrixXf X2 = project_col(X0.array() + I1.array());
      I2 = X2.array() - (X0.array() + I1.array());

      const auto change = (X2.array() - X.array()).abs().mean();
      X = X2;
      if (change < tol) {
        return X;
      }
    }
    return X;
  }

  static Eigen::MatrixXi solve_svt(Eigen::MatrixXf affinity_matrix,
                                   const std::vector<size_t> &dim_group) {
    const bool dual_stochastic = true;
    const int N = affinity_matrix.rows();
    const int max_iter = 500;
    const float alpha = 0.1f;
    const float lambda = 50.0f;
    const float tol = 5e-4f;
    const int p_select = 1;
    float mu = 64.0f;

    affinity_matrix.diagonal() = Eigen::VectorXf::Zero(N);
    affinity_matrix = (affinity_matrix.array() + affinity_matrix.transpose().array()) / 2;

    Eigen::MatrixXf X = affinity_matrix;
    Eigen::MatrixXf Y = Eigen::MatrixXf::Zero(affinity_matrix.rows(), affinity_matrix.cols());
    Eigen::MatrixXf W = alpha - affinity_matrix.array();

    for (int iter = 0; iter < max_iter; iter++) {
      Eigen::MatrixXf X0 = X;

      // Update Q with SVT
      Eigen::JacobiSVD<Eigen::MatrixXf> svd(1.0f / mu * Y.array() + X.array(),
                                            Eigen::ComputeFullU | Eigen::ComputeFullV);

      Eigen::VectorXf diagS = svd.singularValues().array() - static_cast<float>(lambda) / mu;
      diagS = diagS.array().max(0.0f);
      const Eigen::MatrixXf Q = svd.matrixU() * diagS.asDiagonal() * svd.matrixV().transpose();

      // Update X
      X = Q.array() - (W.array() + Y.array()) / mu;

      // Project X
      for (size_t i = 0; i < dim_group.size() - 1; i++) {
        X.block(dim_group[i], dim_group[i], dim_group[i + 1] - dim_group[i],
                dim_group[i + 1] - dim_group[i])
            .array() = 0;
      }
      if (p_select == 1) {
        X.diagonal() = Eigen::VectorXf::Ones(N);
      }
      X = X.array().max(0.0f).min(1.0f);

      if (dual_stochastic) {
        // Projection for double stochastic constraint
        for (size_t i = 0; i < dim_group.size() - 1; i++) {
          const auto row_begin = dim_group[i];
          const auto row_end = dim_group[i + 1];
          for (size_t j = 0; j < dim_group.size() - 1; j++) {
            const auto col_begin = dim_group[j];
            const auto col_end = dim_group[j + 1];

            if (row_end > row_begin && col_end > col_begin) {
              X.block(row_begin, col_begin, row_end - row_begin, col_end - col_begin).array() =
                  proj2dpam(X.block(row_begin, col_begin, row_end - row_begin, col_end - col_begin),
                            1e-2);
            }
          }
        }
      }

      X = (X.array() + X.transpose().array()) / 2;

      // Update Y
      Y = Y.array() + mu * (X.array() - Q.array());

      // Test if convergence
      const auto p_res = (X.array() - Q.array()).matrix().norm() / N;
      const auto d_res = mu * (X.array() - X0.array()).matrix().norm() / N;

      if (p_res < tol && d_res < tol) {
        break;
      }

      if (p_res > 10 * d_res) {
        mu = 2 * mu;
      } else if (d_res > 10 * p_res) {
        mu = mu / 2;
      }
    }

    X = (X.array() + X.transpose().array()) / 2;

    const auto match = transform_closure(
        X.array().unaryExpr([](float p) { return p > 0.5f; }).template cast<int>());
    return match;
  }

  using pose_id_t = std::pair<size_t, size_t>;

  std::vector<std::vector<pose_id_t>> compute_matches(
      const std::vector<std::vector<pose_joints_t>> &pose_joints_list,
      const std::vector<stargazer::camera_t> &cameras_list) {
    std::vector<pose_joints_t> points_set;
    std::vector<size_t> dim_group;
    for (const auto &persons : pose_joints_list) {
      dim_group.push_back(points_set.size());
      for (const auto &pose_joints : persons) {
        points_set.push_back(pose_joints);
      }
    }
    dim_group.push_back(points_set.size());

    if (points_set.empty()) {
      return {};
    }

    const auto affinity_matrix =
        compute_geometry_affinity(points_set, dim_group, cameras_list, factor);
    const auto match_matrix = solve_svt(affinity_matrix, dim_group);

    const int num_camera_min = 2;

    std::vector<std::vector<pose_id_t>> matched_list;
    for (int j = 0; j < match_matrix.cols(); j++) {
      if (match_matrix.col(j).array().sum() >= num_camera_min) {
        std::vector<pose_id_t> matched;
        for (size_t k = 0; k < dim_group.size() - 1; k++) {
          for (size_t i = dim_group[k]; i < dim_group[k + 1]; i++) {
            if (match_matrix(i, j) > 0) {
              matched.push_back(std::make_pair(k, i - dim_group[k]));
            }
          }
        }
        matched_list.push_back(matched);
      }
    }

    return matched_list;
  }
};

static glm::vec3 triangulate(const std::vector<glm::vec2> &points,
                             const std::vector<stargazer::camera_t> &cameras) {
  assert(points.size() == cameras.size());

  const auto nviews = points.size();
  cv::Mat_<double> design = cv::Mat_<double>::zeros(3 * nviews, 4 + nviews);
  for (size_t i = 0; i < nviews; ++i) {
    cv::Mat camera_mat;
    cv::Mat dist_coeffs;
    get_cv_intrinsic(cameras[i].intrin, camera_mat, dist_coeffs);

    std::vector<cv::Point2d> pt = {cv::Point2d(points[i].x, points[i].y)};
    std::vector<cv::Point2d> undistort_pt;
    cv::undistortPoints(pt, undistort_pt, camera_mat, dist_coeffs);

    const auto &proj = cameras[i].extrin.transform_matrix();

    for (size_t m = 0; m < 3; ++m) {
      for (size_t n = 0; n < 4; ++n) {
        design(3 * i + m, n) = -proj[n][m];
      }
    }
    design(3 * i + 0, 4 + i) = undistort_pt[0].x;
    design(3 * i + 1, 4 + i) = undistort_pt[0].y;
    design(3 * i + 2, 4 + i) = 1.0;
  }

  cv::Mat x_and_alphas;
  cv::SVD::solveZ(design, x_and_alphas);

  const glm::vec3 point3d(x_and_alphas.at<double>(0, 0) / x_and_alphas.at<double>(0, 3),
                          x_and_alphas.at<double>(0, 1) / x_and_alphas.at<double>(0, 3),
                          x_and_alphas.at<double>(0, 2) / x_and_alphas.at<double>(0, 3));

  return point3d;
}

std::vector<glm::vec3> mvpose::inference(const std::vector<cv::Mat> &images_list,
                                         const std::vector<stargazer::camera_t> &cameras_list) {
  std::vector<std::vector<cv::Rect2f>> rects(images_list.size());

  {
    std::vector<roi_data> rois;
    inference_det->process(images_list, rois);

    for (size_t i = 0; i < images_list.size(); i++) {
      const auto roi = rois.at(i);
      const auto num_people = dnn_inference_det::num_people;

      std::vector<float> dets(5 * num_people);
      std::vector<int64_t> labels(num_people);

      inference_det->copy_dets_to_cpu(dets.data(), i);
      inference_det->copy_labels_to_cpu(labels.data(), i);

      const auto &&resized_size = cv::Size(640, 640);
      const auto trans = get_transform(cv::Point2f(roi.center[0], roi.center[1]),
                                       cv::Size2f(roi.scale[0], roi.scale[1]), resized_size);

      cv::Mat inv_trans;
      cv::invertAffineTransform(trans, inv_trans);

      for (size_t j = 0; j < num_people; j++) {
        if (labels[j] == 0) {
          const auto score = dets[j * 5 + 4];
          if (score > 0.3f) {
            const auto bbox_left = dets[j * 5];
            const auto bbox_top = dets[j * 5 + 1];
            const auto bbox_right = dets[j * 5 + 2];
            const auto bbox_bottom = dets[j * 5 + 3];

            const auto bbox_left_top = inv_trans * cv::Point2f(bbox_left, bbox_top);
            const auto bbox_right_bottom = inv_trans * cv::Point2f(bbox_right, bbox_bottom);

            rects[i].push_back(cv::Rect2f(bbox_left_top, bbox_right_bottom));
          }
        }
      }
    }
  }

  std::vector<std::vector<roi_data>> rois;
  inference_pose->process(images_list, rects, rois);

  size_t k = 0;

  std::vector<std::vector<pose_joints_t>> pose_joints_list(images_list.size());
  for (size_t i = 0; i < images_list.size(); i++) {
    for (size_t j = 0; j < rects[i].size(); j++, k++) {
      const auto roi = rois.at(i).at(j);

      const auto &&resized_size = cv::Size(288, 384);
      const auto num_joints = dnn_inference_pose::num_joints;
      const auto extend_width = resized_size.width * 2;
      const auto extend_height = resized_size.height * 2;

      std::vector<float> simcc_x(extend_width * num_joints);
      std::vector<float> simcc_y(extend_height * num_joints);

      inference_pose->copy_simcc_x_to_cpu(simcc_x.data(), k);
      inference_pose->copy_simcc_y_to_cpu(simcc_y.data(), k);

      const auto trans = get_transform(cv::Point2f(roi.center[0], roi.center[1]),
                                       cv::Size2f(roi.scale[0], roi.scale[1]), resized_size);

      cv::Mat inv_trans;
      cv::invertAffineTransform(trans, inv_trans);

      pose_joints_t pose_joints;
      for (int k = 0; k < num_joints; ++k) {
        const auto x_biggest_iter = std::max_element(
            simcc_x.begin() + k * extend_width, simcc_x.begin() + k * extend_width + extend_width);
        const auto max_x_pos = std::distance(simcc_x.begin() + k * extend_width, x_biggest_iter);
        const auto pose_x = max_x_pos / 2.0f;
        const auto score_x = *x_biggest_iter;

        const auto y_biggest_iter =
            std::max_element(simcc_y.begin() + k * extend_height,
                             simcc_y.begin() + k * extend_height + extend_height);
        const auto max_y_pos = std::distance(simcc_y.begin() + k * extend_height, y_biggest_iter);
        const auto pose_y = max_y_pos / 2.0f;
        const auto score_y = *y_biggest_iter;

        const auto score = (score_x + score_y) / 2;
        // const auto score = std::max(score_x, score_y);

        const auto pose = inv_trans * cv::Point2f(pose_x, pose_y);
        const auto joint = std::make_tuple(pose, score);
        pose_joints.emplace_back(joint);
      }

      pose_joints_list[i].push_back(pose_joints);
    }
  }

  const auto matched_list = matcher->compute_matches(pose_joints_list, cameras_list);

  glm::mat4 axis(1.0f);
  std::vector<glm::vec3> markers;
  for (const auto &matched : matched_list) {
    for (size_t j = 5; j < 17; j++) {
      std::vector<glm::vec2> pts;
      std::vector<stargazer::camera_t> cams;
      for (std::size_t i = 0; i < matched.size(); i++) {
        const auto pt = std::get<0>(pose_joints_list[matched[i].first][matched[i].second][j]);
        const auto score = std::get<1>(pose_joints_list[matched[i].first][matched[i].second][j]);
        if (score > 0.7f) {
          pts.push_back(glm::vec2(pt.x, pt.y));
          cams.push_back(cameras_list[matched[i].first]);
        }
      }
      if (pts.size() >= 2) {
        const auto marker = triangulate(pts, cams);
        markers.push_back(marker);
      }
    }
  }
  return markers;
}

static void load_model(std::string model_path, std::vector<uint8_t> &data) {
  std::ifstream ifs;
  ifs.open(model_path, std::ios_base::in | std::ios_base::binary);
  if (ifs.fail()) {
    spdlog::error("File open error: %s", model_path);
    std::quick_exit(0);
  }

  ifs.seekg(0, std::ios::end);
  const auto length = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  data.resize(length);

  ifs.read((char *)data.data(), length);
  if (ifs.fail()) {
    spdlog::error("File read error: %s", model_path);
    std::quick_exit(0);
  }
}

mvpose::mvpose() {
  std::vector<uint8_t> det_model_data;
  {
    const auto model_path = "../data/mvpose/rtmdet_m_640-8xb32_coco-person-fp16.engine";
    std::vector<uint8_t> data;
    load_model(model_path, data);

    det_model_data = std::move(data);
  }

  inference_det.reset(new dnn_inference_det_trt(det_model_data, 5));

  std::vector<uint8_t> pose_model_data;
  {
    const auto model_path =
        "../data/mvpose/rtmpose-l_8xb32-270e_coco-wholebody-384x288-fp16.engine";
    std::vector<uint8_t> data;
    load_model(model_path, data);

    pose_model_data = std::move(data);
  }

  inference_pose.reset(new dnn_inference_pose_trt(pose_model_data, 100));

  matcher.reset(new mvpose_matcher());
}

mvpose::~mvpose() = default;
}  // namespace stargazer::mvpose