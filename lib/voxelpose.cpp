#include "voxelpose.hpp"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#include "graph_proc.h"
#include "graph_proc_cv.h"
#include "graph_proc_img.h"
#include "graph_proc_tensor.h"
#include "preprocess.hpp"
#include "voxelpose_cuda.hpp"

using namespace stargazer;

#define CUDA_SAFE_CALL(func)                                                                       \
  do {                                                                                             \
    cudaError_t err = (func);                                                                      \
    if (err != cudaSuccess) {                                                                      \
      fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, \
              __FILE__, __LINE__);                                                                 \
      exit(err);                                                                                   \
    }                                                                                              \
  } while (0)

namespace stargazer::voxelpose {
class dnn_inference {
 public:
  virtual void inference(const float *input) = 0;
  virtual const float *get_output_data() const = 0;
};

class dnn_inference_heatmap {
 public:
  virtual void process(const std::vector<cv::Mat> &images, std::vector<roi_data> &rois) = 0;
  virtual void inference(size_t num_views) = 0;
  virtual const float *get_heatmaps() const = 0;
  virtual int get_heatmap_width() const = 0;
  virtual int get_heatmap_height() const = 0;
};
}  // namespace stargazer::voxelpose

#define ENABLE_ONNXRUNTIME

#ifdef ENABLE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>

namespace stargazer::voxelpose {
class ort_dnn_inference : public dnn_inference {
  std::vector<uint8_t> model_data;

  Ort::Env env{ORT_LOGGING_LEVEL_WARNING};
  Ort::Session session;
  Ort::IoBinding io_binding;
  Ort::MemoryInfo info_cuda{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};

  float *input_data = nullptr;
  float *output_data = nullptr;

  std::vector<std::string> input_node_names;
  std::vector<std::string> output_node_names;

  std::unordered_map<std::string, std::vector<int64_t>> input_node_dims;
  std::unordered_map<std::string, std::vector<int64_t>> output_node_dims;

 public:
  ort_dnn_inference(const std::vector<uint8_t> &model_data)
      : session(nullptr), io_binding(nullptr) {
    namespace fs = std::filesystem;

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

      CUDA_SAFE_CALL(cudaMalloc(&input_data, input_size * sizeof(float)));
    }

    {
      const auto dims = output_node_dims.at(output_node_names[0]);
      const auto output_size =
          std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

      CUDA_SAFE_CALL(cudaMalloc(&output_data, output_size * sizeof(float)));
    }
  }

  void inference(const float *input) {
    assert(input_node_names.size() == 1);
    assert(input_node_names[0] == "input");

    assert(output_node_names.size() == 1);
    assert(output_node_names[0] == "output");

    std::vector<const char *> input_node_names;
    {
      input_node_names.push_back(this->input_node_names[0].c_str());
    }

    std::vector<const char *> output_node_names;
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

      CUDA_SAFE_CALL(
          cudaMemcpy(input_data, input, input_size * sizeof(float), cudaMemcpyDeviceToDevice));

      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(info_cuda, input_data, input_size,
                                                                dims.data(), dims.size());

      io_binding.BindInput(input_node_names[0], input_tensor);

      input_tensors.emplace_back(std::move(input_tensor));
    }

    std::vector<Ort::Value> output_tensors;
    {
      const auto dims = output_node_dims.at(output_node_names[0]);
      const auto output_size =
          std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

      Ort::Value output_tensor =
          Ort::Value::CreateTensor(info_cuda, output_data, output_size, dims.data(), dims.size());

      io_binding.BindOutput(output_node_names[0], output_tensor);

      output_tensors.emplace_back(std::move(output_tensor));
    }

    io_binding.SynchronizeInputs();

    session.Run(Ort::RunOptions{nullptr}, io_binding);

    io_binding.SynchronizeOutputs();
  }

  const float *get_output_data() const { return output_data; }
};

class ort_dnn_inference_heatmap : public dnn_inference_heatmap {
  std::vector<uint8_t> model_data;

  Ort::Env env{ORT_LOGGING_LEVEL_WARNING};
  Ort::Session session;
  Ort::IoBinding io_binding;
  Ort::MemoryInfo info_cuda{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};

  float *input_data = nullptr;
  float *output_data = nullptr;

  std::vector<std::string> input_node_names;
  std::vector<std::string> output_node_names;

  std::unordered_map<std::string, std::vector<int64_t>> input_node_dims;
  std::unordered_map<std::string, std::vector<int64_t>> output_node_dims;

  static const int max_input_image_width = 1920;
  static const int max_input_image_height = 1080;

  int image_width = 960;
  int image_height = 512;

  uint8_t *input_image_data = nullptr;

 public:
  ort_dnn_inference_heatmap(const std::vector<uint8_t> &model_data, size_t max_views)
      : session(nullptr), io_binding(nullptr) {
    namespace fs = std::filesystem;

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

    const auto input_size = 960 * 512 * 3 * max_views;
    CUDA_SAFE_CALL(cudaMalloc(&input_data, input_size * sizeof(float)));

    const auto output_size = 240 * 128 * 15 * max_views;
    CUDA_SAFE_CALL(cudaMalloc(&output_data, output_size * sizeof(float)));

    CUDA_SAFE_CALL(cudaMalloc(&input_image_data,
                              max_input_image_width * max_input_image_height * 3 * max_views));
  }

  ~ort_dnn_inference_heatmap() {
    CUDA_SAFE_CALL(cudaFree(input_data));
    CUDA_SAFE_CALL(cudaFree(output_data));
    CUDA_SAFE_CALL(cudaFree(input_image_data));
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

      roi_data roi = {{scale.width, scale.height}, rotation, {center.x, center.y}};
      rois.push_back(roi);

      const std::array<float, 3> mean = {0.485, 0.456, 0.406};
      const std::array<float, 3> std = {0.229, 0.224, 0.225};

      CUDA_SAFE_CALL(cudaMemcpy2D(input_image_data + i * input_image_width * 3 * input_image_height,
                                  input_image_width * 3, data.data, data.step, data.cols * 3,
                                  data.rows, cudaMemcpyHostToDevice));

      preprocess_cuda(input_image_data + i * input_image_width * 3 * input_image_height,
                      input_image_width, input_image_height, input_image_width * 3,
                      input_data + i * 960 * 512 * 3, 960, 512, 960, mean, std);
    }

    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    inference(images.size());
  }

  void inference(size_t num_views) {
    assert(input_node_names.size() == 1);
    assert(input_node_names[0] == "input");

    assert(output_node_names.size() == 1);
    assert(output_node_names[0] == "output");

    std::vector<const char *> input_node_names;
    {
      input_node_names.push_back(this->input_node_names[0].c_str());
    }

    std::vector<const char *> output_node_names;
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

      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(info_cuda, input_data, input_size,
                                                                dims.data(), dims.size());

      io_binding.BindInput(input_node_names[0], input_tensor);

      input_tensors.emplace_back(std::move(input_tensor));
    }

    std::vector<Ort::Value> output_tensors;
    {
      auto dims = output_node_dims.at(output_node_names[0]);
      dims[0] = num_views;
      const auto output_size =
          std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

      Ort::Value output_tensor =
          Ort::Value::CreateTensor(info_cuda, output_data, output_size, dims.data(), dims.size());

      io_binding.BindOutput(output_node_names[0], output_tensor);

      output_tensors.emplace_back(std::move(output_tensor));
    }

    io_binding.SynchronizeInputs();

    session.Run(Ort::RunOptions{nullptr}, io_binding);

    io_binding.SynchronizeOutputs();
  }

  const float *get_heatmaps() const { return output_data; }

  int get_heatmap_width() const { return 240; }

  int get_heatmap_height() const { return 128; }
};
}  // namespace stargazer::voxelpose
#endif

#include <opencv2/dnn/dnn.hpp>

namespace stargazer::voxelpose {
class cv_dnn_inference : public dnn_inference {
  cv::dnn::Net net;

  std::vector<std::string> input_node_names;
  std::vector<std::string> output_node_names;

  std::unordered_map<std::string, std::vector<int64_t>> input_node_dims;
  std::unordered_map<std::string, std::vector<int64_t>> output_node_dims;

  std::vector<float> input_data;
  float *output_data = nullptr;

 public:
  cv_dnn_inference(const std::vector<uint8_t> &model_data) {
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
    CUDA_SAFE_CALL(cudaMalloc(&output_data, output_size * sizeof(float)));
  }

  void inference(const float *input) {
    std::vector<cv::dnn::MatShape> input_layer_shapes;
    std::vector<cv::dnn::MatShape> output_layer_shapes;
    net.getLayerShapes(cv::dnn::MatShape(), 0, input_layer_shapes, output_layer_shapes);

    assert(input_layer_shapes.size() == 1);
    assert(output_layer_shapes.size() == 1);

    CUDA_SAFE_CALL(cudaMemcpy(input_data.data(), input, input_data.size(), cudaMemcpyDeviceToHost));

    cv::Mat input_mat(input_layer_shapes[0], CV_32FC1, (void *)input_data.data());
    net.setInput(input_mat);
    const auto output_mat = net.forward();

    CUDA_SAFE_CALL(cudaMemcpy(output_data, output_mat.data, output_mat.total() * sizeof(float),
                              cudaMemcpyHostToDevice));
  }

  const float *get_output_data() const { return output_data; }
};

class cv_dnn_inference_heatmap : public dnn_inference_heatmap {
  cv::dnn::Net net;

  std::vector<std::string> input_node_names;
  std::vector<std::string> output_node_names;

  std::unordered_map<std::string, std::vector<int64_t>> input_node_dims;
  std::unordered_map<std::string, std::vector<int64_t>> output_node_dims;

  float *input_data = nullptr;
  float *output_data = nullptr;
  std::vector<float> input_data_cpu;

  static const int max_input_image_width = 1920;
  static const int max_input_image_height = 1080;

  int image_width = 960;
  int image_height = 512;

  uint8_t *input_image_data = nullptr;

 public:
  cv_dnn_inference_heatmap(const std::vector<uint8_t> &model_data, size_t max_views) {
    const auto backends = cv::dnn::getAvailableBackends();
    net = cv::dnn::readNetFromONNX(model_data);

    const auto input_size = 960 * 512 * 3 * max_views;
    CUDA_SAFE_CALL(cudaMalloc(&input_data, input_size * sizeof(float)));
    input_data_cpu.resize(input_size);

    const auto output_size = 240 * 128 * 15 * max_views;
    CUDA_SAFE_CALL(cudaMalloc(&output_data, output_size * sizeof(float)));

    CUDA_SAFE_CALL(cudaMalloc(&input_image_data,
                              max_input_image_width * max_input_image_height * 3 * max_views));
  }

  ~cv_dnn_inference_heatmap() {}

  void process(const std::vector<cv::Mat> &images, std::vector<roi_data> &rois) {
    const auto &&image_size = cv::Size(960, 512);

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

      roi_data roi = {{scale.width, scale.height}, rotation, {center.x, center.y}};
      rois.push_back(roi);

      const std::array<float, 3> mean = {0.485, 0.456, 0.406};
      const std::array<float, 3> std = {0.229, 0.224, 0.225};

      CUDA_SAFE_CALL(cudaMemcpy2D(input_image_data + i * input_image_width * 3 * input_image_height,
                                  input_image_width * 3, data.data, data.step, data.cols * 3,
                                  data.rows, cudaMemcpyHostToDevice));

      preprocess_cuda(input_image_data + i * input_image_width * 3 * input_image_height,
                      input_image_width, input_image_height, input_image_width * 3,
                      input_data + i * 960 * 512 * 3, 960, 512, 960, mean, std);
    }

    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    inference(images.size());
  }

  void inference(size_t num_views) {
    CUDA_SAFE_CALL(cudaMemcpy(input_data_cpu.data(), input_data, num_views * 960 * 512 * 3,
                              cudaMemcpyDeviceToHost));

    const cv::dnn::MatShape input_shape = {static_cast<int>(num_views), 3, 512, 960};

    cv::Mat input_mat(input_shape, CV_32FC1, (void *)input_data_cpu.data());
    net.setInput(input_mat);
    const auto output_mat = net.forward();

    CUDA_SAFE_CALL(cudaMemcpy(output_data, output_mat.data, output_mat.total() * sizeof(float),
                              cudaMemcpyHostToDevice));
  }

  const float *get_heatmaps() const { return output_data; }

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
  void set_grid_size(const std::array<float, 3> &value) { grid_size = value; }
  std::array<float, 3> get_grid_center() const { return grid_center; }
  void set_grid_center(const std::array<float, 3> &value) { grid_center = value; }
  std::array<int32_t, 3> get_cube_size() const { return cube_size; }
  void set_cube_size(const std::array<int32_t, 3> &value) { cube_size = value; }

  static coalsack::tensor<float, 4> max_pool(const coalsack::tensor<float, 4> &inputs,
                                             size_t kernel = 3) {
    const auto padding = (kernel - 1) / 2;
    const auto max = inputs.max_pool3d(kernel, 1, padding, 1);
    const auto keep = inputs.transform(max, [](const float value1, const float value2, auto...) {
      return value1 == value2 ? value1 : 0.f;
    });
    return keep;
  }

  static coalsack::tensor<uint64_t, 2> get_index(const coalsack::tensor<uint64_t, 1> &indices,
                                                 const std::array<uint64_t, 3> &shape) {
    const auto result = indices.transform_expand<1>({3}, [shape](const uint64_t value, auto...) {
      const auto index_x = value / (shape[1] * shape[0]);
      const auto index_y = value % (shape[1] * shape[0]) / shape[0];
      const auto index_z = value % shape[0];
      return std::array<uint64_t, 3>{index_x, index_y, index_z};
    });
    return result;
  }

  coalsack::tensor<float, 2> get_real_loc(const coalsack::tensor<uint64_t, 2> &index) {
    const auto loc =
        index.cast<float>().transform([this](const float value, const size_t i, const size_t j) {
          return value / (cube_size[i] - 1) * grid_size[i] + grid_center[i] - grid_size[i] / 2.0f;
        });
    return loc;
  }

  coalsack::tensor<float, 2> get_centers(const coalsack::tensor<float, 5> &src) {
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

static void load_model(std::string model_path, std::vector<uint8_t> &data) {
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

  ifs.read((char *)data.data(), length);
  if (ifs.fail()) {
    std::cerr << "File read error: " << model_path << "\n";
    std::quick_exit(0);
  }
}

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
  std::vector<uint8_t> backbone_model_data;
  {
    const auto model_path = "../data/voxelpose/backbone.onnx";
    std::vector<uint8_t> data;
    load_model(model_path, data);

    backbone_model_data = std::move(data);
  }

#ifdef ENABLE_ONNXRUNTIME
  inference_heatmap.reset(new ort_dnn_inference_heatmap(backbone_model_data, 5));
#else
  inference_heatmap.reset(new cv_dnn_inference_heatmap(backbone_model_data, 5));
#endif

  std::vector<uint8_t> proposal_v2v_net_model_data;
  {
    const auto model_path = "../data/voxelpose/proposal_v2v_net.onnx";
    std::vector<uint8_t> data;
    load_model(model_path, data);

    proposal_v2v_net_model_data = std::move(data);
  }

#ifdef ENABLE_ONNXRUNTIME
  inference_proposal.reset(new ort_dnn_inference(proposal_v2v_net_model_data));
#else
  inference_proposal.reset(new cv_dnn_inference(proposal_v2v_net_model_data));
#endif

  std::vector<uint8_t> pose_v2v_net_model_data;
  {
    const auto model_path = "../data/voxelpose/pose_v2v_net.onnx";
    std::vector<uint8_t> data;
    load_model(model_path, data);

    pose_v2v_net_model_data = std::move(data);
  }

#ifdef ENABLE_ONNXRUNTIME
  inference_pose.reset(new ort_dnn_inference(pose_v2v_net_model_data));
#else
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
void voxelpose::set_grid_size(const std::array<float, 3> &value) { grid_size = value; }
void voxelpose::set_grid_center(const std::array<float, 3> &value) { grid_center = value; }

std::vector<glm::vec3> voxelpose::inference(const std::vector<cv::Mat> &images_list,
                                            const std::vector<camera_data> &cameras_list) {
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
  CUDA_SAFE_CALL(cudaMemcpy(proposal.get_data(), inference_proposal->get_output_data(),
                            proposal.get_size() * sizeof(float), cudaMemcpyDeviceToHost));

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

      CUDA_SAFE_CALL(cudaMemcpy(&joints[0][0], joint_extract->get_joints(), 3 * 15 * sizeof(float),
                                cudaMemcpyDeviceToHost));

      glm::mat4 basis(1.f);
      basis[0] = glm::vec4(1.f, 0.f, 0.f, 0.f);
      basis[1] = glm::vec4(0.f, 0.f, -1.f, 0.f);
      basis[2] = glm::vec4(0.f, 1.f, 0.f, 0.f);

      for (const auto &joint : joints) {
        points.push_back(basis * glm::vec4(joint / 1000.0f, 1.0f));
      }
    }
  }

  return points;
}

const float *voxelpose::get_heatmaps() const { return inference_heatmap->get_heatmaps(); }

void voxelpose::copy_heatmap_to(size_t num_views, float *data) const {
  const auto heatmap_size =
      get_heatmap_width() * get_heatmap_height() * get_num_joints() * num_views;
  CUDA_SAFE_CALL(
      cudaMemcpy(data, get_heatmaps(), heatmap_size * sizeof(float), cudaMemcpyDeviceToHost));
}
uint32_t voxelpose::get_heatmap_width() const { return inference_heatmap->get_heatmap_width(); }
uint32_t voxelpose::get_heatmap_height() const { return inference_heatmap->get_heatmap_height(); }
uint32_t voxelpose::get_num_joints() const { return 15; }
}  // namespace stargazer::voxelpose