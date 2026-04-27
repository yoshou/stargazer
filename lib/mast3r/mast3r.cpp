#include "mast3r.hpp"

#include <onnxruntime_cxx_api.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "dust3r.hpp"

namespace {

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

bool contains(const std::string& text, const std::string& needle) {
  return text.find(needle) != std::string::npos;
}

size_t tensor_elements(const std::vector<int64_t>& shape) {
  size_t elements = 1;
  for (int64_t dim : shape) {
    if (dim < 0) {
      throw std::runtime_error("MASt3R: dynamic output shape is not supported at runtime");
    }
    elements *= static_cast<size_t>(dim);
  }
  return elements;
}

struct output_tensor {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<float> data;
};

int view_score(const std::string& name, int view) {
  const auto n = lower(name);
  int score = 0;
  const auto view_digit = std::to_string(view);
  if (contains(n, "pred" + view_digit)) score += 20;
  if (contains(n, "view" + view_digit)) score += 16;
  if (contains(n, "_" + view_digit)) score += 6;
  if (contains(n, std::to_string(view - 1))) score += 2;
  return score;
}

int pts_score(const output_tensor& tensor, int view) {
  const auto n = lower(tensor.name);
  const auto elems = tensor.data.size();
  if (elems != static_cast<size_t>(stargazer::mast3r::ONNX_H * stargazer::mast3r::ONNX_W * 3)) {
    return std::numeric_limits<int>::min();
  }
  if (tensor.shape.empty() || tensor.shape.back() != 3) {
    return std::numeric_limits<int>::min();
  }

  int score = view_score(n, view);
  if (contains(n, "pts3d")) score += 40;
  if (contains(n, "point")) score += 20;
  if (contains(n, "other") && view == 2) score += 8;
  if (contains(n, "desc")) score -= 100;
  if (contains(n, "conf")) score -= 80;
  return score;
}

int conf_score(const output_tensor& tensor, int view) {
  const auto n = lower(tensor.name);
  const auto elems = tensor.data.size();
  if (elems != static_cast<size_t>(stargazer::mast3r::ONNX_H * stargazer::mast3r::ONNX_W)) {
    return std::numeric_limits<int>::min();
  }

  int score = view_score(n, view);
  if (contains(n, "conf")) score += 40;
  if (contains(n, "desc")) score -= 80;
  if (contains(n, "pts") || contains(n, "point")) score -= 80;
  return score;
}

const output_tensor& select_tensor(const std::vector<output_tensor>& tensors, int view,
                                   bool want_points) {
  const output_tensor* best = nullptr;
  int best_score = std::numeric_limits<int>::min();
  for (const auto& tensor : tensors) {
    const int score = want_points ? pts_score(tensor, view) : conf_score(tensor, view);
    if (score > best_score) {
      best = &tensor;
      best_score = score;
    }
  }

  if (!best || best_score == std::numeric_limits<int>::min()) {
    throw std::runtime_error(want_points ? "MASt3R: could not find pts3d output"
                                         : "MASt3R: could not find confidence output");
  }
  return *best;
}

std::vector<const char*> c_names(const std::vector<std::string>& names) {
  std::vector<const char*> result;
  result.reserve(names.size());
  for (const auto& name : names) result.push_back(name.c_str());
  return result;
}

}  // namespace

namespace stargazer::mast3r {

std::vector<float> preprocess_image(const cv::Mat& bgr, const camera_intrin_t& intrin) {
  return dust3r::preprocess_image(bgr, intrin);
}

std::vector<float> preprocess_image(const cv::Mat& bgr) { return dust3r::preprocess_image(bgr); }

struct mast3r_inference::impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING};
  Ort::Session session{nullptr};
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  static constexpr int64_t batch = 1;
  static constexpr int64_t ch = 3;
  static constexpr int64_t H = ONNX_H;
  static constexpr int64_t W = ONNX_W;
  static constexpr size_t img_elems = batch * ch * H * W;

  explicit impl(const std::string& model_path) {
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
      spdlog::warn("MASt3R: CUDA EP not available: {}", e.what());
    }
#else
    try {
      OrtMIGraphXProviderOptions mgx{};
      mgx.device_id = 0;
      mgx.migraphx_arena_extend_strategy = 1;
      mgx.migraphx_mem_limit = 4ULL * 1024 * 1024 * 1024;
      session_options.AppendExecutionProvider_MIGraphX(mgx);
    } catch (const Ort::Exception& e) {
      spdlog::warn("MASt3R: MIGraphX EP not available: {}", e.what());
    }
#endif

    session = Ort::Session(env, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < session.GetInputCount(); ++i) {
      auto name = session.GetInputNameAllocated(i, allocator);
      input_names.emplace_back(name.get());
    }
    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
      auto name = session.GetOutputNameAllocated(i, allocator);
      output_names.emplace_back(name.get());
    }

    if (input_names.size() < 2) {
      throw std::runtime_error("MASt3R: ONNX model must have at least two image inputs");
    }

    spdlog::info("MASt3R: loaded model with {} inputs and {} outputs", input_names.size(),
                 output_names.size());
    for (const auto& name : output_names) {
      spdlog::debug("MASt3R: output '{}'", name);
    }
  }

  inference_result run(const std::vector<float>& img1_nchw, const std::vector<float>& img2_nchw) {
    if (img1_nchw.size() != img_elems || img2_nchw.size() != img_elems) {
      throw std::runtime_error("MASt3R: invalid preprocessed image shape");
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 4> img_dims{batch, ch, H, W};

    auto input1 =
        Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(img1_nchw.data()),
                                        img1_nchw.size(), img_dims.data(), img_dims.size());
    auto input2 =
        Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(img2_nchw.data()),
                                        img2_nchw.size(), img_dims.data(), img_dims.size());

    std::array<Ort::Value, 2> inputs{std::move(input1), std::move(input2)};
    auto input_name_ptrs = c_names(input_names);
    auto output_name_ptrs = c_names(output_names);

    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_name_ptrs.data(), inputs.data(),
                               inputs.size(), output_name_ptrs.data(), output_name_ptrs.size());

    std::vector<output_tensor> tensors;
    tensors.reserve(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (!outputs[i].IsTensor()) continue;
      const auto type_info = outputs[i].GetTensorTypeAndShapeInfo();
      if (type_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) continue;

      output_tensor tensor;
      tensor.name = output_names[i];
      tensor.shape = type_info.GetShape();
      const size_t elements = tensor_elements(tensor.shape);
      const float* data = outputs[i].GetTensorData<float>();
      tensor.data.assign(data, data + elements);
      spdlog::debug("MASt3R: output '{}' elements={} rank={}", tensor.name, elements,
                    tensor.shape.size());
      tensors.push_back(std::move(tensor));
    }

    inference_result result;
    result.pts3d_1 = select_tensor(tensors, 1, true).data;
    result.conf_1 = select_tensor(tensors, 1, false).data;
    result.pts3d_2 = select_tensor(tensors, 2, true).data;
    result.conf_2 = select_tensor(tensors, 2, false).data;
    return result;
  }
};

mast3r_inference::mast3r_inference(const std::string& model_path)
    : pimpl_(std::make_unique<impl>(model_path)) {}

mast3r_inference::~mast3r_inference() = default;

inference_result mast3r_inference::inference(const std::vector<float>& img1_nchw,
                                             const std::vector<float>& img2_nchw) {
  return pimpl_->run(img1_nchw, img2_nchw);
}

}  // namespace stargazer::mast3r
