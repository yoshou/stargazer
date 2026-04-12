#pragma once

#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "coalsack/core/graph_proc.h"
#include "coalsack/image/frame_message.h"
#include "coalsack/image/graph_proc_cv.h"
#include "coalsack/image/image_message.h"
#include "dust3r.hpp"
#include "dust3r_alignment.hpp"
#include "dust3r_calibration.hpp"
#include "dust3r_optimizer.hpp"
#include "messages.hpp"
#include "parameters.hpp"

namespace stargazer {

using namespace coalsack;

// Calibration node based on DUSt3R.
// Receives images without prior intrinsics/extrinsics and performs
// BA-based calibration to produce updated camera_t for each camera.
class dust3r_calibration_node : public coalsack::graph_node {
  mutable std::mutex cameras_mtx_;
  // Camera data stored only for ID mapping (camera_name → field_key).
  std::unordered_map<std::string, std::string> camera_name_to_id_;
  std::vector<std::string> camera_names_ordered_;

  std::atomic<bool> armed_{false};

  std::string model_path_;

  mutable std::mutex engine_mtx_;
  std::unique_ptr<dust3r::dust3r_inference> engine_;

  graph_edge_ptr output_;

 public:
  dust3r_calibration_node()
      : coalsack::graph_node(), output_(std::make_shared<coalsack::graph_edge>(this)) {
    set_output(output_);
  }

  virtual ~dust3r_calibration_node() = default;

  virtual std::string get_proc_name() const override { return "dust3r_calibration"; }

  void set_model_path(const std::string& path) { model_path_ = path; }

  template <typename Archive>
  void serialize(Archive&) {}

  virtual void process(std::string input_name, coalsack::graph_message_ptr message) override {
    if (input_name == "estimate") {
      armed_.store(true);
      spdlog::info("dust3r_calibration_node: armed, waiting for next frame");
      return;
    }

    static constexpr std::string_view camera_prefix = "camera.";
    if (input_name.rfind(camera_prefix.data(), 0) == 0) {
      const auto cam_name = input_name.substr(camera_prefix.size());
      if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message)) {
        for (const auto& [field_name, field] : obj_msg->get_fields()) {
          if (std::dynamic_pointer_cast<camera_message>(field)) {
            std::lock_guard lock(cameras_mtx_);
            // Record id mapping; register name if not seen yet
            if (camera_name_to_id_.find(cam_name) == camera_name_to_id_.end()) {
              camera_names_ordered_.push_back(cam_name);
            }
            camera_name_to_id_[cam_name] = field_name;
            break;
          }
        }
      } else if (auto cam_msg = std::dynamic_pointer_cast<camera_message>(message)) {
        (void)cam_msg;
        std::lock_guard lock(cameras_mtx_);
        if (camera_name_to_id_.find(cam_name) == camera_name_to_id_.end()) {
          camera_names_ordered_.push_back(cam_name);
          camera_name_to_id_[cam_name] = cam_name;
        }
      }
      return;
    }

    auto obj_msg = std::dynamic_pointer_cast<object_message>(message);
    if (!obj_msg) return;

    if (!armed_.load()) return;

    std::unordered_map<std::string, std::string> id_map;
    std::vector<std::string> names;
    {
      std::lock_guard lock(cameras_mtx_);
      id_map = camera_name_to_id_;
      names = camera_names_ordered_;
    }

    const auto& obj = *obj_msg;
    std::vector<std::pair<std::string, cv::Mat>> named_images;

    for (const auto& [name, field] : obj.get_fields()) {
      auto img_frame = std::dynamic_pointer_cast<frame_message<coalsack::image>>(field);
      if (!img_frame) continue;
      const auto& img_data = img_frame->get_data();
      if (img_data.empty()) continue;

      const int cv_type = convert_to_cv_type(img_data.get_format());
      cv::Mat mat(static_cast<int>(img_data.get_height()), static_cast<int>(img_data.get_width()),
                  cv_type, const_cast<uint8_t*>(img_data.get_data()), img_data.get_stride());

      cv::Mat bgr;
      if (img_data.get_format() == image_format::B8G8R8_UINT ||
          img_data.get_format() == image_format::B8G8R8A8_UINT) {
        if (img_data.get_format() == image_format::B8G8R8A8_UINT)
          cv::cvtColor(mat, bgr, cv::COLOR_BGRA2BGR);
        else
          bgr = mat.clone();
      } else if (img_data.get_format() == image_format::R8G8B8_UINT) {
        cv::cvtColor(mat, bgr, cv::COLOR_RGB2BGR);
      } else if (img_data.get_format() == image_format::Y8_UINT) {
        cv::cvtColor(mat, bgr, cv::COLOR_GRAY2BGR);
      } else {
        bgr = mat.clone();
      }

      named_images.emplace_back(name, std::move(bgr));
    }

    // Keep only cameras we know about (have received id mapping for),
    // in consistent order
    std::vector<std::pair<std::string, cv::Mat>> ordered_images;
    ordered_images.reserve(names.size());
    for (const auto& n : names) {
      for (auto& [img_name, img] : named_images) {
        if (img_name == n) {
          ordered_images.emplace_back(n, std::move(img));
          break;
        }
      }
    }

    const int N = static_cast<int>(ordered_images.size());
    if (N < 2) {
      spdlog::info("dust3r_calibration_node: need ≥2 cameras ({} ready), skipping frame", N);
      return;
    }

    // Rebuild names to match ordered_images
    std::vector<std::string> ordered_names;
    std::vector<cv::Mat> bgr_images;
    ordered_names.reserve(N);
    bgr_images.reserve(N);
    for (auto& [n, img] : ordered_images) {
      ordered_names.push_back(n);
      bgr_images.push_back(std::move(img));
    }

    const int orig_W = bgr_images[0].cols;
    const int orig_H = bgr_images[0].rows;

    armed_.store(false);
    spdlog::info("dust3r_calibration_node: processing {} cameras ({}x{})", N, orig_W, orig_H);

    // Preprocess: resize only, no undistortion
    std::vector<std::vector<float>> preprocessed(N);
    for (int i = 0; i < N; ++i) {
      preprocessed[i] = dust3r::preprocess_image(bgr_images[i]);
    }

    {
      std::lock_guard lock(engine_mtx_);
      if (!engine_) {
        if (model_path_.empty()) {
          spdlog::error("dust3r_calibration_node: model_path not set");
          return;
        }
        spdlog::info("dust3r_calibration_node: loading model from {}", model_path_);
        engine_ = std::make_unique<dust3r::dust3r_inference>(model_path_);
        spdlog::info("dust3r_calibration_node: model loaded");
      }
    }

    std::vector<dust3r::pair_result> pair_results;
    pair_results.reserve(static_cast<size_t>(N * (N - 1)));
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        if (i == j) continue;
        auto res = engine_->inference(preprocessed[i], preprocessed[j]);
        dust3r::pair_result pr;
        pr.idx1 = i;
        pr.idx2 = j;
        pr.view1.camera_name = ordered_names[i];
        pr.view1.pts3d = std::move(res.pts3d_1);
        pr.view1.conf = std::move(res.conf_1);
        pr.view2.camera_name = ordered_names[j];
        pr.view2.pts3d = std::move(res.pts3d_2);
        pr.view2.conf = std::move(res.conf_2);
        pair_results.push_back(std::move(pr));
      }
    }

    // Global alignment (no known intrinsics/extrinsics)
    std::unordered_map<std::string, dust3r::aligned_pose> poses;
    try {
      poses = dust3r::align_global(ordered_names, pair_results);
      poses = refine_global_alignment(ordered_names, pair_results, poses);
    } catch (const std::exception& ex) {
      spdlog::error("dust3r_calibration_node: alignment failed: {}", ex.what());
      return;
    }

    // Merge shared points and run bundle adjustment
    auto ba_points = dust3r::merge_shared_points(ordered_names, pair_results, poses, orig_W, orig_H,
                                                 /*conf_threshold=*/3.0f,
                                                 /*subsample=*/8,
                                                 /*merge_threshold=*/0.05f);
    dust3r::filter_observations(ba_points, 3);

    if (ba_points.empty()) {
      spdlog::error("dust3r_calibration_node: no shared points after filtering, aborting");
      return;
    }

    // Outlier removal based on initial reprojection error
    dust3r::filter_outlier_observations(ba_points, ordered_names, poses, orig_W, orig_H, 3.0f);
    dust3r::filter_observations(ba_points, 2);  // Re-filter: remove points that lost enough cameras

    double rmse_before = 0.0, rmse_after = 0.0;
    std::vector<camera_t> cam_results;
    try {
      cam_results = dust3r::run_bundle_adjustment(ordered_names, poses, ba_points, orig_W, orig_H,
                                                  &rmse_before, &rmse_after);
    } catch (const std::exception& ex) {
      spdlog::error("dust3r_calibration_node: BA failed: {}", ex.what());
      return;
    }

    spdlog::info("dust3r_calibration_node: BA done, RMSE {} → {} pixels", rmse_before, rmse_after);

    if (cam_results.empty()) {
      spdlog::error("dust3r_calibration_node: BA returned no cameras");
      return;
    }

    auto out_obj = std::make_shared<object_message>();
    for (int i = 0; i < N; ++i) {
      const std::string& cam_name = ordered_names[i];
      const auto id_it = id_map.find(cam_name);
      const std::string& field_key = (id_it != id_map.end()) ? id_it->second : cam_name;

      auto cam_msg = std::make_shared<camera_message>(cam_results[i]);
      out_obj->add_field(field_key, cam_msg);
    }

    output_->send(out_obj);
    spdlog::info("dust3r_calibration_node: emitted calibration for {} cameras", N);
  }
};

}  // namespace stargazer

COALSACK_REGISTER_NODE(stargazer::dust3r_calibration_node, coalsack::graph_node)
