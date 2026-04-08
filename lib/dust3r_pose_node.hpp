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
#include "dust3r_optimizer.hpp"
#include "messages.hpp"
#include "parameters.hpp"

namespace stargazer {

using namespace coalsack;

class dust3r_pose_node : public coalsack::graph_node {
  mutable std::mutex cameras_mtx_;
  std::unordered_map<std::string, camera_t> cameras_;
  std::unordered_map<std::string, std::string> camera_name_to_id_;

  std::atomic<bool> armed_{false};

  std::string model_path_;

  mutable std::mutex engine_mtx_;
  std::unique_ptr<dust3r::dust3r_inference> engine_;

  graph_edge_ptr output_;

 public:
  dust3r_pose_node()
      : coalsack::graph_node(), output_(std::make_shared<coalsack::graph_edge>(this)) {
    set_output(output_);
  }

  virtual ~dust3r_pose_node() = default;

  virtual std::string get_proc_name() const override { return "dust3r_pose"; }

  void set_model_path(const std::string& path) { model_path_ = path; }

  template <typename Archive>
  void serialize(Archive&) {}

  virtual void process(std::string input_name, coalsack::graph_message_ptr message) override {
    if (input_name == "estimate") {
      armed_.store(true);
      spdlog::info("dust3r_pose_node: armed, waiting for next frame");
      return;
    }

    static constexpr std::string_view camera_prefix = "camera.";
    if (input_name.rfind(camera_prefix.data(), 0) == 0) {
      const auto cam_name = input_name.substr(camera_prefix.size());
      if (auto cam_msg = std::dynamic_pointer_cast<camera_message>(message)) {
        std::lock_guard lock(cameras_mtx_);
        cameras_[cam_name] = cam_msg->get_camera();
        return;
      }
      if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message)) {
        for (const auto& [field_name, field] : obj_msg->get_fields()) {
          if (auto cam_msg2 = std::dynamic_pointer_cast<camera_message>(field)) {
            std::lock_guard lock(cameras_mtx_);
            cameras_[cam_name] = cam_msg2->get_camera();
            camera_name_to_id_[cam_name] = field_name;
            break;
          }
        }
      }
      return;
    }

    auto obj_msg = std::dynamic_pointer_cast<object_message>(message);
    if (!obj_msg) return;

    if (!armed_.load()) return;

    std::unordered_map<std::string, camera_t> cameras;
    {
      std::lock_guard lock(cameras_mtx_);
      cameras = cameras_;
    }

    const auto& obj = *obj_msg;
    std::vector<std::pair<std::string, cv::Mat>> named_images;

    for (const auto& [name, field] : obj.get_fields()) {
      if (cameras.find(name) == cameras.end()) continue;

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

    std::sort(named_images.begin(), named_images.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<std::string> names;
    std::vector<cv::Mat> bgr_images;
    names.reserve(named_images.size());
    bgr_images.reserve(named_images.size());
    for (auto& [n, img] : named_images) {
      names.push_back(n);
      bgr_images.push_back(std::move(img));
    }

    const int N = static_cast<int>(names.size());
    const int expected = static_cast<int>(cameras.size());
    if (N < 2 || N < expected) {
      spdlog::info("dust3r_pose_node: waiting for cameras ({}/{} ready), skipping frame", N,
                   expected);
      return;
    }

    armed_.store(false);
    spdlog::info("dust3r_pose_node: processing frame");

    std::vector<std::vector<float>> preprocessed(N);
    for (int i = 0; i < N; ++i) {
      preprocessed[i] = dust3r::preprocess_image(bgr_images[i], cameras.at(names[i]).intrin);
    }

    {
      std::lock_guard lock(engine_mtx_);
      if (!engine_) {
        if (model_path_.empty()) {
          spdlog::error("dust3r_pose_node: model_path not set");
          return;
        }
        spdlog::info("dust3r_pose_node: loading model from {}", model_path_);
        engine_ = std::make_unique<dust3r::dust3r_inference>(model_path_);
        spdlog::info("dust3r_pose_node: model loaded");
      }
    }

    std::vector<dust3r::pair_result> pair_results;
    pair_results.reserve(N * (N - 1));

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        if (i == j) continue;

        auto res = engine_->inference(preprocessed[i], preprocessed[j]);

        dust3r::pair_result pr;
        pr.idx1 = i;
        pr.idx2 = j;
        pr.view1.camera_name = names[i];
        pr.view1.pts3d = std::move(res.pts3d_1);
        pr.view1.conf = std::move(res.conf_1);
        pr.view2.camera_name = names[j];
        pr.view2.pts3d = std::move(res.pts3d_2);
        pr.view2.conf = std::move(res.conf_2);

        pair_results.push_back(std::move(pr));
      }
    }

    std::unordered_map<std::string, dust3r::aligned_pose> poses;
    try {
      poses = dust3r::align_global(names, pair_results, cameras);
    } catch (const std::exception& ex) {
      spdlog::error("dust3r_pose_node: alignment failed: {}", ex.what());
      return;
    }

    try {
      poses = dust3r::refine_global_alignment(names, pair_results, poses, cameras);
    } catch (const std::exception& ex) {
      spdlog::warn("dust3r_pose_node: refinement skipped: {}", ex.what());
    }

    auto out_obj = std::make_shared<object_message>();

    for (const auto& [cam_name, pose] : poses) {
      auto it = cameras.find(cam_name);
      if (it == cameras.end()) continue;

      camera_t cam = it->second;
      const glm::mat3 R_w2c = glm::transpose(pose.rotation);
      cam.extrin.rotation = R_w2c;
      cam.extrin.translation = -(R_w2c * pose.translation);

      auto cam_msg = std::make_shared<camera_message>(cam);
      const auto id_it = camera_name_to_id_.find(cam_name);
      const auto& field_key = (id_it != camera_name_to_id_.end()) ? id_it->second : cam_name;
      out_obj->add_field(field_key, cam_msg);
    }

    output_->send(out_obj);
    spdlog::info("dust3r_pose_node: emitted poses for {} cameras", poses.size());
  }
};

}  // namespace stargazer

COALSACK_REGISTER_NODE(stargazer::dust3r_pose_node, coalsack::graph_node)
