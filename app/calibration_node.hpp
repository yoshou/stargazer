#pragma once

#include <spdlog/spdlog.h>

#include <glm/glm.hpp>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "calibration.hpp"
#include "graph_proc.h"
#include "messages.hpp"
#include "parameters.hpp"

namespace stargazer {

using namespace coalsack;
using namespace stargazer::calibration;

class calibration_node : public graph_node {
  bool only_extrinsic;
  bool robust;

  observed_points_frames observed_frames;

  mutable std::mutex cameras_mtx;
  std::vector<std::string> camera_names;
  std::unordered_map<std::string, camera_t> cameras;
  std::unordered_map<std::string, camera_t> calibrated_cameras;

  graph_edge_ptr output;

 public:
  calibration_node()
      : graph_node(),
        only_extrinsic(true),
        robust(false),
        output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "calibration"; }

  void set_cameras(const std::unordered_map<std::string, camera_t>& cameras) {
    this->cameras = cameras;
  }

  void set_only_extrinsic(bool only_extrinsic) { this->only_extrinsic = only_extrinsic; }

  void set_robust(bool robust) { this->robust = robust; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cameras, only_extrinsic, robust);
  }

  size_t get_num_frames(std::string name) const { return observed_frames.get_num_points(name); }

  const std::vector<observed_points_t> get_observed_points(std::string name) const {
    return observed_frames.get_observed_points(name);
  }

  void calibrate() {
    if (!initialize_cameras(cameras, camera_names, observed_frames)) {
      spdlog::error("Failed to initialize cameras");
      return;
    }

    bundle_adjust_data ba_data;

    prepare_bundle_adjustment(camera_names, cameras, observed_frames, ba_data);

    bundle_adjustment(ba_data, only_extrinsic, robust);

    {
      calibrated_cameras = cameras;

      {
        for (size_t i = 0; i < camera_names.size(); i++) {
          calibrated_cameras[camera_names[i]].extrin.rotation =
              glm::mat3(ba_data.get_camera_extrinsic(i));
          calibrated_cameras[camera_names[i]].extrin.translation =
              glm::vec3(ba_data.get_camera_extrinsic(i)[3]);
        }
      }

      if (!only_extrinsic) {
        for (size_t i = 0; i < camera_names.size(); i++) {
          const auto intrin = &ba_data.mutable_camera(i)[6];
          calibrated_cameras[camera_names[i]].intrin.fx = intrin[0];
          calibrated_cameras[camera_names[i]].intrin.fy = intrin[1];
          calibrated_cameras[camera_names[i]].intrin.cx = intrin[2];
          calibrated_cameras[camera_names[i]].intrin.cy = intrin[3];
          calibrated_cameras[camera_names[i]].intrin.coeffs[0] = intrin[4];
          calibrated_cameras[camera_names[i]].intrin.coeffs[1] = intrin[5];
          calibrated_cameras[camera_names[i]].intrin.coeffs[4] = intrin[6];
          calibrated_cameras[camera_names[i]].intrin.coeffs[2] = intrin[7];
          calibrated_cameras[camera_names[i]].intrin.coeffs[3] = intrin[8];
        }
      }
    }
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (input_name == "calibrate") {
      camera_names.clear();

      if (auto camera_msg = std::dynamic_pointer_cast<object_message>(message)) {
        for (const auto& [name, field] : camera_msg->get_fields()) {
          if (auto camera_msg = std::dynamic_pointer_cast<camera_message>(field)) {
            std::lock_guard lock(cameras_mtx);
            cameras[name] = camera_msg->get_camera();
            camera_names.push_back(name);
          }
        }
      }

      calibrate();

      std::shared_ptr<object_message> msg(new object_message());
      for (const auto& [name, camera] : calibrated_cameras) {
        std::shared_ptr<camera_message> camera_msg(new camera_message(camera));
        msg->add_field(name, camera_msg);
      }
      output->send(msg);

      return;
    }

    if (auto points_msg = std::dynamic_pointer_cast<float2_list_message>(message)) {
      std::vector<glm::vec2> points;
      for (const auto& pt : points_msg->get_data()) {
        points.emplace_back(pt.x, pt.y);
      }
      observed_frames.add_frame_points(points_msg->get_frame_number(), input_name, points);
    }
  }
};

}  // namespace stargazer

COALSACK_REGISTER_NODE(stargazer::calibration_node, coalsack::graph_node)
