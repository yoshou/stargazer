#pragma once

#include <glm/gtc/matrix_inverse.hpp>
#include <mutex>

#include "coalsack/core/graph_proc.h"
#include "messages.hpp"
#include "reconstruction.hpp"

namespace stargazer {

class epipolar_reconstruct_node : public coalsack::graph_node {
  mutable std::mutex cameras_mtx;
  std::map<std::string, camera_t> cameras;
  mutable std::mutex axis_mtx;
  glm::mat4 axis;
  coalsack::graph_edge_ptr output;

 public:
  epipolar_reconstruct_node()
      : coalsack::graph_node(),
        cameras(),
        axis(),
        output(std::make_shared<coalsack::graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "epipolar_reconstruct"; }

  void set_cameras(const std::map<std::string, camera_t>& new_cameras) {
    std::lock_guard lock(cameras_mtx);
    cameras = new_cameras;
  }

  void set_axis(const glm::mat4& new_axis) {
    std::lock_guard lock(axis_mtx);
    axis = new_axis;
  }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cameras, axis);
  }

  virtual std::optional<coalsack::property_value> get_property(
      const std::string& key) const override {
    if (key == "axis") {
      std::lock_guard lock(axis_mtx);
      coalsack::mat4 result;
      for (int i = 0; i < 16; ++i) result.data[i] = (&axis[0][0])[i];
      return result;
    }
    static constexpr std::string_view camera_prefix = "camera.";
    if (key.rfind(camera_prefix.data(), 0) == 0) {
      const auto camera_name = key.substr(camera_prefix.size());
      std::lock_guard lock(cameras_mtx);
      auto it = cameras.find(camera_name);
      if (it != cameras.end()) {
        const auto& cam = it->second;
        coalsack::camera_t result;
        result.width = static_cast<int>(cam.width);
        result.height = static_cast<int>(cam.height);
        result.ppx = cam.intrin.cx;
        result.ppy = cam.intrin.cy;
        result.fx = cam.intrin.fx;
        result.fy = cam.intrin.fy;
        for (int i = 0; i < 5; ++i) result.coeffs[i] = cam.intrin.coeffs[i];
        const glm::mat4 inv = glm::inverse(cam.extrin.transform_matrix());
        for (int i = 0; i < 16; ++i) result.pose.data[i] = (&inv[0][0])[i];
        return result;
      }
    }
    return std::nullopt;
  }

  virtual void process(std::string input_name, coalsack::graph_message_ptr message) override {
    if (input_name == "cameras") {
      if (auto camera_msg = std::dynamic_pointer_cast<object_message>(message)) {
        for (const auto& [name, field] : camera_msg->get_fields()) {
          if (auto camera_msg = std::dynamic_pointer_cast<camera_message>(field)) {
            std::lock_guard lock(cameras_mtx);
            cameras[name] = camera_msg->get_camera();
          }
        }
      }

      return;
    }
    if (input_name == "axis") {
      if (auto mat4_msg = std::dynamic_pointer_cast<mat4_message>(message)) {
        std::lock_guard lock(axis_mtx);
        axis = mat4_msg->get_data();
      }

      return;
    }

    if (auto frame_msg = std::dynamic_pointer_cast<frame_message<object_message>>(message)) {
      const auto obj_msg = frame_msg->get_data();

      std::vector<std::vector<glm::vec2>> camera_pts;
      std::vector<camera_t> camera_list;

      std::map<std::string, camera_t> cameras;
      glm::mat4 axis;
      {
        std::lock_guard lock(cameras_mtx);
        cameras = this->cameras;
      }
      {
        std::lock_guard lock(axis_mtx);
        axis = this->axis;
      }

      for (const auto& [name, field] : obj_msg.get_fields()) {
        if (auto points_msg = std::dynamic_pointer_cast<float2_list_message>(field)) {
          if (cameras.find(name) == cameras.end()) {
            continue;
          }
          const auto& camera = cameras.at(name);
          std::vector<glm::vec2> pts;
          for (const auto& pt : points_msg->get_data()) {
            pts.push_back(glm::vec2(pt.x, pt.y));
          }
          camera_pts.push_back(pts);
          camera_list.push_back(camera);
        }
      }

      const auto markers = stargazer::reconstruction::reconstruct(camera_list, camera_pts, axis);

      auto marker_msg = std::make_shared<float3_list_message>();
      std::vector<float3> marker_data;
      for (const auto& marker : markers) {
        marker_data.push_back({marker.x, marker.y, marker.z});
      }
      marker_msg->set_data(marker_data);
      marker_msg->set_frame_number(frame_msg->get_frame_number());
      output->send(marker_msg);
    }
  }
};

}  // namespace stargazer

COALSACK_REGISTER_NODE(stargazer::epipolar_reconstruct_node, coalsack::graph_node)
