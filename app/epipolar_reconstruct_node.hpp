#pragma once

#include <mutex>

#include "graph_proc.h"
#include "messages.hpp"
#include "reconstruction.hpp"

namespace stargazer {

using namespace coalsack;

class epipolar_reconstruct_node : public graph_node {
  mutable std::mutex cameras_mtx;
  std::map<std::string, camera_t> cameras;
  mutable std::mutex axis_mtx;
  glm::mat4 axis;
  graph_edge_ptr output;

 public:
  epipolar_reconstruct_node()
      : graph_node(), cameras(), axis(), output(std::make_shared<graph_edge>(this)) {
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

  virtual void process(std::string input_name, graph_message_ptr message) override {
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
