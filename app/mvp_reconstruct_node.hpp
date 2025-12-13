#pragma once

#include <spdlog/spdlog.h>

#include <glm/glm.hpp>
#include <mutex>
#include <opencv2/opencv.hpp>

#include "graph_proc.h"
#include "graph_proc_cv.h"
#include "image_reconstruct_node.hpp"
#include "messages.hpp"
#include "mvp.hpp"
#include "parameters.hpp"

namespace stargazer {

using namespace coalsack;

class mvp_reconstruct_node : public image_reconstruct_node {
  mutable std::mutex cameras_mtx;
  std::map<std::string, camera_t> cameras;
  glm::mat4 axis;
  graph_edge_ptr output;

  std::vector<std::string> names;
  mutable std::mutex features_mtx;

  stargazer::mvp::mvp pose_estimator;

 public:
  mvp_reconstruct_node()
      : image_reconstruct_node(), cameras(), axis(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "mvp_reconstruct"; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cameras, axis);
  }

  std::vector<glm::vec3> reconstruct(const std::map<std::string, camera_t>& cameras,
                                     const std::map<std::string, cv::Mat>& frame,
                                     const glm::mat4& axis) {
    using namespace stargazer::mvp;

    std::vector<std::string> names;
    std::vector<cv::Mat> images_list;
    std::vector<camera_data> cameras_list;

    // MVP requires exactly 5 views
    if (frame.size() != 5) {
      return std::vector<glm::vec3>();
    }

    for (const auto& [camera_name, image] : frame) {
      names.push_back(camera_name);
    }

    for (size_t i = 0; i < frame.size(); i++) {
      const auto name = names[i];

      if (cameras.find(name) == cameras.end()) {
        spdlog::warn("Camera {} not found", name);
        return std::vector<glm::vec3>();
      }

      const auto& cam = cameras.at(name);

      // Convert camera_t to mvp::camera_data
      // Convert back to original Panoptic coordinate system for MVP model
      camera_data mvp_cam;
      mvp_cam.fx = cam.intrin.fx;
      mvp_cam.fy = cam.intrin.fy;
      mvp_cam.cx = cam.intrin.cx;
      mvp_cam.cy = cam.intrin.cy;

      for (int j = 0; j < 5; j++) {
        mvp_cam.dist_coeff[j] = cam.intrin.coeffs[j];
      }

      // cam.extrin.rotation is glm::mat3 (column-major)
      // Convert to row-major and apply flip_yz inverse transform
      double R_stored[3][3];
      for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
          R_stored[row][col] = cam.extrin.rotation[col][row];
        }
      }

      // Apply flip_yz = [[1,0,0],[0,-1,0],[0,0,-1]] to restore original rotation
      for (int row = 0; row < 3; row++) {
        mvp_cam.rotation[row][0] = R_stored[row][0];
        mvp_cam.rotation[row][1] = -R_stored[row][1];
        mvp_cam.rotation[row][2] = -R_stored[row][2];
      }

      // Translation in mm
      mvp_cam.translation[0] = cam.extrin.translation.x * 1000.0;  // m to mm
      mvp_cam.translation[1] = cam.extrin.translation.y * 1000.0;
      mvp_cam.translation[2] = cam.extrin.translation.z * 1000.0;

      cameras_list.push_back(mvp_cam);
      images_list.push_back(frame.at(name));
    }

    std::array<float, 3> grid_center = {0.0f, -500.0f, 800.0f};
    pose_estimator.set_grid_center(grid_center);

    const auto points = pose_estimator.inference(images_list, cameras_list);

    {
      std::lock_guard lock(features_mtx);
      this->names = names;
    }

    return points;
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
        axis = mat4_msg->get_data();
      }

      return;
    }

    if (auto frame_msg = std::dynamic_pointer_cast<frame_message<object_message>>(message)) {
      const auto obj_msg = frame_msg->get_data();

      std::map<std::string, camera_t> cameras;
      {
        std::lock_guard lock(cameras_mtx);
        cameras = this->cameras;
      }

      std::map<std::string, cv::Mat> images;

      for (const auto& [name, field] : obj_msg.get_fields()) {
        if (auto img_msg = std::dynamic_pointer_cast<image_message>(field)) {
          if (cameras.find(name) == cameras.end()) {
            continue;
          }
          const auto& image = img_msg->get_image();
          cv::Mat img(image.get_height(), image.get_width(), convert_to_cv_type(image.get_format()),
                      (void*)image.get_data(), image.get_stride());
          images[name] = img;
        }
      }

      const auto markers = reconstruct(cameras, images, axis);

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

  std::map<std::string, cv::Mat> get_features() const override {
    // MVP does not expose intermediate features like heatmaps
    return std::map<std::string, cv::Mat>();
  }
};

}  // namespace stargazer

COALSACK_REGISTER_NODE(stargazer::mvp_reconstruct_node, stargazer::image_reconstruct_node)
