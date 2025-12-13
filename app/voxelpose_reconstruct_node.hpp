#pragma once

#include <glm/glm.hpp>
#include <mutex>
#include <opencv2/opencv.hpp>

#include "graph_proc.h"
#include "graph_proc_cv.h"
#include "graph_proc_tensor.h"
#include "image_reconstruct_node.hpp"
#include "messages.hpp"
#include "parameters.hpp"
#include "voxelpose.hpp"

#define PANOPTIC

namespace stargazer {

using namespace coalsack;

class voxelpose_reconstruct_node : public image_reconstruct_node {
  mutable std::mutex cameras_mtx;
  std::map<std::string, camera_t> cameras;
  glm::mat4 axis;
  graph_edge_ptr output;

  std::vector<std::string> names;
  coalsack::tensor<float, 4> features;
  mutable std::mutex features_mtx;

  stargazer::voxelpose::voxelpose pose_estimator;

 public:
  voxelpose_reconstruct_node()
      : image_reconstruct_node(), cameras(), axis(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "voxelpose_reconstruct"; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cameras, axis);
  }

  std::vector<glm::vec3> reconstruct(const std::map<std::string, camera_t>& cameras,
                                     const std::map<std::string, cv::Mat>& frame,
                                     const glm::mat4& axis) {
    using namespace stargazer::voxelpose;

    std::vector<std::string> names;
    std::vector<cv::Mat> images_list;
    std::vector<camera_data> cameras_list;

    if (frame.size() <= 1) {
      return std::vector<glm::vec3>();
    }

    for (const auto& [camera_name, image] : frame) {
      names.push_back(camera_name);
    }

    for (size_t i = 0; i < frame.size(); i++) {
      const auto name = names[i];
      camera_data camera;

      const auto& src_camera = cameras.at(name);

      camera.fx = src_camera.intrin.fx;
      camera.fy = src_camera.intrin.fy;
      camera.cx = src_camera.intrin.cx;
      camera.cy = src_camera.intrin.cy;
      camera.k[0] = src_camera.intrin.coeffs[0];
      camera.k[1] = src_camera.intrin.coeffs[1];
      camera.k[2] = src_camera.intrin.coeffs[4];
      camera.p[0] = src_camera.intrin.coeffs[2];
      camera.p[1] = src_camera.intrin.coeffs[3];

      glm::mat4 gl_to_cv(1.f);
      gl_to_cv[0] = glm::vec4(1.f, 0.f, 0.f, 0.f);
      gl_to_cv[1] = glm::vec4(0.f, -1.f, 0.f, 0.f);
      gl_to_cv[2] = glm::vec4(0.f, 0.f, -1.f, 0.f);

      glm::mat4 m(1.f);
      m[0] = glm::vec4(1.f, 0.f, 0.f, 0.f);
      m[1] = glm::vec4(0.f, 0.f, 1.f, 0.f);
      m[2] = glm::vec4(0.f, -1.f, 0.f, 0.f);

      const auto camera_pose =
          axis * glm::inverse(src_camera.extrin.transform_matrix() * gl_to_cv * m);

      for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
          camera.rotation[i][j] = camera_pose[i][j];
        }
        camera.translation[i] = camera_pose[3][i] * 1000.0;
      }

      cameras_list.push_back(camera);
      images_list.push_back(frame.at(name));
    }

#ifdef PANOPTIC
    std::array<float, 3> grid_center = {0.0f, -500.0f, 800.0f};
#else
    std::array<float, 3> grid_center = {0.0f, 0.0f, 0.0f};
#endif

    pose_estimator.set_grid_center(grid_center);

    const auto points = pose_estimator.inference(images_list, cameras_list);

    coalsack::tensor<float, 4> heatmaps(
        {pose_estimator.get_heatmap_width(), pose_estimator.get_heatmap_height(),
         pose_estimator.get_num_joints(), (uint32_t)images_list.size()});
    pose_estimator.copy_heatmap_to(images_list.size(), heatmaps.get_data());

    {
      std::lock_guard lock(features_mtx);
      this->names = names;
      this->features = std::move(heatmaps);
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
    coalsack::tensor<float, 4> features;
    std::vector<std::string> names;
    {
      std::lock_guard lock(features_mtx);
      features = this->features;
      names = this->names;
    }
    std::map<std::string, cv::Mat> result;
    if (features.get_size() == 0) {
      return result;
    }
    for (size_t i = 0; i < names.size(); i++) {
      const auto name = names[i];
      const auto heatmap =
          features
              .view<3>({features.shape[0], features.shape[1], features.shape[2], 0},
                       {0, 0, 0, static_cast<uint32_t>(i)})
              .contiguous()
              .sum<1>({2});

      cv::Mat heatmap_mat;
      cv::Mat(heatmap.shape[1], heatmap.shape[0], CV_32FC1, (float*)heatmap.get_data())
          .clone()
          .convertTo(heatmap_mat, CV_8U, 255);
      cv::resize(heatmap_mat, heatmap_mat, cv::Size(960, 540));
      cv::cvtColor(heatmap_mat, heatmap_mat, cv::COLOR_GRAY2BGR);

      result[name] = heatmap_mat;
    }
    return result;
  }
};

}  // namespace stargazer

COALSACK_REGISTER_NODE(stargazer::voxelpose_reconstruct_node, stargazer::image_reconstruct_node)
