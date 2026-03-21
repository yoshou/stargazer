#pragma once

#include <glm/glm.hpp>
#include <mutex>
#include <opencv2/opencv.hpp>

#include "coalsack/core/graph_proc.h"
#include "coalsack/image/graph_proc_cv.h"
#include "coalsack/tensor/graph_proc_tensor.h"
#include "image_reconstruct_node.hpp"
#include "messages.hpp"
#include "mvpose.hpp"
#include "parameters.hpp"

namespace stargazer {

using namespace coalsack;

class mvpose_reconstruct_node : public image_reconstruct_node {
  mutable std::mutex cameras_mtx;
  std::map<std::string, camera_t> cameras;
  mutable std::mutex axis_mtx;
  glm::mat4 axis;
  graph_edge_ptr output;

  stargazer::mvpose::mvpose pose_estimator;

 public:
  mvpose_reconstruct_node()
      : image_reconstruct_node(), cameras(), axis(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "mvpose_reconstruct"; }

  void set_cameras(const std::map<std::string, camera_t>& new_cameras) override {
    std::lock_guard lock(cameras_mtx);
    cameras = new_cameras;
  }

  void set_axis(const glm::mat4& new_axis) override {
    std::lock_guard lock(axis_mtx);
    axis = new_axis;
  }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cameras, axis);
  }

  stargazer::mvpose::mvpose_inference_result reconstruct(
      const std::map<std::string, camera_t>& cameras, const std::map<std::string, cv::Mat>& frame,
      const glm::mat4& axis, std::vector<std::string>& out_view_names) {
    using namespace stargazer::mvpose;

    std::vector<cv::Mat> images_list;
    std::vector<camera_t> cameras_list;

    if (frame.size() <= 1) {
      return {};
    }

    for (const auto& [camera_name, image] : frame) {
      out_view_names.push_back(camera_name);
    }

    for (size_t i = 0; i < frame.size(); i++) {
      const auto name = out_view_names[i];

      camera_t camera;

      const auto& src_camera = cameras.at(name);

      camera.intrin.fx = src_camera.intrin.fx;
      camera.intrin.fy = src_camera.intrin.fy;
      camera.intrin.cx = src_camera.intrin.cx;
      camera.intrin.cy = src_camera.intrin.cy;
      camera.intrin.coeffs[0] = src_camera.intrin.coeffs[0];
      camera.intrin.coeffs[1] = src_camera.intrin.coeffs[1];
      camera.intrin.coeffs[2] = src_camera.intrin.coeffs[2];
      camera.intrin.coeffs[3] = src_camera.intrin.coeffs[3];
      camera.intrin.coeffs[4] = src_camera.intrin.coeffs[4];

      const auto camera_pose = src_camera.extrin.transform_matrix() * glm::inverse(axis);

      for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
          camera.extrin.rotation[i][j] = camera_pose[i][j];
        }
        camera.extrin.translation[i] = camera_pose[3][i];
      }

      cameras_list.push_back(camera);
      images_list.push_back(frame.at(name));
    }

    const auto infer_result = pose_estimator.inference_with_matches(images_list, cameras_list);

    return infer_result;
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
      if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message)) {
        for (const auto& [id, field] : obj_msg->get_fields()) {
          if (auto scene_msg = std::dynamic_pointer_cast<scene_message>(field)) {
            std::lock_guard lock(axis_mtx);
            axis = scene_msg->get_scene().axis;
          }
        }
      }

      return;
    }

    static constexpr std::string_view single_camera_prefix = "camera.";
    if (input_name.rfind(single_camera_prefix.data(), 0) == 0) {
      const auto camera_name = input_name.substr(single_camera_prefix.size());
      if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message)) {
        for (const auto& [id, field] : obj_msg->get_fields()) {
          if (auto cam_msg = std::dynamic_pointer_cast<camera_message>(field)) {
            std::lock_guard lock(cameras_mtx);
            cameras[camera_name] = cam_msg->get_camera();
          }
        }
      }
      return;
    }

    if (auto frame_msg = std::dynamic_pointer_cast<frame_message<object_message>>(message)) {
      const auto obj_msg = frame_msg->get_data();

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

      std::vector<std::string> view_names;
      const auto infer_result = reconstruct(cameras, images, axis, view_names);

      // Convert mvpose-specific result to generic format
      reconstruction_result_t generic_result;
      generic_result.num_keypoints = infer_result.num_joints;

      // Convert 3D points
      generic_result.points3d.reserve(infer_result.points3d.size());
      for (const auto& p : infer_result.points3d) {
        generic_result.points3d.push_back({p.x, p.y, p.z});
      }

      // Convert per-view 2D detections
      generic_result.views.reserve(infer_result.views.size());
      for (size_t i = 0; i < infer_result.views.size(); ++i) {
        view_result_t view;
        view.name = view_names[i];
        view.detections.reserve(infer_result.views[i].poses.size());

        for (const auto& mvpose_pose : infer_result.views[i].poses) {
          detection2d_t detection;
          detection.bbox.left = mvpose_pose.bbox.x;
          detection.bbox.top = mvpose_pose.bbox.y;
          detection.bbox.right = mvpose_pose.bbox.x + mvpose_pose.bbox.width;
          detection.bbox.bottom = mvpose_pose.bbox.y + mvpose_pose.bbox.height;
          detection.bbox_score = mvpose_pose.bbox_score;

          detection.keypoints.reserve(mvpose_pose.joints.size());
          for (const auto& joint : mvpose_pose.joints) {
            detection.keypoints.push_back({joint.x, joint.y});
          }
          detection.scores = mvpose_pose.scores;

          view.detections.push_back(std::move(detection));
        }
        generic_result.views.push_back(std::move(view));
      }

      // Convert matches
      generic_result.matches = infer_result.matched_list;

      auto result_msg = std::make_shared<reconstruction_result_message>();
      result_msg->set_result(generic_result);
      result_msg->set_cameras(cameras);
      result_msg->set_axis(axis);
      result_msg->set_frame_number(frame_msg->get_frame_number());
      result_msg->set_timestamp(frame_msg->get_timestamp());

      output->send(result_msg);
    }
  }

};

}  // namespace stargazer

COALSACK_REGISTER_NODE(stargazer::mvpose_reconstruct_node, stargazer::image_reconstruct_node)
