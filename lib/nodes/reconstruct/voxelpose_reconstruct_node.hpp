/// @file voxelpose_reconstruct_node.hpp
/// @brief VoxelPose voxel-space 3D human pose reconstruction node.
/// @ingroup reconstruction_nodes
#pragma once

#include <glm/glm.hpp>
#include <mutex>
#include <opencv2/opencv.hpp>

#include "coalsack/core/graph_proc.h"
#include "coalsack/image/graph_proc_cv.h"
#include "coalsack/tensor/graph_proc_tensor.h"
#include "image_reconstruct_node.hpp"
#include "messages.hpp"
#include "parameters.hpp"
#include "voxelpose.hpp"

#define PANOPTIC

namespace stargazer {

using namespace coalsack;

/// @brief Voxel-space 3D human pose reconstruction using the VoxelPose neural network model.
/// @details Builds a 3D voxel feature volume from multi-view image frames and runs
///          VoxelPose inference to locate 3D joint positions.  Camera calibration is
///          kept current via @b "cameras", @b "axis", and @b "camera.*" inputs.  Each
///          complete image batch produces one `reconstruction_result_message` on
///          @b "default".
///
/// @par Inputs
/// - @b "cameras"  — `object_message` — bulk camera parameter update
/// - @b "axis"     — `object_message` — scene coordinate axis
/// - @b "camera.*" — `object_message` — per-camera parameter update
/// - @b "default" — `frame_message<object_message>` — named image frames per camera
///
/// @par Outputs
/// - @b "default" — `reconstruction_result_message` — 3D joint positions
///
/// @par Properties
/// (none)
///
/// @see mvpose_reconstruct_node, mvp_reconstruct_node, image_reconstruct_node
class voxelpose_reconstruct_node : public image_reconstruct_node {
  mutable std::mutex cameras_mtx;
  std::map<std::string, camera_t> cameras;
  mutable std::mutex axis_mtx;
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

      const auto markers = reconstruct(cameras, images, axis);

      reconstruction_result_t generic_result;
      generic_result.num_keypoints = markers.size();
      generic_result.points3d.reserve(markers.size());
      for (const auto& marker : markers) {
        generic_result.points3d.push_back({marker.x, marker.y, marker.z});
      }

      coalsack::tensor<float, 4> features;
      std::vector<std::string> names;
      {
        std::lock_guard lock(features_mtx);
        features = this->features;
        names = this->names;
      }

      if (features.get_size() != 0) {
        for (size_t i = 0; i < names.size(); i++) {
          const auto heatmap =
              features
                  .view<3>({features.shape[0], features.shape[1], features.shape[2], 0},
                           {0, 0, 0, static_cast<uint32_t>(i)})
                  .contiguous()
                  .sum<1>({2});

          cv::Mat heatmap_mat;
          cv::Mat(static_cast<int>(heatmap.shape[1]), static_cast<int>(heatmap.shape[0]), CV_32FC1,
                  (float*)heatmap.get_data())
              .clone()
              .convertTo(heatmap_mat, CV_8U, 255);

          coalsack::image heatmap_image(static_cast<std::uint32_t>(heatmap_mat.cols),
                                        static_cast<std::uint32_t>(heatmap_mat.rows),
                                        static_cast<std::uint32_t>(heatmap_mat.elemSize()),
                                        static_cast<std::uint32_t>(heatmap_mat.step),
                                        heatmap_mat.data);
          heatmap_image.set_format(coalsack::image_format::Y8_UINT);
          generic_result.heatmaps.emplace(names[i], std::move(heatmap_image));
        }
      }

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

COALSACK_REGISTER_NODE(stargazer::voxelpose_reconstruct_node, stargazer::image_reconstruct_node)
