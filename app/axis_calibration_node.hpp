#pragma once

#include <glm/glm.hpp>
#include <iostream>
#include <map>
#include <mutex>
#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "calibration.hpp"
#include "graph_proc.h"
#include "messages.hpp"
#include "parameters.hpp"
#include "point_data.hpp"
#include "reconstruction.hpp"
#include "triangulation.hpp"

namespace stargazer {

using namespace coalsack;
using namespace stargazer::calibration;
using namespace stargazer::reconstruction;

static inline void detect_aruco_marker(cv::Mat image, std::vector<std::vector<cv::Point2f>>& points,
                                       std::vector<int>& ids) {
  cv::aruco::DetectorParameters detector_params = cv::aruco::DetectorParameters();
  cv::aruco::RefineParameters refine_params = cv::aruco::RefineParameters();
  const auto dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
  const auto detector = cv::aruco::ArucoDetector(dictionary, detector_params, refine_params);

  points.clear();
  ids.clear();
  detector.detectMarkers(image, points, ids);
}

class axis_reconstruction {
  glm::mat4 axis;
  std::map<std::string, camera_t> cameras;

 public:
  void set_camera(const std::string& name, const camera_t& camera) { cameras[name] = camera; }

  glm::mat4 get_axis() const { return axis; }

  void set_axis(const glm::mat4& axis) { this->axis = axis; }

  static bool compute_axis(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::mat4& axis) {
    if (!(std::abs(glm::dot(p1 - p0, p2 - p0)) < 0.01)) {
      return false;
    }
    const auto origin = p0;
    const auto e1 = p1 - p0;
    const auto e2 = p2 - p0;

    glm::vec3 x_axis = e1;
    glm::vec3 y_axis = e2;

    auto z_axis = glm::cross(glm::normalize(x_axis), glm::normalize(y_axis));
    z_axis = glm::normalize(z_axis);

    const auto y_axis_length = 0.196f;
    const auto scale = y_axis_length / glm::length(y_axis);
    x_axis = glm::normalize(x_axis);
    y_axis = glm::normalize(y_axis);

    axis = glm::mat4(1.0f);

    axis[0] = glm::vec4(x_axis / scale, 0.0f);
    axis[1] = glm::vec4(y_axis / scale, 0.0f);
    axis[2] = glm::vec4(z_axis / scale, 0.0f);
    axis[3] = glm::vec4(origin, 1.0f);

    axis = glm::inverse(axis);

    return true;
  }

  static bool detect_axis(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::mat4& axis,
                          float x_axis_length = 0.14f, float y_axis_length = 0.17f) {
    glm::vec3 origin;
    glm::vec3 e1, e2;

    // Find origin
    if (std::abs(glm::dot(p1 - p0, p2 - p0)) < 0.01) {
      origin = p0;
      e1 = p1 - p0;
      e2 = p2 - p0;
    } else if (std::abs(glm::dot(p0 - p1, p2 - p1)) < 0.01) {
      origin = p1;
      e1 = p0 - p1;
      e2 = p2 - p1;
    } else if (std::abs(glm::dot(p0 - p2, p1 - p2)) < 0.01) {
      origin = p2;
      e1 = p0 - p2;
      e2 = p1 - p2;
    } else {
      return false;
    }

    glm::vec3 x_axis, y_axis;
    if (glm::length(e1) < glm::length(e2)) {
      x_axis = e1;
      y_axis = e2;
    } else {
      x_axis = e2;
      y_axis = e1;
    }

    auto z_axis = glm::cross(x_axis, y_axis);
    z_axis = glm::normalize(z_axis);

    const auto scale = x_axis_length / glm::length(x_axis);
    if ((std::abs(scale - y_axis_length / glm::length(y_axis)) / scale) > 0.05) {
      return false;
    }
    x_axis = glm::normalize(x_axis);
    y_axis = glm::normalize(y_axis);

    axis[0] = glm::vec4(x_axis / scale, 0.0f);
    axis[1] = glm::vec4(y_axis / scale, 0.0f);
    axis[2] = glm::vec4(z_axis / scale, 0.0f);
    axis[3] = glm::vec4(origin, 1.0f);

    axis = glm::inverse(axis);

    return true;
  }

  void push_frame(const std::map<std::string, cv::Mat>& frame) {
    std::map<std::string, std::vector<point_data>> points;

    for (const auto& [name, image] : frame) {
      std::vector<int> marker_ids;
      std::vector<std::vector<cv::Point2f>> marker_corners;
      detect_aruco_marker(image, marker_corners, marker_ids);

      for (size_t i = 0; i < marker_ids.size(); i++) {
        if (marker_ids[i] == 0) {
          auto& corner_points = points[name];
          for (size_t j = 0; j < 3; j++) {
            point_data point{};
            point.point.x = marker_corners[i][j].x;
            point.point.y = marker_corners[i][j].y;
            corner_points.push_back(point);
          }
        }
      }
    }

    std::vector<glm::vec3> markers;
    for (size_t j = 0; j < 3; j++) {
      std::vector<glm::vec2> pts;
      std::vector<camera_t> cams;

      for (const auto& [name, camera] : cameras) {
        pts.push_back(points[name][j].point);
        cams.push_back(camera);
      }
      const auto marker = triangulate(pts, cams);
      markers.push_back(marker);
    }

    if (markers.size() == 3) {
      if (!compute_axis(markers[1], markers[0], markers[2], axis)) {
        std::cout << "Failed to compute axis" << std::endl;
        return;
      }
    }

    glm::mat4 basis(1.f);
    basis[0] = glm::vec4(-1.f, 0.f, 0.f, 0.f);
    basis[1] = glm::vec4(0.f, 0.f, 1.f, 0.f);
    basis[2] = glm::vec4(0.f, 1.f, 0.f, 0.f);

    glm::mat4 cv_to_gl(1.f);
    cv_to_gl[0] = glm::vec4(1.f, 0.f, 0.f, 0.f);
    cv_to_gl[1] = glm::vec4(0.f, -1.f, 0.f, 0.f);
    cv_to_gl[2] = glm::vec4(0.f, 0.f, -1.f, 0.f);

    // z down -> z up -> opengl
    axis = basis * axis;
    // axis = basis * cv_to_gl * extrinsic_calib.axis;
  }

  void push_frame(const std::map<std::string, std::vector<point_data>>& frame) {
    const auto markers = reconstruct(cameras, frame);

    glm::mat4 axis;
    if (markers.size() == 3) {
      if (detect_axis(markers[0], markers[1], markers[2], axis)) {
        glm::mat4 basis(1.f);
        basis[0] = glm::vec4(-1.f, 0.f, 0.f, 0.f);
        basis[1] = glm::vec4(0.f, 0.f, 1.f, 0.f);
        basis[2] = glm::vec4(0.f, 1.f, 0.f, 0.f);

        // z up -> opengl
        axis = basis * axis;

        this->axis = axis;

        return;
      }
    }

    {
      constexpr size_t grid_width = 2;
      constexpr size_t grid_height = 9;
      constexpr size_t num_points = grid_width * grid_height;

      std::vector<std::vector<cv::Point2f>> image_points_nview;
      std::vector<camera_t> cams;
      for (const auto& [name, camera] : cameras) {
        std::vector<cv::Point2f> image_points;
        for (const auto& pt : frame.at(name)) {
          image_points.push_back(cv::Point2f(pt.point.x, pt.point.y));
        }
        if (image_points.size() == num_points) {
          image_points_nview.push_back(image_points);
          cams.push_back(camera);
        }
      }

      if (cams.size() < 2) {
        return;
      }

      std::vector<glm::vec3> markers;
      for (size_t i = 0; i < num_points; i++) {
        std::vector<glm::vec2> pts;

        for (size_t j = 0; j < cams.size(); j++) {
          pts.push_back(glm::vec2(image_points_nview[j][i].x, image_points_nview[j][i].y));
        }

        const auto marker = triangulate(pts, cams);
        markers.push_back(marker);
      }

      const auto grid_size_x = 0.01825f;
      const auto grid_size_y = 0.01825f;
      const auto x_axis_length = grid_size_x * 2.0f * (grid_width - 1);
      const auto y_axis_length = grid_size_y * (grid_height - 1);

      // Points are expected to be in the following order:
      // 0 1
      //  2 3
      // 4 5
      //  6 7
      // ...

      if (detect_axis(markers[num_points - grid_width], markers[num_points - 1], markers[0], axis,
                      x_axis_length, y_axis_length)) {
        glm::mat4 basis(1.f);
        basis[0] = glm::vec4(-1.f, 0.f, 0.f, 0.f);
        basis[1] = glm::vec4(0.f, 0.f, 1.f, 0.f);
        basis[2] = glm::vec4(0.f, 1.f, 0.f, 0.f);

        // z up -> opengl
        axis = basis * axis;

        this->axis = axis;

        return;
      }
    }
  }
};

class axis_calibration_node : public graph_node {
  observed_points_frames observed_frames;

  mutable std::mutex cameras_mtx;
  std::vector<std::string> camera_names;
  std::unordered_map<std::string, camera_t> cameras;

  axis_reconstruction reconstructor;

  graph_edge_ptr output;

 public:
  axis_calibration_node() : graph_node(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "calibration"; }

  void set_cameras(const std::unordered_map<std::string, camera_t>& cameras) {
    this->cameras = cameras;
  }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cameras);
  }

  size_t get_num_frames(std::string name) const { return observed_frames.get_num_points(name); }

  const std::vector<observed_points_t> get_observed_points(std::string name) const {
    return observed_frames.get_observed_points(name);
  }

  void calibrate() {
    for (const auto& [name, camera] : cameras) {
      reconstructor.set_camera(name, camera);
    }

    {
      for (size_t f = 0; f < observed_frames.get_num_frames(); f++) {
        std::map<std::string, std::vector<point_data>> frame;
        for (const auto& camera_name : camera_names) {
          std::vector<point_data> points_data;
          const auto point = observed_frames.get_observed_point(camera_name, f);
          for (const auto& pt : point.points) {
            points_data.push_back(point_data{pt, 0, 0});
          }
          frame[camera_name] = points_data;
        }

        reconstructor.push_frame(frame);
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
      scene_t scene;
      scene.axis = reconstructor.get_axis();

      std::shared_ptr<scene_message> msg(new scene_message(scene));
      output->send(msg);

      return;
    }

    if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message)) {
      for (const auto& [name, field] : obj_msg->get_fields()) {
        if (auto points_msg = std::dynamic_pointer_cast<float2_list_message>(field)) {
          std::vector<glm::vec2> points;
          for (const auto& pt : points_msg->get_data()) {
            points.emplace_back(pt.x, pt.y);
          }
          observed_frames.add_frame_points(points_msg->get_frame_number(), name, points);
        }
      }
    }
  }
};

}  // namespace stargazer

COALSACK_REGISTER_NODE(stargazer::axis_calibration_node, coalsack::graph_node)
