#pragma once

#include <cereal/types/map.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "graph_proc.h"
#include "graph_proc_img.h"
#include "parameters.hpp"

namespace stargazer {

using namespace coalsack;

struct se3 {
  glm::vec3 position;
  glm::quat rotation;

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(position, rotation);
  }
};

struct camera_intrinsics {
  glm::vec2 focal_length;
  glm::vec2 principal_point;
  glm::ivec2 image_size;
  struct {
    double k1, k2, p1, p2, k3;
  } distortion;

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(focal_length, principal_point, image_size, distortion.k1, distortion.k2,
            distortion.p1, distortion.p2, distortion.k3);
  }
};

enum class image_data_format {
  UNKNOWN = 0,
  JPEG = 1,
  PNG = 2,
  RAW = 3
};

struct camera_image {
  std::vector<uint8_t> image_data;
  glm::ivec2 image_size;
  image_data_format format;
  camera_intrinsics intrinsics;

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(image_data, image_size, format, intrinsics);
  }
};

struct inertial {
  glm::vec3 acceleration;
  glm::vec3 gyroscope;
  glm::vec3 magnetometer;
  glm::vec3 gravity;

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(acceleration, gyroscope, magnetometer, gravity);
  }
};

struct float2 {
  float x;
  float y;

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(x, y);
  }
};

struct float3 {
  float x;
  float y;
  float z;

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(x, y, z);
  }
};

// Generic 2D bounding box
struct bbox2d_t {
  float left;
  float top;
  float right;
  float bottom;

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(left, top, right, bottom);
  }
};

// Generic 2D detection result
struct detection2d_t {
  bbox2d_t bbox;
  float bbox_score;
  std::vector<float2> keypoints;  // (x, y) coordinates
  std::vector<float> scores;      // per-keypoint confidence

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(bbox, bbox_score, keypoints, scores);
  }
};

// Per-view 2D detection results
struct view_result_t {
  std::string name;
  std::vector<detection2d_t> detections;

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(name, detections);
  }
};

// Multi-view correspondence (view_index, detection_index)
using detection_id_t = std::pair<size_t, size_t>;
using match_t = std::vector<detection_id_t>;

// Complete reconstruction result
struct reconstruction_result_t {
  std::vector<float3> points3d;      // Final 3D keypoints
  std::vector<view_result_t> views;  // Per-view 2D detections
  std::vector<match_t> matches;      // Multi-view correspondences
  size_t num_keypoints;

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(points3d, views, matches, num_keypoints);
  }
};

using float2_list_message = frame_message<std::vector<float2>>;
using float3_list_message = frame_message<std::vector<float3>>;
using mat4_message = frame_message<glm::mat4>;
using se3_list_message = frame_message<std::vector<se3>>;
using camera_image_list_message = frame_message<std::vector<camera_image>>;
using inertial_list_message = frame_message<std::vector<inertial>>;

class camera_message : public graph_message {
  camera_t camera;

 public:
  camera_message() : graph_message(), camera() {}

  camera_message(const camera_t& camera) : graph_message(), camera(camera) {}

  static std::string get_type() { return "camera"; }

  camera_t get_camera() const { return camera; }

  void set_camera(const camera_t& value) { camera = value; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(camera);
  }
};

class scene_message : public graph_message {
  scene_t scene;

 public:
  scene_message() : graph_message(), scene() {}

  scene_message(const scene_t& scene) : graph_message(), scene(scene) {}

  static std::string get_type() { return "scene"; }

  scene_t get_scene() const { return scene; }

  void set_scene(const scene_t& value) { scene = value; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(scene);
  }
};

// Generic multi-view reconstruction result message
class reconstruction_result_message : public frame_message_base {
  reconstruction_result_t result;
  std::map<std::string, camera_t> cameras;
  glm::mat4 axis;

 public:
  void set_result(const reconstruction_result_t& result) { this->result = result; }
  void set_cameras(const std::map<std::string, camera_t>& cameras) { this->cameras = cameras; }
  void set_axis(const glm::mat4& axis) { this->axis = axis; }

  const reconstruction_result_t& get_result() const { return result; }
  const std::map<std::string, camera_t>& get_cameras() const { return cameras; }
  const glm::mat4& get_axis() const { return axis; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(result, cameras, axis);
  }
};

}  // namespace stargazer

COALSACK_REGISTER_MESSAGE(stargazer::float2_list_message, coalsack::frame_message_base)
COALSACK_REGISTER_MESSAGE(stargazer::float3_list_message, coalsack::frame_message_base)
COALSACK_REGISTER_MESSAGE(stargazer::mat4_message, coalsack::frame_message_base)
COALSACK_REGISTER_MESSAGE(stargazer::se3_list_message, coalsack::frame_message_base)
COALSACK_REGISTER_MESSAGE(stargazer::camera_image_list_message, coalsack::frame_message_base)
COALSACK_REGISTER_MESSAGE(stargazer::inertial_list_message, coalsack::frame_message_base)
COALSACK_REGISTER_MESSAGE(stargazer::camera_message, coalsack::graph_message)
COALSACK_REGISTER_MESSAGE(stargazer::scene_message, coalsack::graph_message)
COALSACK_REGISTER_MESSAGE(coalsack::frame_message<coalsack::object_message>,
                          coalsack::frame_message_base)
COALSACK_REGISTER_MESSAGE(stargazer::reconstruction_result_message, coalsack::frame_message_base)
