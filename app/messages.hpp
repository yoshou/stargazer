#pragma once

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

using float2_list_message = frame_message<std::vector<float2>>;
using float3_list_message = frame_message<std::vector<float3>>;
using mat4_message = frame_message<glm::mat4>;
using se3_list_message = frame_message<std::vector<se3>>;

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

}  // namespace stargazer

COALSACK_REGISTER_MESSAGE(stargazer::float2_list_message, coalsack::frame_message_base)
COALSACK_REGISTER_MESSAGE(stargazer::float3_list_message, coalsack::frame_message_base)
COALSACK_REGISTER_MESSAGE(stargazer::mat4_message, coalsack::frame_message_base)
COALSACK_REGISTER_MESSAGE(stargazer::se3_list_message, coalsack::frame_message_base)
COALSACK_REGISTER_MESSAGE(stargazer::camera_message, coalsack::graph_message)
COALSACK_REGISTER_MESSAGE(stargazer::scene_message, coalsack::graph_message)
COALSACK_REGISTER_MESSAGE(coalsack::frame_message<coalsack::object_message>,
                          coalsack::frame_message_base)
