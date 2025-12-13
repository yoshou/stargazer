#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "graph_proc.h"
#include "graph_proc_img.h"

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

}  // namespace stargazer

COALSACK_REGISTER_MESSAGE(stargazer::float2_list_message, coalsack::frame_message_base)
COALSACK_REGISTER_MESSAGE(stargazer::float3_list_message, coalsack::frame_message_base)
COALSACK_REGISTER_MESSAGE(stargazer::mat4_message, coalsack::frame_message_base)
COALSACK_REGISTER_MESSAGE(stargazer::se3_list_message, coalsack::frame_message_base)
