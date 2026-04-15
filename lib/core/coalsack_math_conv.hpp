#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <vector>

#include "coalsack/camera/camera.h"
#include "parameters.hpp"

namespace stargazer {

inline coalsack::mat4 to_coalsack(const glm::mat4& m) {
  coalsack::mat4 result;
  for (int i = 0; i < 16; ++i) result.data[i] = (&m[0][0])[i];
  return result;
}

inline glm::mat4 to_glm(const coalsack::mat4& m) {
  glm::mat4 result;
  for (int i = 0; i < 16; ++i) (&result[0][0])[i] = m.data[i];
  return result;
}

inline coalsack::vec3 to_coalsack(const glm::vec3& v) { return {v.x, v.y, v.z}; }

inline glm::vec3 to_glm(const coalsack::vec3& v) { return {v.x, v.y, v.z}; }

inline std::vector<coalsack::vec3> to_coalsack(const std::vector<glm::vec3>& vs) {
  std::vector<coalsack::vec3> result;
  result.reserve(vs.size());
  for (const auto& v : vs) result.push_back(to_coalsack(v));
  return result;
}

inline std::vector<glm::vec3> to_glm(const std::vector<coalsack::vec3>& vs) {
  std::vector<glm::vec3> result;
  result.reserve(vs.size());
  for (const auto& v : vs) result.push_back(to_glm(v));
  return result;
}

inline coalsack::camera_t to_coalsack(const stargazer::camera_t& cam) {
  coalsack::camera_t result;
  result.width = static_cast<int>(cam.width);
  result.height = static_cast<int>(cam.height);
  result.ppx = cam.intrin.cx;
  result.ppy = cam.intrin.cy;
  result.fx = cam.intrin.fx;
  result.fy = cam.intrin.fy;
  for (int i = 0; i < 5; ++i) result.coeffs[i] = cam.intrin.coeffs[i];
  result.pose = to_coalsack(glm::inverse(cam.extrin.transform_matrix()));
  return result;
}

inline stargazer::camera_t to_stargazer(const coalsack::camera_t& cam) {
  stargazer::camera_t result;
  result.width = static_cast<uint32_t>(cam.width);
  result.height = static_cast<uint32_t>(cam.height);
  result.intrin.cx = cam.ppx;
  result.intrin.cy = cam.ppy;
  result.intrin.fx = cam.fx;
  result.intrin.fy = cam.fy;
  for (int i = 0; i < 5; ++i) result.intrin.coeffs[i] = cam.coeffs[i];
  const glm::mat4 transform = glm::inverse(to_glm(cam.pose));
  result.extrin.rotation = glm::mat3(transform);
  result.extrin.translation = glm::vec3(transform[3]);
  return result;
}

}  // namespace stargazer
