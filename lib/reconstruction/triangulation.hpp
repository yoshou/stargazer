#pragma once

#include <glm/glm.hpp>
#include <vector>

#include "parameters.hpp"

namespace stargazer::reconstruction {
glm::vec3 triangulate(const glm::vec2 pt1, const glm::vec2 pt2, const camera_t &cm1,
                      const camera_t &cm2);

glm::vec3 triangulate_undistorted(const glm::vec2 pt1, const glm::vec2 pt2, const camera_t &cm1,
                                  const camera_t &cm2);

glm::vec3 triangulate(const std::vector<glm::vec2> &points, const std::vector<camera_t> &cameras);
}  // namespace stargazer::reconstruction
