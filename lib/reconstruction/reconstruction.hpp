#pragma once

#include <glm/glm.hpp>
#include <vector>

#include "parameters.hpp"
#include "point_data.hpp"

namespace stargazer::reconstruction {
std::vector<glm::vec3> reconstruct(
    const std::map<std::string, stargazer::camera_t>& cameras,
    const std::map<std::string, std::vector<stargazer::point_data>>& frame,
    glm::mat4 axis = glm::mat4(1.0f));

std::vector<glm::vec3> reconstruct(const std::vector<stargazer::camera_t>& camera_list,
                                   const std::vector<std::vector<glm::vec2>>& camera_pts,
                                   glm::mat4 axis = glm::mat4(1.0f));
}  // namespace stargazer::reconstruction