#pragma once

#include <glm/glm.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "dust3r.hpp"

namespace stargazer::dust3r {

struct view_pointcloud {
  std::string camera_name;
  std::vector<float> pts3d;  // [H*W*3] flat XYZ in HWC order
  std::vector<float> conf;   // [H*W] confidence
};

struct pair_result {
  int idx1;
  int idx2;
  view_pointcloud view1;  // pts3d_1 / conf_1 in the frame of view1
  view_pointcloud view2;  // pts3d_2 / conf_2 in the frame of view1
};

struct aligned_pose {
  glm::mat3 rotation;     // cam-to-world rotation
  glm::vec3 translation;  // cam-to-world translation
  float scale = 1.0f;     // similarity scale from local camera frame to world gauge
};

std::unordered_map<std::string, aligned_pose> align_global(
    const std::vector<std::string>& camera_names, const std::vector<pair_result>& pair_results);

std::unordered_map<std::string, aligned_pose> align_global(
    const std::vector<std::string>& camera_names, const std::vector<pair_result>& pair_results,
    const std::unordered_map<std::string, camera_t>& cameras);

}  // namespace stargazer::dust3r
