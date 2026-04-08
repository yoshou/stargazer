#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "dust3r_alignment.hpp"
#include "parameters.hpp"

namespace stargazer::dust3r {

std::unordered_map<std::string, aligned_pose> refine_global_alignment(
    const std::vector<std::string>& camera_names, const std::vector<pair_result>& pair_results,
    const std::unordered_map<std::string, aligned_pose>& initial_poses);

std::unordered_map<std::string, aligned_pose> refine_global_alignment(
    const std::vector<std::string>& camera_names, const std::vector<pair_result>& pair_results,
    const std::unordered_map<std::string, aligned_pose>& initial_poses,
    const std::unordered_map<std::string, camera_t>& cameras);

}  // namespace stargazer::dust3r
