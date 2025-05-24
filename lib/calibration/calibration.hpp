#pragma once

#include <glm/glm.hpp>
#include <vector>

#include "parameters.hpp"

struct observed_points_t {
  size_t camera_idx;
  std::vector<glm::vec2> points;
};

void zip_points(const std::vector<observed_points_t> &points1,
                const std::vector<observed_points_t> &points2,
                std::vector<std::pair<glm::vec2, glm::vec2>> &corresponding_points);

void zip_points(const std::vector<observed_points_t> &points1,
                const std::vector<observed_points_t> &points2,
                const std::vector<observed_points_t> &points3,
                std::vector<std::tuple<glm::vec2, glm::vec2, glm::vec2>> &corresponding_points);

float compute_diff_camera_angle(const glm::mat3 &r1, const glm::mat3 &r2);

glm::mat4 estimate_relative_pose(
    const std::vector<std::pair<glm::vec2, glm::vec2>> &corresponding_points,
    const stargazer::camera_t &base_camera, const stargazer::camera_t &target_camera,
    bool use_lmeds = false);

glm::mat4 estimate_pose(
    const std::vector<std::tuple<glm::vec2, glm::vec2, glm::vec2>> &corresponding_points,
    const stargazer::camera_t &base_camera1, const stargazer::camera_t &base_camera2,
    const stargazer::camera_t &target_camera);