#pragma once

#include <glm/glm.hpp>
#include <vector>

#include "bundle_adjust_data.hpp"
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

class observed_points_frames {
  mutable std::mutex frames_mtx;
  std::unordered_map<uint32_t, uint32_t> timestamp_to_index;
  std::map<std::string, size_t> camera_name_to_index;
  std::map<std::string, std::vector<observed_points_t>> observed_frames;
  std::map<std::string, size_t> num_points;

 public:
  const std::vector<observed_points_t> get_observed_points(std::string name) const;

  const observed_points_t get_observed_point(std::string name, size_t frame) const;

  size_t get_num_frames() const;

  size_t get_num_points(std::string name) const;

  void add_frame_points(uint32_t timestamp, std::string name, const std::vector<glm::vec2> &points);
};

bool initialize_cameras(std::unordered_map<std::string, stargazer::camera_t> &cameras,
                        const std::vector<std::string> &camera_names,
                        const observed_points_frames &observed_frames);

void prepare_bundle_adjustment(const std::vector<std::string> &camera_names,
                               const std::unordered_map<std::string, stargazer::camera_t> &cameras,
                               const observed_points_frames &observed_frames,
                               stargazer::calibration::bundle_adjust_data &ba_data);
