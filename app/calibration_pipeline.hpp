#pragma once

#include <glm/glm.hpp>
#include <iostream>
#include <map>
#include <memory>
#include <random>

#include "bundle_adjust_data.hpp"
#include "config.hpp"
#include "parameters.hpp"
#include "point_data.hpp"

struct observed_points_t {
  size_t camera_idx;
  std::vector<glm::vec2> points;
};

class calibration_pipeline {
  class impl;
  std::unique_ptr<impl> pimpl;

 public:
  void add_calibrated(
      std::function<void(const std::unordered_map<std::string, stargazer::camera_t> &)> f);
  void clear_calibrated();

  void set_camera(const std::string &name, const stargazer::camera_t &camera);

  size_t get_camera_size() const;

  const std::unordered_map<std::string, stargazer::camera_t> &get_cameras() const;

  std::unordered_map<std::string, stargazer::camera_t> &get_cameras();

  const std::unordered_map<std::string, stargazer::camera_t> &get_calibrated_cameras() const;

  calibration_pipeline();
  virtual ~calibration_pipeline();

  size_t get_num_frames(std::string name) const;
  const std::vector<observed_points_t> get_observed_points(std::string name) const;

  void push_frame(const std::map<std::string, std::vector<stargazer::point_data>> &frame);
  void run(const std::vector<stargazer::node_info> &infos);
  void stop();

  void calibrate();
};

class intrinsic_calibration_pipeline {
  class impl;
  std::unique_ptr<impl> pimpl;

 public:
  intrinsic_calibration_pipeline();
  virtual ~intrinsic_calibration_pipeline();

  double get_rms() const;

  void set_image_size(int width, int height);

  const stargazer::camera_t &get_calibrated_camera() const;

  size_t get_num_frames() const;

  void push_frame(const std::vector<stargazer::point_data> &frame);
  void push_frame(const cv::Mat &frame);

  void run(const std::vector<stargazer::node_info> &infos);
  void stop();

  void calibrate();
};

class axis_calibration_pipeline {
  class impl;
  std::unique_ptr<impl> pimpl;

 public:
  axis_calibration_pipeline(std::shared_ptr<stargazer::parameters_t> parameters);
  virtual ~axis_calibration_pipeline();

  void set_camera(const std::string &name, const stargazer::camera_t &camera);

  size_t get_camera_size() const;

  const std::unordered_map<std::string, stargazer::camera_t> &get_cameras() const;

  std::unordered_map<std::string, stargazer::camera_t> &get_cameras();

  size_t get_num_frames(std::string name) const;
  const std::vector<observed_points_t> get_observed_points(std::string name) const;

  void push_frame(const std::map<std::string, std::vector<stargazer::point_data>> &frame);

  void run(const std::vector<stargazer::node_info> &infos);
  void stop();

  void calibrate();
};
