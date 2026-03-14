#pragma once

#include <map>
#include <memory>

#include <opencv2/opencv.hpp>

#include "calibration.hpp"
#include "config.hpp"
#include "parameters.hpp"
#include "point_data.hpp"

class intrinsic_calibration_pipeline {
  class impl;
  std::unique_ptr<impl> pimpl;

 public:
  intrinsic_calibration_pipeline();
  virtual ~intrinsic_calibration_pipeline();

  double get_rms() const;

  void set_image_size(int width, int height);

  const stargazer::camera_t& get_calibrated_camera() const;

  size_t get_num_frames() const;

  void push_frame(const std::vector<stargazer::point_data>& frame);
  void push_frame(const cv::Mat& frame);

  void run(const std::vector<stargazer::node_def>& nodes);
  void stop();

  void calibrate();
};
