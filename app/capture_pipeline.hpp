#pragma once

#include <glm/glm.hpp>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

#include "config.hpp"

struct marker_data {
  float x, y, r;
};

struct marker_frame_data {
  std::vector<marker_data> markers;
  double timestamp;
  uint64_t frame_number;
};

class capture_pipeline {
  class impl;

  std::unique_ptr<impl> pimpl;

 public:
  capture_pipeline();
  capture_pipeline(const std::map<std::string, cv::Mat>& masks);
  virtual ~capture_pipeline();

  void run(const std::vector<stargazer::node_def>& nodes);
  void stop();

  std::map<std::string, cv::Mat> get_frames() const;
  std::map<std::string, cv::Mat> get_masks() const;

  void gen_mask();
  void clear_mask();

  void enable_marker_collecting(std::string name);
  void disable_marker_collecting(std::string name);

  void add_marker_received(std::function<void(const std::map<std::string, marker_frame_data>&)> f);
  void clear_marker_received();
  void add_image_received(std::function<void(const std::map<std::string, cv::Mat>&)> f);
  void clear_image_received();
};
