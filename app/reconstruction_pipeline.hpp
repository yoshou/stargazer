#pragma once

#include <glm/glm.hpp>
#include <map>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "config.hpp"
#include "parameters.hpp"
#include "point_data.hpp"

class multiview_point_reconstruction_pipeline {
  std::map<std::string, stargazer::camera_t> cameras;
  glm::mat4 axis;
  class impl;
  std::unique_ptr<impl> pimpl;

  using frame_type = std::map<std::string, std::vector<stargazer::point_data>>;

 public:
  multiview_point_reconstruction_pipeline();
  virtual ~multiview_point_reconstruction_pipeline();

  void push_frame(const frame_type &frame);
  void run();
  void stop();

  std::vector<glm::vec3> get_markers() const;
  virtual std::map<std::string, stargazer::camera_t> get_cameras() const { return cameras; }
  virtual stargazer::camera_t &get_camera(const std::string &name) { return cameras.at(name); }
  void set_camera(const std::string &name, const stargazer::camera_t &camera);
  virtual glm::mat4 get_axis() const { return axis; }
  void set_axis(const glm::mat4 &axis);
};

class multiview_image_reconstruction_pipeline {
  class impl;
  std::unique_ptr<impl> pimpl;

  using frame_type = std::map<std::string, cv::Mat>;

 public:
  multiview_image_reconstruction_pipeline();
  virtual ~multiview_image_reconstruction_pipeline();

  void push_frame(const frame_type &frame);
  void run(const std::vector<node_info> &infos);
  void stop();

  std::vector<glm::vec3> get_markers() const;
  std::map<std::string, cv::Mat> get_features() const;
  void set_camera(const std::string &name, const stargazer::camera_t &camera);
  void set_axis(const glm::mat4 &axis);
};