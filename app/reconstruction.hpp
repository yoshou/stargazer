#pragma once

#include <glm/glm.hpp>
#include <map>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "config_file.hpp"
#include "parameters.hpp"
#include "point_data.hpp"

std::vector<glm::vec3> reconstruct(
    const std::map<std::string, stargazer::camera_t> &cameras,
    const std::map<std::string, std::vector<stargazer::point_data>> &frame,
    glm::mat4 axis = glm::mat4(1.0f));

class multiview_point_reconstruction {
  std::map<std::string, stargazer::camera_t> cameras;
  glm::mat4 axis;

 public:
  using frame_type = std::map<std::string, std::vector<stargazer::point_data>>;

  multiview_point_reconstruction() = default;
  virtual ~multiview_point_reconstruction() = default;

  virtual void push_frame(const frame_type &frame) = 0;
  virtual void run() = 0;
  virtual void stop() = 0;

  virtual std::vector<glm::vec3> get_markers() const = 0;

  virtual std::map<std::string, stargazer::camera_t> get_cameras() const { return cameras; }
  virtual void set_camera(const std::string &name, const stargazer::camera_t &camera) {
    cameras[name] = camera;
  }
  virtual const stargazer::camera_t &get_camera(const std::string &name) const {
    return cameras.at(name);
  }
  virtual stargazer::camera_t &get_camera(const std::string &name) { return cameras.at(name); }
  virtual void set_axis(const glm::mat4 &axis) { this->axis = axis; }
  virtual glm::mat4 get_axis() const { return axis; }
};

class epipolar_reconstruction : public multiview_point_reconstruction {
  class impl;
  std::unique_ptr<impl> pimpl;

 public:
  epipolar_reconstruction();
  virtual ~epipolar_reconstruction();

  void push_frame(const frame_type &frame);
  void run();
  void stop();

  std::vector<glm::vec3> get_markers() const;
  void set_camera(const std::string &name, const stargazer::camera_t &camera) override;
  void set_axis(const glm::mat4 &axis) override;
};

class multiview_image_reconstruction {
  class impl;
  std::unique_ptr<impl> pimpl;

  using frame_type = std::map<std::string, cv::Mat>;

 public:
  multiview_image_reconstruction();
  virtual ~multiview_image_reconstruction();

  void push_frame(const frame_type &frame);
  void run(const std::vector<node_info> &infos);
  void stop();

  std::vector<glm::vec3> get_markers() const;
  std::map<std::string, cv::Mat> get_features() const;
  void set_camera(const std::string &name, const stargazer::camera_t &camera);
  void set_axis(const glm::mat4 &axis);
};