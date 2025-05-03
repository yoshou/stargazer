#pragma once

#include <fstream>
#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace stargazer::calibration {
class bundle_adjust_data {
  static constexpr auto num_camera_parameters = 18;

 public:
  bundle_adjust_data()
      : num_cameras_(0), num_points_(0), num_observations_(0), num_parameters_(0) {}

  void resize_observations(size_t num_observations);
  void resize_parameters(size_t num_cameras, size_t num_points);

  void add_point(const double *parameters) {
    for (size_t i = 0; i < 3; i++) {
      parameters_.push_back(parameters[i]);
    }

    num_points_++;
    num_parameters_ = num_camera_parameters * num_cameras_ + 3 * num_points_;
  }
  void add_camera(const double *parameters) {
    for (size_t i = 0; i < num_camera_parameters; i++) {
      parameters_.push_back(parameters[i]);
    }

    num_cameras_++;
    num_parameters_ = num_camera_parameters * num_cameras_ + 3 * num_points_;
  }
  void add_observation(const double *observation, size_t camera_idx, size_t point_idx) {
    for (size_t i = 0; i < 2; i++) {
      observations_.push_back(observation[i]);
    }
    point_index_.push_back(point_idx);
    camera_index_.push_back(camera_idx);

    num_observations_++;
  }

  int num_observations() const { return num_observations_; }
  int num_cameras() const { return num_cameras_; }
  int num_points() const { return num_points_; }
  const double *observations() const { return observations_.data(); }
  double *mutable_cameras() { return parameters_.data(); }
  double *mutable_points() { return parameters_.data() + num_camera_parameters * num_cameras_; }
  double *mutable_point(int i) {
    return parameters_.data() + num_camera_parameters * num_cameras_ + 3 * i;
  }
  double *mutable_camera(int i) { return parameters_.data() + num_camera_parameters * i; }
  double *mutable_camera_for_observation(int i) {
    return mutable_cameras() + camera_index_[i] * num_camera_parameters;
  }
  double *mutable_point_for_observation(int i) { return mutable_points() + point_index_[i] * 3; }
  int *point_index() { return point_index_.data(); }
  int *camera_index() { return camera_index_.data(); };

  bool load_txt(std::istream &ifs);

  bool save_txt(std::ostream &ofs);

  bool load_txt(const std::string &filename);

  bool save_txt(const std::string &filename);

  bool load_json(std::istream &ifs);

  bool save_json(std::ostream &ofs);

  bool load_json(const std::string &filename);

  bool save_json(const std::string &filename);

  glm::mat4 get_camera_extrinsic(std::size_t i);

 private:
  int num_cameras_;
  int num_points_;
  int num_observations_;
  int num_parameters_;

  std::vector<int> point_index_;
  std::vector<int> camera_index_;
  std::vector<double> observations_;
  std::vector<double> parameters_;
};
}  // namespace stargazer::calibration