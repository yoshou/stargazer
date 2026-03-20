#pragma once

#include <map>
#include <memory>
#include <optional>

#include "bundle_adjust_data.hpp"
#include "calibration.hpp"
#include "coalsack/core/graph_node.h"
#include "config.hpp"
#include "parameters.hpp"
#include "point_data.hpp"

class extrinsic_calibration_pipeline {
  class impl;
  std::unique_ptr<impl> pimpl;

 public:
  void add_calibrated(
      std::function<void(const std::unordered_map<std::string, stargazer::camera_t>&)> f);
  void clear_calibrated();

  void set_camera(const std::string& name, const stargazer::camera_t& camera);

  size_t get_camera_size() const;

  const std::unordered_map<std::string, stargazer::camera_t>& get_cameras() const;

  const std::unordered_map<std::string, stargazer::camera_t>& get_calibrated_cameras() const;

  extrinsic_calibration_pipeline();
  virtual ~extrinsic_calibration_pipeline();

  size_t get_num_frames(std::string name) const;
  const std::vector<stargazer::calibration::observed_points_t> get_observed_points(
      std::string name) const;

  void push_frame(const std::map<std::string, std::vector<stargazer::point_data>>& frame);
  void run(const std::vector<stargazer::node_def>& nodes);
  void stop();

  void calibrate();

  std::optional<coalsack::property_value> get_node_property(const std::string& node_name,
                                                            const std::string& key) const;
};
