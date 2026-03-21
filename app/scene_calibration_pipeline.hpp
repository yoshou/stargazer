#pragma once

#include <map>
#include <memory>
#include <optional>

#include "coalsack/core/graph_node.h"
#include "calibration.hpp"
#include "config.hpp"
#include "parameters.hpp"
#include "point_data.hpp"

class scene_calibration_pipeline {
  class impl;
  std::unique_ptr<impl> pimpl;

 public:
  scene_calibration_pipeline(std::shared_ptr<stargazer::parameters_t> parameters);
  virtual ~scene_calibration_pipeline();

  void push_frame(const std::map<std::string, std::vector<stargazer::point_data>>& frame);

  void run(const std::vector<stargazer::node_def>& nodes);
  void stop();

  void calibrate();

  std::optional<coalsack::property_value> get_node_property(const std::string& node_name,
                                                            const std::string& key) const;
};
