#pragma once

#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>

#include "calibration.hpp"
#include "coalsack/core/graph_node.h"
#include "config.hpp"
#include "parameters.hpp"
#include "point_data.hpp"

class intrinsic_calibration_pipeline {
  class impl;
  std::unique_ptr<impl> pimpl;

 public:
  explicit intrinsic_calibration_pipeline(std::shared_ptr<stargazer::parameters_t> parameters);
  virtual ~intrinsic_calibration_pipeline();

  void push_frame(const std::vector<stargazer::point_data>& frame);
  void push_frame(const cv::Mat& frame);

  void run(const std::vector<stargazer::node_def>& nodes);
  void stop();

  void dispatch_action(const std::string& action_node_name);

  std::optional<coalsack::property_value> get_node_property(const std::string& node_name,
                                                            const std::string& key) const;
};
