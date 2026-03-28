#pragma once

#include <map>
#include <memory>
#include <optional>

#include "calibration.hpp"
#include "coalsack/core/graph_node.h"
#include "config.hpp"
#include "parameters.hpp"

class intrinsic_calibration_pipeline {
  class impl;
  std::unique_ptr<impl> pimpl;

 public:
  explicit intrinsic_calibration_pipeline(std::shared_ptr<stargazer::parameters_t> parameters);
  virtual ~intrinsic_calibration_pipeline();

  void run(const std::vector<stargazer::node_def>& nodes);
  void start();
  void pause();
  void stop();

  void dispatch_action(const std::string& action_node_name);

  std::optional<coalsack::property_value> get_node_property(const std::string& node_name,
                                                            const std::string& key) const;
};
