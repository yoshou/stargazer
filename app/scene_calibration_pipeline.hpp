#pragma once

#include <memory>
#include <optional>
#include <string>

#include "coalsack/core/graph_node.h"
#include "config.hpp"
#include "parameters.hpp"

class scene_calibration_pipeline {
  class impl;
  std::unique_ptr<impl> pimpl;

 public:
  explicit scene_calibration_pipeline(std::shared_ptr<stargazer::parameters_t> parameters);
  virtual ~scene_calibration_pipeline();

  void run(const std::vector<stargazer::node_def>& nodes);
  void start();
  void pause();
  void stop();

  void enable_marker_collecting(const std::string& name);
  void disable_marker_collecting(const std::string& name);

  void dispatch_action(const std::string& action_node_name);

  std::optional<coalsack::property_value> get_node_property(const std::string& node_name,
                                                            const std::string& key) const;
};
