#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "coalsack/core/graph_node.h"
#include "config.hpp"
#include "parameters.hpp"

class multiview_point_reconstruction_pipeline {
  class impl;
  std::unique_ptr<impl> pimpl;

 public:
  explicit multiview_point_reconstruction_pipeline(
      std::shared_ptr<stargazer::parameters_t> parameters);
  virtual ~multiview_point_reconstruction_pipeline();

  void run(const std::vector<stargazer::node_def>& nodes);
  void start();
  void pause();
  void stop();

  std::optional<coalsack::property_value> get_node_property(const std::string& node_name,
                                                            const std::string& key) const;
};
