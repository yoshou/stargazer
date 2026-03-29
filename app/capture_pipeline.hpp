#pragma once

#include <memory>
#include <optional>
#include <string>

#include "coalsack/core/graph_node.h"
#include "config.hpp"

class capture_pipeline {
  class impl;

  std::unique_ptr<impl> pimpl;

 public:
  capture_pipeline();
  virtual ~capture_pipeline();

  void run(const std::vector<stargazer::node_def>& nodes);
  void start();
  void pause();
  void stop();

  std::optional<coalsack::property_value> get_node_property(const std::string& node_name,
                                                            const std::string& key) const;
};
