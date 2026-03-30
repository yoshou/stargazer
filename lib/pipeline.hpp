#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "coalsack/core/graph_node.h"
#include "config.hpp"
#include "parameters.hpp"

namespace stargazer {

class pipeline {
  class impl;
  std::unique_ptr<impl> pimpl;

 public:
  pipeline();
  explicit pipeline(std::shared_ptr<parameters_t> parameters);
  ~pipeline();

  void run(const std::vector<node_def>& nodes);
  void start();
  void pause();
  void stop();

  std::optional<coalsack::property_value> get_node_property(const std::string& node_name,
                                                            const std::string& key) const;
  void dispatch_action(const std::string& action_id);
  void enable_marker_collecting(const std::string& name);
  void disable_marker_collecting(const std::string& name);
};

}  // namespace stargazer
