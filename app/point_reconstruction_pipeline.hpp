#pragma once

#include <glm/glm.hpp>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "coalsack/core/graph_node.h"
#include "config.hpp"
#include "parameters.hpp"
#include "point_data.hpp"

class multiview_point_reconstruction_pipeline {
  class impl;
  std::unique_ptr<impl> pimpl;

  using frame_type = std::map<std::string, std::vector<stargazer::point_data>>;

 public:
  explicit multiview_point_reconstruction_pipeline(
      std::shared_ptr<stargazer::parameters_t> parameters);
  virtual ~multiview_point_reconstruction_pipeline();

  void push_frame(const frame_type& frame);
  void run(const std::vector<stargazer::node_def>& nodes);
  void stop();

  std::optional<coalsack::property_value> get_node_property(const std::string& node_name,
                                                            const std::string& key) const;
};
