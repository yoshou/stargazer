#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "coalsack/core/graph_node.h"
#include "config.hpp"
#include "parameters.hpp"

class multiview_image_reconstruction_pipeline {
  class impl;
  std::unique_ptr<impl> pimpl;

  using frame_type = std::map<std::string, cv::Mat>;

 public:
  multiview_image_reconstruction_pipeline();
  virtual ~multiview_image_reconstruction_pipeline();

  void push_frame(const frame_type& frame);
  void run(const std::vector<stargazer::node_def>& nodes);
  void stop();

  std::vector<glm::vec3> get_markers() const;
  std::map<std::string, cv::Mat> get_features() const;
  void set_camera(const std::string& name, const stargazer::camera_t& camera);
  void set_axis(const glm::mat4& axis);
  std::optional<coalsack::property_value> get_node_property(const std::string& node_name,
                                                            const std::string& key) const;
};
