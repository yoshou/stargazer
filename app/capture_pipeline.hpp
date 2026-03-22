#pragma once

#include <glm/glm.hpp>
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>

#include "coalsack/core/graph_node.h"
#include "config.hpp"

struct marker_data {
  float x, y, r;
};

struct marker_frame_data {
  std::vector<marker_data> markers;
  double timestamp;
  uint64_t frame_number;
};

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

  void dispatch_action(const std::string& action_id);

  void enable_marker_collecting(std::string name);
  void disable_marker_collecting(std::string name);

  void add_marker_received(std::function<void(const std::map<std::string, marker_frame_data>&)> f);
  void clear_marker_received();
  void add_image_received(std::function<void(const std::map<std::string, cv::Mat>&)> f);
  void clear_image_received();

  std::optional<coalsack::property_value> get_node_property(const std::string& node_name,
                                                            const std::string& key) const;
};
