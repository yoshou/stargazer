#pragma once

#include <glm/glm.hpp>
#include <map>
#include <opencv2/core.hpp>
#include <string>

#include "graph_proc.h"
#include "parameters.hpp"

namespace stargazer {

using namespace coalsack;

class image_reconstruct_node : public graph_node {
 public:
  virtual std::map<std::string, cv::Mat> get_features() const = 0;
  virtual void set_cameras(const std::map<std::string, camera_t>& cameras) = 0;
  virtual void set_axis(const glm::mat4& axis) = 0;
};

}  // namespace stargazer

COALSACK_REGISTER_NODE(stargazer::image_reconstruct_node, coalsack::graph_node)
