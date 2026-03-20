#pragma once

#include <glm/glm.hpp>
#include <map>
#include <string>

#include "coalsack/core/graph_proc.h"
#include "parameters.hpp"

namespace stargazer {

using namespace coalsack;

class image_reconstruct_node : public graph_node {
 public:
  virtual void set_cameras(const std::map<std::string, stargazer::camera_t>& cameras) = 0;
  virtual void set_axis(const glm::mat4& axis) = 0;
};

}  // namespace stargazer

COALSACK_REGISTER_NODE(stargazer::image_reconstruct_node, coalsack::graph_node)
