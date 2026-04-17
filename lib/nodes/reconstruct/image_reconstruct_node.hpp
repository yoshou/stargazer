/// @file image_reconstruct_node.hpp
/// @brief Abstract base class for multi-view image-based reconstruction nodes.
/// @ingroup reconstruction_nodes
#pragma once

#include <glm/glm.hpp>
#include <map>
#include <string>

#include "coalsack/core/graph_proc.h"
#include "parameters.hpp"

namespace stargazer {

using namespace coalsack;

/// @brief Abstract base class for multi-view image-based reconstruction nodes.
/// @details Defines the common interface for reconstruction nodes that operate on
///          calibrated multi-camera image streams.  Derived classes must implement
///          `set_cameras()` and `set_axis()` to receive camera calibration updates.
///
/// @par Inputs
/// (defined by derived classes)
///
/// @par Outputs
/// (defined by derived classes)
///
/// @par Properties
/// (defined by derived classes)
///
/// @see epipolar_reconstruct_node, mvpose_reconstruct_node, voxelpose_reconstruct_node
class image_reconstruct_node : public graph_node {
 public:
  virtual void set_cameras(const std::map<std::string, stargazer::camera_t>& cameras) = 0;
  virtual void set_axis(const glm::mat4& axis) = 0;
};

}  // namespace stargazer

COALSACK_REGISTER_NODE(stargazer::image_reconstruct_node, coalsack::graph_node)
