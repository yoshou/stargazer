/// @file reconstruction_result_markers_node.hpp
/// @brief Extracts 3D point positions from a reconstruction result message.
/// @ingroup reconstruction_nodes
#pragma once

#include <memory>
#include <string>

#include "coalsack/core/graph_proc_registry.h"
#include "messages.hpp"

namespace stargazer {

/// @brief Extracts 3D marker positions from a reconstruction result message.
/// @details Receives a `reconstruction_result_message` on @b "default", extracts the
///          3D point list, and re-emits it as a `float3_list_message` on @b "default".
///          This allows downstream marker-based nodes (e.g. `marker_property_node`) to
///          consume reconstruction output without depending on the heavy result type.
///
/// @par Inputs
/// - @b "default" — `reconstruction_result_message` — 3D reconstruction output
///
/// @par Outputs
/// - @b "default" — `float3_list_message` — extracted 3D point positions
///
/// @par Properties
/// (none)
///
/// @see marker_property_node, epipolar_reconstruct_node
class reconstruction_result_markers_node : public coalsack::graph_node {
  coalsack::graph_edge_ptr output;

 public:
  reconstruction_result_markers_node()
      : graph_node(), output(std::make_shared<coalsack::graph_edge>(this)) {
    set_output(output);
  }

  std::string get_proc_name() const override { return "reconstruction_result_markers"; }

  template <typename Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }

  void process([[maybe_unused]] std::string input_name,
               coalsack::graph_message_ptr message) override {
    const auto result_msg = std::dynamic_pointer_cast<reconstruction_result_message>(message);
    if (!result_msg) {
      output->send(message);
      return;
    }

    auto marker_msg = std::make_shared<float3_list_message>();
    marker_msg->set_data(result_msg->get_result().points3d);
    marker_msg->set_frame_number(result_msg->get_frame_number());
    marker_msg->set_timestamp(result_msg->get_timestamp());
    output->send(marker_msg);
  }
};

}  // namespace stargazer

COALSACK_REGISTER_NODE(stargazer::reconstruction_result_markers_node, coalsack::graph_node)