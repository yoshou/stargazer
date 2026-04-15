#pragma once

#include <memory>
#include <string>

#include "coalsack/core/graph_proc_registry.h"
#include "messages.hpp"

namespace stargazer {

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