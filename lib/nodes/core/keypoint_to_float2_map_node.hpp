#pragma once

#include <memory>
#include <string>

#include "coalsack/core/graph_proc.h"
#include "coalsack/image/graph_proc_cv.h"
#include "messages.hpp"

namespace stargazer {

class keypoint_to_float2_map_node : public coalsack::graph_node {
 public:
  keypoint_to_float2_map_node() : graph_node() {
    set_output(std::make_shared<coalsack::graph_edge>(this));
  }

  virtual ~keypoint_to_float2_map_node() = default;

  virtual std::string get_proc_name() const override { return "keypoint_to_float2_map"; }

  virtual void process(std::string input_name, coalsack::graph_message_ptr message) override {
    (void)input_name;
    if (const auto obj_msg = std::dynamic_pointer_cast<coalsack::object_message>(message)) {
      auto out_msg = std::make_shared<coalsack::object_message>();
      for (const auto& [name, field] : obj_msg->get_fields()) {
        if (const auto kp_msg =
                std::dynamic_pointer_cast<coalsack::keypoint_frame_message>(field)) {
          auto float2_msg = std::make_shared<stargazer::float2_list_message>();
          std::vector<stargazer::float2> float2_data;
          for (const auto& kp : kp_msg->get_data()) {
            float2_data.push_back({kp.pt_x, kp.pt_y});
          }
          float2_msg->set_data(float2_data);
          float2_msg->set_timestamp(kp_msg->get_timestamp());
          float2_msg->set_frame_number(kp_msg->get_frame_number());
          out_msg->add_field(name, float2_msg);
        }
      }
      get_output()->send(out_msg);
    }
  }
};

}  // namespace stargazer
