#pragma once

#include <string>

#include "coalsack/core/graph_proc.h"
#include "coalsack/image/frame_message.h"
#include "messages.hpp"

namespace stargazer {

class object_to_frame_node : public coalsack::graph_node {
 public:
  object_to_frame_node() : graph_node() {
    set_output(std::make_shared<coalsack::graph_edge>(this));
  }

  virtual ~object_to_frame_node() = default;

  virtual std::string get_proc_name() const override { return "object_to_frame"; }

  template <typename Archive>
  void serialize(Archive&) {}

  virtual void process(std::string input_name, coalsack::graph_message_ptr message) override {
    (void)input_name;
    if (const auto obj_msg = std::dynamic_pointer_cast<coalsack::object_message>(message)) {
      uint64_t frame_number = 0;
      double timestamp = 0.0;
      for (const auto& [name, field] : obj_msg->get_fields()) {
        if (const auto fm = std::dynamic_pointer_cast<coalsack::frame_message_base>(field)) {
          frame_number = fm->get_frame_number();
          timestamp = fm->get_timestamp();
          break;
        }
      }
      auto frame_msg = std::make_shared<coalsack::frame_message<coalsack::object_message>>();
      frame_msg->set_data(*obj_msg);
      frame_msg->set_frame_number(frame_number);
      frame_msg->set_timestamp(timestamp);
      get_output()->send(frame_msg);
    }
  }
};

}  // namespace stargazer

COALSACK_REGISTER_NODE(stargazer::object_to_frame_node, coalsack::graph_node)
