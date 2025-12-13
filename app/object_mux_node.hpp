#pragma once

#include <string>

#include "graph_proc.h"
#include "messages.hpp"

namespace stargazer {

using namespace coalsack;

class sentinel_message : public graph_message {
 public:
  sentinel_message() {}

  static std::string get_type() { return "sentinel"; }

  template <typename Archive>
  void serialize(Archive& archive) {}
};

class object_mux_node : public graph_node {
  graph_edge_ptr output;

 public:
  object_mux_node() : graph_node(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "object_mux"; }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message)) {
      for (const auto& [name, field] : obj_msg->get_fields()) {
        auto msg = std::make_shared<object_message>();
        msg->add_field(name, field);
        output->send(msg);
      }
      auto msg = std::make_shared<sentinel_message>();
      output->send(msg);
    }
  }
};

}  // namespace stargazer

COALSACK_REGISTER_NODE(stargazer::object_mux_node, coalsack::graph_node)
