#pragma once

#include <spdlog/spdlog.h>

#include <string>
#include <vector>

#include "graph_proc.h"
#include "messages.hpp"

namespace stargazer {

using namespace coalsack;

class object_map_node : public graph_node {
 public:
  object_map_node() : graph_node() {}

  virtual std::string get_proc_name() const override { return "object_map"; }

  template <typename Archive>
  void save(Archive& archive) const {
    std::vector<std::string> output_names;
    auto outputs = get_outputs();
    for (auto output : outputs) {
      output_names.push_back(output.first);
    }
    archive(output_names);
  }

  template <typename Archive>
  void load(Archive& archive) {
    std::vector<std::string> output_names;
    archive(output_names);
    for (auto output_name : output_names) {
      set_output(std::make_shared<graph_edge>(this), output_name);
    }
  }

  graph_edge_ptr add_output(const std::string& name) {
    auto outputs = get_outputs();
    auto it = outputs.find(name);
    if (it == outputs.end()) {
      auto output = std::make_shared<graph_edge>(this);
      set_output(output, name);
      return output;
    }
    return it->second;
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message)) {
      for (const auto& [name, field] : obj_msg->get_fields()) {
        if (auto frame_msg = std::dynamic_pointer_cast<frame_message_base>(field)) {
          try {
            const auto output = get_output(name);
            output->send(field);
          } catch (const std::exception& e) {
            spdlog::error(e.what());
          }
        }
      }
    }
  }
};

}  // namespace stargazer

COALSACK_REGISTER_NODE(stargazer::object_map_node, coalsack::graph_node)
