#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <string>

#include "coalsack/core/graph_proc.h"

namespace stargazer {

class gate_node : public coalsack::graph_node {
  std::atomic<bool> enabled_{false};

 public:
  gate_node() : graph_node() {
    set_output(std::make_shared<coalsack::graph_edge>(this));
  }

  virtual ~gate_node() = default;

  virtual std::string get_proc_name() const override { return "gate"; }

  void set_enabled(bool value) { enabled_.store(value); }
  bool is_enabled() const { return enabled_.load(); }

  virtual std::optional<coalsack::property_value> get_property(
      const std::string& key) const override {
    if (key == "enabled") {
      return enabled_.load();
    }
    return std::nullopt;
  }

  virtual void process(std::string input_name, coalsack::graph_message_ptr message) override {
    (void)input_name;
    if (enabled_.load()) {
      get_output()->send(message);
    }
  }
};

}  // namespace stargazer
