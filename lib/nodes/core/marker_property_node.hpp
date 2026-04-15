#pragma once

#include <mutex>
#include <vector>

#include "coalsack/camera/camera.h"
#include "coalsack/core/graph_proc.h"
#include "messages.hpp"

namespace stargazer {

// Receives float3_list_message and exposes a "markers" property as std::vector<coalsack::vec3>.
class marker_property_node : public coalsack::graph_node {
  mutable std::mutex mtx;
  std::vector<coalsack::vec3> markers;
  std::int64_t received_count = 0;
  coalsack::graph_edge_ptr output;

 public:
  marker_property_node()
      : coalsack::graph_node(), output(std::make_shared<coalsack::graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "marker_property"; }

  template <typename Archive>
  void serialize(Archive& archive) {}

  virtual std::optional<coalsack::property_value> get_property(
      const std::string& key) const override {
    if (key == "markers") {
      std::lock_guard lock(mtx);
      return markers;
    }
    if (key == "received") {
      std::lock_guard lock(mtx);
      return received_count;
    }
    return std::nullopt;
  }

  virtual void process(std::string input_name, coalsack::graph_message_ptr message) override {
    if (auto msg = std::dynamic_pointer_cast<float3_list_message>(message)) {
      std::vector<coalsack::vec3> pts;
      pts.reserve(msg->get_data().size());
      for (const auto& p : msg->get_data()) {
        pts.push_back({p.x, p.y, p.z});
      }
      {
        std::lock_guard lock(mtx);
        markers = std::move(pts);
        received_count++;
      }
      output->send(message);
    }
  }
};

}  // namespace stargazer

COALSACK_REGISTER_NODE(stargazer::marker_property_node, coalsack::graph_node)
