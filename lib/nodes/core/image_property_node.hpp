/// @file image_property_node.hpp
/// @brief Image caching and property exposure node.
/// @ingroup core_nodes
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>

#include "coalsack/core/graph_proc_registry.h"
#include "coalsack/image/frame_message.h"

/// @brief Image passthrough node that caches the latest received image as a property.
/// @details Forwards every `frame_message<image>` received on @b "default" unchanged.
///          The latest frame is also stored internally and exposed as the `image`
///          property (for UI display) and as a `received` counter property.
///
/// @par Inputs
/// - @b "default" — `frame_message<image>` — image frame to cache and forward
///
/// @par Outputs
/// - @b "default" — `frame_message<image>` — passthrough of the input frame
///
/// @par Properties
/// (none)
///
/// @see marker_property_node
class image_property_node : public coalsack::graph_node {
  coalsack::graph_edge_ptr output;
  mutable std::mutex image_mtx;
  std::shared_ptr<coalsack::image> current_image;
  std::atomic<std::int64_t> received_count;

 public:
  image_property_node()
      : graph_node(),
        output(std::make_shared<coalsack::graph_edge>(this)),
        image_mtx(),
        current_image(),
        received_count(0) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "image_property"; }

  template <typename Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }

  virtual std::optional<coalsack::property_value> get_property(
      const std::string& key) const override {
    if (key == "received") {
      return received_count.load();
    }
    if (key == "image") {
      std::lock_guard lock(image_mtx);
      if (current_image) {
        return current_image;
      }
    }
    return std::nullopt;
  }

  virtual void process([[maybe_unused]] std::string input_name,
                       coalsack::graph_message_ptr message) override {
    if (const auto image_frame =
            std::dynamic_pointer_cast<coalsack::frame_message<coalsack::image>>(message)) {
      {
        std::lock_guard lock(image_mtx);
        current_image = std::make_shared<coalsack::image>(image_frame->get_data());
      }
      received_count.fetch_add(1);
    }
    output->send(message);
  }
};

COALSACK_REGISTER_NODE(image_property_node, coalsack::graph_node)