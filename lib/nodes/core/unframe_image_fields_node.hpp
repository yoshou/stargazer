/// @file unframe_image_fields_node.hpp
/// @brief Converts frame_message<image> object fields to bare image_message fields.
/// @ingroup core_nodes
#pragma once

#include <string>

#include "coalsack/core/graph_proc.h"
#include "coalsack/image/frame_message.h"
#include "coalsack/image/image_message.h"

namespace stargazer {

// Converts frame_message<object_message> whose fields are frame_message<image>
// into frame_message<object_message> whose fields are image_message.
// This bridges approximate_time_sync (outputs frame_message<image> fields)
// with reconstruct nodes that expect image_message fields.
/// @brief Converts `frame_message<image>` fields in an object to bare `image_message` fields.
/// @details Receives a `frame_message<object_message>` on @b "default" and iterates
///          over each field.  Fields that are `frame_message<image>` are unwrapped to
///          `image_message`; all other fields are passed through unchanged.  The
///          resulting `frame_message<object_message>` is forwarded on @b "default".
///
/// @par Inputs
/// - @b "default" — `frame_message<object_message>` — framed object with mixed field types
///
/// @par Outputs
/// - @b "default" — `frame_message<object_message>` — object with unwrapped image fields
///
/// @par Properties
/// (none)
///
/// @see object_to_frame_node
class unframe_image_fields_node : public coalsack::graph_node {
 public:
  unframe_image_fields_node() : graph_node() {
    set_output(std::make_shared<coalsack::graph_edge>(this));
  }

  virtual ~unframe_image_fields_node() = default;

  virtual std::string get_proc_name() const override { return "unframe_image_fields"; }

  template <typename Archive>
  void serialize(Archive&) {}

  virtual void process(std::string, coalsack::graph_message_ptr message) override {
    using coalsack::frame_message;
    using coalsack::image;
    using coalsack::image_message;
    using coalsack::object_message;

    if (const auto fm = std::dynamic_pointer_cast<frame_message<object_message>>(message)) {
      auto new_obj = std::make_shared<object_message>();
      for (const auto& [name, field] : fm->get_data().get_fields()) {
        if (const auto img_frame = std::dynamic_pointer_cast<frame_message<image>>(field)) {
          auto img_msg = std::make_shared<image_message>();
          img_msg->set_image(img_frame->get_data());
          new_obj->add_field(name, img_msg);
        } else {
          new_obj->add_field(name, field);
        }
      }
      auto new_fm = std::make_shared<frame_message<object_message>>();
      new_fm->set_data(*new_obj);
      new_fm->set_frame_number(fm->get_frame_number());
      new_fm->set_timestamp(fm->get_timestamp());
      get_output()->send(new_fm);
    }
  }
};

}  // namespace stargazer

COALSACK_REGISTER_NODE(stargazer::unframe_image_fields_node, coalsack::graph_node)
