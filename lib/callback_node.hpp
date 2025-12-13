#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "graph_proc.h"
#include "graph_proc_registry.h"

using namespace coalsack;

class callback_node;

class callback_list : public resource_base {
  using callback_func = std::function<void(const callback_node*, std::string, graph_message_ptr)>;
  std::vector<callback_func> callbacks;

 public:
  virtual std::string get_name() const { return "callback_list"; }

  void add(callback_func callback) { callbacks.push_back(callback); }

  void invoke(const callback_node* node, std::string input_name, graph_message_ptr message) const {
    for (auto& callback : callbacks) {
      callback(node, input_name, message);
    }
  }
};

class callback_node : public graph_node {
 public:
  enum class callback_type { unknown, image, marker, object };

 private:
  std::string callback_name;
  std::string camera_name;
  callback_type type;

 public:
  callback_node() : graph_node(), callback_name(), camera_name(), type(callback_type::unknown) {}

  virtual ~callback_node() = default;

  virtual std::string get_proc_name() const override { return "callback"; }

  void set_callback_name(const std::string& value) { callback_name = value; }
  std::string get_callback_name() const { return callback_name; }

  void set_camera_name(const std::string& value) { camera_name = value; }
  std::string get_camera_name() const { return camera_name; }

  void set_callback_type(callback_type value) { type = value; }
  callback_type get_callback_type() const { return type; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(callback_name);
    archive(camera_name);
    archive(type);
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (const auto resource = resources->get("callback_list")) {
      if (const auto callbacks = std::dynamic_pointer_cast<callback_list>(resource)) {
        callbacks->invoke(this, input_name, message);
      }
    }
  }
};

COALSACK_REGISTER_NODE(callback_node, graph_node)
