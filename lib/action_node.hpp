#pragma once

#include <optional>
#include <string>

#include "coalsack/core/graph_proc.h"
#include "coalsack/core/graph_proc_registry.h"

using namespace coalsack;

class action_node : public graph_node {
  std::string action_id_;
  std::string label_;
  std::string icon_;

  graph_edge_ptr output_;

 public:
  action_node() : graph_node(), output_(std::make_shared<graph_edge>(this)) { set_output(output_); }
  virtual ~action_node() = default;

  virtual std::string get_proc_name() const override { return "action"; }

  void set_action_id(const std::string& value) { action_id_ = value; }
  std::string get_action_id() const { return action_id_; }

  void set_label(const std::string& value) { label_ = value; }
  std::string get_label() const { return label_; }

  void set_icon(const std::string& value) { icon_ = value; }
  std::string get_icon() const { return icon_; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(action_id_);
    archive(label_);
    archive(icon_);
  }

  virtual std::optional<property_value> get_property(const std::string& key) const override {
    if (key == "action_id") return action_id_;
    if (key == "label") return label_;
    if (key == "icon") return icon_;
    return std::nullopt;
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    (void)input_name;
    output_->send(message);
  }
};

COALSACK_REGISTER_NODE(action_node, graph_node)
