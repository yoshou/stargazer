#pragma once

#include "coalsack/core/graph_proc.h"
#include "messages.hpp"
#include "parameter_resource.hpp"

namespace stargazer {

class store_parameter_node : public coalsack::graph_node {
  std::shared_ptr<parameter_resource> param_res_;

 public:
  store_parameter_node() : coalsack::graph_node() {}

  virtual std::string get_proc_name() const override { return "store_parameter"; }

  virtual void run() override {
    const auto res = resources->get(parameter_resource::resource_name());
    if (!res) return;
    param_res_ = std::static_pointer_cast<parameter_resource>(res);
  }

  virtual void stop() override { param_res_.reset(); }

  virtual void process(std::string input_name, coalsack::graph_message_ptr message) override {
    if (!param_res_) return;

    auto obj_msg = std::dynamic_pointer_cast<object_message>(message);
    if (!obj_msg) return;

    auto& params = param_res_->get_parameters();
    for (const auto& [id, field] : obj_msg->get_fields()) {
      if (auto cam_msg = std::dynamic_pointer_cast<camera_message>(field)) {
        params.update_camera(id, cam_msg->get_camera());
      } else if (auto scene_msg = std::dynamic_pointer_cast<scene_message>(field)) {
        params.update_scene(id, scene_msg->get_scene());
      }
    }
    params.save();
  }
};

}  // namespace stargazer
