#pragma once

#include <atomic>
#include <thread>

#include "coalsack/core/graph_proc.h"
#include "messages.hpp"
#include "parameter_resource.hpp"

namespace stargazer {

class load_parameter_node : public coalsack::graph_node {
  std::string id_;
  coalsack::graph_edge_ptr output_;
  std::shared_ptr<std::thread> monitor_thread_;
  std::atomic_bool running_{false};
  std::shared_ptr<parameter_resource> param_res_;

  void send_current() {
    if (!param_res_) return;
    const auto& params = param_res_->get_parameters();
    if (!params.contains(id_)) return;

    const auto value = params.at(id_);
    if (std::holds_alternative<camera_t>(value)) {
      auto msg = std::make_shared<camera_message>(std::get<camera_t>(value));
      output_->send(msg);
    } else if (std::holds_alternative<scene_t>(value)) {
      const auto& scene = std::get<scene_t>(value);
      auto msg = std::make_shared<mat4_message>();
      msg->set_data(scene.axis);
      output_->send(msg);
    }
  }

 public:
  load_parameter_node()
      : coalsack::graph_node(), output_(std::make_shared<coalsack::graph_edge>(this)) {
    set_output(output_);
  }

  void set_id(const std::string& value) { id_ = value; }

  virtual std::string get_proc_name() const override { return "load_parameter"; }

  virtual void run() override {
    const auto res = resources->get(parameter_resource::resource_name());
    if (!res) return;
    param_res_ = std::static_pointer_cast<parameter_resource>(res);

    // Initial send (synchronous — delivered to downstream process() immediately)
    send_current();

    // Monitor for updates (e.g., post-calibration) and re-send
    running_.store(true);
    uint64_t last_version = param_res_->get_parameters().get_version();
    monitor_thread_ = std::make_shared<std::thread>([this, last_version]() mutable {
      while (running_.load()) {
        last_version =
            param_res_->get_parameters().wait_for_change(last_version, running_);
        if (!running_.load()) break;
        send_current();
      }
    });
  }

  virtual void stop() override {
    running_.store(false);
    if (param_res_) {
      param_res_->get_parameters().notify_all();
    }
    if (monitor_thread_ && monitor_thread_->joinable()) {
      monitor_thread_->join();
    }
    monitor_thread_.reset();
    param_res_.reset();
  }
};

}  // namespace stargazer

