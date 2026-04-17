/// @file load_parameter_node.hpp
/// @brief Camera/scene parameter resource reader node.
/// @ingroup io_nodes
#pragma once

#include <atomic>
#include <thread>

#include "coalsack/core/graph_proc.h"
#include "messages.hpp"
#include "parameter_resource.hpp"

namespace stargazer {

/// @brief Reads the latest camera or scene parameters from a parameter resource.
/// @details Monitors the parameter resource identified by `id` and emits an
///          `object_message` containing the current `camera_t` or `scene_t` whenever
///          the resource is updated.  Also emits on `run()` so downstream nodes
///          receive an initial value.  Monitoring runs on a background thread.
///
/// @par Inputs
/// (none — resource-driven source node)
///
/// @par Outputs
/// - @b "default" — `object_message` — current parameters as `camera_t` or `scene_t`
///
/// @par Properties
/// - `id` (`std::string`, default `""`) — identifier of the parameter resource to watch
///
/// @see store_parameter_node
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
    auto obj_msg = std::make_shared<object_message>();
    if (std::holds_alternative<camera_t>(value)) {
      obj_msg->add_field(id_, std::make_shared<camera_message>(std::get<camera_t>(value)));
    } else if (std::holds_alternative<scene_t>(value)) {
      obj_msg->add_field(id_, std::make_shared<scene_message>(std::get<scene_t>(value)));
    }
    output_->send(obj_msg);
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
        last_version = param_res_->get_parameters().wait_for_change(last_version, running_);
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
