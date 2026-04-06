#pragma once

#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "coalsack/camera/camera.h"
#include "coalsack/core/graph_node.h"
#include "config.hpp"
#include "pipeline_control.grpc.pb.h"
#include "pipeline_control_command.hpp"

namespace stargazer {

struct pipeline_action_info {
  std::string id;
  std::string label;
  std::string icon;
};

class PipelineControlServiceImpl final : public PipelineControl::Service {
  pipeline_command_queue* cmd_queue_{nullptr};
  std::atomic<bool>* running_{nullptr};
  std::atomic<bool>* collecting_{nullptr};
  std::function<std::optional<coalsack::property_value>(const std::string&, const std::string&)>
      get_property_;

  std::vector<node_def> nodes_;
  std::vector<pipeline_action_info> actions_;

  // Convert coalsack::image to JPEG bytes
  static std::string image_to_jpeg_bytes(const std::shared_ptr<coalsack::image>& img) {
    if (!img || img->empty()) {
      return {};
    }
    int cv_type = -1;
    switch (img->get_format()) {
      case coalsack::image_format::Y8_UINT:
        cv_type = CV_8UC1;
        break;
      case coalsack::image_format::R8G8B8_UINT:
        cv_type = CV_8UC3;
        break;
      case coalsack::image_format::B8G8R8_UINT:
        cv_type = CV_8UC3;
        break;
      case coalsack::image_format::R8G8B8A8_UINT:
      case coalsack::image_format::B8G8R8A8_UINT:
        cv_type = CV_8UC4;
        break;
      default:
        return {};
    }
    cv::Mat frame(static_cast<int>(img->get_height()), static_cast<int>(img->get_width()),
                  cv_type, const_cast<uint8_t*>(img->get_data()),
                  static_cast<size_t>(img->get_stride()));
    cv::Mat rgb;
    if (img->get_format() == coalsack::image_format::B8G8R8_UINT) {
      cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    } else if (img->get_format() == coalsack::image_format::B8G8R8A8_UINT) {
      cv::cvtColor(frame, rgb, cv::COLOR_BGRA2RGBA);
    } else {
      rgb = frame.clone();
    }
    std::vector<uchar> buf;
    cv::imencode(".jpg", rgb, buf);
    return std::string(buf.begin(), buf.end());
  }

  // Execute a command synchronously via the command queue
  grpc::Status dispatch_command(pipeline_command_type type, const std::string& param = {}) {
    if (!cmd_queue_) {
      return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Command queue not initialized");
    }
    auto future = cmd_queue_->enqueue(type, param);
    auto result = future.get();
    if (result.has_value()) {
      return grpc::Status(grpc::StatusCode::INTERNAL, *result);
    }
    return grpc::Status::OK;
  }

 public:
  void setup(pipeline_command_queue* queue, std::atomic<bool>* running,
             std::atomic<bool>* collecting,
             std::function<std::optional<coalsack::property_value>(const std::string&,
                                                                    const std::string&)>
                 get_property,
             std::vector<node_def> nodes) {
    cmd_queue_ = queue;
    running_ = running;
    collecting_ = collecting;
    get_property_ = std::move(get_property);
    nodes_ = std::move(nodes);

    // Extract actions from action nodes
    std::unordered_map<std::string, pipeline_action_info> seen;
    for (const auto& node : nodes_) {
      try {
        if (node.get_type() != node_type::action) continue;
      } catch (...) {
        continue;
      }
      pipeline_action_info act;
      act.id = node.contains_param("action_id")
                   ? node.get_param<std::string>("action_id")
                   : node.name;
      act.label = node.contains_param("label") ? node.get_param<std::string>("label") : act.id;
      act.icon = node.contains_param("icon") ? node.get_param<std::string>("icon") : "";
      if (seen.find(act.id) == seen.end()) {
        seen[act.id] = act;
        actions_.push_back(act);
      }
    }
  }

  grpc::Status Start(grpc::ServerContext*, const google::protobuf::Empty*,
                     google::protobuf::Empty*) override {
    return dispatch_command(pipeline_command_type::start);
  }

  grpc::Status Stop(grpc::ServerContext*, const google::protobuf::Empty*,
                    google::protobuf::Empty*) override {
    return dispatch_command(pipeline_command_type::stop);
  }

  grpc::Status GetStatus(grpc::ServerContext*, const google::protobuf::Empty*,
                         PipelineStatus* response) override {
    response->set_running(running_ ? running_->load() : false);
    response->set_collecting(collecting_ ? collecting_->load() : false);
    return grpc::Status::OK;
  }

  grpc::Status EnableCollecting(grpc::ServerContext*, const google::protobuf::Empty*,
                                google::protobuf::Empty*) override {
    return dispatch_command(pipeline_command_type::enable_collecting);
  }

  grpc::Status DisableCollecting(grpc::ServerContext*, const google::protobuf::Empty*,
                                 google::protobuf::Empty*) override {
    return dispatch_command(pipeline_command_type::disable_collecting);
  }

  grpc::Status DispatchAction(grpc::ServerContext*, const ActionRequest* request,
                              google::protobuf::Empty*) override {
    return dispatch_command(pipeline_command_type::dispatch_action, request->action_id());
  }

  grpc::Status ListActions(grpc::ServerContext*, const google::protobuf::Empty*,
                           ActionList* response) override {
    for (const auto& action : actions_) {
      auto* info = response->add_actions();
      info->set_id(action.id);
      info->set_label(action.label);
      info->set_icon(action.icon);
    }
    return grpc::Status::OK;
  }

  grpc::Status GetNodeProperty(grpc::ServerContext*, const PropertyRequest* request,
                               PropertyResponse* response) override {
    if (!get_property_) {
      return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Property getter not initialized");
    }
    const auto value = get_property_(request->node_name(), request->key());
    if (!value.has_value()) {
      response->set_found(false);
      return grpc::Status::OK;
    }
    response->set_found(true);
    std::visit(
        [&](const auto& v) {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, std::string>) {
            response->set_string_value(v);
          } else if constexpr (std::is_same_v<T, std::int64_t>) {
            response->set_int_value(v);
          } else if constexpr (std::is_same_v<T, double>) {
            response->set_double_value(v);
          } else if constexpr (std::is_same_v<T, bool>) {
            response->set_bool_value(v);
          } else if constexpr (std::is_same_v<T, std::shared_ptr<coalsack::image>>) {
            auto bytes = image_to_jpeg_bytes(v);
            response->set_image_value(std::move(bytes));
          } else if constexpr (std::is_same_v<T, coalsack::camera_t>) {
            auto* cam = response->mutable_camera_value();
            cam->set_fx(v.fx);
            cam->set_fy(v.fy);
            cam->set_cx(v.ppx);
            cam->set_cy(v.ppy);
            cam->set_width(v.width);
            cam->set_height(v.height);
            cam->set_k1(v.coeffs[0]);
            cam->set_k2(v.coeffs[1]);
            cam->set_p1(v.coeffs[2]);
            cam->set_p2(v.coeffs[3]);
            cam->set_k3(v.coeffs[4]);
            for (int i = 0; i < 16; ++i) {
              cam->add_pose(v.pose.data[i]);
            }
          } else if constexpr (std::is_same_v<T, coalsack::mat4>) {
            auto* m = response->mutable_mat4_value();
            for (int i = 0; i < 16; ++i) {
              m->add_values(v.data[i]);
            }
          } else if constexpr (std::is_same_v<T, std::vector<coalsack::vec3>>) {
            auto* list = response->mutable_vec3_list_value();
            for (const auto& p : v) {
              auto* pt = list->add_points();
              pt->set_x(p.x);
              pt->set_y(p.y);
              pt->set_z(p.z);
            }
          }
        },
        value.value());
    return grpc::Status::OK;
  }

  grpc::Status ListProperties(grpc::ServerContext*, const NodeRequest* request,
                              PropertyList* response) override {
    const std::string& filter = request->node_name();
    for (const auto& node : nodes_) {
      if (!filter.empty() && node.name != filter) {
        continue;
      }
      for (const auto& prop : node.properties) {
        auto* info = response->add_properties();
        info->set_node_name(node.name);
        info->set_key(prop.source_key);
        info->set_label(prop.label);
        // Derive type_hint from target
        if (prop.target == "image" || prop.target == "contrail" || prop.target == "point") {
          info->set_type_hint("image");
        } else if (!prop.default_value.has_value()) {
          info->set_type_hint("unknown");
        } else {
          std::visit(
              [&](const auto& dv) {
                using T = std::decay_t<decltype(dv)>;
                if constexpr (std::is_same_v<T, std::string>)
                  info->set_type_hint("string");
                else if constexpr (std::is_same_v<T, std::int64_t>)
                  info->set_type_hint("int");
                else if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>)
                  info->set_type_hint("double");
                else if constexpr (std::is_same_v<T, bool>)
                  info->set_type_hint("bool");
                else
                  info->set_type_hint("unknown");
              },
              *prop.default_value);
        }
      }
    }
    return grpc::Status::OK;
  }

  grpc::Status ListNodes(grpc::ServerContext*, const google::protobuf::Empty*,
                         NodeList* response) override {
    for (const auto& node : nodes_) {
      auto* info = response->add_nodes();
      info->set_name(node.name);
      node_type type;
      try {
        type = node.get_type();
      } catch (...) {
        info->set_type("unknown");
        continue;
      }
      switch (type) {
        case node_type::image_property:
          info->set_type("image_property");
          break;
        case node_type::marker_property:
          info->set_type("marker_property");
          break;
        case node_type::extrinsic_calibration:
          info->set_type("extrinsic_calibration");
          break;
        case node_type::intrinsic_calibration:
          info->set_type("intrinsic_calibration");
          break;
        case node_type::epipolar_reconstruction:
          info->set_type("epipolar_reconstruction");
          break;
        case node_type::action:
          info->set_type("action");
          break;
        case node_type::gate:
          info->set_type("gate");
          break;
        case node_type::grpc_server:
          info->set_type("grpc_server");
          break;
        default:
          info->set_type("other");
          break;
      }
    }
    return grpc::Status::OK;
  }
};

class pipeline_control_server {
  std::string address_;
  std::atomic_bool running_;
  std::shared_ptr<std::thread> server_th_;
  std::unique_ptr<grpc::Server> server_;
  std::unique_ptr<PipelineControlServiceImpl> service_;

 public:
  explicit pipeline_control_server(const std::string& address)
      : address_(address), running_(false), service_(std::make_unique<PipelineControlServiceImpl>()) {}

  ~pipeline_control_server() { stop(); }

  PipelineControlServiceImpl* service() { return service_.get(); }

  void run() {
    running_ = true;
    grpc::ServerBuilder builder;
    builder.AddListeningPort(address_, grpc::InsecureServerCredentials());
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    if (server_) {
      spdlog::info("PipelineControl gRPC server listening on {}", address_);
    } else {
      spdlog::error("Failed to start PipelineControl gRPC server on {}", address_);
      running_ = false;
      return;
    }
    server_th_.reset(new std::thread([this]() { server_->Wait(); }));
  }

  void stop() {
    if (running_.load()) {
      running_.store(false);
    }
    if (server_) {
      server_->Shutdown(std::chrono::system_clock::now());
      if (server_th_ && server_th_->joinable()) {
        server_th_->join();
      }
    }
  }
};

}  // namespace stargazer
