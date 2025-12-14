#include "reconstruction_pipeline.hpp"

#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <spdlog/spdlog.h>
#include <sqlite3.h>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <regex>

#include "callback_node.hpp"
#include "capture_pipeline.hpp"
#include "correspondance.hpp"
#include "dump_se3_node.hpp"
#include "epipolar_reconstruct_node.hpp"
#include "glm_json.hpp"
#include "glm_serialize.hpp"
#include "graph_builder.hpp"
#include "graph_proc.h"
#include "graph_proc_cv.h"
#include "graph_proc_img.h"
#include "graph_proc_tensor.h"
#include "grpc_server_node.hpp"
#include "image_reconstruct_node.hpp"
#include "messages.hpp"
#include "mvp.hpp"
#include "mvp_reconstruct_node.hpp"
#include "mvpose.hpp"
#include "mvpose_reconstruct_node.hpp"
#include "parameters.hpp"
#include "point_data.hpp"
#include "reconstruction.hpp"
#include "sensor.grpc.pb.h"
#include "triangulation.hpp"
#include "utils.hpp"
#include "voxelpose.hpp"
#include "voxelpose_reconstruct_node.hpp"

using namespace coalsack;
using namespace stargazer;

class multiview_point_reconstruction_pipeline::impl {
  graph_proc graph;

  std::atomic_bool running;

  mutable std::mutex markers_mtx;
  std::vector<glm::vec3> markers;
  std::vector<std::function<void(const std::vector<glm::vec3>&)>> markers_received;

  std::shared_ptr<epipolar_reconstruct_node> reconstruct_node;
  std::shared_ptr<graph_node> input_node;

  std::map<std::string, camera_t> cameras;
  glm::mat4 axis;

 public:
  void add_markers_received(std::function<void(const std::vector<glm::vec3>&)> f) {
    std::lock_guard lock(markers_mtx);
    markers_received.push_back(f);
  }

  void clear_markers_received() {
    std::lock_guard lock(markers_mtx);
    markers_received.clear();
  }

  impl()
      : graph(),
        running(false),
        markers(),
        markers_received(),
        reconstruct_node(),
        input_node(),
        cameras(),
        axis(1.0f) {}

  void set_camera(const std::string& name, const camera_t& camera) {
    cameras[name] = camera;
    if (reconstruct_node) {
      std::map<std::string, camera_t> updated_cameras = cameras;
      reconstruct_node->set_cameras(updated_cameras);
    }
  }

  void set_axis(const glm::mat4& new_axis) {
    axis = new_axis;
    if (reconstruct_node) {
      reconstruct_node->set_axis(axis);
    }
  }

  using frame_type = std::map<std::string, std::vector<point_data>>;

  void push_frame(const frame_type& frame) {
    if (!running) {
      return;
    }

    auto msg = std::make_shared<object_message>();
    for (const auto& [name, field] : frame) {
      auto float2_msg = std::make_shared<float2_list_message>();
      std::vector<float2> float2_data;
      for (const auto& pt : field) {
        float2_data.push_back({pt.point.x, pt.point.y});
      }
      float2_msg->set_data(float2_data);
      msg->add_field(name, float2_msg);
    }

    auto frame_msg = std::make_shared<frame_message<object_message>>();
    frame_msg->set_data(*msg);

    if (input_node) {
      graph.process(input_node.get(), frame_msg);
    }
  }

  void run(const std::vector<node_def>& nodes) {
    // Group nodes by subgraph instance
    std::map<std::string, std::vector<node_def>> nodes_by_subgraph;
    for (const auto& node : nodes) {
      nodes_by_subgraph[node.subgraph_instance].push_back(node);
    }

    // Create empty subgraphs first
    std::map<std::string, std::shared_ptr<subgraph>> subgraphs;
    for (const auto& [subgraph_name, nodes] : nodes_by_subgraph) {
      subgraphs[subgraph_name] = std::make_shared<subgraph>();
    }

    std::unordered_map<std::string, graph_node_ptr> node_map;

    // Build graph using common function
    stargazer::build_graph_from_json(nodes, subgraphs, node_map);

    // Extract specific nodes from the graph
    for (const auto& node : nodes) {
      if (node.get_type() == node_type::frame_number_numbering) {
        if (input_node) {
          spdlog::warn("Multiple frame_number_numbering nodes found, using the first one");
        } else {
          input_node =
              std::dynamic_pointer_cast<frame_number_numbering_node>(node_map.at(node.name));
        }
      } else if (node.get_type() == node_type::epipolar_reconstruction) {
        if (reconstruct_node) {
          spdlog::warn("Multiple epipolar_reconstruction nodes found, using the first one");
        } else {
          reconstruct_node =
              std::dynamic_pointer_cast<epipolar_reconstruct_node>(node_map.at(node.name));
          // Set cameras and axis for reconstruction node
          if (reconstruct_node) {
            reconstruct_node->set_cameras(cameras);
            reconstruct_node->set_axis(axis);
          }
        }
      }
    }

    const auto callbacks = std::make_shared<callback_list>();

    callbacks->add(
        [this](const callback_node* node, std::string input_name, graph_message_ptr message) {
          if (node->get_callback_name() == "markers") {
            if (const auto markers_msg = std::dynamic_pointer_cast<float3_list_message>(message)) {
              std::vector<glm::vec3> markers;
              for (const auto& marker : markers_msg->get_data()) {
                markers.push_back(glm::vec3(marker.x, marker.y, marker.z));
              }

              {
                std::lock_guard lock(markers_mtx);
                this->markers = markers;
              }

              for (const auto& f : markers_received) {
                f(markers);
              }
            }
          }
        });

    // Deploy all subgraphs
    for (const auto& [subgraph_name, subgraph_ptr] : subgraphs) {
      graph.deploy(subgraph_ptr);
    }
    graph.get_resources()->add(callbacks);
    graph.run();

    running = true;
  }

  void stop() {
    running.store(false);
    graph.stop();
  }

  std::vector<glm::vec3> get_markers() const {
    std::vector<glm::vec3> result;

    {
      std::lock_guard lock(markers_mtx);
      result = this->markers;
    }

    return result;
  }
};

multiview_point_reconstruction_pipeline::multiview_point_reconstruction_pipeline()
    : pimpl(new impl()) {}
multiview_point_reconstruction_pipeline::~multiview_point_reconstruction_pipeline() = default;

void multiview_point_reconstruction_pipeline::push_frame(const frame_type& frame) {
  pimpl->push_frame(frame);
}

void multiview_point_reconstruction_pipeline::run(const std::vector<node_def>& nodes) {
  pimpl->run(nodes);
}

void multiview_point_reconstruction_pipeline::stop() { pimpl->stop(); }

std::vector<glm::vec3> multiview_point_reconstruction_pipeline::get_markers() const {
  return pimpl->get_markers();
}

void multiview_point_reconstruction_pipeline::set_camera(const std::string& name,
                                                         const camera_t& camera) {
  cameras[name] = camera;
  pimpl->set_camera(name, camera);
}
void multiview_point_reconstruction_pipeline::set_axis(const glm::mat4& axis) {
  this->axis = axis;
  pimpl->set_axis(axis);
}

class multiview_image_reconstruction_pipeline::impl {
  graph_proc graph;

  std::atomic_bool running;

  mutable std::mutex markers_mtx;
  std::vector<glm::vec3> markers;
  std::vector<std::function<void(const std::vector<glm::vec3>&)>> markers_received;

  std::shared_ptr<image_reconstruct_node> reconstruct_node;
  std::shared_ptr<graph_node> input_node;

  std::map<std::string, camera_t> cameras;
  glm::mat4 axis;

 public:
  void add_markers_received(std::function<void(const std::vector<glm::vec3>&)> f) {
    std::lock_guard lock(markers_mtx);
    markers_received.push_back(f);
  }

  void clear_markers_received() {
    std::lock_guard lock(markers_mtx);
    markers_received.clear();
  }

  impl()
      : graph(),
        running(false),
        markers(),
        markers_received(),
        reconstruct_node(),
        input_node(),
        cameras(),
        axis(1.0f) {}

  void set_camera(const std::string& name, const camera_t& camera) {
    cameras[name] = camera;
    if (reconstruct_node) {
      std::map<std::string, camera_t> updated_cameras = cameras;
      reconstruct_node->set_cameras(updated_cameras);
    }
  }

  void set_axis(const glm::mat4& new_axis) {
    axis = new_axis;
    if (reconstruct_node) {
      reconstruct_node->set_axis(axis);
    }
  }

  using frame_type = std::map<std::string, cv::Mat>;

  static image_format convert_to_image_format(int type) {
    switch (type) {
      case CV_8UC1:
        return image_format::Y8_UINT;
      case CV_8UC3:
        return image_format::B8G8R8_UINT;
      case CV_8UC4:
        return image_format::B8G8R8A8_UINT;
      default:
        throw std::runtime_error("Invalid image format");
    }
  }

  void push_frame(const frame_type& frame) {
    if (!running) {
      return;
    }

    auto msg = std::make_shared<object_message>();
    for (const auto& [name, field] : frame) {
      auto img_msg = std::make_shared<image_message>();

      image img(static_cast<std::uint32_t>(field.size().width),
                static_cast<std::uint32_t>(field.size().height),
                static_cast<std::uint32_t>(field.elemSize()),
                static_cast<std::uint32_t>(field.step), (const uint8_t*)field.data);
      img.set_format(convert_to_image_format(field.type()));

      img_msg->set_image(std::move(img));
      msg->add_field(name, img_msg);
    }

    auto frame_msg = std::make_shared<frame_message<object_message>>();
    frame_msg->set_data(*msg);

    if (input_node) {
      graph.process(input_node.get(), frame_msg);
    }
  }

  void run(const std::vector<node_def>& nodes) {
    // Group nodes by subgraph instance
    std::map<std::string, std::vector<node_def>> nodes_by_subgraph;
    for (const auto& node : nodes) {
      nodes_by_subgraph[node.subgraph_instance].push_back(node);
    }

    // Create empty subgraphs first
    std::map<std::string, std::shared_ptr<subgraph>> subgraphs;
    for (const auto& [subgraph_name, nodes] : nodes_by_subgraph) {
      subgraphs[subgraph_name] = std::make_shared<subgraph>();
    }

    std::unordered_map<std::string, graph_node_ptr> node_map;

    // Build graph using common function
    stargazer::build_graph_from_json(nodes, subgraphs, node_map);

    // Extract specific nodes from the graph
    for (const auto& node : nodes) {
      if (node.get_type() == node_type::frame_number_numbering) {
        if (input_node) {
          spdlog::warn("Multiple frame_number_numbering nodes found, using the first one");
        } else {
          input_node =
              std::dynamic_pointer_cast<frame_number_numbering_node>(node_map.at(node.name));
        }
      } else if (node.get_type() == node_type::voxelpose_reconstruction) {
        if (reconstruct_node) {
          spdlog::warn("Multiple reconstruction nodes found, using the first one");
        } else {
          reconstruct_node =
              std::dynamic_pointer_cast<voxelpose_reconstruct_node>(node_map.at(node.name));
          // Set cameras and axis for reconstruction node
          if (reconstruct_node) {
            reconstruct_node->set_cameras(cameras);
            reconstruct_node->set_axis(axis);
          }
        }
      } else if (node.get_type() == node_type::mvpose_reconstruction) {
        if (reconstruct_node) {
          spdlog::warn("Multiple reconstruction nodes found, using the first one");
        } else {
          reconstruct_node =
              std::dynamic_pointer_cast<mvpose_reconstruct_node>(node_map.at(node.name));
          // Set cameras and axis for reconstruction node
          if (reconstruct_node) {
            reconstruct_node->set_cameras(cameras);
            reconstruct_node->set_axis(axis);
          }
        }
      } else if (node.get_type() == node_type::mvp_reconstruction) {
        if (reconstruct_node) {
          spdlog::warn("Multiple reconstruction nodes found, using the first one");
        } else {
          reconstruct_node =
              std::dynamic_pointer_cast<mvp_reconstruct_node>(node_map.at(node.name));
          // Set cameras and axis for reconstruction node
          if (reconstruct_node) {
            reconstruct_node->set_cameras(cameras);
            reconstruct_node->set_axis(axis);
          }
        }
      }
    }

    if (!input_node) {
      spdlog::warn("frame_number_numbering node not found in image reconstruction pipeline");
    }
    if (!reconstruct_node) {
      spdlog::warn("reconstruction node not found in image reconstruction pipeline");
    }

    const auto callbacks = std::make_shared<callback_list>();

    callbacks->add(
        [this](const callback_node* node, std::string input_name, graph_message_ptr message) {
          if (node->get_callback_name() == "markers") {
            if (const auto markers_msg = std::dynamic_pointer_cast<float3_list_message>(message)) {
              std::vector<glm::vec3> markers;
              for (const auto& marker : markers_msg->get_data()) {
                markers.push_back(glm::vec3(marker.x, marker.y, marker.z));
              }

              {
                std::lock_guard lock(markers_mtx);
                this->markers = markers;
              }

              for (const auto& f : markers_received) {
                f(markers);
              }
            }
          }
        });

    // Deploy all subgraphs
    for (const auto& [subgraph_name, subgraph_ptr] : subgraphs) {
      graph.deploy(subgraph_ptr);
    }
    graph.get_resources()->add(callbacks);
    graph.run();

    running = true;
  }

  void stop() {
    running.store(false);
    graph.stop();
  }

  std::vector<glm::vec3> get_markers() const {
    std::vector<glm::vec3> result;

    {
      std::lock_guard lock(markers_mtx);
      result = this->markers;
    }

    return result;
  }

  std::map<std::string, cv::Mat> get_features() const { return reconstruct_node->get_features(); }
};

multiview_image_reconstruction_pipeline::multiview_image_reconstruction_pipeline()
    : pimpl(new impl()) {}
multiview_image_reconstruction_pipeline::~multiview_image_reconstruction_pipeline() = default;

void multiview_image_reconstruction_pipeline::push_frame(const frame_type& frame) {
  pimpl->push_frame(frame);
}

void multiview_image_reconstruction_pipeline::run(const std::vector<node_def>& nodes) {
  pimpl->run(nodes);
}

void multiview_image_reconstruction_pipeline::stop() { pimpl->stop(); }

std::map<std::string, cv::Mat> multiview_image_reconstruction_pipeline::get_features() const {
  return pimpl->get_features();
}

std::vector<glm::vec3> multiview_image_reconstruction_pipeline::get_markers() const {
  return pimpl->get_markers();
}

void multiview_image_reconstruction_pipeline::set_camera(const std::string& name,
                                                         const camera_t& camera) {
  pimpl->set_camera(name, camera);
}

void multiview_image_reconstruction_pipeline::set_axis(const glm::mat4& axis) {
  pimpl->set_axis(axis);
}