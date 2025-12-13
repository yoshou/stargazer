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
      : graph(), running(false), markers(), markers_received(), reconstruct_node(), input_node() {}

  void set_camera(const std::string& name, const camera_t& camera) {
    auto camera_msg = std::make_shared<camera_message>(camera);
    camera_msg->set_camera(camera);

    auto obj_msg = std::make_shared<object_message>();
    obj_msg->add_field(name, camera_msg);

    if (reconstruct_node) {
      graph.process(reconstruct_node.get(), "cameras", obj_msg);
    }
  }

  void set_axis(const glm::mat4& axis) {
    auto mat4_msg = std::make_shared<mat4_message>();
    mat4_msg->set_data(axis);

    if (reconstruct_node) {
      graph.process(reconstruct_node.get(), "axis", mat4_msg);
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

  void run() {
    std::shared_ptr<subgraph> g(new subgraph());

    std::shared_ptr<frame_number_numbering_node> n4(new frame_number_numbering_node());
    g->add_node(n4);

    input_node = n4;

    std::shared_ptr<parallel_queue_node> n6(new parallel_queue_node());
    n6->set_input(n4->get_output());
    g->add_node(n6);

    std::shared_ptr<epipolar_reconstruct_node> n1(new epipolar_reconstruct_node());
    n1->set_input(n6->get_output());
    g->add_node(n1);

    reconstruct_node = n1;

    std::shared_ptr<frame_number_ordering_node> n5(new frame_number_ordering_node());
    n5->set_input(n1->get_output());
    g->add_node(n5);

    std::shared_ptr<callback_node> n2(new callback_node());
    n2->set_input(n5->get_output());
    g->add_node(n2);

    n2->set_callback_name("markers");

    std::shared_ptr<grpc_server_node> n3(new grpc_server_node());
    n3->set_input(n5->get_output(), "sphere");
    g->add_node(n3);

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

    graph.deploy(g);
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

void multiview_point_reconstruction_pipeline::run() { pimpl->run(); }

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
      : graph(), running(false), markers(), markers_received(), reconstruct_node(), input_node() {}

  void set_camera(const std::string& name, const camera_t& camera) {
    auto camera_msg = std::make_shared<camera_message>(camera);
    camera_msg->set_camera(camera);

    auto obj_msg = std::make_shared<object_message>();
    obj_msg->add_field(name, camera_msg);

    if (reconstruct_node) {
      graph.process(reconstruct_node.get(), "cameras", obj_msg);
    }
  }

  void set_axis(const glm::mat4& axis) {
    auto mat4_msg = std::make_shared<mat4_message>();
    mat4_msg->set_data(axis);

    if (reconstruct_node) {
      graph.process(reconstruct_node.get(), "axis", mat4_msg);
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

  void run(const std::vector<node_info>& infos) {
    std::cout << "=== Reconstruction Pipeline Graph Structure ===" << std::endl;
    for (const auto& info : infos) {
      std::cout << "Node: " << info.name << " (type: " << static_cast<int>(info.get_type()) << ")"
                << std::endl;

      // Print node parameters
      if (info.contains_param("fps")) {
        std::cout << "  fps: " << info.get_param<double>("fps") << std::endl;
      }
      if (info.contains_param("interval")) {
        std::cout << "  interval: " << info.get_param<double>("interval") << std::endl;
      }
      if (info.contains_param("num_threads")) {
        std::cout << "  num_threads: " << info.get_param<std::int64_t>("num_threads") << std::endl;
      }
      if (info.contains_param("callback_name")) {
        std::cout << "  callback_name: " << info.get_param<std::string>("callback_name")
                  << std::endl;
      }

      for (const auto& [input_name, source_name] : info.inputs) {
        std::cout << "  Input '" << input_name << "' <- '" << source_name << "'" << std::endl;
      }
      for (const auto& output_name : info.outputs) {
        std::cout << "  Output: '" << output_name << "'" << std::endl;
      }
    }
    std::cout << "==============================================" << std::endl;

    std::shared_ptr<subgraph> g(new subgraph());

    std::unordered_map<std::string, graph_node_ptr> node_map;

    for (const auto& info : infos) {
      graph_node_ptr node;

      switch (info.get_type()) {
        case node_type::frame_number_numbering: {
          auto n = std::make_shared<frame_number_numbering_node>();
          node = n;
          input_node = n;
          break;
        }
        case node_type::parallel_queue: {
          auto n = std::make_shared<parallel_queue_node>();
          if (info.contains_param("num_threads")) {
            n->set_num_threads(static_cast<size_t>(info.get_param<std::int64_t>("num_threads")));
          }
          node = n;
          break;
        }
        case node_type::voxelpose_reconstruction: {
          auto n = std::make_shared<voxelpose_reconstruct_node>();
          node = n;
          reconstruct_node = n;
          break;
        }
        case node_type::mvpose_reconstruction: {
          auto n = std::make_shared<mvpose_reconstruct_node>();
          node = n;
          reconstruct_node = n;
          break;
        }
        case node_type::mvp_reconstruction: {
          auto n = std::make_shared<mvp_reconstruct_node>();
          node = n;
          reconstruct_node = n;
          break;
        }
        case node_type::frame_number_ordering: {
          auto n = std::make_shared<frame_number_ordering_node>();
          node = n;
          break;
        }
        case node_type::callback: {
          auto n = std::make_shared<callback_node>();
          if (info.contains_param("callback_name")) {
            n->set_callback_name(info.get_param<std::string>("callback_name"));
          }
          node = n;
          break;
        }
        case node_type::grpc_server: {
          auto n = std::make_shared<grpc_server_node>();
          if (info.contains_param("address")) {
            n->set_address(info.get_param<std::string>("address"));
          }
          node = n;
          break;
        }
        case node_type::frame_demux: {
          auto n = std::make_shared<frame_demux_node>();
          for (const auto& output_name : info.outputs) {
            n->add_output(output_name);
          }
          node = n;
          break;
        }
        case node_type::dump_se3: {
          auto n = std::make_shared<dump_se3_node>();
          if (info.contains_param("db_path")) {
            n->set_db_path(info.get_param<std::string>("db_path"));
          }
          if (info.contains_param("topic_name")) {
            n->set_name(info.get_param<std::string>("topic_name"));
          }
          node = n;
          break;
        }
        default:
          throw std::runtime_error("Unknown node type: " + info.name);
      }

      node_map[info.name] = node;
      g->add_node(node);
    }

    for (const auto& info : infos) {
      if (info.inputs.empty()) {
        continue;
      }

      auto target_node = node_map.at(info.name);

      for (const auto& [input_name, source_name] : info.inputs) {
        size_t pos = source_name.find(':');
        if (pos != std::string::npos) {
          auto node_name = source_name.substr(0, pos);
          auto output_name = source_name.substr(pos + 1);
          auto source_node = node_map.at(node_name);
          target_node->set_input(source_node->get_output(output_name), input_name);
        } else {
          auto source_node = node_map.at(source_name);
          target_node->set_input(source_node->get_output(), input_name);
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

    graph.deploy(g);
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

void multiview_image_reconstruction_pipeline::run(const std::vector<node_info>& infos) {
  pimpl->run(infos);
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