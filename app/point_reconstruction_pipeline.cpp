#include "point_reconstruction_pipeline.hpp"

#include <spdlog/spdlog.h>

#include "callback_node.hpp"
#include "coalsack/core/graph_proc.h"
#include "epipolar_reconstruct_node.hpp"
#include "glm_serialize.hpp"
#include "graph_builder.hpp"
#include "grpc_server_node.hpp"
#include "messages.hpp"
#include "parameter_resource.hpp"
#include "parameters.hpp"
#include "point_data.hpp"

using namespace coalsack;
using namespace stargazer;

class multiview_point_reconstruction_pipeline::impl {
  graph_proc graph;
  std::unordered_map<std::string, graph_node_ptr> node_map;

  std::atomic_bool running;

  mutable std::mutex markers_mtx;
  std::vector<glm::vec3> markers;
  std::vector<std::function<void(const std::vector<glm::vec3>&)>> markers_received;

  std::shared_ptr<graph_node> input_node;
  std::shared_ptr<parameters_t> parameters_;

 public:
  void add_markers_received(std::function<void(const std::vector<glm::vec3>&)> f) {
    std::lock_guard lock(markers_mtx);
    markers_received.push_back(f);
  }

  void clear_markers_received() {
    std::lock_guard lock(markers_mtx);
    markers_received.clear();
  }

  explicit impl(std::shared_ptr<parameters_t> parameters)
      : graph(),
        running(false),
        markers(),
        markers_received(),
        input_node(),
        parameters_(std::move(parameters)) {}

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

    std::unordered_map<std::string, graph_node_ptr> built_node_map;

    // Build graph using common function
    stargazer::build_graph_from_json(nodes, subgraphs, built_node_map);
    node_map = built_node_map;

    // Find input node
    for (const auto& node : nodes) {
      if (node.get_type() == node_type::frame_number_numbering) {
        if (!input_node) {
          input_node =
              std::dynamic_pointer_cast<frame_number_numbering_node>(built_node_map.at(node.name));
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
    if (parameters_) {
      graph.get_resources()->add(std::make_shared<parameter_resource>(parameters_));
    }
    graph.initialize();
    graph.run();

    running = true;
  }

  void stop() {
    running.store(false);
    graph.stop();
    graph.finalize();
  }

  std::optional<property_value> get_node_property(const std::string& node_name,
                                                  const std::string& key) const {
    const auto found = node_map.find(node_name);
    if (found == node_map.end() || !found->second) {
      return std::nullopt;
    }
    return found->second->get_property(key);
  }
};

multiview_point_reconstruction_pipeline::multiview_point_reconstruction_pipeline(
    std::shared_ptr<stargazer::parameters_t> parameters)
    : pimpl(new impl(std::move(parameters))) {}
multiview_point_reconstruction_pipeline::~multiview_point_reconstruction_pipeline() = default;

void multiview_point_reconstruction_pipeline::push_frame(const frame_type& frame) {
  pimpl->push_frame(frame);
}

void multiview_point_reconstruction_pipeline::run(const std::vector<node_def>& nodes) {
  pimpl->run(nodes);
}

void multiview_point_reconstruction_pipeline::stop() { pimpl->stop(); }

std::optional<property_value> multiview_point_reconstruction_pipeline::get_node_property(
    const std::string& node_name, const std::string& key) const {
  return pimpl->get_node_property(node_name, key);
}
