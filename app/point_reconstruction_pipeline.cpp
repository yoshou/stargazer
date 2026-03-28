#include "point_reconstruction_pipeline.hpp"

#include <spdlog/spdlog.h>

#include <set>
#include <unordered_map>

#include "callback_node.hpp"
#include "coalsack/core/graph_proc.h"
#include "coalsack/core/graph_proc_client.h"
#include "coalsack/image/image_nodes.h"
#include "glm_serialize.hpp"
#include "graph_builder.hpp"
#include "messages.hpp"
#include "parameter_resource.hpp"
#include "parameters.hpp"
#include "point_data.hpp"

using namespace coalsack;
using namespace stargazer;

class multiview_point_reconstruction_pipeline::impl {
  std::unordered_map<std::string, std::shared_ptr<graph_node>> node_map;

  graph_proc local_graph;
  std::shared_ptr<subgraph> local_subgraph;
  asio::io_context io_context;
  graph_proc_client client;
  std::unique_ptr<std::thread> io_thread;

  bool has_remote_subgraphs = false;

  std::shared_ptr<stargazer::parameters_t> parameters_;

  bool is_local_node(const graph_node* node) const {
    return node && local_subgraph && node->get_parent() == local_subgraph.get();
  }

 public:
  impl()
      : node_map(),
        local_graph(),
        local_subgraph(),
        io_context(),
        client(),
        io_thread(),
        has_remote_subgraphs(false),
        parameters_() {}

  void deploy(const std::vector<node_def>& nodes) {
    node_map.clear();
    local_graph = graph_proc();
    local_subgraph.reset();
    client = graph_proc_client();
    io_context.restart();
    has_remote_subgraphs = false;

    std::map<std::string, std::vector<node_def>> nodes_by_subgraph;
    for (const auto& node : nodes) {
      nodes_by_subgraph[node.subgraph_instance].push_back(node);
    }

    std::unordered_map<std::string, std::shared_ptr<graph_node>> global_node_map;

    std::map<std::string, std::shared_ptr<subgraph>> subgraphs;
    for (const auto& [subgraph_name, sg_nodes] : nodes_by_subgraph) {
      subgraphs[subgraph_name] = std::make_shared<subgraph>();
    }

    stargazer::build_graph_from_json(nodes, subgraphs, global_node_map);
    node_map = global_node_map;

    const auto callbacks = std::make_shared<callback_list>();

    std::map<std::string, std::pair<std::string, uint16_t>> remote_subgraph_deploy_info;
    std::vector<std::string> local_subgraph_names;

    for (const auto& [subgraph_name, subgraph_nodes] : nodes_by_subgraph) {
      bool is_remote = false;
      std::string deploy_address = "127.0.0.1";
      uint16_t deploy_port = 0;

      for (const auto& node : subgraph_nodes) {
        if (node.contains_param("deploy_port")) {
          is_remote = true;
          deploy_port = static_cast<uint16_t>(node.get_param<std::int64_t>("deploy_port"));
        }
        if (node.contains_param("address")) {
          deploy_address = node.get_param<std::string>("address");
        }
      }

      if (is_remote) {
        remote_subgraph_deploy_info[subgraph_name] = std::make_pair(deploy_address, deploy_port);
      } else {
        local_subgraph_names.push_back(subgraph_name);
      }
    }

    if (!local_subgraph_names.empty()) {
      local_subgraph = std::make_shared<subgraph>();
      for (const auto& subgraph_name : local_subgraph_names) {
        local_subgraph->merge(*subgraphs.at(subgraph_name));
      }
    }

    std::map<std::pair<std::string, uint16_t>, std::vector<std::string>> address_groups;
    for (const auto& [subgraph_name, deploy_key] : remote_subgraph_deploy_info) {
      address_groups[deploy_key].push_back(subgraph_name);
    }

    std::map<std::string, std::shared_ptr<subgraph>> deploy_subgraphs;
    std::map<std::string, std::string> original_to_merged;
    std::map<std::string, std::pair<std::string, uint16_t>> merged_deploy_info;

    for (const auto& [deploy_key, subgraph_names] : address_groups) {
      if (subgraph_names.size() == 1) {
        const auto& name = subgraph_names[0];
        deploy_subgraphs[name] = subgraphs[name];
        original_to_merged[name] = name;
        merged_deploy_info[name] = deploy_key;
      } else {
        auto merged = std::make_shared<subgraph>();
        std::string merged_name = "merged_" + std::to_string(deploy_key.second);
        for (const auto& name : subgraph_names) {
          merged->merge(*subgraphs[name]);
          original_to_merged[name] = merged_name;
        }
        deploy_subgraphs[merged_name] = merged;
        merged_deploy_info[merged_name] = deploy_key;
      }
    }

    std::map<std::string, std::set<std::string>> merged_deps;
    for (const auto& node : nodes) {
      const auto& target_sg = node.subgraph_instance;
      if (remote_subgraph_deploy_info.find(target_sg) == remote_subgraph_deploy_info.end()) {
        continue;
      }
      const auto& target_merged = original_to_merged[target_sg];
      for (const auto& [input_name, source_name] : node.inputs) {
        size_t pos = source_name.find(':');
        auto source_node_name =
            (pos != std::string::npos) ? source_name.substr(0, pos) : source_name;
        for (const auto& source_node : nodes) {
          if (source_node.name == source_node_name) {
            const auto& source_sg = source_node.subgraph_instance;
            if (remote_subgraph_deploy_info.find(source_sg) == remote_subgraph_deploy_info.end()) {
              break;
            }
            const auto& source_merged = original_to_merged[source_sg];
            if (source_merged != target_merged) {
              merged_deps[target_merged].insert(source_merged);
            }
            break;
          }
        }
      }
    }

    std::vector<std::string> deploy_order;
    std::set<std::string> deployed;
    std::set<std::string> visiting;

    std::function<void(const std::string&)> visit = [&](const std::string& merged_name) {
      if (deployed.count(merged_name)) return;
      if (visiting.count(merged_name)) {
        throw std::runtime_error("Circular dependency in subgraph dependencies");
      }
      visiting.insert(merged_name);
      if (merged_deps.count(merged_name)) {
        for (const auto& dep : merged_deps[merged_name]) {
          visit(dep);
        }
      }
      visiting.erase(merged_name);
      deployed.insert(merged_name);
      deploy_order.push_back(merged_name);
    };

    for (const auto& [merged_name, _] : deploy_subgraphs) {
      visit(merged_name);
    }

    local_graph.get_resources()->add(callbacks);
    if (parameters_) {
      local_graph.get_resources()->add(std::make_shared<parameter_resource>(parameters_));
    }

    if (local_subgraph && local_subgraph->get_node_count() > 0) {
      local_graph.deploy(local_subgraph);
      local_graph.initialize();
    }

    if (!deploy_order.empty()) {
      has_remote_subgraphs = true;
      for (const auto& merged_name : deploy_order) {
        const auto& [deploy_address, deploy_port] = merged_deploy_info[merged_name];
        client.deploy(io_context, deploy_address, deploy_port, deploy_subgraphs[merged_name]);
      }
      io_thread.reset(new std::thread([this] { io_context.run(); }));
      client.initialize();
    }
  }

  void run() {
    if (local_subgraph && local_subgraph->get_node_count() > 0) {
      local_graph.run();
    }
    if (has_remote_subgraphs) {
      client.run();
    }
  }

  void stop_streaming() {
    if (has_remote_subgraphs) {
      client.stop();
    }
    if (local_subgraph && local_subgraph->get_node_count() > 0) {
      local_graph.stop();
    }
  }

  void finalize() {
    if (has_remote_subgraphs) {
      client.finalize();
      has_remote_subgraphs = false;
    }
    if (local_subgraph && local_subgraph->get_node_count() > 0) {
      local_graph.finalize();
    }
    io_context.stop();
    if (io_thread && io_thread->joinable()) {
      io_thread->join();
    }
    io_thread.reset();
    local_subgraph.reset();
  }

  std::optional<property_value> get_node_property(const std::string& node_name,
                                                  const std::string& key) const {
    const auto found = node_map.find(node_name);
    if (found == node_map.end() || !found->second) {
      return std::nullopt;
    }
    if (!is_local_node(found->second.get())) {
      return std::nullopt;
    }
    return found->second->get_property(key);
  }

  void set_parameters(std::shared_ptr<stargazer::parameters_t> p) { parameters_ = std::move(p); }
};

multiview_point_reconstruction_pipeline::multiview_point_reconstruction_pipeline(
    std::shared_ptr<stargazer::parameters_t> parameters)
    : pimpl(std::make_unique<impl>()) {
  pimpl->set_parameters(std::move(parameters));
}
multiview_point_reconstruction_pipeline::~multiview_point_reconstruction_pipeline() = default;

void multiview_point_reconstruction_pipeline::run(const std::vector<node_def>& nodes) {
  pimpl->deploy(nodes);
}

void multiview_point_reconstruction_pipeline::start() { pimpl->run(); }

void multiview_point_reconstruction_pipeline::pause() { pimpl->stop_streaming(); }

void multiview_point_reconstruction_pipeline::stop() { pimpl->finalize(); }

std::optional<property_value> multiview_point_reconstruction_pipeline::get_node_property(
    const std::string& node_name, const std::string& key) const {
  return pimpl->get_node_property(node_name, key);
}

