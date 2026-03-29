#include "capture_pipeline.hpp"

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <regex>
#include <set>

#include "coalsack/core/graph_proc.h"
#include "coalsack/core/graph_proc_client.h"
#include "coalsack/core/graph_proc_server.h"
#include "coalsack/ext/graph_proc_depthai.h"
#include "coalsack/ext/graph_proc_jpeg.h"
#include "coalsack/ext/graph_proc_libcamera.h"
#include "coalsack/ext/graph_proc_rs_d435.h"
#include "dump_blob_node.hpp"
#include "graph_builder.hpp"
#include "load_blob_node.hpp"
#include "load_panoptic_node.hpp"

using namespace coalsack;
using namespace stargazer;

class capture_pipeline::impl {
  std::unordered_map<std::string, std::shared_ptr<graph_node>> node_map;
  graph_proc local_graph;
  std::shared_ptr<subgraph> local_subgraph;
  asio::io_context io_context;
  graph_proc_client client;
  std::unique_ptr<std::thread> io_thread;

  bool has_remote_subgraphs = false;

  bool is_local_node(const graph_node* node) const {
    return node && local_subgraph && node->get_parent() == local_subgraph.get();
  }

  void process_node(const graph_node* node, const std::string& input_name,
                    const graph_message_ptr& message) {
    if (!node) {
      return;
    }
    if (is_local_node(node)) {
      local_graph.process(node, input_name, message);
      return;
    }
    client.process(node, input_name, message);
  }

 public:
  impl()
      : node_map(),
        local_graph(),
        local_subgraph(),
        io_context(),
        client(),
        io_thread(),
        has_remote_subgraphs(false) {}

  void deploy(const std::vector<node_def>& nodes) {
    node_map.clear();
    local_graph = graph_proc();
    local_subgraph.reset();
    client = graph_proc_client();
    io_context.restart();
    has_remote_subgraphs = false;

    // Group nodes by subgraph instance
    std::map<std::string, std::vector<node_def>> nodes_by_subgraph;
    for (const auto& node : nodes) {
      nodes_by_subgraph[node.subgraph_instance].push_back(node);
    }

    // Create a global node map shared across all subgraphs
    std::unordered_map<std::string, std::shared_ptr<graph_node>> global_node_map;

    // Create empty subgraphs first
    std::map<std::string, std::shared_ptr<subgraph>> subgraphs;
    for (const auto& [subgraph_name, nodes] : nodes_by_subgraph) {
      subgraphs[subgraph_name] = std::make_shared<subgraph>();
    }

    // Build all subgraphs in one pass
    stargazer::build_graph_from_json(nodes, subgraphs, global_node_map);
    node_map = global_node_map;

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

    // Group remote subgraphs by deploy address and create merged subgraphs
    std::map<std::pair<std::string, uint16_t>, std::vector<std::string>> address_groups;

    for (const auto& [subgraph_name, deploy_key] : remote_subgraph_deploy_info) {
      address_groups[deploy_key].push_back(subgraph_name);
    }

    // Create merged subgraphs and mapping from original to merged group name
    std::map<std::string, std::shared_ptr<subgraph>> deploy_subgraphs;
    std::map<std::string, std::string> original_to_merged;
    std::map<std::string, std::pair<std::string, uint16_t>> merged_deploy_info;
    std::map<std::string, std::vector<std::string>> merged_subgraph_members;

    for (const auto& [deploy_key, subgraph_names] : address_groups) {
      if (subgraph_names.size() == 1) {
        // Single subgraph - no merge needed
        const auto& name = subgraph_names[0];
        deploy_subgraphs[name] = subgraphs[name];
        original_to_merged[name] = name;
        merged_deploy_info[name] = deploy_key;
        merged_subgraph_members[name] = {name};
      } else {
        // Multiple subgraphs - merge them
        auto merged = std::make_shared<subgraph>();
        std::string merged_name = "merged_" + std::to_string(deploy_key.second);

        for (const auto& name : subgraph_names) {
          merged->merge(*subgraphs[name]);
          original_to_merged[name] = merged_name;
        }

        deploy_subgraphs[merged_name] = merged;
        merged_deploy_info[merged_name] = deploy_key;
        merged_subgraph_members[merged_name] = subgraph_names;
      }
    }

    // Build dependency graph based on merged remote subgraphs
    std::map<std::string, std::set<std::string>> merged_dependencies;

    for (const auto& node : nodes) {
      const auto& target_subgraph = node.subgraph_instance;
      if (remote_subgraph_deploy_info.find(target_subgraph) == remote_subgraph_deploy_info.end()) {
        continue;
      }
      const auto& target_merged = original_to_merged[target_subgraph];

      for (const auto& [input_name, source_name] : node.inputs) {
        size_t pos = source_name.find(':');
        std::string source_node_name =
            (pos != std::string::npos) ? source_name.substr(0, pos) : source_name;

        for (const auto& source_node : nodes) {
          if (source_node.name == source_node_name) {
            const auto& source_subgraph = source_node.subgraph_instance;
            if (remote_subgraph_deploy_info.find(source_subgraph) ==
                remote_subgraph_deploy_info.end()) {
              break;
            }
            const auto& source_merged = original_to_merged[source_subgraph];

            if (source_merged != target_merged) {
              merged_dependencies[target_merged].insert(source_merged);
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
        throw std::runtime_error("Circular dependency detected in subgraph dependencies");
      }

      visiting.insert(merged_name);

      if (merged_dependencies.count(merged_name)) {
        for (const auto& dep : merged_dependencies[merged_name]) {
          visit(dep);
        }
      }

      visiting.erase(merged_name);
      deployed.insert(merged_name);
      deploy_order.push_back(merged_name);
    };

    for (const auto& [merged_name, graph] : deploy_subgraphs) {
      visit(merged_name);
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

  void stop() {
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
    if (found == node_map.end() || !found->second || !is_local_node(found->second.get())) {
      return std::nullopt;
    }
    return found->second->get_property(key);
  }
};

capture_pipeline::capture_pipeline() : pimpl(new impl()) {}
capture_pipeline::~capture_pipeline() = default;

void capture_pipeline::run(const std::vector<node_def>& nodes) { pimpl->deploy(nodes); }
void capture_pipeline::start() { pimpl->run(); }
void capture_pipeline::pause() { pimpl->stop(); }
void capture_pipeline::stop() { pimpl->finalize(); }

std::optional<property_value> capture_pipeline::get_node_property(const std::string& node_name,
                                                                  const std::string& key) const {
  return pimpl->get_node_property(node_name, key);
}
