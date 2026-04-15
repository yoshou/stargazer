#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "coalsack/core/graph_proc.h"
#include "config.hpp"

namespace stargazer {

// Forward declarations for node types
void build_graph_from_json(
    const std::vector<node_def>& nodes,
    std::map<std::string, std::shared_ptr<coalsack::subgraph>>& subgraphs,
    std::unordered_map<std::string, std::shared_ptr<coalsack::graph_node>>& node_map);

}  // namespace stargazer
