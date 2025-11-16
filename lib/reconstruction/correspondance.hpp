#pragma once

#include <glm/glm.hpp>
#include <vector>

#include "parameters.hpp"

namespace stargazer::reconstruction {
struct node_t {
  glm::vec2 pt;
  std::size_t camera_idx;
  std::size_t point_idx;
};

using node_index_list_t = std::vector<size_t>;

using adj_list_t = std::vector<node_index_list_t>;

void compute_observations(const adj_list_t &adj, std::vector<node_index_list_t> &connected_graphs);

bool has_soft_correspondance(const std::vector<node_t> &nodes, const node_index_list_t &graph);

void compute_hard_correspondance(const std::vector<node_t> &nodes, adj_list_t &adj);

void remove_ambiguous_observations(const std::vector<node_t> &nodes, adj_list_t &adj,
                                   const std::vector<camera_t> &cameras,
                                   double world_thresh = 0.03);

void find_correspondance(const std::vector<std::vector<glm::vec2>> &pts,
                         const std::vector<camera_t> &cameras, std::vector<node_t> &nodes,
                         adj_list_t &adj, double screen_thresh);

void save_graphs(const std::vector<node_t> &nodes, const adj_list_t &adj,
                 const std::string &prefix);
}  // namespace stargazer::reconstruction
