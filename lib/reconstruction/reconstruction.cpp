#include "reconstruction.hpp"

#include <vector>

#include "correspondance.hpp"
#include "parameters.hpp"
#include "point_data.hpp"
#include "triangulation.hpp"

namespace stargazer::reconstruction {
std::vector<glm::vec3> reconstruct(const std::vector<stargazer::camera_t>& camera_list,
                                   const std::vector<std::vector<glm::vec2>>& camera_pts,
                                   glm::mat4 axis) {
  if (camera_list.size() < 2) {
    return {};
  }
  std::vector<stargazer::reconstruction::node_t> nodes;
  stargazer::reconstruction::adj_list_t adj;

  const auto thresh = 1.0;
  stargazer::reconstruction::find_correspondance(camera_pts, camera_list, nodes, adj, thresh);

#if 0
  stargazer::reconstruction::compute_hard_correspondance(nodes, adj);

  std::vector<std::vector<std::size_t>> connected_components;
  stargazer::reconstruction::compute_observations(adj, connected_components);
#else
  stargazer::reconstruction::remove_ambiguous_observations(nodes, adj, camera_list, 0.01);

  std::vector<std::vector<std::size_t>> connected_components;
  stargazer::reconstruction::compute_observations(adj, connected_components);
#endif

  bool all_hard_correspondance = true;
  for (std::size_t i = 0; i < connected_components.size(); i++) {
    const auto& connected_graph = connected_components[i];
    const auto has_ambigious =
        stargazer::reconstruction::has_soft_correspondance(nodes, connected_graph);
    if (has_ambigious) {
      all_hard_correspondance = false;
      break;
    }
  }

  if (!all_hard_correspondance) {
    std::cout << "Can't find correspondance points on frame" << std::endl;
  }

  std::vector<glm::vec3> markers;
  for (auto& g : connected_components) {
    if (g.size() < 2) {
      continue;
    }

    std::vector<glm::vec2> pts;
    std::vector<stargazer::camera_t> cams;

    for (std::size_t i = 0; i < g.size(); i++) {
      pts.push_back(nodes[g[i]].pt);
      cams.push_back(camera_list[nodes[g[i]].camera_idx]);
    }
    const auto marker = stargazer::reconstruction::triangulate(pts, cams);
    markers.push_back(glm::vec3(axis * glm::vec4(marker, 1.0f)));
  }

  return markers;
}

std::vector<glm::vec3> reconstruct(
    const std::map<std::string, stargazer::camera_t>& cameras,
    const std::map<std::string, std::vector<stargazer::point_data>>& frame, glm::mat4 axis) {
  std::vector<std::vector<glm::vec2>> camera_pts;
  std::vector<stargazer::camera_t> camera_list;
  for (const auto& [camera_name, camera] : cameras) {
    std::vector<glm::vec2> pts;
    for (const auto& pt : frame.at(camera_name)) {
      pts.push_back(pt.point);
    }
    camera_pts.push_back(pts);
    camera_list.push_back(camera);
  }
  return reconstruct(camera_list, camera_pts, axis);
}
}  // namespace stargazer::reconstruction