#include "dust3r_optimizer.hpp"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <numeric>
#include <opencv2/calib3d.hpp>
#include <optional>
#include <stdexcept>
#include <vector>

namespace stargazer::dust3r {
namespace {

constexpr int SUBSAMPLE = 1;
const int sample_cols = ONNX_W / SUBSAMPLE;
const int sample_rows = ONNX_H / SUBSAMPLE;
const int sample_count = sample_cols * sample_rows;
constexpr double min_depth = 1e-6;
constexpr double base_pair_scale = 0.5;
constexpr double final_log_scale_reference_weight = 100.0;
constexpr int alternating_iterations = 8;
constexpr int pose_refine_iterations = 50;
constexpr int final_joint_refine_iterations = 100;

float transform_confidence(float confidence) {
  if (!std::isfinite(confidence) || confidence <= 0.0f) {
    return 0.0f;
  }
  return std::log(confidence);
}

struct camera_params {
  std::array<double, 7> values{};
};

struct image_pose_params {
  std::array<double, 6> values{};
};

struct sampled_view {
  std::vector<double> points;
  std::vector<double> weights;
  std::vector<double> raw_weights;
};

struct sampled_depths {
  std::vector<std::array<double, 1>> values;
};

bool procrustes_similarity(const std::vector<Eigen::Vector3f>& src,
                           const std::vector<Eigen::Vector3f>& dst,
                           const std::vector<float>& weights, float& scale, Eigen::Matrix3f& R,
                           Eigen::Vector3f& t);

std::array<double, 3> get_point(const sampled_view& sampled, int sample_index);

struct union_find {
  std::vector<int> parent;
  std::vector<int> rank_;

  explicit union_find(int n) : parent(n), rank_(n, 0) {
    std::iota(parent.begin(), parent.end(), 0);
  }

  int find(int x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }

  bool unite(int a, int b) {
    a = find(a);
    b = find(b);
    if (a == b) return false;
    if (rank_[a] < rank_[b]) std::swap(a, b);
    parent[b] = a;
    if (rank_[a] == rank_[b]) ++rank_[a];
    return true;
  }
};

float sanitize_confidence_raw(float confidence) {
  return (std::isfinite(confidence) && confidence > 0.0f) ? confidence : 0.0f;
}

sampled_view sample_view(const view_pointcloud& view) {
  sampled_view sampled;
  const int n = sample_count;
  sampled.points.resize(n * 3);
  sampled.weights.resize(n);
  sampled.raw_weights.resize(n);
  int sample_index = 0;
  for (int row = 0; row < ONNX_H; row += SUBSAMPLE) {
    for (int col = 0; col < ONNX_W; col += SUBSAMPLE) {
      const int px = row * ONNX_W + col;
      sampled.points[sample_index * 3 + 0] = static_cast<double>(view.pts3d[px * 3 + 0]);
      sampled.points[sample_index * 3 + 1] = static_cast<double>(view.pts3d[px * 3 + 1]);
      sampled.points[sample_index * 3 + 2] = static_cast<double>(view.pts3d[px * 3 + 2]);
      sampled.weights[sample_index] = static_cast<double>(transform_confidence(view.conf[px]));
      sampled.raw_weights[sample_index] =
          static_cast<double>(sanitize_confidence_raw(view.conf[px]));
      ++sample_index;
    }
  }
  return sampled;
}

float mean_confidence_raw(const std::vector<float>& confidences) {
  if (confidences.empty()) {
    return 0.0f;
  }
  double sum = 0.0;
  for (float confidence : confidences) {
    if (std::isfinite(confidence) && confidence > 0.0f) {
      sum += confidence;
    }
  }
  return static_cast<float>(sum / static_cast<double>(confidences.size()));
}

std::vector<Eigen::Vector3f> sampled_points_to_eigen(const sampled_view& sampled) {
  const int n = static_cast<int>(sampled.raw_weights.size());
  std::vector<Eigen::Vector3f> points;
  points.reserve(n);
  for (int sample_index = 0; sample_index < n; ++sample_index) {
    points.emplace_back(static_cast<float>(sampled.points[sample_index * 3 + 0]),
                        static_cast<float>(sampled.points[sample_index * 3 + 1]),
                        static_cast<float>(sampled.points[sample_index * 3 + 2]));
  }
  return points;
}

std::vector<float> sampled_raw_weights_to_float(const sampled_view& sampled) {
  const int n = static_cast<int>(sampled.raw_weights.size());
  std::vector<float> weights;
  weights.reserve(n);
  for (int sample_index = 0; sample_index < n; ++sample_index) {
    weights.push_back(static_cast<float>(sampled.raw_weights[sample_index]));
  }
  return weights;
}

std::vector<std::array<double, 3>> transform_points(const sampled_view& sampled, float scale,
                                                    const Eigen::Matrix3f& R,
                                                    const Eigen::Vector3f& t) {
  const int n = static_cast<int>(sampled.raw_weights.size());
  std::vector<std::array<double, 3>> transformed(n);
  for (int sample_index = 0; sample_index < n; ++sample_index) {
    const Eigen::Vector3f point(static_cast<float>(sampled.points[sample_index * 3 + 0]),
                                static_cast<float>(sampled.points[sample_index * 3 + 1]),
                                static_cast<float>(sampled.points[sample_index * 3 + 2]));
    const Eigen::Vector3f world = scale * (R * point) + t;
    transformed[sample_index] = {static_cast<double>(world.x()), static_cast<double>(world.y()),
                                 static_cast<double>(world.z())};
  }
  return transformed;
}

std::vector<Eigen::Vector3f> world_points_to_eigen(
    const std::vector<std::array<double, 3>>& points) {
  std::vector<Eigen::Vector3f> out;
  out.reserve(points.size());
  for (const auto& point : points) {
    out.emplace_back(static_cast<float>(point[0]), static_cast<float>(point[1]),
                     static_cast<float>(point[2]));
  }
  return out;
}

float compute_pair_scale_normalization(
    const std::vector<pair_result>& pair_results,
    const std::vector<std::optional<std::vector<std::array<double, 3>>>>& world_points) {
  std::vector<double> log_scales;
  log_scales.reserve(pair_results.size());

  for (const auto& pair_result : pair_results) {
    if (!world_points[pair_result.idx1].has_value()) {
      continue;
    }
    const sampled_view sampled = sample_view(pair_result.view1);
    const auto sampled_points = sampled_points_to_eigen(sampled);
    const auto sampled_raw_weights = sampled_raw_weights_to_float(sampled);
    const auto world = world_points_to_eigen(*world_points[pair_result.idx1]);

    float scale = 1.0f;
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    if (!procrustes_similarity(sampled_points, world, sampled_raw_weights, scale, R, t)) {
      continue;
    }
    if (std::isfinite(scale) && scale > 1e-8f) {
      log_scales.push_back(std::log(static_cast<double>(scale)));
    }
  }

  if (log_scales.empty()) {
    return 1.0f;
  }

  const double mean_log_scale = std::accumulate(log_scales.begin(), log_scales.end(), 0.0) /
                                static_cast<double>(log_scales.size());
  return static_cast<float>(std::exp(std::log(base_pair_scale) - mean_log_scale));
}

std::vector<std::optional<std::vector<std::array<double, 3>>>> build_mst_world_points(
    const std::vector<std::string>& camera_names, const std::vector<pair_result>& pair_results) {
  struct directed_edge {
    int src;
    int dst;
    float weight;
  };
  struct undirected_edge {
    int u;
    int v;
    float weight;
    int preferred_src;
    int preferred_dst;
  };

  const int N = static_cast<int>(camera_names.size());
  std::map<std::pair<int, int>, const pair_result*> pair_lookup;
  for (const auto& pair_result : pair_results) {
    pair_lookup[{pair_result.idx1, pair_result.idx2}] = &pair_result;
  }

  std::vector<undirected_edge> edges;
  edges.reserve(N * (N - 1) / 2);
  for (int u = 0; u < N; ++u) {
    for (int v = u + 1; v < N; ++v) {
      const auto it_uv = pair_lookup.find({u, v});
      const auto it_vu = pair_lookup.find({v, u});
      if (it_uv == pair_lookup.end() || it_vu == pair_lookup.end()) {
        continue;
      }

      const float weight_uv = mean_confidence_raw(it_uv->second->view1.conf) *
                              mean_confidence_raw(it_uv->second->view2.conf);
      const float weight_vu = mean_confidence_raw(it_vu->second->view1.conf) *
                              mean_confidence_raw(it_vu->second->view2.conf);
      if (weight_uv >= weight_vu) {
        edges.push_back({u, v, weight_uv, u, v});
      } else {
        edges.push_back({u, v, weight_vu, v, u});
      }
    }
  }

  std::sort(edges.begin(), edges.end(), [](const undirected_edge& lhs, const undirected_edge& rhs) {
    return lhs.weight > rhs.weight;
  });

  union_find uf(N);
  std::vector<directed_edge> mst;
  mst.reserve(std::max(0, N - 1));
  for (const auto& edge : edges) {
    if (uf.unite(edge.u, edge.v)) {
      mst.push_back({edge.preferred_src, edge.preferred_dst, edge.weight});
      if (static_cast<int>(mst.size()) == N - 1) {
        break;
      }
    }
  }

  if (mst.empty()) {
    throw std::runtime_error("dust3r_optimizer: failed to build MST world points");
  }

  std::sort(mst.begin(), mst.end(), [](const directed_edge& lhs, const directed_edge& rhs) {
    return lhs.weight > rhs.weight;
  });

  std::vector<std::optional<std::vector<std::array<double, 3>>>> world_points(N);
  const pair_result& init_pair = *pair_lookup.at({mst.front().src, mst.front().dst});
  const sampled_view init_view1 = sample_view(init_pair.view1);
  const sampled_view init_view2 = sample_view(init_pair.view2);
  world_points[mst.front().src].emplace(sample_count);
  world_points[mst.front().dst].emplace(sample_count);
  for (int sample_index = 0; sample_index < sample_count; ++sample_index) {
    (*world_points[mst.front().src])[sample_index] = get_point(init_view1, sample_index);
    (*world_points[mst.front().dst])[sample_index] = get_point(init_view2, sample_index);
  }

  std::vector<bool> done(N, false);
  done[mst.front().src] = true;
  done[mst.front().dst] = true;

  std::vector<directed_edge> todo(mst.begin() + 1, mst.end());
  while (!todo.empty()) {
    bool progressed = false;
    for (auto it = todo.begin(); it != todo.end();) {
      const pair_result& pair_result = *pair_lookup.at({it->src, it->dst});
      const sampled_view sampled_src = sample_view(pair_result.view1);
      const sampled_view sampled_dst = sample_view(pair_result.view2);

      if (done[it->src] && !done[it->dst]) {
        const auto src_points = sampled_points_to_eigen(sampled_src);
        const auto src_raw_weights = sampled_raw_weights_to_float(sampled_src);
        const auto dst_world = world_points_to_eigen(*world_points[it->src]);

        float scale = 1.0f;
        Eigen::Matrix3f R;
        Eigen::Vector3f t;
        if (!procrustes_similarity(src_points, dst_world, src_raw_weights, scale, R, t)) {
          throw std::runtime_error(
              "dust3r_optimizer: failed to propagate MST world points from source");
        }
        world_points[it->dst] = transform_points(sampled_dst, scale, R, t);
        done[it->dst] = true;
        it = todo.erase(it);
        progressed = true;
        continue;
      }

      if (done[it->dst] && !done[it->src]) {
        const auto dst_points = sampled_points_to_eigen(sampled_dst);
        const auto dst_raw_weights = sampled_raw_weights_to_float(sampled_dst);
        const auto src_world = world_points_to_eigen(*world_points[it->dst]);

        float scale = 1.0f;
        Eigen::Matrix3f R;
        Eigen::Vector3f t;
        if (!procrustes_similarity(dst_points, src_world, dst_raw_weights, scale, R, t)) {
          throw std::runtime_error(
              "dust3r_optimizer: failed to propagate MST world points from destination");
        }
        world_points[it->src] = transform_points(sampled_src, scale, R, t);
        done[it->src] = true;
        it = todo.erase(it);
        progressed = true;
        continue;
      }

      ++it;
    }

    if (!progressed) {
      throw std::runtime_error("dust3r_optimizer: stalled while propagating MST world points");
    }
  }

  const float scale_normalization = compute_pair_scale_normalization(pair_results, world_points);
  for (auto& maybe_points : world_points) {
    if (!maybe_points.has_value()) {
      continue;
    }
    for (auto& point : *maybe_points) {
      point[0] *= scale_normalization;
      point[1] *= scale_normalization;
      point[2] *= scale_normalization;
    }
  }

  return world_points;
}

bool procrustes_similarity(const std::vector<Eigen::Vector3f>& src,
                           const std::vector<Eigen::Vector3f>& dst,
                           const std::vector<float>& weights, float& scale, Eigen::Matrix3f& R,
                           Eigen::Vector3f& t) {
  if (src.size() != dst.size() || src.size() != weights.size() || src.empty()) {
    return false;
  }

  float wsum = 0.0f;
  Eigen::Vector3f mu_src = Eigen::Vector3f::Zero();
  Eigen::Vector3f mu_dst = Eigen::Vector3f::Zero();
  for (size_t i = 0; i < src.size(); ++i) {
    wsum += weights[i];
    mu_src += weights[i] * src[i];
    mu_dst += weights[i] * dst[i];
  }
  if (wsum < 1e-8f) {
    return false;
  }
  mu_src /= wsum;
  mu_dst /= wsum;

  Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
  float src_variance = 0.0f;
  for (size_t i = 0; i < src.size(); ++i) {
    const Eigen::Vector3f centered_src = src[i] - mu_src;
    const Eigen::Vector3f centered_dst = dst[i] - mu_dst;
    cov += weights[i] * centered_src * centered_dst.transpose();
    src_variance += weights[i] * centered_src.squaredNorm();
  }
  if (src_variance < 1e-8f) {
    return false;
  }

  Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3f U = svd.matrixU();
  Eigen::Matrix3f V = svd.matrixV();
  const float diag_last = (V * U.transpose()).determinant() > 0.0f ? 1.0f : -1.0f;
  Eigen::DiagonalMatrix<float, 3> D(1.0f, 1.0f, diag_last);

  R = V * D * U.transpose();
  scale = (svd.singularValues().array() * D.diagonal().array()).sum() / src_variance;
  t = mu_dst - scale * R * mu_src;
  return true;
}

double estimate_focal_from_calibration(const camera_t& camera) {
  cv::Mat K = (cv::Mat_<double>(3, 3) << camera.intrin.fx, 0.0, camera.intrin.cx, 0.0,
               camera.intrin.fy, camera.intrin.cy, 0.0, 0.0, 1.0);
  cv::Mat dist_coeffs(1, 5, CV_64F);
  for (int i = 0; i < 5; ++i)
    dist_coeffs.at<double>(0, i) = static_cast<double>(camera.intrin.coeffs[i]);

  const cv::Mat new_K = cv::getOptimalNewCameraMatrix(
      K, dist_coeffs, cv::Size(static_cast<int>(camera.width), static_cast<int>(camera.height)),
      0.0);
  const double scale_x = static_cast<double>(ONNX_W) / static_cast<double>(camera.width);
  const double scale_y = static_cast<double>(ONNX_H) / static_cast<double>(camera.height);
  return (new_K.at<double>(0, 0) * scale_x + new_K.at<double>(1, 1) * scale_y) * 0.5;
}

std::pair<double, double> estimate_pp_from_calibration(const camera_t& camera) {
  cv::Mat K = (cv::Mat_<double>(3, 3) << camera.intrin.fx, 0.0, camera.intrin.cx, 0.0,
               camera.intrin.fy, camera.intrin.cy, 0.0, 0.0, 1.0);
  cv::Mat dist_coeffs(1, 5, CV_64F);
  for (int i = 0; i < 5; ++i)
    dist_coeffs.at<double>(0, i) = static_cast<double>(camera.intrin.coeffs[i]);

  const cv::Mat new_K = cv::getOptimalNewCameraMatrix(
      K, dist_coeffs, cv::Size(static_cast<int>(camera.width), static_cast<int>(camera.height)),
      0.0);
  const double scale_x = static_cast<double>(ONNX_W) / static_cast<double>(camera.width);
  const double scale_y = static_cast<double>(ONNX_H) / static_cast<double>(camera.height);
  return {new_K.at<double>(0, 2) * scale_x, new_K.at<double>(1, 2) * scale_y};
}

camera_params to_camera_params(const aligned_pose& pose) {
  camera_params params;

  const double rotation_matrix[9] = {
      static_cast<double>(pose.rotation[0][0]), static_cast<double>(pose.rotation[0][1]),
      static_cast<double>(pose.rotation[0][2]), static_cast<double>(pose.rotation[1][0]),
      static_cast<double>(pose.rotation[1][1]), static_cast<double>(pose.rotation[1][2]),
      static_cast<double>(pose.rotation[2][0]), static_cast<double>(pose.rotation[2][1]),
      static_cast<double>(pose.rotation[2][2]),
  };
  ceres::RotationMatrixToAngleAxis(rotation_matrix, params.values.data());
  const double safe_scale = std::max(1e-6f, pose.scale);
  params.values[3] = static_cast<double>(pose.translation.x) / safe_scale;
  params.values[4] = static_cast<double>(pose.translation.y) / safe_scale;
  params.values[5] = static_cast<double>(pose.translation.z) / safe_scale;
  params.values[6] = std::log(safe_scale);
  return params;
}

image_pose_params to_image_pose_params(const aligned_pose& pose) {
  image_pose_params params;
  const double rotation_matrix[9] = {
      static_cast<double>(pose.rotation[0][0]), static_cast<double>(pose.rotation[0][1]),
      static_cast<double>(pose.rotation[0][2]), static_cast<double>(pose.rotation[1][0]),
      static_cast<double>(pose.rotation[1][1]), static_cast<double>(pose.rotation[1][2]),
      static_cast<double>(pose.rotation[2][0]), static_cast<double>(pose.rotation[2][1]),
      static_cast<double>(pose.rotation[2][2]),
  };
  ceres::RotationMatrixToAngleAxis(rotation_matrix, params.values.data());
  params.values[3] = pose.translation.x;
  params.values[4] = pose.translation.y;
  params.values[5] = pose.translation.z;
  return params;
}

aligned_pose from_camera_params(const camera_params& params) {
  double rot[9];
  ceres::AngleAxisToRotationMatrix(params.values.data(), rot);

  const double scale = std::exp(params.values[6]);
  aligned_pose pose;
  pose.rotation =
      glm::mat3(static_cast<float>(rot[0]), static_cast<float>(rot[1]), static_cast<float>(rot[2]),
                static_cast<float>(rot[3]), static_cast<float>(rot[4]), static_cast<float>(rot[5]),
                static_cast<float>(rot[6]), static_cast<float>(rot[7]), static_cast<float>(rot[8]));
  pose.translation = glm::vec3(static_cast<float>(scale * params.values[3]),
                               static_cast<float>(scale * params.values[4]),
                               static_cast<float>(scale * params.values[5]));
  pose.scale = static_cast<float>(scale);
  return pose;
}

aligned_pose from_image_pose_params(const image_pose_params& params) {
  double rot[9];
  ceres::AngleAxisToRotationMatrix(params.values.data(), rot);

  aligned_pose pose;
  pose.rotation =
      glm::mat3(static_cast<float>(rot[0]), static_cast<float>(rot[1]), static_cast<float>(rot[2]),
                static_cast<float>(rot[3]), static_cast<float>(rot[4]), static_cast<float>(rot[5]),
                static_cast<float>(rot[6]), static_cast<float>(rot[7]), static_cast<float>(rot[8]));
  pose.translation =
      glm::vec3(static_cast<float>(params.values[3]), static_cast<float>(params.values[4]),
                static_cast<float>(params.values[5]));
  pose.scale = 1.0f;
  return pose;
}

std::array<double, 3> get_point(const sampled_view& sampled, int sample_index) {
  return {sampled.points[sample_index * 3 + 0], sampled.points[sample_index * 3 + 1],
          sampled.points[sample_index * 3 + 2]};
}

std::array<double, 3> transform_point(const aligned_pose& pose,
                                      const std::array<double, 3>& point) {
  std::array<double, 3> result{};
  for (int row = 0; row < 3; ++row) {
    result[row] = static_cast<double>(pose.translation[row]);
    for (int col = 0; col < 3; ++col) {
      result[row] += static_cast<double>(pose.scale * pose.rotation[col][row]) * point[col];
    }
  }
  return result;
}

std::array<double, 3> transform_point(const camera_params& params,
                                      const std::array<double, 3>& point) {
  return transform_point(from_camera_params(params), point);
}

std::array<double, 3> unproject_sample(double pixel_x, double pixel_y, double focal, double depth,
                                       double cx, double cy) {
  return {
      (pixel_x - cx) / focal * depth,
      (pixel_y - cy) / focal * depth,
      depth,
  };
}

std::array<double, 3> rotate_point(const glm::mat3& rotation, const std::array<double, 3>& point) {
  std::array<double, 3> result{};
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      result[row] += static_cast<double>(rotation[col][row]) * point[col];
    }
  }
  return result;
}

std::array<double, 3> subtract_points(const std::array<double, 3>& lhs,
                                      const std::array<double, 3>& rhs) {
  return {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
}

double dot_point(const std::array<double, 3>& lhs, const std::array<double, 3>& rhs) {
  return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

std::array<double, 3> localize_point(const aligned_pose& pose,
                                     const std::array<double, 3>& world_point) {
  const std::array<double, 3> centered = {
      world_point[0] - static_cast<double>(pose.translation.x),
      world_point[1] - static_cast<double>(pose.translation.y),
      world_point[2] - static_cast<double>(pose.translation.z),
  };

  std::array<double, 3> local{};
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      local[row] += static_cast<double>(pose.rotation[row][col]) * centered[col];
    }
  }
  return local;
}

struct sample_reference {
  size_t edge_index;
  bool uses_view1;
};

std::vector<std::vector<sample_reference>> build_sample_references(
    const std::vector<std::string>& camera_names, const std::vector<pair_result>& pair_results) {
  std::vector<std::vector<sample_reference>> references(camera_names.size());
  for (size_t edge_index = 0; edge_index < pair_results.size(); ++edge_index) {
    const auto& pair_result = pair_results[edge_index];
    references[pair_result.idx1].push_back({edge_index, true});
    references[pair_result.idx2].push_back({edge_index, false});
  }
  return references;
}

void update_depthmaps_closed_form(
    const std::vector<std::vector<sample_reference>>& sample_references,
    const std::vector<sampled_view>& sampled_views_i,
    const std::vector<sampled_view>& sampled_views_j,
    const std::vector<image_pose_params>& image_params,
    const std::vector<camera_params>& pair_params, const std::vector<double>& focals,
    const std::vector<std::pair<double, double>>& pps, std::vector<sampled_depths>& image_depths) {
  for (size_t image_index = 0; image_index < sample_references.size(); ++image_index) {
    const aligned_pose image_pose = from_image_pose_params(image_params[image_index]);
    const std::array<double, 3> image_translation = {
        static_cast<double>(image_pose.translation.x),
        static_cast<double>(image_pose.translation.y),
        static_cast<double>(image_pose.translation.z),
    };

    const int n_depths = static_cast<int>(image_depths[image_index].values.size());
    for (int sample_index = 0; sample_index < n_depths; ++sample_index) {
      const int row = (sample_index / sample_cols) * SUBSAMPLE;
      const int col = (sample_index % sample_cols) * SUBSAMPLE;
      const std::array<double, 3> local_unit_ray =
          unproject_sample(static_cast<double>(col), static_cast<double>(row), focals[image_index],
                           1.0, pps[image_index].first, pps[image_index].second);
      const std::array<double, 3> world_ray = rotate_point(image_pose.rotation, local_unit_ray);

      double numerator = 0.0;
      double denominator = 0.0;
      for (const auto& reference : sample_references[image_index]) {
        const auto& sampled_view = reference.uses_view1 ? sampled_views_i[reference.edge_index]
                                                        : sampled_views_j[reference.edge_index];
        const double weight = sampled_view.weights[sample_index];
        if (weight <= 0.0) {
          continue;
        }

        const std::array<double, 3> pair_world = transform_point(
            pair_params[reference.edge_index], get_point(sampled_view, sample_index));
        const std::array<double, 3> centered = subtract_points(pair_world, image_translation);
        numerator += weight * dot_point(world_ray, centered);
        denominator += weight * dot_point(world_ray, world_ray);
      }

      double updated_depth = std::exp(image_depths[image_index].values[sample_index][0]);
      if (denominator > 0.0) {
        updated_depth = std::max(min_depth, numerator / denominator);
      }
      image_depths[image_index].values[sample_index][0] = std::log(updated_depth);
    }
  }
}

struct image_pair_point_cost {
  double pixel_x;
  double pixel_y;
  double focal;
  double cx;
  double cy;
  std::array<double, 3> pair_point;
  double sqrt_weight;

  template <typename T>
  bool operator()(const T* const image_pose, const T* const log_depth, const T* const pair_pose,
                  T* residuals) const {
    const T depth = ceres::exp(log_depth[0]);
    const T local_point[3] = {
        (T(pixel_x) - T(cx)) / T(focal) * depth,
        (T(pixel_y) - T(cy)) / T(focal) * depth,
        depth,
    };

    T image_world[3];
    ceres::AngleAxisRotatePoint(image_pose, local_point, image_world);
    image_world[0] += image_pose[3];
    image_world[1] += image_pose[4];
    image_world[2] += image_pose[5];

    const T pair_local[3] = {T(pair_point[0]), T(pair_point[1]), T(pair_point[2])};
    T pair_rotated[3];
    ceres::AngleAxisRotatePoint(pair_pose, pair_local, pair_rotated);
    const T pair_scale = ceres::exp(pair_pose[6]);

    const T w = T(sqrt_weight);
    for (int axis = 0; axis < 3; ++axis) {
      const T pair_world = pair_scale * (pair_rotated[axis] + pair_pose[3 + axis]);
      residuals[axis] = w * (image_world[axis] - pair_world);
    }
    return true;
  }
};

void normalize_pair_scales(std::vector<camera_params>& pair_params) {
  if (pair_params.empty()) {
    return;
  }
  double mean_log_scale = 0.0;
  for (const auto& pair_param : pair_params) {
    mean_log_scale += pair_param.values[6];
  }
  mean_log_scale /= static_cast<double>(pair_params.size());
  const double delta = std::log(base_pair_scale) - mean_log_scale;
  for (auto& pair_param : pair_params) {
    pair_param.values[6] += delta;
  }
}

void add_image_pair_residuals(ceres::Problem& problem, const std::vector<pair_result>& pair_results,
                              const std::vector<sampled_view>& sampled_views_i,
                              const std::vector<sampled_view>& sampled_views_j,
                              const std::vector<double>& focals,
                              const std::vector<std::pair<double, double>>& pps,
                              std::vector<image_pose_params>& image_params,
                              std::vector<sampled_depths>& image_depths,
                              std::vector<camera_params>& pair_params) {
  for (size_t edge_index = 0; edge_index < pair_results.size(); ++edge_index) {
    const auto& pair_result = pair_results[edge_index];
    const auto& sampled_i = sampled_views_i[edge_index];
    const auto& sampled_j = sampled_views_j[edge_index];

    const int n_samples = static_cast<int>(sampled_i.weights.size());
    for (int sample_index = 0; sample_index < n_samples; ++sample_index) {
      const int row = (sample_index / sample_cols) * SUBSAMPLE;
      const int col = (sample_index % sample_cols) * SUBSAMPLE;

      if (sampled_i.weights[sample_index] > 0.0) {
        auto* cost = new ceres::AutoDiffCostFunction<image_pair_point_cost, 3, 6, 1, 7>(
            new image_pair_point_cost{
                static_cast<double>(col), static_cast<double>(row), focals[pair_result.idx1],
                pps[pair_result.idx1].first, pps[pair_result.idx1].second,
                get_point(sampled_i, sample_index), std::sqrt(sampled_i.weights[sample_index])});
        problem.AddResidualBlock(cost, nullptr, image_params[pair_result.idx1].values.data(),
                                 image_depths[pair_result.idx1].values[sample_index].data(),
                                 pair_params[edge_index].values.data());
      }

      if (sampled_j.weights[sample_index] > 0.0) {
        auto* cost = new ceres::AutoDiffCostFunction<image_pair_point_cost, 3, 6, 1, 7>(
            new image_pair_point_cost{
                static_cast<double>(col), static_cast<double>(row), focals[pair_result.idx2],
                pps[pair_result.idx2].first, pps[pair_result.idx2].second,
                get_point(sampled_j, sample_index), std::sqrt(sampled_j.weights[sample_index])});
        problem.AddResidualBlock(cost, nullptr, image_params[pair_result.idx2].values.data(),
                                 image_depths[pair_result.idx2].values[sample_index].data(),
                                 pair_params[edge_index].values.data());
      }
    }
  }
}

camera_params initialize_pair_pose(const sampled_view& sampled_view,
                                   const std::vector<std::array<double, 3>>& world_points) {
  const int n = static_cast<int>(sampled_view.raw_weights.size());
  std::vector<Eigen::Vector3f> src;
  std::vector<Eigen::Vector3f> dst;
  std::vector<float> weights;
  src.reserve(n);
  dst.reserve(n);
  weights.reserve(n);

  for (int sample_index = 0; sample_index < n; ++sample_index) {
    const double weight = sampled_view.raw_weights[sample_index];
    if (weight <= 0.0) {
      continue;
    }
    const auto point = get_point(sampled_view, sample_index);
    src.emplace_back(static_cast<float>(point[0]), static_cast<float>(point[1]),
                     static_cast<float>(point[2]));
    dst.emplace_back(static_cast<float>(world_points[sample_index][0]),
                     static_cast<float>(world_points[sample_index][1]),
                     static_cast<float>(world_points[sample_index][2]));
    weights.push_back(static_cast<float>(weight));
  }

  float scale = 1.0f;
  Eigen::Matrix3f R;
  Eigen::Vector3f t;
  if (!procrustes_similarity(src, dst, weights, scale, R, t)) {
    throw std::runtime_error("dust3r_optimizer: failed to initialize pair pose");
  }

  aligned_pose pose;
  pose.rotation =
      glm::mat3(R(0, 0), R(1, 0), R(2, 0), R(0, 1), R(1, 1), R(2, 1), R(0, 2), R(1, 2), R(2, 2));
  pose.translation = glm::vec3(t(0), t(1), t(2));
  pose.scale = scale;
  return to_camera_params(pose);
}

struct pair_point_consistency_cost {
  std::array<double, 3> point_in_a;
  std::array<double, 3> point_in_b;
  double sqrt_weight;

  template <typename T>
  bool operator()(const T* const cam_a, const T* const cam_b, T* residuals) const {
    T rotated_a[3];
    T rotated_b[3];
    T local_a[3] = {T(point_in_a[0]), T(point_in_a[1]), T(point_in_a[2])};
    T local_b[3] = {T(point_in_b[0]), T(point_in_b[1]), T(point_in_b[2])};

    ceres::AngleAxisRotatePoint(cam_a, local_a, rotated_a);
    ceres::AngleAxisRotatePoint(cam_b, local_b, rotated_b);

    const T scale_a = ceres::exp(cam_a[6]);
    const T scale_b = ceres::exp(cam_b[6]);
    const T w = T(sqrt_weight);

    for (int axis = 0; axis < 3; ++axis) {
      const T world_a = scale_a * (rotated_a[axis] + cam_a[3 + axis]);
      const T world_b = scale_b * (rotated_b[axis] + cam_b[3 + axis]);
      residuals[axis] = w * (world_a - world_b);
    }
    return true;
  }
};

struct log_scale_reference_cost {
  double reference_log_scale;
  double sqrt_weight;

  template <typename T>
  bool operator()(const T* const pair_param, T* residuals) const {
    residuals[0] = T(sqrt_weight) * (pair_param[6] - T(reference_log_scale));
    return true;
  }
};

std::vector<double> capture_log_scales(const std::vector<camera_params>& pair_params) {
  std::vector<double> log_scales;
  log_scales.reserve(pair_params.size());
  for (const auto& pair_param : pair_params) {
    log_scales.push_back(pair_param.values[6]);
  }
  return log_scales;
}

void add_log_scale_reference_priors(ceres::Problem& problem,
                                    std::vector<camera_params>& pair_params,
                                    const std::vector<double>& reference_log_scales,
                                    double weight) {
  if (weight <= 0.0 || pair_params.size() != reference_log_scales.size()) {
    return;
  }
  for (size_t index = 0; index < pair_params.size(); ++index) {
    auto* cost = new ceres::AutoDiffCostFunction<log_scale_reference_cost, 1, 7>(
        new log_scale_reference_cost{reference_log_scales[index], std::sqrt(weight)});
    problem.AddResidualBlock(cost, nullptr, pair_params[index].values.data());
  }
}

void add_pair_residuals(ceres::Problem& problem, camera_params& cam_a, camera_params& cam_b,
                        const pair_result& pr_ab, const pair_result& pr_ba) {
  for (int row = 0; row < ONNX_H; row += SUBSAMPLE) {
    for (int col = 0; col < ONNX_W; col += SUBSAMPLE) {
      const int px = row * ONNX_W + col;

      const float w_v =
          transform_confidence(pr_ba.view1.conf[px]) * transform_confidence(pr_ab.view2.conf[px]);
      if (w_v > 0.0f) {
        const float* point_in_v = pr_ba.view1.pts3d.data() + px * 3;
        const float* point_in_u = pr_ab.view2.pts3d.data() + px * 3;
        auto* cost = new ceres::AutoDiffCostFunction<pair_point_consistency_cost, 3, 7, 7>(
            new pair_point_consistency_cost{{point_in_u[0], point_in_u[1], point_in_u[2]},
                                            {point_in_v[0], point_in_v[1], point_in_v[2]},
                                            std::sqrt(static_cast<double>(w_v))});
        problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), cam_a.values.data(),
                                 cam_b.values.data());
      }

      const float w_u =
          transform_confidence(pr_ab.view1.conf[px]) * transform_confidence(pr_ba.view2.conf[px]);
      if (w_u > 0.0f) {
        const float* point_in_u = pr_ab.view1.pts3d.data() + px * 3;
        const float* point_in_v = pr_ba.view2.pts3d.data() + px * 3;
        auto* cost = new ceres::AutoDiffCostFunction<pair_point_consistency_cost, 3, 7, 7>(
            new pair_point_consistency_cost{{point_in_u[0], point_in_u[1], point_in_u[2]},
                                            {point_in_v[0], point_in_v[1], point_in_v[2]},
                                            std::sqrt(static_cast<double>(w_u))});
        problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), cam_a.values.data(),
                                 cam_b.values.data());
      }
    }
  }
}

}  // namespace

std::unordered_map<std::string, aligned_pose> refine_global_alignment(
    const std::vector<std::string>& camera_names, const std::vector<pair_result>& pair_results,
    const std::unordered_map<std::string, aligned_pose>& initial_poses) {
  if (camera_names.empty()) {
    return {};
  }

  std::map<std::pair<int, int>, const pair_result*> pair_lookup;
  for (const auto& pair_result : pair_results) {
    pair_lookup[{pair_result.idx1, pair_result.idx2}] = &pair_result;
  }

  std::vector<camera_params> camera_params(camera_names.size());
  for (size_t index = 0; index < camera_names.size(); ++index) {
    const auto pose_it = initial_poses.find(camera_names[index]);
    if (pose_it == initial_poses.end()) {
      throw std::runtime_error("dust3r_optimizer: missing initial pose");
    }
    camera_params[index] = to_camera_params(pose_it->second);
  }

  ceres::Problem problem;

  for (auto& camera_param : camera_params) {
    problem.AddParameterBlock(camera_param.values.data(), 7);
  }

  for (int i = 0; i < static_cast<int>(camera_names.size()); ++i) {
    for (int j = i + 1; j < static_cast<int>(camera_names.size()); ++j) {
      const auto it_ij = pair_lookup.find({i, j});
      const auto it_ji = pair_lookup.find({j, i});
      if (it_ij == pair_lookup.end() || it_ji == pair_lookup.end()) {
        continue;
      }
      add_pair_residuals(problem, camera_params[i], camera_params[j], *it_ij->second,
                         *it_ji->second);
    }
  }

  if (camera_names.size() > 1) {
    problem.SetParameterBlockConstant(camera_params.front().values.data());
  }

  ceres::Solver::Options options;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = 8;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::unordered_map<std::string, aligned_pose> refined;
  for (size_t index = 0; index < camera_names.size(); ++index) {
    refined[camera_names[index]] = from_camera_params(camera_params[index]);
    // Preserve focal/cx/cy from initial poses (not optimised in this path)
    const auto it = initial_poses.find(camera_names[index]);
    if (it != initial_poses.end()) {
      refined[camera_names[index]].focal = it->second.focal;
      refined[camera_names[index]].cx = it->second.cx;
      refined[camera_names[index]].cy = it->second.cy;
    }
  }
  return refined;
}

std::unordered_map<std::string, aligned_pose> refine_global_alignment(
    const std::vector<std::string>& camera_names, const std::vector<pair_result>& pair_results,
    const std::unordered_map<std::string, aligned_pose>& initial_poses,
    const std::unordered_map<std::string, camera_t>& cameras) {
  if (camera_names.empty()) {
    return {};
  }

  std::map<std::pair<int, int>, const pair_result*> pair_lookup;
  for (const auto& pair_result : pair_results) {
    pair_lookup[{pair_result.idx1, pair_result.idx2}] = &pair_result;
  }

  std::vector<image_pose_params> image_params(camera_names.size());
  std::vector<sampled_depths> image_depths(camera_names.size());
  std::vector<std::vector<std::array<double, 3>>> world_points(camera_names.size());
  std::vector<double> focals(camera_names.size(), 1.0);
  std::vector<std::pair<double, double>> pps(
      camera_names.size(), {static_cast<double>(ONNX_W) * 0.5, static_cast<double>(ONNX_H) * 0.5});
  const auto sample_references = build_sample_references(camera_names, pair_results);

  const auto propagated_world_points = build_mst_world_points(camera_names, pair_results);

  for (size_t index = 0; index < camera_names.size(); ++index) {
    const auto pose_it = initial_poses.find(camera_names[index]);
    const auto camera_it = cameras.find(camera_names[index]);
    if (pose_it == initial_poses.end() || camera_it == cameras.end()) {
      throw std::runtime_error("dust3r_optimizer: missing initial pose or camera");
    }

    image_params[index] = to_image_pose_params(pose_it->second);
    focals[index] = estimate_focal_from_calibration(camera_it->second);
    pps[index] = estimate_pp_from_calibration(camera_it->second);
    if (!propagated_world_points[index].has_value()) {
      throw std::runtime_error("dust3r_optimizer: missing propagated world points");
    }

    image_depths[index].values.resize(sample_count);
    world_points[index].resize(sample_count);
    const aligned_pose initial_pose = pose_it->second;
    for (int sample_index = 0; sample_index < sample_count; ++sample_index) {
      world_points[index][sample_index] = (*propagated_world_points[index])[sample_index];
      const auto local_point = localize_point(initial_pose, world_points[index][sample_index]);
      image_depths[index].values[sample_index][0] = std::log(std::max(min_depth, local_point[2]));
    }
  }

  std::vector<camera_params> pair_params(pair_results.size());
  std::vector<sampled_view> sampled_views_i(pair_results.size());
  std::vector<sampled_view> sampled_views_j(pair_results.size());
  for (size_t edge_index = 0; edge_index < pair_results.size(); ++edge_index) {
    const auto& pair_result = pair_results[edge_index];
    sampled_views_i[edge_index] = sample_view(pair_result.view1);
    sampled_views_j[edge_index] = sample_view(pair_result.view2);
    pair_params[edge_index] =
        initialize_pair_pose(sampled_views_i[edge_index], world_points[pair_result.idx1]);
  }
  normalize_pair_scales(pair_params);

  for (int alternating_iteration = 0; alternating_iteration < alternating_iterations;
       ++alternating_iteration) {
    update_depthmaps_closed_form(sample_references, sampled_views_i, sampled_views_j, image_params,
                                 pair_params, focals, pps, image_depths);

    ceres::Problem problem;
    for (size_t index = 0; index < camera_names.size(); ++index) {
      problem.AddParameterBlock(image_params[index].values.data(), 6);
      const int n_depths = static_cast<int>(image_depths[index].values.size());
      for (int sample_index = 0; sample_index < n_depths; ++sample_index) {
        problem.AddParameterBlock(image_depths[index].values[sample_index].data(), 1);
        problem.SetParameterBlockConstant(image_depths[index].values[sample_index].data());
      }
    }
    for (auto& pair_param : pair_params) {
      problem.AddParameterBlock(pair_param.values.data(), 7);
    }

    add_image_pair_residuals(problem, pair_results, sampled_views_i, sampled_views_j, focals, pps,
                             image_params, image_depths, pair_params);

    problem.SetParameterBlockConstant(image_params.front().values.data());

    ceres::Solver::Options options;
    options.max_num_iterations = pose_refine_iterations;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 8;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    spdlog::info("dust3r alternating[{}]: initial_cost={:.6e} final_cost={:.6e}",
                 alternating_iteration, summary.initial_cost, summary.final_cost);
  }

  {
    const std::vector<double> final_log_scale_references = capture_log_scales(pair_params);

    ceres::Problem problem;
    for (size_t index = 0; index < camera_names.size(); ++index) {
      problem.AddParameterBlock(image_params[index].values.data(), 6);
      const int n_depths = static_cast<int>(image_depths[index].values.size());
      for (int sample_index = 0; sample_index < n_depths; ++sample_index) {
        problem.AddParameterBlock(image_depths[index].values[sample_index].data(), 1);
      }
    }
    for (auto& pair_param : pair_params) {
      problem.AddParameterBlock(pair_param.values.data(), 7);
    }

    add_image_pair_residuals(problem, pair_results, sampled_views_i, sampled_views_j, focals, pps,
                             image_params, image_depths, pair_params);
    add_log_scale_reference_priors(problem, pair_params, final_log_scale_references,
                                   final_log_scale_reference_weight);

    problem.SetParameterBlockConstant(image_params.front().values.data());

    ceres::Solver::Options options;
    options.max_num_iterations = final_joint_refine_iterations;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 8;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    spdlog::info("dust3r final_joint: initial_cost={:.6e} final_cost={:.6e}", summary.initial_cost,
                 summary.final_cost);
    normalize_pair_scales(pair_params);
  }

  std::unordered_map<std::string, aligned_pose> refined;
  for (size_t index = 0; index < camera_names.size(); ++index) {
    refined[camera_names[index]] = from_image_pose_params(image_params[index]);
    // Preserve focal/cx/cy from initial poses
    const auto it = initial_poses.find(camera_names[index]);
    if (it != initial_poses.end()) {
      refined[camera_names[index]].focal = it->second.focal;
      refined[camera_names[index]].cx = it->second.cx;
      refined[camera_names[index]].cy = it->second.cy;
    }
  }
  return refined;
}

}  // namespace stargazer::dust3r
