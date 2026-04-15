#include "dust3r_alignment.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <opencv2/calib3d.hpp>
#include <optional>
#include <queue>
#include <stdexcept>

namespace stargazer::dust3r {

static constexpr int H = ONNX_H;
static constexpr int W = ONNX_W;
static constexpr int HW = H * W;
static constexpr float min_confidence = 3.0f;
static constexpr float base_pair_scale = 0.5f;

struct union_find {
  std::vector<int> parent, rank_;
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

static bool procrustes_similarity(const std::vector<Eigen::Vector3f>& src,
                                  const std::vector<Eigen::Vector3f>& dst,
                                  const std::vector<float>& weights, float& scale,
                                  Eigen::Matrix3f& R, Eigen::Vector3f& t) {
  assert(src.size() == dst.size());
  assert(src.size() == weights.size());

  float wsum = 0.0f;
  Eigen::Vector3f mu_src = Eigen::Vector3f::Zero();
  Eigen::Vector3f mu_dst = Eigen::Vector3f::Zero();

  for (size_t i = 0; i < src.size(); ++i) {
    wsum += weights[i];
    mu_src += weights[i] * src[i];
    mu_dst += weights[i] * dst[i];
  }
  if (wsum < 1e-8f) return false;
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
  if (src_variance < 1e-8f) return false;

  Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3f U = svd.matrixU();
  Eigen::Matrix3f V = svd.matrixV();

  float diag_last = (V * U.transpose()).determinant() > 0.0f ? 1.0f : -1.0f;
  Eigen::DiagonalMatrix<float, 3> D(1.0f, 1.0f, diag_last);

  R = V * D * U.transpose();
  scale = (svd.singularValues().array() * D.diagonal().array()).sum() / src_variance;
  t = mu_dst - scale * R * mu_src;
  return true;
}

static constexpr int SUBSAMPLE = 1;  // use every pixel → 288*512 = 147456 points per pair

struct sampled_view {
  std::vector<Eigen::Vector3f> points;
  std::vector<float> weights;
  std::vector<cv::Point2f> pixels;
};

static float sanitize_confidence(float confidence) {
  return (std::isfinite(confidence) && confidence > 0.0f) ? confidence : 0.0f;
}

static sampled_view sample_view(const view_pointcloud& view) {
  sampled_view sampled;
  sampled.points.reserve(HW / (SUBSAMPLE * SUBSAMPLE));
  sampled.weights.reserve(HW / (SUBSAMPLE * SUBSAMPLE));
  sampled.pixels.reserve(HW / (SUBSAMPLE * SUBSAMPLE));

  for (int row = 0; row < H; row += SUBSAMPLE) {
    for (int col = 0; col < W; col += SUBSAMPLE) {
      const int px = row * W + col;
      const float* src = view.pts3d.data() + px * 3;
      sampled.points.emplace_back(src[0], src[1], src[2]);
      sampled.weights.push_back(sanitize_confidence(view.conf[px]));
      sampled.pixels.emplace_back(static_cast<float>(col), static_cast<float>(row));
    }
  }
  return sampled;
}

static std::vector<Eigen::Vector3f> transform_points(const std::vector<Eigen::Vector3f>& src,
                                                     float scale, const Eigen::Matrix3f& R,
                                                     const Eigen::Vector3f& t) {
  std::vector<Eigen::Vector3f> dst;
  dst.reserve(src.size());
  for (const auto& point : src) {
    dst.push_back(scale * (R * point) + t);
  }
  return dst;
}

static float mean_confidence(const std::vector<float>& confidences) {
  if (confidences.empty()) return 0.0f;
  float sum = 0.0f;
  for (float confidence : confidences) {
    sum += sanitize_confidence(confidence);
  }
  return sum / static_cast<float>(confidences.size());
}

static float estimate_focal_from_calibration(const camera_t& camera) {
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
  return static_cast<float>((new_K.at<double>(0, 0) * scale_x + new_K.at<double>(1, 1) * scale_y) *
                            0.5);
}

static std::pair<float, float> estimate_pp_from_calibration(const camera_t& camera) {
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
  return {static_cast<float>(new_K.at<double>(0, 2) * scale_x),
          static_cast<float>(new_K.at<double>(1, 2) * scale_y)};
}

static float estimate_focal_from_view(const sampled_view& sampled) {
  std::vector<std::pair<float, float>> ratios;
  ratios.reserve(sampled.points.size() * 2);

  const float cx = static_cast<float>(W) * 0.5f;
  const float cy = static_cast<float>(H) * 0.5f;
  for (size_t index = 0; index < sampled.points.size(); ++index) {
    const auto& point = sampled.points[index];
    const float weight = sampled.weights[index];
    if (!(std::isfinite(weight) && weight > min_confidence) ||
        !(std::isfinite(point.z()) && point.z() > 1e-6f)) {
      continue;
    }

    const float dx = sampled.pixels[index].x - cx;
    const float dy = sampled.pixels[index].y - cy;
    if (std::abs(point.x()) > 1e-6f && std::abs(dx) > 1e-6f) {
      const float focal_x = std::abs(dx * point.z() / point.x());
      if (std::isfinite(focal_x) && focal_x > 1e-3f) ratios.emplace_back(focal_x, weight);
    }
    if (std::abs(point.y()) > 1e-6f && std::abs(dy) > 1e-6f) {
      const float focal_y = std::abs(dy * point.z() / point.y());
      if (std::isfinite(focal_y) && focal_y > 1e-3f) ratios.emplace_back(focal_y, weight);
    }
  }

  if (ratios.empty()) return static_cast<float>(std::max(W, H));

  std::sort(ratios.begin(), ratios.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

  float total_weight = 0.0f;
  for (const auto& [_, weight] : ratios) total_weight += weight;

  float prefix_weight = 0.0f;
  for (const auto& [ratio, weight] : ratios) {
    prefix_weight += weight;
    if (prefix_weight >= total_weight * 0.5f) return ratio;
  }
  return ratios.back().first;
}

static std::optional<Eigen::Matrix4f> recover_pose_with_pnp(
    const std::vector<Eigen::Vector3f>& world_points, const sampled_view& sampled, float focal,
    float cx = static_cast<float>(W) * 0.5f, float cy = static_cast<float>(H) * 0.5f) {
  std::vector<cv::Point3f> object_points;
  std::vector<cv::Point2f> image_points;
  object_points.reserve(world_points.size());
  image_points.reserve(world_points.size());

  for (size_t index = 0; index < world_points.size(); ++index) {
    if (index >= sampled.weights.size()) break;
    if (!(std::isfinite(sampled.weights[index]) && sampled.weights[index] > min_confidence))
      continue;

    const auto& world_point = world_points[index];
    if (!std::isfinite(world_point.x()) || !std::isfinite(world_point.y()) ||
        !std::isfinite(world_point.z())) {
      continue;
    }
    object_points.emplace_back(world_point.x(), world_point.y(), world_point.z());
    image_points.push_back(sampled.pixels[index]);
  }

  if (object_points.size() < 4) return std::nullopt;

  cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal, 0.0, static_cast<double>(cx), 0.0,
                           focal, static_cast<double>(cy), 0.0, 0.0, 1.0);
  cv::Mat rvec;
  cv::Mat tvec;
  cv::Mat inliers;
  bool success = false;
  try {
    success = cv::solvePnPRansac(object_points, image_points, camera_matrix, cv::noArray(), rvec,
                                 tvec, false, 10, 5.0, 0.99, inliers, cv::SOLVEPNP_SQPNP);
  } catch (const cv::Exception&) {
    return std::nullopt;
  }
  if (!success) return std::nullopt;

  cv::Mat rotation_world_to_camera;
  cv::Rodrigues(rvec, rotation_world_to_camera);

  Eigen::Matrix3f R_w2c;
  Eigen::Vector3f t_w2c;
  for (int row = 0; row < 3; ++row) {
    t_w2c(row) = static_cast<float>(tvec.at<double>(row, 0));
    for (int col = 0; col < 3; ++col) {
      R_w2c(row, col) = static_cast<float>(rotation_world_to_camera.at<double>(row, col));
    }
  }

  const Eigen::Matrix3f R_c2w = R_w2c.transpose();
  const Eigen::Vector3f t_c2w = -(R_c2w * t_w2c);

  Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
  pose.block<3, 3>(0, 0) = R_c2w;
  pose.block<3, 1>(0, 3) = t_c2w;
  return pose;
}

static std::optional<Eigen::Matrix4f> recover_pose_with_similarity(
    const sampled_view& sampled_local, const std::vector<Eigen::Vector3f>& world_points) {
  float scale = 1.0f;
  Eigen::Matrix3f R;
  Eigen::Vector3f t;
  if (!procrustes_similarity(sampled_local.points, world_points, sampled_local.weights, scale, R,
                             t)) {
    return std::nullopt;
  }

  Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
  pose.block<3, 3>(0, 0) = R;
  pose.block<3, 1>(0, 3) = t;
  return pose;
}

static float compute_scale_normalization(
    const std::vector<pair_result>& pair_results,
    const std::vector<std::optional<std::vector<Eigen::Vector3f>>>& world_points) {
  std::vector<float> pair_scales;
  pair_scales.reserve(pair_results.size());

  for (const auto& pair_result : pair_results) {
    if (!world_points[pair_result.idx1].has_value()) continue;
    const sampled_view sampled_view1 = sample_view(pair_result.view1);
    float scale = 1.0f;
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    if (!procrustes_similarity(sampled_view1.points, *world_points[pair_result.idx1],
                               sampled_view1.weights, scale, R, t)) {
      continue;
    }
    if (std::isfinite(scale) && scale > 1e-6f) pair_scales.push_back(scale);
  }

  if (pair_scales.empty()) return 1.0f;

  double log_sum = 0.0;
  for (float scale : pair_scales) log_sum += std::log(static_cast<double>(scale));
  return static_cast<float>(
      std::exp(std::log(static_cast<double>(base_pair_scale)) - log_sum / pair_scales.size()));
}

static std::unordered_map<std::string, aligned_pose> align_global_impl(
    const std::vector<std::string>& camera_names, const std::vector<pair_result>& pair_results,
    const std::unordered_map<std::string, camera_t>* cameras) {
  const int N = static_cast<int>(camera_names.size());
  if (N == 0) return {};

  std::map<std::pair<int, int>, const pair_result*> pair_lookup;
  std::vector<const pair_result*> best_outgoing_pair(N, nullptr);
  std::vector<float> best_outgoing_score(N, -1.0f);
  for (const auto& pr : pair_results) {
    pair_lookup[{pr.idx1, pr.idx2}] = &pr;
    const float score = mean_confidence(pr.view1.conf);
    if (score > best_outgoing_score[pr.idx1]) {
      best_outgoing_score[pr.idx1] = score;
      best_outgoing_pair[pr.idx1] = &pr;
    }
  }

  struct directed_edge {
    int src, dst;
    float weight;
  };

  struct undirected_edge {
    int u, v;
    float weight;
    int preferred_src;
    int preferred_dst;
  };

  std::vector<undirected_edge> edges;
  edges.reserve(N * (N - 1) / 2);

  for (int u = 0; u < N; ++u) {
    for (int v = u + 1; v < N; ++v) {
      auto it_uv = pair_lookup.find({u, v});
      auto it_vu = pair_lookup.find({v, u});
      if (it_uv == pair_lookup.end() || it_vu == pair_lookup.end()) continue;

      const pair_result& pr_uv = *it_uv->second;
      const pair_result& pr_vu = *it_vu->second;

      const float weight_uv = mean_confidence(pr_uv.view1.conf) * mean_confidence(pr_uv.view2.conf);
      const float weight_vu = mean_confidence(pr_vu.view1.conf) * mean_confidence(pr_vu.view2.conf);

      if (weight_uv >= weight_vu) {
        edges.push_back({u, v, weight_uv, u, v});
      } else {
        edges.push_back({u, v, weight_vu, v, u});
      }
    }
  }

  if (edges.empty()) {
    throw std::runtime_error("dust3r_alignment: no valid edges found");
  }

  std::sort(edges.begin(), edges.end(),
            [](const undirected_edge& a, const undirected_edge& b) { return a.weight > b.weight; });

  union_find uf(N);
  std::vector<directed_edge> mst;
  mst.reserve(N - 1);
  for (const auto& e : edges) {
    if (uf.unite(e.u, e.v)) {
      mst.push_back({e.preferred_src, e.preferred_dst, e.weight});
      if (static_cast<int>(mst.size()) == N - 1) break;
    }
  }

  if (mst.empty()) {
    throw std::runtime_error("dust3r_alignment: failed to build MST");
  }

  std::sort(mst.begin(), mst.end(),
            [](const directed_edge& a, const directed_edge& b) { return a.weight > b.weight; });

  std::vector<std::optional<std::vector<Eigen::Vector3f>>> world_points(N);
  std::vector<std::optional<Eigen::Matrix4f>> image_poses(N);

  const directed_edge root_edge = mst.front();
  const pair_result& init_pair = *pair_lookup.at({root_edge.src, root_edge.dst});
  const sampled_view init_view1 = sample_view(init_pair.view1);
  const sampled_view init_view2 = sample_view(init_pair.view2);
  world_points[root_edge.src] = init_view1.points;
  world_points[root_edge.dst] = init_view2.points;
  image_poses[root_edge.src] = Eigen::Matrix4f::Identity();

  std::vector<bool> done(N, false);
  done[root_edge.src] = true;
  done[root_edge.dst] = true;

  std::vector<directed_edge> todo(mst.begin() + 1, mst.end());
  while (!todo.empty()) {
    bool progressed = false;

    for (auto it = todo.begin(); it != todo.end();) {
      const pair_result& pr = *pair_lookup.at({it->src, it->dst});
      const sampled_view sampled_src = sample_view(pr.view1);
      const sampled_view sampled_dst = sample_view(pr.view2);

      if (done[it->src] && !done[it->dst]) {
        float scale = 1.0f;
        Eigen::Matrix3f R;
        Eigen::Vector3f t;
        if (!procrustes_similarity(sampled_src.points, *world_points[it->src], sampled_src.weights,
                                   scale, R, t)) {
          throw std::runtime_error("dust3r_alignment: failed to align MST edge source");
        }
        world_points[it->dst] = transform_points(sampled_dst.points, scale, R, t);
        if (!image_poses[it->src].has_value()) {
          Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
          pose.block<3, 3>(0, 0) = R;
          pose.block<3, 1>(0, 3) = t;
          image_poses[it->src] = pose;
        }
        done[it->dst] = true;
        it = todo.erase(it);
        progressed = true;
        continue;
      }

      if (done[it->dst] && !done[it->src]) {
        float scale = 1.0f;
        Eigen::Matrix3f R;
        Eigen::Vector3f t;
        if (!procrustes_similarity(sampled_dst.points, *world_points[it->dst], sampled_dst.weights,
                                   scale, R, t)) {
          throw std::runtime_error("dust3r_alignment: failed to align MST edge destination");
        }
        world_points[it->src] = transform_points(sampled_src.points, scale, R, t);
        if (!image_poses[it->src].has_value()) {
          Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
          pose.block<3, 3>(0, 0) = R;
          pose.block<3, 1>(0, 3) = t;
          image_poses[it->src] = pose;
        }
        done[it->src] = true;
        it = todo.erase(it);
        progressed = true;
        continue;
      }

      ++it;
    }

    if (!progressed) {
      throw std::runtime_error("dust3r_alignment: stalled while propagating MST point clouds");
    }
  }

  for (int i = 0; i < N; ++i) {
    if (!world_points[i].has_value()) {
      throw std::runtime_error("dust3r_alignment: missing propagated world points");
    }
    if (image_poses[i].has_value()) continue;

    const pair_result* representative_pair = best_outgoing_pair[i];
    if (representative_pair == nullptr) {
      image_poses[i] = Eigen::Matrix4f::Identity();
      continue;
    }

    const sampled_view sampled_local = sample_view(representative_pair->view1);
    float focal = 0.0f;
    float cx = static_cast<float>(W) * 0.5f;
    float cy = static_cast<float>(H) * 0.5f;
    if (cameras != nullptr) {
      auto camera_it = cameras->find(camera_names[i]);
      if (camera_it != cameras->end()) {
        focal = estimate_focal_from_calibration(camera_it->second);
        const auto [pp_cx, pp_cy] = estimate_pp_from_calibration(camera_it->second);
        cx = pp_cx;
        cy = pp_cy;
      }
    }
    if (!(std::isfinite(focal) && focal > 1e-3f)) {
      focal = estimate_focal_from_view(sampled_local);
    }

    image_poses[i] = recover_pose_with_pnp(*world_points[i], sampled_local, focal, cx, cy);
    if (!image_poses[i].has_value() && cameras == nullptr) {
      image_poses[i] = recover_pose_with_similarity(sampled_local, *world_points[i]);
    }
    if (!image_poses[i].has_value()) {
      image_poses[i] = Eigen::Matrix4f::Identity();
    }
  }

  const float s_factor = compute_scale_normalization(pair_results, world_points);
  for (int i = 0; i < N; ++i) {
    for (auto& point : *world_points[i]) {
      point *= s_factor;
    }
    (*image_poses[i]).block<3, 1>(0, 3) *= s_factor;
  }

  std::unordered_map<std::string, aligned_pose> result;
  for (int i = 0; i < N; ++i) {
    const pair_result* representative_pair = best_outgoing_pair[i];
    float scale = s_factor;
    float estimated_focal = static_cast<float>(std::max(W, H));
    float estimated_cx = static_cast<float>(W) * 0.5f;
    float estimated_cy = static_cast<float>(H) * 0.5f;

    if (cameras != nullptr) {
      auto camera_it = cameras->find(camera_names[i]);
      if (camera_it != cameras->end()) {
        const float cal_focal = estimate_focal_from_calibration(camera_it->second);
        if (std::isfinite(cal_focal) && cal_focal > 1e-3f) {
          estimated_focal = cal_focal;
        }
        const auto [pp_cx, pp_cy] = estimate_pp_from_calibration(camera_it->second);
        estimated_cx = pp_cx;
        estimated_cy = pp_cy;
      }
    }

    if (representative_pair != nullptr) {
      const sampled_view sampled_local = sample_view(representative_pair->view1);
      float estimated_scale = 1.0f;
      Eigen::Matrix3f R;
      Eigen::Vector3f t;
      if (procrustes_similarity(sampled_local.points, *world_points[i], sampled_local.weights,
                                estimated_scale, R, t) &&
          std::isfinite(estimated_scale) && estimated_scale > 1e-6f) {
        scale = estimated_scale;
      }
      if (cameras == nullptr || !(std::isfinite(estimated_focal) && estimated_focal > 1e-3f)) {
        const float view_focal = estimate_focal_from_view(sampled_local);
        if (std::isfinite(view_focal) && view_focal > 1e-3f) {
          estimated_focal = view_focal;
        }
      }
    }

    aligned_pose pose;
    const Eigen::Matrix3f rotation = (*image_poses[i]).block<3, 3>(0, 0);
    const Eigen::Vector3f translation = (*image_poses[i]).block<3, 1>(0, 3);
    pose.rotation =
        glm::mat3(rotation(0, 0), rotation(1, 0), rotation(2, 0), rotation(0, 1), rotation(1, 1),
                  rotation(2, 1), rotation(0, 2), rotation(1, 2), rotation(2, 2));
    pose.translation = glm::vec3(translation(0), translation(1), translation(2));
    pose.scale = scale;
    pose.focal = estimated_focal;
    pose.cx = estimated_cx;
    pose.cy = estimated_cy;
    result[camera_names[i]] = pose;
  }
  return result;
}

std::unordered_map<std::string, aligned_pose> align_global(
    const std::vector<std::string>& camera_names, const std::vector<pair_result>& pair_results) {
  return align_global_impl(camera_names, pair_results, nullptr);
}

std::unordered_map<std::string, aligned_pose> align_global(
    const std::vector<std::string>& camera_names, const std::vector<pair_result>& pair_results,
    const std::unordered_map<std::string, camera_t>& cameras) {
  return align_global_impl(camera_names, pair_results, &cameras);
}

}  // namespace stargazer::dust3r
