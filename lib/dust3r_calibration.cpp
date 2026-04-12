#include "dust3r_calibration.hpp"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <numeric>
#include <opencv2/flann.hpp>

#include "dust3r.hpp"

namespace stargazer::dust3r {

void dust3r_to_original(float u_512, float v_512, int W_orig, int H_orig, float& u_orig,
                        float& v_orig) {
  u_orig = u_512 * static_cast<float>(W_orig) / static_cast<float>(ONNX_W);
  v_orig = v_512 * static_cast<float>(H_orig) / static_cast<float>(ONNX_H);
}

void dust3r_focal_to_original(float focal_512, int W_orig, int H_orig, float& fx, float& fy) {
  fx = focal_512 * static_cast<float>(W_orig) / static_cast<float>(ONNX_W);
  fy = focal_512 * static_cast<float>(H_orig) / static_cast<float>(ONNX_H);
}

namespace {

// Transform a local camera-frame point to world frame using an aligned_pose.
// world = t_c2w + scale * R_c2w @ local_pt
static std::array<float, 3> pose_transform(const aligned_pose& pose, const float* local_pt3) {
  std::array<float, 3> world{};
  for (int row = 0; row < 3; ++row) {
    world[row] = pose.translation[row];
    for (int col = 0; col < 3; ++col) {
      world[row] += pose.scale * pose.rotation[col][row] * local_pt3[col];
    }
  }
  return world;
}

// Compute the depth (z-component in camera frame) of a world point in a given camera.
// depth = R_c2w[:,2] · (world − t_c2w)  = sum_r pose.rotation[2][r] * (world[r] -
// pose.translation[r]) Equivalent to (R_wc @ world + t_wc)[2] used by Ceres.
static float camera_depth(const aligned_pose& pose, const std::array<float, 3>& world) {
  float depth = 0.0f;
  for (int r = 0; r < 3; ++r) {
    depth += pose.rotation[2][r] * (world[r] - pose.translation[r]);
  }
  return depth;
}

// Find the best (highest mean view1 confidence) pair where camera_idx is idx1.
static const pair_result* best_pair_for_camera(const std::vector<pair_result>& pairs,
                                               int camera_idx) {
  const pair_result* best = nullptr;
  float best_score = -1.0f;
  for (const auto& pr : pairs) {
    if (pr.idx1 != camera_idx) continue;
    float score = 0.0f;
    for (float c : pr.view1.conf) score += c;
    if (!pr.view1.conf.empty()) score /= static_cast<float>(pr.view1.conf.size());
    if (score > best_score) {
      best_score = score;
      best = &pr;
    }
  }
  return best;
}

// Number of camera parameters: [aa(0..2), t(3..5), f(6), cx(7), cy(8), k1,k2,k3,p1,p2(9..13),
// k4,k5,k6(14..16)]
static constexpr int kNumCamParams = 17;

// Pack aligned_pose + initial intrinsics (original image space) into 17-param array.
// cam = [aa(3), t(3), f, cx, cy, k1, k2, k3, p1, p2, k4, k5, k6]
// Camera convention: p_cam = R_wc @ p_world + t_wc
static void pack_camera_params(const aligned_pose& pose, float f, float cx, float cy, double* cam) {
  // R_wc stored in column-major order for Ceres RotationMatrixToAngleAxis.
  // Ceres column-major: array[k*3+j] = R[row=j, col=k]
  // R_wc[j,k] = R_c2w^T[j,k] = R_c2w[k,j] = glm rotation[col=j][row=k] = pose.rotation[j][k]
  // R_wc column-major: rot_wc_col_major[k*3+j] = R_wc[j,k] = pose.rotation[j][k]
  double rot_wc_col_major[9];
  for (int j = 0; j < 3; ++j)
    for (int k = 0; k < 3; ++k)
      rot_wc_col_major[k * 3 + j] = static_cast<double>(pose.rotation[j][k]);
  ceres::RotationMatrixToAngleAxis(rot_wc_col_major, cam);  // [0..2] = aa(R_wc)

  // t_wc = -(R_wc @ t_c2w)
  // R_wc[i,j] = R_c2w^T[i,j] = R_c2w[j,i] = pose.rotation[col=i][row=j] = pose.rotation[i][j]
  // (R_wc @ t_c2w)[i] = sum_j R_wc[i,j] * t_c2w[j] = sum_j pose.rotation[i][j] * t_c2w[j]
  for (int i = 0; i < 3; ++i) {
    double val = 0.0;
    for (int j = 0; j < 3; ++j) {
      val += static_cast<double>(pose.rotation[i][j]) * static_cast<double>(pose.translation[j]);
    }
    cam[3 + i] = -val;  // [3..5] = t_wc
  }

  cam[6] = static_cast<double>(f);   // f (single focal, fx=fy assumed)
  cam[7] = static_cast<double>(cx);  // cx
  cam[8] = static_cast<double>(cy);  // cy
  // [9..16]: k1,k2,k3,p1,p2,k4,k5,k6 = all zero initially
  for (int i = 9; i < kNumCamParams; ++i) cam[i] = 0.0;
}

// Unpack 17-param Ceres camera → camera_t.
// cam = [aa(3), t(3), f, cx, cy, k1, k2, k3, p1, p2, k4, k5, k6]
// coeffs layout in camera_intrin_t: [k1, k2, p1, p2, k3] (OpenCV order)
static camera_t unpack_camera_params(const double* cam, int orig_W, int orig_H) {
  double rot[9];
  ceres::AngleAxisToRotationMatrix(cam, rot);
  // Ceres returns column-major: rot[k*3+j] = R[row=j, col=k]
  // R_wc[r,c] = rot[c*3+r]
  // glm col-major: R_wc_glm[col=c][row=r] = R_wc[r,c] = rot[c*3+r]
  glm::mat3 R_wc_glm;
  for (int c = 0; c < 3; ++c)
    for (int r = 0; r < 3; ++r) R_wc_glm[c][r] = static_cast<float>(rot[c * 3 + r]);

  camera_t result{};
  result.intrin.fx = static_cast<float>(cam[6]);  // f (single focal)
  result.intrin.fy = static_cast<float>(cam[6]);  // fx = fy = f
  result.intrin.cx = static_cast<float>(cam[7]);
  result.intrin.cy = static_cast<float>(cam[8]);
  // Remap distortion coefficients to OpenCV order [k1, k2, p1, p2, k3]
  result.intrin.coeffs[0] = static_cast<float>(cam[9]);   // k1
  result.intrin.coeffs[1] = static_cast<float>(cam[10]);  // k2
  result.intrin.coeffs[2] = static_cast<float>(cam[12]);  // p1
  result.intrin.coeffs[3] = static_cast<float>(cam[13]);  // p2
  result.intrin.coeffs[4] = static_cast<float>(cam[11]);  // k3
  result.width = static_cast<uint32_t>(orig_W);
  result.height = static_cast<uint32_t>(orig_H);
  result.extrin.rotation = R_wc_glm;  // R_wc (world-to-camera), matches dust3r_pose_node convention
  result.extrin.translation = glm::vec3(static_cast<float>(cam[3]), static_cast<float>(cam[4]),
                                        static_cast<float>(cam[5]));  // t_wc
  return result;
}

// Weighted reprojection error cost function (5-param distortion, no radial denominator).
// residuals = sqrt(weight) * (projected - observed)
struct weighted_reprojection_error {
  weighted_reprojection_error(double obs_u, double obs_v, double weight)
      : obs_u_(obs_u), obs_v_(obs_v), sqrt_weight_(std::sqrt(weight)) {}

  template <typename T>
  bool operator()(const T* const cam, const T* const pt, T* res) const {
    // cam = [aa(3), t(3), f, cx, cy, k1, k2, k3, p1, p2, k4, k5, k6]
    T p[3];
    ceres::AngleAxisRotatePoint(cam, pt, p);
    p[0] += cam[3];
    p[1] += cam[4];
    p[2] += cam[5];

    const T xn = p[0] / p[2];
    const T yn = p[1] / p[2];
    const T r2 = xn * xn + yn * yn;
    const T k1 = cam[9], k2 = cam[10], k3 = cam[11];
    const T p1 = cam[12], p2 = cam[13];
    const T dist = T(1.0) + (k1 + (k2 + k3 * r2) * r2) * r2;
    const T xd = xn * dist + T(2.0) * p1 * xn * yn + p2 * (r2 + T(2.0) * xn * xn);
    const T yd = yn * dist + p1 * (r2 + T(2.0) * yn * yn) + T(2.0) * p2 * xn * yn;
    const T pred_u = cam[6] * xd + cam[7];  // f * xd + cx
    const T pred_v = cam[6] * yd + cam[8];  // f * yd + cy

    res[0] = T(sqrt_weight_) * (pred_u - T(obs_u_));
    res[1] = T(sqrt_weight_) * (pred_v - T(obs_v_));
    return true;
  }

  static ceres::CostFunction* create(double obs_u, double obs_v, double weight) {
    return new ceres::AutoDiffCostFunction<weighted_reprojection_error, 2, kNumCamParams, 3>(
        new weighted_reprojection_error(obs_u, obs_v, weight));
  }

 private:
  double obs_u_, obs_v_, sqrt_weight_;
};

}  // anonymous namespace

std::vector<ba_point> merge_shared_points(
    const std::vector<std::string>& camera_names, const std::vector<pair_result>& pair_results,
    const std::unordered_map<std::string, aligned_pose>& poses, int orig_W, int orig_H,
    float conf_threshold, int subsample, float merge_threshold) {
  const int N = static_cast<int>(camera_names.size());
  if (N == 0) return {};

  // Collect per-camera world-space point clouds from each camera's best pair (view1)
  struct cam_sample {
    std::array<float, 3> world;
    float u_orig, v_orig;
    float conf;
  };
  std::vector<std::vector<cam_sample>> per_cam(N);

  for (int cam_i = 0; cam_i < N; ++cam_i) {
    const auto* pr = best_pair_for_camera(pair_results, cam_i);
    if (pr == nullptr) continue;

    const auto pose_it = poses.find(camera_names[cam_i]);
    if (pose_it == poses.end()) continue;
    const aligned_pose& pose = pose_it->second;

    const float* pts3d = pr->view1.pts3d.data();
    const float* conf = pr->view1.conf.data();

    for (int h = 0; h < ONNX_H; h += subsample) {
      for (int w = 0; w < ONNX_W; w += subsample) {
        const int idx = h * ONNX_W + w;
        const float c = conf[idx];
        if (!(c > conf_threshold)) continue;

        const float* lp = pts3d + idx * 3;
        if (!std::isfinite(lp[0]) || !std::isfinite(lp[1]) || !std::isfinite(lp[2])) continue;

        float u_orig, v_orig;
        dust3r_to_original(static_cast<float>(w), static_cast<float>(h), orig_W, orig_H, u_orig,
                           v_orig);
        if (u_orig < 0 || u_orig >= orig_W || v_orig < 0 || v_orig >= orig_H) continue;

        const auto world = pose_transform(pose, lp);
        if (!std::isfinite(world[0]) || !std::isfinite(world[1]) || !std::isfinite(world[2]))
          continue;

        // Ensure the world point has positive depth in THIS camera (sanity check; should always
        // hold)
        if (camera_depth(pose, world) < 1e-3f) continue;

        per_cam[cam_i].push_back({world, u_orig, v_orig, c});
      }
    }
    spdlog::debug("dust3r_calibration: camera {} ({}) has {} valid samples", cam_i,
                  camera_names[cam_i], per_cam[cam_i].size());
  }

  // Seed shared points from camera 0
  std::vector<ba_point> shared;
  shared.reserve(per_cam[0].size() * 4);

  for (const auto& s : per_cam[0]) {
    ba_point pt;
    pt.x = s.world[0];
    pt.y = s.world[1];
    pt.z = s.world[2];
    pt.conf = s.conf;
    pt.observations.push_back({0, s.u_orig, s.v_orig, std::sqrt(s.conf)});
    shared.push_back(std::move(pt));
  }

  const float merge_threshold_sq = merge_threshold * merge_threshold;

  // Merge cameras 1..N-1 into shared set
  for (int cam_i = 1; cam_i < N; ++cam_i) {
    if (per_cam[cam_i].empty()) continue;

    // Build flann index on current shared world points
    const int K = static_cast<int>(shared.size());
    cv::Mat flann_data(K, 3, CV_32F);
    for (int k = 0; k < K; ++k) {
      flann_data.at<float>(k, 0) = shared[k].x;
      flann_data.at<float>(k, 1) = shared[k].y;
      flann_data.at<float>(k, 2) = shared[k].z;
    }
    cv::flann::Index flann_idx(flann_data, cv::flann::KDTreeIndexParams(4));

    cv::Mat query_mat(1, 3, CV_32F);
    cv::Mat nn_indices(1, 1, CV_32S);
    cv::Mat nn_dists(1, 1, CV_32F);

    static constexpr float kMinDepth = 0.05f;  // world units; skip obs where depth < this
    const auto& pose_cam_i = poses.at(camera_names[cam_i]);

    for (const auto& s : per_cam[cam_i]) {
      query_mat.at<float>(0, 0) = s.world[0];
      query_mat.at<float>(0, 1) = s.world[1];
      query_mat.at<float>(0, 2) = s.world[2];

      flann_idx.knnSearch(query_mat, nn_indices, nn_dists, 1, cv::flann::SearchParams(-1));

      const float dist_sq = nn_dists.at<float>(0, 0);
      const int nn_idx = nn_indices.at<int>(0, 0);

      bool do_merge = (nn_idx >= 0 && nn_idx < K && dist_sq < merge_threshold_sq);
      if (do_merge) {
        // Check that the matched seed point has sufficient depth in camera cam_i
        const std::array<float, 3> matched_world = {shared[nn_idx].x, shared[nn_idx].y,
                                                    shared[nn_idx].z};
        if (camera_depth(pose_cam_i, matched_world) < kMinDepth) {
          do_merge = false;  // Seed world point is too close to / behind camera cam_i
        }
      }

      if (do_merge) {
        // Merge: confidence-weighted average of 3D position
        const float old_conf = shared[nn_idx].conf;
        const float new_conf = old_conf + s.conf;
        shared[nn_idx].x = (shared[nn_idx].x * old_conf + s.world[0] * s.conf) / new_conf;
        shared[nn_idx].y = (shared[nn_idx].y * old_conf + s.world[1] * s.conf) / new_conf;
        shared[nn_idx].z = (shared[nn_idx].z * old_conf + s.world[2] * s.conf) / new_conf;
        shared[nn_idx].conf = new_conf;
        shared[nn_idx].observations.push_back({cam_i, s.u_orig, s.v_orig, std::sqrt(s.conf)});
      } else {
        // New shared point
        ba_point pt;
        pt.x = s.world[0];
        pt.y = s.world[1];
        pt.z = s.world[2];
        pt.conf = s.conf;
        pt.observations.push_back({cam_i, s.u_orig, s.v_orig, std::sqrt(s.conf)});
        shared.push_back(std::move(pt));
      }
    }
    spdlog::debug("dust3r_calibration: after merging camera {}, shared size={}", cam_i,
                  shared.size());
  }

  spdlog::info("dust3r_calibration: merged {} shared points, {} total observations", shared.size(),
               std::accumulate(shared.begin(), shared.end(), 0, [](int sum, const ba_point& p) {
                 return sum + static_cast<int>(p.observations.size());
               }));
  return shared;
}

void filter_observations(std::vector<ba_point>& points, int min_cameras) {
  std::vector<ba_point> filtered;
  filtered.reserve(points.size());

  int removed = 0;
  for (auto& pt : points) {
    // Count unique cameras observing this point
    std::vector<int> cam_ids;
    cam_ids.reserve(pt.observations.size());
    for (const auto& obs : pt.observations) {
      cam_ids.push_back(obs.camera_idx);
    }
    std::sort(cam_ids.begin(), cam_ids.end());
    cam_ids.erase(std::unique(cam_ids.begin(), cam_ids.end()), cam_ids.end());

    if (static_cast<int>(cam_ids.size()) >= min_cameras) {
      filtered.push_back(std::move(pt));
    } else {
      ++removed;
    }
  }

  const int kept = static_cast<int>(filtered.size());
  spdlog::info("dust3r_calibration: filter_observations: kept {} / {} points ({} removed)", kept,
               kept + removed, removed);
  points = std::move(filtered);
}

int filter_outlier_observations(std::vector<ba_point>& points,
                                const std::vector<std::string>& camera_names,
                                const std::unordered_map<std::string, aligned_pose>& poses,
                                int orig_W, int orig_H, float k) {
  const int N = static_cast<int>(camera_names.size());
  if (N == 0 || points.empty()) return 0;

  // Build initial camera parameter arrays (same logic as run_bundle_adjustment)
  std::vector<std::array<double, kNumCamParams>> cam_params(N);
  for (int i = 0; i < N; ++i) {
    const auto it = poses.find(camera_names[i]);
    if (it == poses.end()) continue;
    const aligned_pose& pose = it->second;
    float fx, fy;
    dust3r_focal_to_original(pose.focal, orig_W, orig_H, fx, fy);
    const float f = (fx + fy) * 0.5f;  // single focal (fx=fy for 16:9 images)
    float cx, cy;
    dust3r_to_original(pose.cx, pose.cy, orig_W, orig_H, cx, cy);
    pack_camera_params(pose, f, cx, cy, cam_params[i].data());
  }

  // Build flat list of (point_idx, obs_idx, error) for robust threshold computation
  std::vector<double> all_errors;
  all_errors.reserve(points.size() * 4);

  for (size_t k_pt = 0; k_pt < points.size(); ++k_pt) {
    const std::array<double, 3> pt = {points[k_pt].x, points[k_pt].y, points[k_pt].z};
    for (const auto& obs : points[k_pt].observations) {
      const auto& cam = cam_params[obs.camera_idx];
      double p[3];
      ceres::AngleAxisRotatePoint(cam.data(), pt.data(), p);
      p[0] += cam[3];
      p[1] += cam[4];
      p[2] += cam[5];
      if (p[2] <= 1e-6) {
        all_errors.push_back(std::numeric_limits<double>::infinity());
        continue;
      }
      const double xn = p[0] / p[2], yn = p[1] / p[2];
      const double r2 = xn * xn + yn * yn;
      const double dist = 1.0 + (cam[9] + (cam[10] + cam[11] * r2) * r2) * r2;
      const double xd = xn * dist + 2.0 * cam[12] * xn * yn + cam[13] * (r2 + 2.0 * xn * xn);
      const double yd = yn * dist + cam[12] * (r2 + 2.0 * yn * yn) + 2.0 * cam[13] * xn * yn;
      const double eu = cam[6] * xd + cam[7] - static_cast<double>(obs.u_orig);  // f*xd+cx
      const double ev = cam[6] * yd + cam[8] - static_cast<double>(obs.v_orig);  // f*yd+cy
      all_errors.push_back(std::sqrt(eu * eu + ev * ev));
    }
  }

  if (all_errors.empty()) return 0;

  // Compute median + k*MAD threshold over finite errors
  std::vector<double> finite_errors;
  finite_errors.reserve(all_errors.size());
  for (double e : all_errors)
    if (std::isfinite(e)) finite_errors.push_back(e);
  if (finite_errors.empty()) return 0;

  std::sort(finite_errors.begin(), finite_errors.end());
  const double median = finite_errors[finite_errors.size() / 2];

  std::vector<double> abs_dev;
  abs_dev.reserve(finite_errors.size());
  for (double e : finite_errors) abs_dev.push_back(std::abs(e - median));
  std::sort(abs_dev.begin(), abs_dev.end());
  const double mad = abs_dev[abs_dev.size() / 2];
  const double threshold = median + static_cast<double>(k) * mad;

  // Remove observations above threshold
  int removed = 0;
  size_t error_idx = 0;
  for (auto& pt : points) {
    std::vector<ba_observation> kept;
    kept.reserve(pt.observations.size());
    for (const auto& obs : pt.observations) {
      const double e = all_errors[error_idx++];
      if (e <= threshold) {
        kept.push_back(obs);
      } else {
        ++removed;
      }
    }
    pt.observations = std::move(kept);
  }

  spdlog::info(
      "dust3r_calibration: outlier removal: median={:.2f}px MAD={:.2f}px "
      "threshold={:.2f}px removed={} (k={})",
      median, mad, threshold, removed, k);
  return removed;
}

std::vector<camera_t> run_bundle_adjustment(
    const std::vector<std::string>& camera_names,
    const std::unordered_map<std::string, aligned_pose>& poses, std::vector<ba_point>& points,
    int orig_W, int orig_H, double* out_initial_rmse, double* out_final_rmse,
    double* out_initial_ceres_cost) {
  const int N = static_cast<int>(camera_names.size());
  if (N == 0 || points.empty()) return {};

  // Pack camera parameters
  std::vector<std::array<double, kNumCamParams>> cam_params(N);
  for (int i = 0; i < N; ++i) {
    const auto it = poses.find(camera_names[i]);
    if (it == poses.end()) {
      spdlog::error("dust3r_calibration: missing pose for {}", camera_names[i]);
      return {};
    }
    const aligned_pose& pose = it->second;
    float fx, fy;
    dust3r_focal_to_original(pose.focal, orig_W, orig_H, fx, fy);
    const float f = (fx + fy) * 0.5f;  // single focal
    float cx, cy;
    dust3r_to_original(pose.cx, pose.cy, orig_W, orig_H, cx, cy);
    pack_camera_params(pose, f, cx, cy, cam_params[i].data());
  }

  // Pack 3D points
  std::vector<std::array<double, 3>> pt_params(points.size());
  for (size_t k = 0; k < points.size(); ++k) {
    pt_params[k] = {points[k].x, points[k].y, points[k].z};
  }

  // Helper: compute RMSE of reprojection error
  auto compute_rmse = [&]() -> double {
    double sum_sq = 0.0;
    int count = 0;
    for (size_t k = 0; k < points.size(); ++k) {
      for (const auto& obs : points[k].observations) {
        const auto& cam = cam_params[obs.camera_idx];
        const auto& pt = pt_params[k];
        double p[3];
        ceres::AngleAxisRotatePoint(cam.data(), pt.data(), p);
        p[0] += cam[3];
        p[1] += cam[4];
        p[2] += cam[5];
        if (p[2] <= 1e-6) continue;
        const double xn = p[0] / p[2], yn = p[1] / p[2];
        const double r2 = xn * xn + yn * yn;
        const double dist = 1.0 + (cam[9] + (cam[10] + cam[11] * r2) * r2) * r2;
        const double xd = xn * dist + 2.0 * cam[12] * xn * yn + cam[13] * (r2 + 2.0 * xn * xn);
        const double yd = yn * dist + cam[12] * (r2 + 2.0 * yn * yn) + 2.0 * cam[13] * xn * yn;
        const double eu = cam[6] * xd + cam[7] - obs.u_orig;  // f*xd + cx
        const double ev = cam[6] * yd + cam[8] - obs.v_orig;  // f*yd + cy
        sum_sq += eu * eu + ev * ev;
        ++count;
      }
    }
    return count > 0 ? std::sqrt(sum_sq / static_cast<double>(count)) : 0.0;
  };

  if (out_initial_rmse) *out_initial_rmse = compute_rmse();

  // Build Ceres problem
  ceres::Problem problem;

  // Add camera parameter blocks
  for (int i = 0; i < N; ++i) {
    problem.AddParameterBlock(cam_params[i].data(), kNumCamParams);
  }

  // Fix camera 0 extrinsics (gauge constraint): params [0..5]
  // Also fix cx(7), cy(8) for all cameras: principal point is not reliably
  // estimable from the DUSt3R point cloud and drifts during BA.
  // Fix k4, k5, k6 (indices 14,15,16) as well — not optimised.
  for (int i = 0; i < N; ++i) {
    std::vector<int> fixed_indices;
    if (i == 0 && N > 1) {
      // camera 0: fix extrinsics + cx + cy + k4/k5/k6
      fixed_indices = {0, 1, 2, 3, 4, 5, 7, 8, 14, 15, 16};
    } else {
      // other cameras: fix cx + cy + k4/k5/k6
      fixed_indices = {7, 8, 14, 15, 16};
    }
    problem.SetManifold(cam_params[i].data(),
                        new ceres::SubsetManifold(kNumCamParams, fixed_indices));
  }

  // Add point parameter blocks and residuals
  for (size_t k = 0; k < points.size(); ++k) {
    problem.AddParameterBlock(pt_params[k].data(), 3);
    for (const auto& obs : points[k].observations) {
      const double weight = static_cast<double>(obs.weight);
      ceres::CostFunction* cost = weighted_reprojection_error::create(
          static_cast<double>(obs.u_orig), static_cast<double>(obs.v_orig), weight * weight);
      problem.AddResidualBlock(cost, nullptr, cam_params[obs.camera_idx].data(),
                               pt_params[k].data());
    }
  }

  spdlog::info("dust3r_calibration: BA setup: {} cameras, {} points, {} observations", N,
               points.size(),
               std::accumulate(points.begin(), points.end(), 0, [](int s, const ba_point& p) {
                 return s + static_cast<int>(p.observations.size());
               }));

  // Solve
  ceres::Solver::Options opts;
  opts.max_num_iterations = 500;
  opts.linear_solver_type = ceres::DENSE_SCHUR;
  opts.preconditioner_type = ceres::SCHUR_JACOBI;
  opts.use_inner_iterations = false;
  opts.use_nonmonotonic_steps = false;
  opts.num_threads = 8;
  opts.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(opts, &problem, &summary);
  spdlog::info("dust3r_calibration: {}", summary.BriefReport());
  if (out_initial_ceres_cost) *out_initial_ceres_cost = summary.initial_cost;

  // Write back optimised 3D points
  for (size_t k = 0; k < points.size(); ++k) {
    points[k].x = static_cast<float>(pt_params[k][0]);
    points[k].y = static_cast<float>(pt_params[k][1]);
    points[k].z = static_cast<float>(pt_params[k][2]);
  }

  if (out_final_rmse) *out_final_rmse = compute_rmse();

  // Unpack optimised camera params into camera_t
  std::vector<camera_t> results(N);
  for (int i = 0; i < N; ++i) {
    results[i] = unpack_camera_params(cam_params[i].data(), orig_W, orig_H);
  }
  return results;
}

}  // namespace stargazer::dust3r
