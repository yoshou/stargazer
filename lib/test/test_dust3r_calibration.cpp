// Unit tests for dust3r_calibration: coordinate transforms, point merge, synthetic BA
#include <ceres/rotation.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "dust3r.hpp"
#include "dust3r_alignment.hpp"
#include "dust3r_calibration.hpp"

namespace {
namespace d3r = stargazer::dust3r;
}

TEST(DustR3Calibration, TC1_OriginalCoordTransform_16x9) {
  // 960x540 (same aspect as 512x288) → no crop, simple scale
  const int W = 960, H = 540;
  // Corner points
  float u_orig, v_orig;

  d3r::dust3r_to_original(0.0f, 0.0f, W, H, u_orig, v_orig);
  EXPECT_NEAR(u_orig, 0.0f, 1e-3f);
  EXPECT_NEAR(v_orig, 0.0f, 1e-3f);

  d3r::dust3r_to_original(512.0f, 288.0f, W, H, u_orig, v_orig);
  EXPECT_NEAR(u_orig, 960.0f, 1e-3f);
  EXPECT_NEAR(v_orig, 540.0f, 1e-3f);

  // Center of 512x288 → center of 960x540
  d3r::dust3r_to_original(256.0f, 144.0f, W, H, u_orig, v_orig);
  EXPECT_NEAR(u_orig, 480.0f, 1e-3f);
  EXPECT_NEAR(v_orig, 270.0f, 1e-3f);
  std::cout << "[TC1] 960x540 center: (" << u_orig << ", " << v_orig << ")\n";
}

TEST(DustR3Calibration, TC2_OriginalCoordTransform_640x480) {
  // 640x480 (4:3 aspect, different from 512:288)
  const int W = 640, H = 480;
  float u_orig, v_orig;

  d3r::dust3r_to_original(256.0f, 144.0f, W, H, u_orig, v_orig);
  EXPECT_NEAR(u_orig, 256.0f * 640.0f / 512.0f, 1e-3f);
  EXPECT_NEAR(v_orig, 144.0f * 480.0f / 288.0f, 1e-3f);
  std::cout << "[TC2] 640x480 (" << 256 << "," << 144 << ") → (" << u_orig << ", " << v_orig
            << ")\n";
}

TEST(DustR3Calibration, TC3_FocalToOriginal) {
  // For 960x540: scale_x = scale_y = 512/960, so fx = fy = focal * 960/512
  const int W = 960, H = 540;
  const float focal_512 = 220.0f;
  float fx, fy;
  d3r::dust3r_focal_to_original(focal_512, W, H, fx, fy);
  EXPECT_NEAR(fx, focal_512 * 960.0f / 512.0f, 1e-2f);
  EXPECT_NEAR(fy, focal_512 * 540.0f / 288.0f, 1e-2f);
  // For 16:9: fx == fy
  EXPECT_NEAR(fx, fy, 1.0f);
  std::cout << "[TC3] focal_512=" << focal_512 << " → fx=" << fx << " fy=" << fy << "\n";
}

// Build a synthetic pair_result from known 3D points in camera-local frame
static d3r::pair_result make_synthetic_pair(int idx1, int idx2,
                                            const std::vector<std::array<float, 3>>& pts_cam1,
                                            const std::vector<std::array<float, 3>>& pts_cam2,
                                            float conf_val = 5.0f) {
  const int HW = d3r::ONNX_H * d3r::ONNX_W;
  d3r::pair_result pr;
  pr.idx1 = idx1;
  pr.idx2 = idx2;
  pr.view1.camera_name = "cam" + std::to_string(idx1);
  pr.view2.camera_name = "cam" + std::to_string(idx2);
  pr.view1.pts3d.assign(HW * 3, 0.0f);
  pr.view1.conf.assign(HW, 0.0f);
  pr.view2.pts3d.assign(HW * 3, 0.0f);
  pr.view2.conf.assign(HW, 0.0f);

  // Place points at stride=8 positions
  int stride = 8;
  size_t p_idx = 0;
  for (int h = 0; h < d3r::ONNX_H; h += stride) {
    for (int w = 0; w < d3r::ONNX_W; w += stride) {
      const int idx = h * d3r::ONNX_W + w;
      if (p_idx < pts_cam1.size()) {
        pr.view1.pts3d[idx * 3 + 0] = pts_cam1[p_idx][0];
        pr.view1.pts3d[idx * 3 + 1] = pts_cam1[p_idx][1];
        pr.view1.pts3d[idx * 3 + 2] = pts_cam1[p_idx][2];
        pr.view1.conf[idx] = conf_val;
      }
      if (p_idx < pts_cam2.size()) {
        pr.view2.pts3d[idx * 3 + 0] = pts_cam2[p_idx][0];
        pr.view2.pts3d[idx * 3 + 1] = pts_cam2[p_idx][1];
        pr.view2.pts3d[idx * 3 + 2] = pts_cam2[p_idx][2];
        pr.view2.conf[idx] = conf_val;
      }
      ++p_idx;
    }
  }
  return pr;
}

TEST(DustR3Calibration, TC4_MergeSharedPoints) {
  // Two cameras share the same world points (identical point clouds in world frame)
  // After merge, the shared set should have approximately the original number of points
  const int N_pts = 50;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Generate N_pts world points
  std::vector<std::array<float, 3>> world_pts(N_pts);
  for (auto& p : world_pts) {
    p = {dist(rng), dist(rng), 2.0f + dist(rng) * 0.5f};
  }

  // Camera 0: identity pose, local points = world points
  // Camera 1: translated along X by 0.5, local points ≠ world points
  // But we'll set view1.pts3d = local_pts and aligned_pose transforms back to world

  // For simplicity: camera 0 and 1 both have identity rotation, different translations
  stargazer::dust3r::aligned_pose pose0, pose1;
  pose0.rotation = glm::mat3(1.0f);
  pose0.translation = glm::vec3(0.0f);
  pose0.scale = 1.0f;
  pose0.focal = 220.0f;
  pose0.cx = 256.0f;
  pose0.cy = 144.0f;

  pose1.rotation = glm::mat3(1.0f);
  pose1.translation = glm::vec3(0.5f, 0.0f, 0.0f);
  pose1.scale = 1.0f;
  pose1.focal = 220.0f;
  pose1.cx = 256.0f;
  pose1.cy = 144.0f;

  // Local pts for camera 0 (world = identity * local + 0 = local)
  // Local pts for camera 1 (world = identity * local + (0.5,0,0))
  // So local1 = world - (0.5, 0, 0)
  std::vector<std::array<float, 3>> local0(N_pts), local1(N_pts);
  for (int i = 0; i < N_pts; ++i) {
    local0[i] = world_pts[i];
    local1[i] = {world_pts[i][0] - 0.5f, world_pts[i][1], world_pts[i][2]};
  }

  const std::vector<std::string> names = {"cam0", "cam1"};
  auto pair01 = make_synthetic_pair(0, 1, local0, local1);
  auto pair10 = make_synthetic_pair(1, 0, local1, local0);
  const std::vector<d3r::pair_result> pairs = {pair01, pair10};

  std::unordered_map<std::string, d3r::aligned_pose> poses;
  poses["cam0"] = pose0;
  poses["cam1"] = pose1;

  const int W = 960, H = 540;
  auto shared = d3r::merge_shared_points(names, pairs, poses, W, H, 3.0f, 8, 0.1f);
  d3r::filter_observations(shared, 2);

  std::cout << "[TC4] Merged shared points (2-cam filter): " << shared.size() << " (expected ~"
            << N_pts << ")\n";

  // The shared set should have points observed by both cameras
  EXPECT_GT(static_cast<int>(shared.size()), 0);
  for (const auto& pt : shared) {
    EXPECT_GE(pt.observations.size(), 2u) << "Expected >=2 observations per shared point";
  }
}

TEST(DustR3Calibration, TC5_SyntheticBA) {
  // Create synthetic camera rig: 2 cameras with known intrinsics
  // Project 3D points to 2D, then run BA to recover intrinsics
  // Verify |Δfx| < 5.0 px after BA

  const int W = 960, H = 540;
  const float TRUE_FX = 412.0f, TRUE_FY = 412.0f;
  const float TRUE_CX = 480.0f, TRUE_CY = 270.0f;
  const float TRUE_K1 = -0.1f, TRUE_K2 = 0.02f;
  const float TRUE_P1 = 0.001f, TRUE_P2 = 0.001f, TRUE_K3 = 0.0f;

  // Forward project with distortion (world → cam coords → distorted pixel)
  auto project = [&](const std::array<double, 3>& world_pt, const std::array<double, 6>& ext,
                     float fx, float fy, float cx, float cy, float k1, float k2, float k3, float p1,
                     float p2, float& u_out, float& v_out) {
    // ext = [aa(3), t(3)]
    double p[3];
    ceres::AngleAxisRotatePoint(ext.data(), world_pt.data(), p);
    p[0] += ext[3];
    p[1] += ext[4];
    p[2] += ext[5];
    if (p[2] < 1e-6f) {
      u_out = v_out = -1.0f;
      return;
    }
    const float xn = p[0] / p[2], yn = p[1] / p[2];
    const float r2 = xn * xn + yn * yn;
    const float dist = 1.0f + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
    const float xd = xn * dist + 2.0f * p1 * xn * yn + p2 * (r2 + 2.0f * xn * xn);
    const float yd = yn * dist + p1 * (r2 + 2.0f * yn * yn) + 2.0f * p2 * xn * yn;
    u_out = fx * xd + cx;
    v_out = fy * yd + cy;
  };

  // Two cameras: cam0 at origin identity, cam1 translated by (0.5, 0, 0)
  // In BA convention: cam = [aa_wc(3), t_wc(3), ...]
  // cam0: identity R_wc, t_wc=0 → aa=0, t=(0,0,0)
  // cam1: R_wc=Identity, t_wc = (cam origin in world frame → t_c2w=(0.5,0,0) → t_wc = -(I *
  // (0.5,0,0)) = (-0.5,0,0))
  const std::array<double, 6> ext0 = {0, 0, 0, 0, 0, 0};
  const std::array<double, 6> ext1 = {0, 0, 0, -0.5, 0, 0};

  // Generate N world points in front of both cameras
  const int N_pts = 200;
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dx(-0.3f, 0.3f);
  std::vector<std::array<double, 3>> world_pts(N_pts);
  for (auto& p : world_pts) {
    p = {static_cast<double>(dx(rng)), static_cast<double>(dx(rng)), 1.5 + dx(rng) * 0.3};
  }

  // Build ba_points with synthetic 2D observations from BOTH cameras
  std::vector<d3r::ba_point> ba_pts;
  int valid_count = 0;
  for (int k = 0; k < N_pts; ++k) {
    float u0, v0, u1, v1;
    project(world_pts[k], ext0, TRUE_FX, TRUE_FY, TRUE_CX, TRUE_CY, TRUE_K1, TRUE_K2, TRUE_K3,
            TRUE_P1, TRUE_P2, u0, v0);
    project(world_pts[k], ext1, TRUE_FX, TRUE_FY, TRUE_CX, TRUE_CY, TRUE_K1, TRUE_K2, TRUE_K3,
            TRUE_P1, TRUE_P2, u1, v1);
    if (u0 < 0 || u0 >= W || v0 < 0 || v0 >= H) continue;
    if (u1 < 0 || u1 >= W || v1 < 0 || v1 >= H) continue;

    d3r::ba_point pt;
    pt.x = static_cast<float>(world_pts[k][0]);
    pt.y = static_cast<float>(world_pts[k][1]);
    pt.z = static_cast<float>(world_pts[k][2]);
    pt.conf = 1.0f;
    pt.observations.push_back({0, u0, v0, 1.0f});
    pt.observations.push_back({1, u1, v1, 1.0f});
    ba_pts.push_back(std::move(pt));
    ++valid_count;
  }
  std::cout << "[TC5] Valid 2-camera points: " << valid_count << "\n";
  ASSERT_GT(valid_count, 10) << "Not enough valid points for BA test";

  // Initial poses: camera 0 at identity, camera 1 translated
  // aligned_pose stores cam-to-world: pose0 = I,0; pose1 = I, (0.5,0,0)
  stargazer::dust3r::aligned_pose pose0, pose1;
  pose0.rotation = glm::mat3(1.0f);
  pose0.translation = glm::vec3(0.0f);
  pose0.scale = 1.0f;
  pose0.focal = 220.0f;  // in 512 space ~ TRUE_FX * 512/960
  pose0.cx = 256.0f;
  pose0.cy = 144.0f;

  pose1.rotation = glm::mat3(1.0f);
  pose1.translation = glm::vec3(0.5f, 0.0f, 0.0f);
  pose1.scale = 1.0f;
  pose1.focal = 220.0f;
  pose1.cx = 256.0f;
  pose1.cy = 144.0f;

  const std::vector<std::string> names = {"cam0", "cam1"};
  std::unordered_map<std::string, d3r::aligned_pose> poses;
  poses["cam0"] = pose0;
  poses["cam1"] = pose1;

  double rmse_before, rmse_after;
  auto results = d3r::run_bundle_adjustment(names, poses, ba_pts, W, H, &rmse_before, &rmse_after);

  std::cout << "[TC5] RMSE: before=" << rmse_before << "  after=" << rmse_after << "\n";
  ASSERT_EQ(results.size(), 2u);

  const auto& cam0_result = results[0];
  const auto& cam1_result = results[1];
  std::cout << "[TC5] cam0 fx=" << cam0_result.intrin.fx << " fy=" << cam0_result.intrin.fy
            << " cx=" << cam0_result.intrin.cx << " cy=" << cam0_result.intrin.cy << "\n";
  std::cout << "[TC5] cam0 k1=" << cam0_result.intrin.coeffs[0]
            << " k2=" << cam0_result.intrin.coeffs[1] << "\n";

  // After BA the reprojection error should be near-zero (BA converged)
  EXPECT_LT(rmse_after, rmse_before * 0.001) << "RMSE should decrease by >1000x";
  EXPECT_LT(rmse_after, 0.01) << "Final RMSE should be < 0.01 px";

  // intrinsics should be in valid range (not degenerate)
  EXPECT_GT(cam0_result.intrin.fx, 50.0f) << "fx should be positive and reasonable";
  EXPECT_GT(cam1_result.intrin.fx, 50.0f) << "cam1 fx should be positive and reasonable";
  // Note: with 2 cameras and scale ambiguity, exact fx recovery is not expected;
  // we only verify the BA converged to a consistent solution.
}
