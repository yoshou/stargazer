#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <array>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "dust3r_alignment.hpp"

namespace {

std::vector<float> fill_sparse_points(const std::vector<Eigen::Vector3f>& src) {
  namespace d3r = stargazer::dust3r;
  static constexpr int HW = d3r::ONNX_H * d3r::ONNX_W;

  std::vector<float> pts(HW * 3, 0.0f);
  for (int row = 0; row < d3r::ONNX_H; row += 8) {
    for (int col = 0; col < d3r::ONNX_W; col += 8) {
      const int px = row * d3r::ONNX_W + col;
      const int sidx = (row / 8 * (d3r::ONNX_W / 8) + col / 8) % static_cast<int>(src.size());
      pts[px * 3 + 0] = src[sidx][0];
      pts[px * 3 + 1] = src[sidx][1];
      pts[px * 3 + 2] = src[sidx][2];
    }
  }
  return pts;
}

std::vector<float> conf_with_value(float value) {
  return std::vector<float>(stargazer::dust3r::ONNX_H * stargazer::dust3r::ONNX_W, value);
}

Eigen::Vector3f world_to_camera(const Eigen::Matrix3f& R_wc, const Eigen::Vector3f& t_wc,
                                const Eigen::Vector3f& p_world) {
  return R_wc.transpose() * (p_world - t_wc);
}

}  // namespace

TEST(DustR3Alignment, TC1_ProcrustesKnownRt) {
  const float angle = 0.5236f;  // 30 deg
  Eigen::Vector3f axis(1, 1, 1);
  axis.normalize();
  const float s = std::sin(angle), c = std::cos(angle), t = 1 - c;
  Eigen::Matrix3f R_true;
  R_true << t * axis[0] * axis[0] + c, t * axis[0] * axis[1] - s * axis[2],
      t * axis[0] * axis[2] + s * axis[1], t * axis[0] * axis[1] + s * axis[2],
      t * axis[1] * axis[1] + c, t * axis[1] * axis[2] - s * axis[0],
      t * axis[0] * axis[2] - s * axis[1], t * axis[1] * axis[2] + s * axis[0],
      t * axis[2] * axis[2] + c;
  Eigen::Vector3f t_true(0.3f, -0.2f, 0.5f);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

  const int N = 1000;
  std::vector<Eigen::Vector3f> pts_in_cam0(N), pts_in_cam1(N);
  for (int i = 0; i < N; ++i) {
    Eigen::Vector3f P_world(dist(rng), dist(rng), dist(rng));
    pts_in_cam0[i] = P_world;
    pts_in_cam1[i] = R_true.transpose() * (P_world - t_true);
  }

  namespace d3r = stargazer::dust3r;
  auto conf_ones = conf_with_value(1.0f);

  d3r::pair_result pr;
  pr.idx1 = 0;
  pr.idx2 = 1;
  pr.view1.camera_name = "cam0";
  pr.view2.camera_name = "cam1";
  pr.view1.pts3d = fill_sparse_points(pts_in_cam0);
  pr.view2.pts3d = fill_sparse_points(pts_in_cam0);
  pr.view1.conf = conf_ones;
  pr.view2.conf = conf_ones;

  d3r::pair_result pr_rev;
  pr_rev.idx1 = 1;
  pr_rev.idx2 = 0;
  pr_rev.view1.camera_name = "cam1";
  pr_rev.view2.camera_name = "cam0";
  pr_rev.view1.pts3d = fill_sparse_points(pts_in_cam1);
  pr_rev.view2.pts3d = fill_sparse_points(pts_in_cam1);
  pr_rev.view1.conf = conf_ones;
  pr_rev.view2.conf = conf_ones;

  std::vector<d3r::pair_result> pairs = {pr, pr_rev};
  std::vector<std::string> names = {"cam0", "cam1"};

  auto poses = d3r::align_global(names, pairs);

  ASSERT_TRUE(poses.count("cam0"));
  ASSERT_TRUE(poses.count("cam1"));

  const auto& p0 = poses["cam0"];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) EXPECT_NEAR(p0.rotation[j][i], (i == j ? 1.0f : 0.0f), 1e-4f);
  EXPECT_NEAR(p0.translation.x, 0.f, 1e-4f);
  EXPECT_NEAR(p0.translation.y, 0.f, 1e-4f);
  EXPECT_NEAR(p0.translation.z, 0.f, 1e-4f);

  const auto& p1 = poses["cam1"];
  Eigen::Matrix3f R_est;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) R_est(r, c) = p1.rotation[c][r];

  const float frob = (R_est - R_true).norm();
  std::cout << "[TC1] Procrustes R recovery Frobenius err = " << frob << "\n";
  std::cout << "[TC1] t recovery: est=(" << p1.translation.x << "," << p1.translation.y << ","
            << p1.translation.z << ")" << "  true=(" << t_true[0] << "," << t_true[1] << ","
            << t_true[2] << ")\n";

  EXPECT_LT(frob, 0.01f) << "Rotation recovery error too large";

  const float t_err = std::sqrt(std::pow(p1.translation.x - t_true[0], 2) +
                                std::pow(p1.translation.y - t_true[1], 2) +
                                std::pow(p1.translation.z - t_true[2], 2));
  std::cout << "[TC1] t recovery err = " << t_err << "\n";
  EXPECT_LT(t_err, 0.01f) << "Translation recovery error too large";
}

TEST(DustR3Alignment, TC2_MSTConnectivity) {
  namespace d3r = stargazer::dust3r;
  static constexpr int HW = d3r::ONNX_H * d3r::ONNX_W;

  std::vector<std::string> names = {"cam0", "cam1", "cam2"};
  std::vector<d3r::pair_result> pairs;

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (i == j) continue;
      d3r::pair_result pr;
      pr.idx1 = i;
      pr.idx2 = j;
      pr.view1.camera_name = names[i];
      pr.view2.camera_name = names[j];
      pr.view1.pts3d.assign(HW * 3, 0.1f);
      pr.view2.pts3d.assign(HW * 3, 0.1f);
      pr.view1.conf.assign(HW, 1.0f);
      pr.view2.conf.assign(HW, 1.0f);
      pairs.push_back(pr);
    }
  }

  auto poses = d3r::align_global(names, pairs);

  ASSERT_EQ(poses.size(), 3u) << "Expected poses for all 3 cameras";
  ASSERT_TRUE(poses.count("cam0"));
  ASSERT_TRUE(poses.count("cam1"));
  ASSERT_TRUE(poses.count("cam2"));

  std::cout << "[TC2] MST connectivity OK, 3 cameras aligned\n";
}

TEST(DustR3Alignment, TC3_RecoversSyntheticMultiCameraRig) {
  namespace d3r = stargazer::dust3r;

  std::mt19937 rng(7);
  std::uniform_real_distribution<float> xy(-2.5f, 2.5f);
  std::uniform_real_distribution<float> z(3.0f, 7.0f);

  std::vector<Eigen::Vector3f> world_points(256);
  for (auto& point : world_points) {
    point = Eigen::Vector3f(xy(rng), xy(rng), z(rng));
  }

  const std::vector<std::string> camera_names = {
      "camera0", "camera1", "camera2", "camera3", "camera4",
  };

  const auto rot_y = [](float angle) {
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    Eigen::Matrix3f R;
    R << c, 0.0f, s, 0.0f, 1.0f, 0.0f, -s, 0.0f, c;
    return R;
  };

  const std::array<Eigen::Matrix3f, 5> rotations = {
      Eigen::Matrix3f::Identity(),
      rot_y(0.18f),
      rot_y(-0.22f),
      rot_y(0.35f),
      rot_y(-0.12f),
  };
  const std::array<Eigen::Vector3f, 5> translations = {
      Eigen::Vector3f(0.0f, 0.0f, 0.0f),
      Eigen::Vector3f(0.3f, -0.1f, 0.2f),
      Eigen::Vector3f(-0.4f, 0.2f, 0.5f),
      Eigen::Vector3f(0.6f, 0.1f, -0.3f),
      Eigen::Vector3f(-0.2f, -0.3f, 0.4f),
  };

  std::vector<std::vector<Eigen::Vector3f>> camera_points(camera_names.size());
  for (size_t cam = 0; cam < camera_names.size(); ++cam) {
    auto& dst = camera_points[cam];
    dst.reserve(world_points.size());
    for (const auto& world_point : world_points) {
      dst.push_back(world_to_camera(rotations[cam], translations[cam], world_point));
    }
  }

  std::vector<d3r::pair_result> pairs;
  for (int i = 0; i < static_cast<int>(camera_names.size()); ++i) {
    for (int j = 0; j < static_cast<int>(camera_names.size()); ++j) {
      if (i == j) continue;
      d3r::pair_result pr;
      pr.idx1 = i;
      pr.idx2 = j;
      pr.view1.camera_name = camera_names[i];
      pr.view2.camera_name = camera_names[j];
      pr.view1.pts3d = fill_sparse_points(camera_points[i]);
      pr.view2.pts3d = fill_sparse_points(camera_points[i]);
      pr.view1.conf = conf_with_value((std::abs(i - j) == 1) ? 2.0f : 0.5f);
      pr.view2.conf = conf_with_value((std::abs(i - j) == 1) ? 2.0f : 0.5f);
      pairs.push_back(std::move(pr));
    }
  }

  const auto poses = d3r::align_global(camera_names, pairs);
  ASSERT_EQ(poses.size(), camera_names.size());

  for (size_t cam = 0; cam < camera_names.size(); ++cam) {
    const auto& pose = poses.at(camera_names[cam]);
    Eigen::Matrix3f estimated_rotation;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        estimated_rotation(r, c) = pose.rotation[c][r];
      }
    }

    const float rotation_error = (estimated_rotation - rotations[cam]).norm();
    const Eigen::Vector3f estimated_translation(pose.translation.x, pose.translation.y,
                                                pose.translation.z);
    const float translation_error = (estimated_translation - translations[cam]).norm();

    std::cout << "[TC3] " << camera_names[cam] << " rotation err=" << rotation_error
              << " translation err=" << translation_error << "\n";

    EXPECT_LT(rotation_error, 0.02f);
    EXPECT_LT(translation_error, 0.02f);
  }
}
