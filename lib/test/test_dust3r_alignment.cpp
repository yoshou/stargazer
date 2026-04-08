#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "dust3r_alignment.hpp"
#include "dust3r_optimizer.hpp"

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

std::vector<float> conf_sparse(float value) {
  namespace d3r = stargazer::dust3r;
  std::vector<float> conf(d3r::ONNX_H * d3r::ONNX_W, 0.0f);
  for (int row = 0; row < d3r::ONNX_H; row += 8) {
    for (int col = 0; col < d3r::ONNX_W; col += 8) {
      conf[row * d3r::ONNX_W + col] = value;
    }
  }
  return conf;
}

Eigen::Vector3f world_to_camera(const Eigen::Matrix3f& R_wc, const Eigen::Vector3f& t_wc,
                                const Eigen::Vector3f& p_world) {
  return R_wc.transpose() * (p_world - t_wc);
}

glm::mat3 to_glm(const Eigen::Matrix3f& matrix) {
  return glm::mat3(matrix(0, 0), matrix(1, 0), matrix(2, 0), matrix(0, 1), matrix(1, 1),
                   matrix(2, 1), matrix(0, 2), matrix(1, 2), matrix(2, 2));
}

float direction_error_deg(const Eigen::Vector3f& estimate, const Eigen::Vector3f& truth) {
  const float estimate_norm = estimate.norm();
  const float truth_norm = truth.norm();
  if (estimate_norm < 1e-8f || truth_norm < 1e-8f) {
    return 0.0f;
  }

  const float cosine = std::clamp(estimate.dot(truth) / (estimate_norm * truth_norm), -1.0f, 1.0f);
  return std::acos(cosine) * 180.0f / static_cast<float>(M_PI);
}

}  // namespace

TEST(DustR3Alignment, TC1_ProcrustesKnownRt) {
  const float angle = 0.5236f;  // 30 deg
  const float scale_true = 2.5f;
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
    pts_in_cam1[i] = (1.0f / scale_true) * (R_true.transpose() * (P_world - t_true));
  }

  namespace d3r = stargazer::dust3r;
  const auto conf_ones = conf_sparse(10.0f);

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

  const Eigen::Vector3f estimated_translation(p1.translation.x, p1.translation.y, p1.translation.z);
  const float t_dir_err = direction_error_deg(estimated_translation, t_true);
  std::cout << "[TC1] t direction err deg = " << t_dir_err << "\n";
  EXPECT_LT(t_dir_err, 0.01f) << "Translation direction recovery error too large";
  EXPECT_GT(p1.scale, 0.0f) << "Scale should stay positive after normalization";
}

TEST(DustR3Alignment, TC2_MSTConnectivity) {
  namespace d3r = stargazer::dust3r;

  std::vector<std::string> names = {"cam0", "cam1", "cam2"};
  std::vector<d3r::pair_result> pairs;

  std::vector<Eigen::Vector3f> synthetic_points;
  synthetic_points.reserve((d3r::ONNX_H / 8) * (d3r::ONNX_W / 8));
  for (int row = 0; row < d3r::ONNX_H; row += 8) {
    for (int col = 0; col < d3r::ONNX_W; col += 8) {
      synthetic_points.emplace_back(static_cast<float>(col) * 0.01f,
                                    static_cast<float>(row) * 0.01f,
                                    1.0f + static_cast<float>(row + col) * 0.001f);
    }
  }

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (i == j) continue;
      d3r::pair_result pr;
      pr.idx1 = i;
      pr.idx2 = j;
      pr.view1.camera_name = names[i];
      pr.view2.camera_name = names[j];
      pr.view1.pts3d = fill_sparse_points(synthetic_points);
      pr.view2.pts3d = fill_sparse_points(synthetic_points);
      pr.view1.conf = conf_sparse(10.0f);
      pr.view2.conf = conf_sparse(10.0f);
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
      Eigen::Matrix3f::Identity(), rot_y(0.18f), rot_y(-0.22f), rot_y(0.35f), rot_y(-0.12f),
  };
  const std::array<Eigen::Vector3f, 5> translations = {
      Eigen::Vector3f(0.0f, 0.0f, 0.0f),   Eigen::Vector3f(0.3f, -0.1f, 0.2f),
      Eigen::Vector3f(-0.4f, 0.2f, 0.5f),  Eigen::Vector3f(0.6f, 0.1f, -0.3f),
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
      const float conf_val = (std::abs(i - j) == 1) ? 10.0f : 4.0f;
      pr.view1.conf = conf_sparse(conf_val);
      pr.view2.conf = conf_sparse(conf_val);
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
    const float translation_direction_error =
        direction_error_deg(estimated_translation, translations[cam]);

    std::cout << "[TC3] " << camera_names[cam] << " rotation err=" << rotation_error
              << " translation_dir_err_deg=" << translation_direction_error << "\n";

    EXPECT_LT(rotation_error, 0.02f);
    EXPECT_LT(translation_direction_error, 0.05f);
  }
}

TEST(DustR3Alignment, TC4_RefineGlobalAlignmentImprovesSimilarityPose) {
  namespace d3r = stargazer::dust3r;

  std::mt19937 rng(13);
  std::uniform_real_distribution<float> xy(-1.5f, 1.5f);
  std::uniform_real_distribution<float> z(3.0f, 5.5f);

  std::vector<Eigen::Vector3f> world_points(192);
  for (auto& point : world_points) {
    point = Eigen::Vector3f(xy(rng), xy(rng), z(rng));
  }

  const auto rot_y = [](float angle) {
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    Eigen::Matrix3f R;
    R << c, 0.0f, s, 0.0f, 1.0f, 0.0f, -s, 0.0f, c;
    return R;
  };

  const std::vector<std::string> camera_names = {"camera0", "camera1", "camera2"};
  const std::array<Eigen::Matrix3f, 3> rotations = {
      Eigen::Matrix3f::Identity(),
      rot_y(0.22f),
      rot_y(-0.18f),
  };
  const std::array<Eigen::Vector3f, 3> translations = {
      Eigen::Vector3f(0.0f, 0.0f, 0.0f),
      Eigen::Vector3f(0.45f, -0.15f, 0.25f),
      Eigen::Vector3f(-0.35f, 0.1f, 0.55f),
  };
  const std::array<float, 3> scales = {1.0f, 1.8f, 0.75f};

  std::vector<std::vector<Eigen::Vector3f>> local_points(camera_names.size());
  for (size_t cam = 0; cam < camera_names.size(); ++cam) {
    for (const auto& point : world_points) {
      local_points[cam].push_back((1.0f / scales[cam]) *
                                  world_to_camera(rotations[cam], translations[cam], point));
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
      pr.view1.pts3d = fill_sparse_points(local_points[i]);
      pr.view2.pts3d = fill_sparse_points(local_points[i]);
      pr.view1.conf = conf_sparse(10.0f);
      pr.view2.conf = conf_sparse(10.0f);
      pairs.push_back(std::move(pr));
    }
  }

  std::unordered_map<std::string, d3r::aligned_pose> initial;
  initial["camera0"] = d3r::aligned_pose{
      to_glm(rotations[0]), glm::vec3(translations[0](0), translations[0](1), translations[0](2)),
      scales[0]};
  initial["camera1"] =
      d3r::aligned_pose{to_glm(rot_y(0.32f)), glm::vec3(0.65f, -0.05f, 0.45f), 1.35f};
  initial["camera2"] =
      d3r::aligned_pose{to_glm(rot_y(-0.05f)), glm::vec3(-0.05f, 0.2f, 0.15f), 1.1f};

  const auto refined = d3r::refine_global_alignment(camera_names, pairs, initial);

  const auto translation_error = [](const d3r::aligned_pose& pose, const Eigen::Vector3f& truth) {
    const Eigen::Vector3f estimate(pose.translation.x, pose.translation.y, pose.translation.z);
    return (estimate - truth).norm();
  };
  const auto rotation_error = [](const d3r::aligned_pose& pose, const Eigen::Matrix3f& truth) {
    Eigen::Matrix3f estimate;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        estimate(r, c) = pose.rotation[c][r];
      }
    }
    return (estimate - truth).norm();
  };

  const float before_cam1 = translation_error(initial.at("camera1"), translations[1]);
  const float after_cam1 = translation_error(refined.at("camera1"), translations[1]);
  const float before_cam2 = translation_error(initial.at("camera2"), translations[2]);
  const float after_cam2 = translation_error(refined.at("camera2"), translations[2]);
  const float before_rot_cam1 = rotation_error(initial.at("camera1"), rotations[1]);
  const float after_rot_cam1 = rotation_error(refined.at("camera1"), rotations[1]);
  const float before_rot_cam2 = rotation_error(initial.at("camera2"), rotations[2]);
  const float after_rot_cam2 = rotation_error(refined.at("camera2"), rotations[2]);

  std::cout << "[TC4] cam1 translation err before=" << before_cam1 << " after=" << after_cam1
            << "\n";
  std::cout << "[TC4] cam2 translation err before=" << before_cam2 << " after=" << after_cam2
            << "\n";
  std::cout << "[TC4] cam1 rotation err before=" << before_rot_cam1 << " after=" << after_rot_cam1
            << "\n";
  std::cout << "[TC4] cam2 rotation err before=" << before_rot_cam2 << " after=" << after_rot_cam2
            << "\n";

  EXPECT_LT(after_cam1, before_cam1);
  EXPECT_LT(after_cam2, before_cam2);
  EXPECT_LT(after_rot_cam1, before_rot_cam1);
  EXPECT_LT(after_rot_cam2, before_rot_cam2);
  EXPECT_NEAR(refined.at("camera1").scale, scales[1], 0.1f);
  EXPECT_NEAR(refined.at("camera2").scale, scales[2], 0.1f);
}
