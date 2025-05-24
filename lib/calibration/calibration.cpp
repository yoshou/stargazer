#include "calibration.hpp"

#include <spdlog/spdlog.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <map>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <vector>

#include "parameters.hpp"
#include "utils.hpp"

using namespace stargazer;

void zip_points(const std::vector<observed_points_t> &points1,
                const std::vector<observed_points_t> &points2,
                std::vector<std::pair<glm::vec2, glm::vec2>> &corresponding_points) {
  const auto size = std::min(points1.size(), points2.size());
  for (size_t i = 0; i < size; i++) {
    if (points1[i].points.size() != points2[i].points.size()) {
      continue;
    }
    for (size_t j = 0; j < points1[i].points.size(); j++) {
      corresponding_points.emplace_back(points1[i].points[j], points2[i].points[j]);
    }
  }
}

void zip_points(const std::vector<observed_points_t> &points1,
                const std::vector<observed_points_t> &points2,
                const std::vector<observed_points_t> &points3,
                std::vector<std::tuple<glm::vec2, glm::vec2, glm::vec2>> &corresponding_points) {
  const auto size = std::min(points1.size(), std::min(points2.size(), points3.size()));
  for (size_t i = 0; i < size; i++) {
    if (points1[i].points.size() == 0 || points2[i].points.size() == 0 ||
        points3[i].points.size() == 0) {
      continue;
    }
    assert(points1[i].points.size() == points2[i].points.size());
    assert(points1[i].points.size() == points3[i].points.size());
    for (size_t j = 0; j < points1[i].points.size(); j++) {
      corresponding_points.emplace_back(points1[i].points[j], points2[i].points[j],
                                        points3[i].points[j]);
    }
  }
}

float compute_diff_camera_angle(const glm::mat3 &r1, const glm::mat3 &r2) {
  const auto r = glm::transpose(r1) * r2;
  const auto r_quat = glm::quat_cast(r);
  return glm::angle(r_quat);
}

glm::mat4 estimate_relative_pose(
    const std::vector<std::pair<glm::vec2, glm::vec2>> &corresponding_points,
    const camera_t &base_camera, const camera_t &target_camera, bool use_lmeds) {
  std::vector<cv::Point2f> points1;
  std::vector<cv::Point2f> points2;
  for (const auto &[point1, point2] : corresponding_points) {
    points1.emplace_back(point1.x, point1.y);
    points2.emplace_back(point2.x, point2.y);
  }

  cv::Mat camera_matrix1, camera_matrix2;
  cv::Mat coeffs1, coeffs2;
  get_cv_intrinsic(base_camera.intrin, camera_matrix1, coeffs1);
  get_cv_intrinsic(target_camera.intrin, camera_matrix2, coeffs2);

  cv::Mat R, t;
  if (use_lmeds) {
    cv::Mat E;
    cv::recoverPose(points1, points2, camera_matrix1, coeffs1, camera_matrix2, coeffs2, E, R, t,
                    cv::LMEDS);
  } else {
    std::vector<cv::Point2f> norm_points1;
    std::vector<cv::Point2f> norm_points2;
    cv::undistortPoints(points1, norm_points1, camera_matrix1, coeffs1);
    cv::undistortPoints(points2, norm_points2, camera_matrix2, coeffs2);

    cv::Mat mask;
    const auto E = cv::findEssentialMat(norm_points1, norm_points2, 1.0, cv::Point2d(0.0, 0.0),
                                        cv::RANSAC, 0.99, 0.003, mask);

    cv::recoverPose(E, norm_points1, norm_points2, R, t, 1.0, cv::Point2d(0.0, 0.0), mask);
  }

  const auto r_mat = cv_to_glm_mat3x3(R);
  const auto t_vec = cv_to_glm_vec3(t);

  return glm::mat4(glm::vec4(r_mat[0], 0.f), glm::vec4(r_mat[1], 0.f), glm::vec4(r_mat[2], 0.f),
                   glm::vec4(t_vec, 1.f));
}

glm::mat4 estimate_pose(
    const std::vector<std::tuple<glm::vec2, glm::vec2, glm::vec2>> &corresponding_points,
    const camera_t &base_camera1, const camera_t &base_camera2, const camera_t &target_camera) {
  std::vector<cv::Point2d> points1;
  std::vector<cv::Point2d> points2;
  std::vector<cv::Point2d> points3;

  cv::Mat camera_matrix1, camera_matrix2, camera_matrix3;
  cv::Mat coeffs1, coeffs2, coeffs3;
  get_cv_intrinsic(base_camera1.intrin, camera_matrix1, coeffs1);
  get_cv_intrinsic(base_camera2.intrin, camera_matrix2, coeffs2);
  get_cv_intrinsic(target_camera.intrin, camera_matrix3, coeffs3);

  std::vector<cv::Point2d> norm_points1;
  std::vector<cv::Point2d> norm_points2;
  std::vector<cv::Point2d> norm_points3;
  for (const auto &[point1, point2, point3] : corresponding_points) {
    points1.emplace_back(point1.x, point1.y);
    points2.emplace_back(point2.x, point2.y);
    points3.emplace_back(point3.x, point3.y);
  }
  cv::undistortPoints(points1, norm_points1, camera_matrix1, coeffs1);
  cv::undistortPoints(points2, norm_points2, camera_matrix2, coeffs2);
  cv::undistortPoints(points3, norm_points3, camera_matrix3, coeffs3);

  cv::Mat point4d;
  cv::triangulatePoints(glm_to_cv_mat3x4(base_camera1.extrin.transform_matrix()),
                        glm_to_cv_mat3x4(base_camera2.extrin.transform_matrix()), norm_points1,
                        norm_points2, point4d);

  std::vector<cv::Point3d> point3d;
  for (size_t i = 0; i < static_cast<size_t>(point4d.cols); i++) {
    point3d.emplace_back(point4d.at<double>(0, i) / point4d.at<double>(3, i),
                         point4d.at<double>(1, i) / point4d.at<double>(3, i),
                         point4d.at<double>(2, i) / point4d.at<double>(3, i));
  }

  // Check reprojection error
  {
    const auto Rt1 = glm_to_cv_mat3x4(base_camera1.extrin.transform_matrix());
    const auto Rt2 = glm_to_cv_mat3x4(base_camera2.extrin.transform_matrix());

    cv::Mat R1 = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat tvec1 = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat rvec1;
    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 3; j++) {
        R1.at<double>(i, j) = Rt1.at<double>(i, j);
      }
    }
    cv::Rodrigues(R1, rvec1);
    tvec1.at<double>(0, 0) = Rt1.at<double>(0, 3);
    tvec1.at<double>(1, 0) = Rt1.at<double>(1, 3);
    tvec1.at<double>(2, 0) = Rt1.at<double>(2, 3);

    cv::Mat R2 = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat tvec2 = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat rvec2;
    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 3; j++) {
        R2.at<double>(i, j) = Rt2.at<double>(i, j);
      }
    }
    cv::Rodrigues(R2, rvec2);
    tvec2.at<double>(0, 0) = Rt2.at<double>(0, 3);
    tvec2.at<double>(1, 0) = Rt2.at<double>(1, 3);
    tvec2.at<double>(2, 0) = Rt2.at<double>(2, 3);

    std::vector<cv::Point2d> proj_points1;
    std::vector<cv::Point2d> proj_points2;
    cv::projectPoints(point3d, rvec1, tvec1, camera_matrix1, coeffs1, proj_points1);
    cv::projectPoints(point3d, rvec2, tvec2, camera_matrix2, coeffs2, proj_points2);

    double error1 = 0.0;
    double error2 = 0.0;
    for (size_t i = 0; i < point3d.size(); i++) {
      error1 += cv::norm(points1[i] - proj_points1[i]);
      error2 += cv::norm(points2[i] - proj_points2[i]);
    }

    error1 /= point3d.size();
    error2 /= point3d.size();

    spdlog::info("reprojection error1: {}", error1);
    spdlog::info("reprojection error2: {}", error2);
  }

  cv::Mat r, t;
  constexpr auto use_extrinsic_guess = false;
  constexpr auto iterations_count = 100;
  constexpr auto reprojection_error = 8.0;
  constexpr auto confidence = 0.99;
  std::vector<int> inliers;
  cv::solvePnPRansac(point3d, points3, camera_matrix3, coeffs3, r, t, use_extrinsic_guess,
                     iterations_count, reprojection_error, confidence, inliers);

  cv::Mat R;
  cv::Rodrigues(r, R);

  // Check reprojection error
  {
    std::vector<cv::Point2d> proj_points3;
    cv::projectPoints(point3d, r, t, camera_matrix3, coeffs3, proj_points3);

    double error3 = 0.0;
    for (int i : inliers) {
      error3 += cv::norm(points3[i] - proj_points3[i]);
    }

    error3 /= inliers.size();

    spdlog::info("num matches: {}", point3d.size());
    spdlog::info("num inliers: {}", inliers.size());
    spdlog::info("reprojection error3: {}", error3);
  }

  const auto r_mat = cv_to_glm_mat3x3(R);
  const auto t_vec = cv_to_glm_vec3(t);

  return glm::mat4(glm::vec4(r_mat[0], 0.f), glm::vec4(r_mat[1], 0.f), glm::vec4(r_mat[2], 0.f),
                   glm::vec4(t_vec, 1.f));
}
