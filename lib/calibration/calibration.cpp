#include "calibration.hpp"

#include <spdlog/spdlog.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <map>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <vector>

#include "parameters.hpp"
#include "triangulation.hpp"
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

const std::vector<observed_points_t> observed_points_frames::get_observed_points(
    std::string name) const {
  static std::vector<observed_points_t> empty;
  if (observed_frames.find(name) == observed_frames.end()) {
    return empty;
  }
  std::vector<observed_points_t> observed_points;
  {
    std::lock_guard<std::mutex> lock(frames_mtx);
    observed_points = observed_frames.at(name);
  }
  return observed_points;
}

const observed_points_t observed_points_frames::get_observed_point(std::string name,
                                                                   size_t frame) const {
  observed_points_t observed_point;
  if (observed_frames.find(name) == observed_frames.end()) {
    return observed_point;
  }
  {
    std::lock_guard<std::mutex> lock(frames_mtx);
    observed_point = observed_frames.at(name).at(frame);
  }
  return observed_point;
}

size_t observed_points_frames::get_num_frames() const { return timestamp_to_index.size(); }

size_t observed_points_frames::get_num_points(std::string name) const {
  if (num_points.find(name) == num_points.end()) {
    return 0;
  }
  return num_points.at(name);
}

void observed_points_frames::add_frame_points(uint32_t timestamp, std::string name,
                                              const std::vector<glm::vec2> &points) {
  if (timestamp_to_index.find(timestamp) == timestamp_to_index.end()) {
    timestamp_to_index.insert(std::make_pair(timestamp, timestamp_to_index.size()));
  }

  const auto index = timestamp_to_index.at(timestamp);

  if (num_points.find(name) == num_points.end()) {
    num_points.insert(std::make_pair(name, 0));
  }

  if (camera_name_to_index.find(name) == camera_name_to_index.end()) {
    camera_name_to_index.insert(std::make_pair(name, camera_name_to_index.size()));
  }

  if (observed_frames.find(name) == observed_frames.end()) {
    if (observed_frames.empty()) {
      std::lock_guard lock(frames_mtx);
      observed_frames.insert(std::make_pair(name, std::vector<observed_points_t>()));
    } else {
      observed_points_t obs = {};
      obs.camera_idx = camera_name_to_index.at(name);
      std::lock_guard lock(frames_mtx);
      observed_frames.insert(std::make_pair(
          name, std::vector<observed_points_t>(observed_frames.begin()->second.size(), obs)));
    }
  }

  for (auto &[name, observed_points] : observed_frames) {
    observed_points.resize(timestamp_to_index.size());
  }

  observed_points_t obs = {};
  obs.camera_idx = camera_name_to_index.at(name);
  for (const auto &pt : points) {
    obs.points.emplace_back(pt);
  }

  {
    std::lock_guard lock(frames_mtx);
    observed_frames[name][index] = obs;
  }

  if (obs.points.size() > 0) {
    num_points.at(name) += 1;
  }
}

bool initialize_cameras(std::unordered_map<std::string, camera_t> &cameras,
                        const std::vector<std::string> &camera_names,
                        const observed_points_frames &observed_frames) {
  std::string base_camera_name1;
  std::string base_camera_name2;
  glm::mat4 base_camera_pose1(1.0f);
  glm::mat4 base_camera_pose2(1.0f);

  bool found_base_pair = false;
  const float min_base_angle = 15.0f;

  for (const auto &camera_name1 : camera_names) {
    if (found_base_pair) {
      break;
    }
    for (const auto &camera_name2 : camera_names) {
      if (camera_name1 == camera_name2) {
        continue;
      }
      if (found_base_pair) {
        break;
      }

      std::vector<std::pair<glm::vec2, glm::vec2>> corresponding_points;
      zip_points(observed_frames.get_observed_points(camera_name1),
                 observed_frames.get_observed_points(camera_name2), corresponding_points);

      const auto pose1 = glm::mat4(1.0);
      const auto pose2 = estimate_relative_pose(corresponding_points, cameras.at(camera_name1),
                                                cameras.at(camera_name2));

      const auto angle = compute_diff_camera_angle(glm::mat3(pose1), glm::mat3(pose2));

      if (glm::degrees(angle) > min_base_angle) {
        base_camera_name1 = camera_name1;
        base_camera_name2 = camera_name2;
        base_camera_pose1 = pose1;
        base_camera_pose2 = pose2;
        found_base_pair = true;
        break;
      }
    }
  }

  if (!found_base_pair) {
    spdlog::error("Failed to find base camera pair");
    return false;
  }

  cameras[base_camera_name1].extrin.rotation = glm::mat3(base_camera_pose1);
  cameras[base_camera_name1].extrin.translation = glm::vec3(base_camera_pose1[3]);
  cameras[base_camera_name2].extrin.rotation = glm::mat3(base_camera_pose2);
  cameras[base_camera_name2].extrin.translation = glm::vec3(base_camera_pose2[3]);

  std::vector<std::string> processed_cameras = {base_camera_name1, base_camera_name2};
  for (const auto &camera_name : camera_names) {
    if (camera_name == base_camera_name1) {
      continue;
    }
    if (camera_name == base_camera_name2) {
      continue;
    }

    std::vector<std::tuple<glm::vec2, glm::vec2, glm::vec2>> corresponding_points;
    zip_points(observed_frames.get_observed_points(base_camera_name1),
               observed_frames.get_observed_points(base_camera_name2),
               observed_frames.get_observed_points(camera_name), corresponding_points);

    if (corresponding_points.size() < 7) {
      continue;
    }

    spdlog::info("Estimate camera pose: {}", camera_name);

    const auto pose = estimate_pose(corresponding_points, cameras.at(base_camera_name1),
                                    cameras.at(base_camera_name2), cameras.at(camera_name));
    cameras[camera_name].extrin.rotation = glm::mat3(pose);
    cameras[camera_name].extrin.translation = glm::vec3(pose[3]);

    processed_cameras.push_back(camera_name);
  }

  if (processed_cameras.size() != camera_names.size()) {
    spdlog::error("Failed to calibrate all cameras");
    return false;
  }

  return true;
}

void prepare_bundle_adjustment(const std::vector<std::string> &camera_names,
                               const std::unordered_map<std::string, camera_t> &cameras,
                               const observed_points_frames &observed_frames,
                               stargazer::calibration::bundle_adjust_data &ba_data) {
  for (const auto &camera_name : camera_names) {
    const auto &camera = cameras.at(camera_name);
    cv::Mat rot_vec;
    cv::Rodrigues(glm_to_cv_mat3(camera.extrin.transform_matrix()), rot_vec);
    const auto trans_vec = camera.extrin.translation;

    std::vector<double> camera_params;
    for (size_t j = 0; j < 3; j++) {
      camera_params.push_back(rot_vec.at<float>(j));
    }
    for (size_t j = 0; j < 3; j++) {
      camera_params.push_back(trans_vec[j]);
    }
    camera_params.push_back(camera.intrin.fx);
    camera_params.push_back(camera.intrin.fy);
    camera_params.push_back(camera.intrin.cx);
    camera_params.push_back(camera.intrin.cy);
    camera_params.push_back(camera.intrin.coeffs[0]);
    camera_params.push_back(camera.intrin.coeffs[1]);
    camera_params.push_back(camera.intrin.coeffs[4]);
    camera_params.push_back(camera.intrin.coeffs[2]);
    camera_params.push_back(camera.intrin.coeffs[3]);
    for (size_t j = 0; j < 3; j++) {
      camera_params.push_back(0.0);
    }

    ba_data.add_camera(camera_params.data());
  }

  {
    constexpr auto reproj_error_threshold = 2.0;

    std::vector<camera_t> camera_list;
    std::map<std::string, size_t> camera_name_to_index;
    for (size_t i = 0; i < camera_names.size(); i++) {
      camera_list.push_back(cameras.at(camera_names[i]));
      camera_name_to_index.insert(std::make_pair(camera_names[i], i));
    }
    size_t point_idx = 0;
    for (size_t f = 0; f < observed_frames.get_num_frames(); f++) {
      std::vector<std::vector<glm::vec2>> pts;
      std::vector<size_t> camera_idxs;
      for (const auto &camera_name : camera_names) {
        const auto point = observed_frames.get_observed_point(camera_name, f);
        if (point.points.size() == 0) {
          continue;
        }
        std::vector<glm::vec2> pt;
        std::copy(point.points.begin(), point.points.end(), std::back_inserter(pt));

        pts.push_back(pt);
        camera_idxs.push_back(camera_name_to_index.at(camera_name));
      }

      if (pts.size() < 2) {
        continue;
      }

      std::vector<camera_t> view_cameras;
      for (const auto i : camera_idxs) {
        view_cameras.push_back(camera_list[i]);
      }

      const auto num_points =
          std::min_element(pts.begin(), pts.end(), [](const auto &a, const auto &b) {
            return a.size() < b.size();
          })->size();

      std::vector<glm::vec3> point3ds(num_points);

      for (size_t i = 0; i < num_points; i++) {
        std::vector<glm::vec2> point2ds;
        for (size_t j = 0; j < pts.size(); j++) {
          point2ds.push_back(pts[j][i]);
        }

        const auto point3d = stargazer::reconstruction::triangulate(point2ds, view_cameras);
        point3ds[i] = point3d;
      }

      for (const auto &point3d : point3ds) {
        std::vector<double> point_params;
        for (size_t i = 0; i < 3; i++) {
          point_params.push_back(point3d[i]);
        }
        ba_data.add_point(point_params.data());
      }

      for (size_t i = 0; i < pts.size(); i++) {
        for (size_t j = 0; j < pts[i].size(); j++) {
          const auto &pt = pts[i][j];
          const auto proj_pt = project(camera_list[camera_idxs[i]], point3ds[j]);

          if (glm::distance(pt, proj_pt) > reproj_error_threshold) {
            continue;
          }

          std::array<double, 2> observation;
          for (size_t k = 0; k < 2; k++) {
            observation[k] = pt[k];
          }
          ba_data.add_observation(observation.data(), camera_idxs[i], point_idx + j);
        }
      }

      point_idx += point3ds.size();
    }
  }
}
