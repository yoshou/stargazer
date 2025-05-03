#include "triangulation.hpp"

#include <Eigen/Core>
#include <glm/glm.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

namespace stargazer::reconstruction {
glm::vec3 triangulate(const glm::vec2 pt1, const glm::vec2 pt2, const camera_t &cm1,
                      const camera_t &cm2) {
  cv::Mat camera_mat1, camera_mat2;
  cv::Mat dist_coeffs1, dist_coeffs2;
  get_cv_intrinsic(cm1.intrin, camera_mat1, dist_coeffs1);
  get_cv_intrinsic(cm2.intrin, camera_mat2, dist_coeffs2);

  std::vector<cv::Point2d> pts1, pts2;
  pts1.push_back(cv::Point2d(pt1.x, pt1.y));
  pts2.push_back(cv::Point2d(pt2.x, pt2.y));

  std::vector<cv::Point2d> undistort_pts1, undistort_pts2;
  cv::undistortPoints(pts1, undistort_pts1, camera_mat1, dist_coeffs1);
  cv::undistortPoints(pts2, undistort_pts2, camera_mat2, dist_coeffs2);

  cv::Mat proj1 = glm_to_cv_mat3x4(cm1.extrin.transform_matrix());
  cv::Mat proj2 = glm_to_cv_mat3x4(cm2.extrin.transform_matrix());

  cv::Mat output;
  cv::triangulatePoints(proj1, proj2, undistort_pts1, undistort_pts2, output);

  return glm::vec3(output.at<double>(0, 0) / output.at<double>(3, 0),
                   output.at<double>(1, 0) / output.at<double>(3, 0),
                   output.at<double>(2, 0) / output.at<double>(3, 0));
}

glm::vec3 triangulate_undistorted(const glm::vec2 pt1, const glm::vec2 pt2, const camera_t &cm1,
                                  const camera_t &cm2) {
  cv::Mat camera_mat1, camera_mat2;
  cv::Mat dist_coeffs1, dist_coeffs2;
  get_cv_intrinsic(cm1.intrin, camera_mat1, dist_coeffs1);
  get_cv_intrinsic(cm2.intrin, camera_mat2, dist_coeffs2);

  std::vector<cv::Point2d> pts1, pts2;
  pts1.push_back(cv::Point2d(pt1.x, pt1.y));
  pts2.push_back(cv::Point2d(pt2.x, pt2.y));

  cv::Mat proj1 = glm_to_cv_mat3x4(cm1.extrin.transform_matrix());
  cv::Mat proj2 = glm_to_cv_mat3x4(cm2.extrin.transform_matrix());

  cv::Mat output;
  cv::triangulatePoints(proj1, proj2, pts1, pts2, output);

  return glm::vec3(output.at<double>(0, 0) / output.at<double>(3, 0),
                   output.at<double>(1, 0) / output.at<double>(3, 0),
                   output.at<double>(2, 0) / output.at<double>(3, 0));
}

glm::vec3 triangulate(const std::vector<glm::vec2> &points, const std::vector<camera_t> &cameras) {
  assert(points.size() == cameras.size());

  const auto nviews = points.size();
  cv::Mat_<double> design = cv::Mat_<double>::zeros(3 * nviews, 4 + nviews);
  for (size_t i = 0; i < nviews; ++i) {
    cv::Mat camera_mat;
    cv::Mat dist_coeffs;
    get_cv_intrinsic(cameras[i].intrin, camera_mat, dist_coeffs);

    std::vector<cv::Point2d> pt = {cv::Point2d(points[i].x, points[i].y)};
    std::vector<cv::Point2d> undistort_pt;
    cv::undistortPoints(pt, undistort_pt, camera_mat, dist_coeffs);

    const auto &proj = cameras[i].extrin.transform_matrix();

    for (size_t m = 0; m < 3; ++m) {
      for (size_t n = 0; n < 4; ++n) {
        design(3 * i + m, n) = -proj[n][m];
      }
    }
    design(3 * i + 0, 4 + i) = undistort_pt[0].x;
    design(3 * i + 1, 4 + i) = undistort_pt[0].y;
    design(3 * i + 2, 4 + i) = 1.0;
  }

  cv::Mat x_and_alphas;
  cv::SVD::solveZ(design, x_and_alphas);

  const glm::vec3 point3d(x_and_alphas.at<double>(0, 0) / x_and_alphas.at<double>(0, 3),
                          x_and_alphas.at<double>(0, 1) / x_and_alphas.at<double>(0, 3),
                          x_and_alphas.at<double>(0, 2) / x_and_alphas.at<double>(0, 3));

  return point3d;
}
}  // namespace stargazer::reconstruction
