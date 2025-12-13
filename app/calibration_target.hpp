#pragma once

#include <glm/glm.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "calibration.hpp"
#include "parameters.hpp"
#include "point_data.hpp"

namespace stargazer {

class calibration_target {
 public:
  virtual std::vector<glm::vec2> detect_points(const std::vector<point_data>& markers) = 0;
  virtual ~calibration_target() = default;
};

template <class T, class F>
void combination(const std::vector<T>& seed, int target_size, F callback) {
  std::vector<int> indices(target_size);
  const int seed_size = seed.size();
  int start_index = 0;
  int size = 0;

  while (size >= 0) {
    for (int i = start_index; i < seed_size; ++i) {
      indices[size++] = i;
      if (size == target_size) {
        std::vector<T> comb(target_size);
        for (int x = 0; x < target_size; ++x) {
          comb[x] = seed[indices[x]];
        }
        if (callback(comb)) return;
        break;
      }
    }
    --size;
    if (size < 0) break;
    start_index = indices[size] + 1;
  }
}

enum class calibration_pattern {
  CHESSBOARD,
  CIRCLES_GRID,
  ASYMMETRIC_CIRCLES_GRID,
};

static inline void calc_board_corner_positions(
    cv::Size board_size, cv::Size2f square_size, std::vector<cv::Point3f>& corners,
    const calibration_pattern pattern_type = calibration_pattern::CHESSBOARD) {
  corners.clear();
  switch (pattern_type) {
    case calibration_pattern::CHESSBOARD:
    case calibration_pattern::CIRCLES_GRID:
      for (int i = 0; i < board_size.height; ++i) {
        for (int j = 0; j < board_size.width; ++j) {
          corners.push_back(cv::Point3f(j * square_size.width, i * square_size.height, 0));
        }
      }
      break;
    case calibration_pattern::ASYMMETRIC_CIRCLES_GRID:
      for (int i = 0; i < board_size.height; i++) {
        for (int j = 0; j < board_size.width; j++) {
          corners.push_back(
              cv::Point3f((2 * j + i % 2) * square_size.width, i * square_size.height, 0));
        }
      }
      break;
    default:
      break;
  }
}

static inline bool detect_calibration_board(
    cv::Mat frame, std::vector<cv::Point2f>& points,
    const calibration_pattern pattern_type = calibration_pattern::CHESSBOARD) {
  if (frame.empty()) {
    return false;
  }
  constexpr auto use_fisheye = false;

  cv::Size board_size;
  switch (pattern_type) {
    case calibration_pattern::CHESSBOARD:
      board_size = cv::Size(10, 7);
      break;
    case calibration_pattern::CIRCLES_GRID:
      board_size = cv::Size(10, 7);
      break;
    case calibration_pattern::ASYMMETRIC_CIRCLES_GRID:
      board_size = cv::Size(4, 11);
      break;
  }
  const auto win_size = 5;

  int chessboard_flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
  if (!use_fisheye) {
    chessboard_flags |= cv::CALIB_CB_FAST_CHECK;
  }

  bool found = false;
  switch (pattern_type) {
    case calibration_pattern::CHESSBOARD:
      found = cv::findChessboardCorners(frame, board_size, points, chessboard_flags);
      break;
    case calibration_pattern::CIRCLES_GRID:
      found = cv::findCirclesGrid(frame, board_size, points);
      break;
    case calibration_pattern::ASYMMETRIC_CIRCLES_GRID: {
      auto params = cv::SimpleBlobDetector::Params();
      params.minDistBetweenBlobs = 3;
      auto detector = cv::SimpleBlobDetector::create(params);
      found =
          cv::findCirclesGrid(frame, board_size, points, cv::CALIB_CB_ASYMMETRIC_GRID, detector);
    } break;
    default:
      found = false;
      break;
  }

  if (found) {
    if (pattern_type == calibration_pattern::CHESSBOARD) {
      cv::Mat gray;
      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
      cv::cornerSubPix(
          gray, points, cv::Size(win_size, win_size), cv::Size(-1, -1),
          cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.0001));
    }
  }

  return found;
}

class three_point_bar_calibration_target : public calibration_target {
  template <typename T>
  static void sort(T& a, T& b, T& c) {
    if (a > b) std::swap(a, b);
    if (b > c) std::swap(b, c);
    if (a > b) std::swap(a, b);
  }

 public:
  virtual std::vector<glm::vec2> detect_points(const std::vector<point_data>& markers) override {
    std::vector<glm::vec2> points;
    combination(markers, 3, [&](const std::vector<point_data>& target_markers) {
      auto x1 = target_markers[0].point.x;
      auto y1 = target_markers[0].point.y;
      auto x2 = target_markers[1].point.x;
      auto y2 = target_markers[1].point.y;
      auto x3 = target_markers[2].point.x;
      auto y3 = target_markers[2].point.y;

      sort(x1, x2, x3);
      sort(y1, y2, y3);

      if (std::abs(x2 - x1) > std::abs(x2 - x3)) {
        std::swap(x1, x3);
      }
      if (std::abs(y2 - y1) > std::abs(y2 - y3)) {
        std::swap(y1, y3);
      }

      const auto la = (y3 - y1) / (x3 - x1);
      const auto lb = -1;
      const auto lc = y1 - la * x1;

      const auto d = std::abs(la * x2 + lb * y2 + lc) / std::sqrt(la * la + lb * lb);

      if (d < 1.0) {
        points.push_back(glm::vec2(x2, y2));
        return true;
      }

      return false;
    });

    return points;
  }
};

class pattern_board_calibration_target : public calibration_target {
  std::vector<cv::Point3f> object_points;
  camera_t camera;

 public:
  pattern_board_calibration_target(const std::vector<cv::Point3f>& object_points,
                                   const camera_t& camera)
      : object_points(object_points), camera(camera) {}

  virtual std::vector<glm::vec2> detect_points(const std::vector<point_data>& markers) override {
    if (markers.size() == object_points.size()) {
      std::vector<cv::Point2f> image_points;

      std::transform(markers.begin(), markers.end(), std::back_inserter(image_points),
                     [](const auto& pt) { return cv::Point2f(pt.point.x, pt.point.y); });

      cv::Mat rvec, tvec;

      cv::Mat camera_matrix;
      cv::Mat dist_coeffs;
      get_cv_intrinsic(camera.intrin, camera_matrix, dist_coeffs);

      cv::solvePnP(object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec);

      std::vector<cv::Point2f> proj_points;
      cv::projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs, proj_points);

      double error = 0.0;
      for (size_t i = 0; i < object_points.size(); i++) {
        error += cv::norm(image_points[i] - proj_points[i]);
      }

      if (error / object_points.size() > 4.0) {
        return {};
      }

      return {glm::vec2(proj_points[0].x, proj_points[0].y)};
    }
    return {};
  }
};

}  // namespace stargazer
