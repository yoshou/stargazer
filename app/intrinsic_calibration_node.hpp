#pragma once

#include <algorithm>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <vector>

#include "calibration_target.hpp"
#include "graph_proc.h"
#include "graph_proc_cv.h"
#include "graph_proc_img.h"
#include "messages.hpp"
#include "parameters.hpp"
#include "point_data.hpp"

namespace stargazer {

using namespace coalsack;

static inline std::vector<size_t> create_random_indices(size_t size) {
  std::vector<size_t> data(size);
  for (size_t i = 0; i < size; i++) {
    data[i] = i;
  }

  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());

  std::shuffle(data.begin(), data.end(), engine);

  return data;
}

class intrinsic_calibration_node : public graph_node {
  std::shared_ptr<graph_edge> output;

  std::vector<std::vector<point_data>> frames;
  camera_t initial_camera;
  camera_t calibrated_camera;
  double rms = 0.0;

 public:
  intrinsic_calibration_node() : graph_node(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "intrinsic_calibration"; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(initial_camera);
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (input_name == "calibrate") {
      if (auto camera_msg = std::dynamic_pointer_cast<camera_message>(message)) {
        initial_camera = camera_msg->get_camera();
      }

      calibrate();

      auto camera_msg = std::make_shared<camera_message>(calibrated_camera);
      output->send(camera_msg);
    }

    if (auto points_msg = std::dynamic_pointer_cast<float2_list_message>(message)) {
      std::vector<point_data> points;
      for (const auto& pt : points_msg->get_data()) {
        points.push_back(point_data{glm::vec2(pt.x, pt.y), 0, 0});
      }
      push_frame(points);
    }

    if (auto image_msg = std::dynamic_pointer_cast<image_message>(message)) {
      const auto& image = image_msg->get_image();
      cv::Mat img(image.get_height(), image.get_width(), convert_to_cv_type(image.get_format()),
                  (void*)image.get_data(), image.get_stride());
      push_frame(img);
    }
  }

  double get_rms() const { return rms; }

  void set_initial_camera(const camera_t& camera) { initial_camera = camera; }

  const camera_t& get_calibrated_camera() const { return calibrated_camera; }

  size_t get_num_frames() const { return frames.size(); }

  void push_frame(const std::vector<point_data>& frame) { frames.push_back(frame); }

  void push_frame(const cv::Mat& frame) {
    std::vector<cv::Point2f> board;
    if (detect_calibration_board(frame, board)) {
      std::vector<point_data> points;
      for (const auto& point : board) {
        points.push_back(point_data{glm::vec2(point.x, point.y), 0, 0});
      }
      push_frame(points);
    }
  }

  void calibrate() {
    const auto image_width = initial_camera.width;
    const auto image_height = initial_camera.height;
    const auto square_size = cv::Size2f(2.41, 2.4);  // TODO: Define as config
    const auto board_size = cv::Size(10, 7);         // TODO: Define as config
    const auto image_size = cv::Size(image_width, image_height);

    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points;

    std::vector<cv::Point3f> object_point;
    calc_board_corner_positions(board_size, square_size, object_point);

    const auto max_num_frames = 100;  // TODO: Define as config

    const auto frame_indices =
        create_random_indices(std::min(frames.size(), static_cast<size_t>(max_num_frames)));

    for (const auto& frame_index : frame_indices) {
      const auto& frame = frames.at(frame_index);

      object_points.push_back(object_point);

      std::vector<cv::Point2f> image_point;
      for (const auto& point : frame) {
        image_point.push_back(cv::Point2f(point.point.x, point.point.y));
      }

      image_points.push_back(image_point);
    }

    std::vector<cv::Mat> rvecs, tvecs;
    cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);

    rms = cv::calibrateCamera(object_points, image_points, image_size, camera_matrix, dist_coeffs,
                              rvecs, tvecs);

    calibrated_camera.intrin.fx = camera_matrix.at<double>(0, 0);
    calibrated_camera.intrin.fy = camera_matrix.at<double>(1, 1);
    calibrated_camera.intrin.cx = camera_matrix.at<double>(0, 2);
    calibrated_camera.intrin.cy = camera_matrix.at<double>(1, 2);
    calibrated_camera.intrin.coeffs[0] = dist_coeffs.at<double>(0);
    calibrated_camera.intrin.coeffs[1] = dist_coeffs.at<double>(1);
    calibrated_camera.intrin.coeffs[2] = dist_coeffs.at<double>(2);
    calibrated_camera.intrin.coeffs[3] = dist_coeffs.at<double>(3);
    calibrated_camera.intrin.coeffs[4] = dist_coeffs.at<double>(4);
    calibrated_camera.width = image_width;
    calibrated_camera.height = image_height;
  }
};

}  // namespace stargazer

COALSACK_REGISTER_NODE(stargazer::intrinsic_calibration_node, coalsack::graph_node)
