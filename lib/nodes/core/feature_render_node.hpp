/// @file feature_render_node.hpp
/// @brief Reconstruction feature heatmap rendering node.
/// @ingroup core_nodes
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <string>

#include "coalsack/core/graph_proc_registry.h"
#include "coalsack/image/image.h"
#include "messages.hpp"

namespace stargazer {

/// @brief Renders reconstruction feature heatmaps as colour-coded images.
/// @details Receives `reconstruction_result_message` frames and maps the per-pixel
///          confidence or depth values to an RGB heatmap.  Output size is configured
///          via `set_camera_name`, `set_width`, and `set_height`.
///
/// @par Inputs
/// - @b "default" — `reconstruction_result_message` — per-pixel feature data
///
/// @par Outputs
/// - @b "default" — `image_message` — colour-mapped heatmap image
///
/// @par Properties
/// - `camera_name` (`std::string`, default `""`) — identifier for the source camera
/// - `width`  (`int`, default `0`) — output image width in pixels
/// - `height` (`int`, default `0`) — output image height in pixels
///
/// @see contrail_render_node, reconstruction_result_markers_node
class feature_render_node : public coalsack::graph_node {
  coalsack::graph_edge_ptr output;
  std::string camera_name;
  int width = 0;
  int height = 0;
  mutable std::mutex image_mtx;
  std::shared_ptr<coalsack::image> current_image;
  std::atomic<std::int64_t> received_count;

  static cv::Scalar color_from_index(size_t index) {
    static const cv::Scalar colors[] = {cv::Scalar(255, 128, 0),  cv::Scalar(0, 200, 255),
                                        cv::Scalar(80, 255, 80),  cv::Scalar(255, 80, 180),
                                        cv::Scalar(180, 80, 255), cv::Scalar(255, 255, 80)};
    return colors[index % (sizeof(colors) / sizeof(colors[0]))];
  }

 public:
  feature_render_node()
      : graph_node(),
        output(std::make_shared<coalsack::graph_edge>(this)),
        camera_name(),
        width(0),
        height(0),
        image_mtx(),
        current_image(),
        received_count(0) {
    set_output(output);
  }

  void set_camera_name(std::string value) { camera_name = std::move(value); }
  void set_width(int value) { width = value; }
  void set_height(int value) { height = value; }

  virtual std::string get_proc_name() const override { return "feature_render"; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(camera_name, width, height);
  }

  virtual std::optional<coalsack::property_value> get_property(
      const std::string& key) const override {
    if (key == "received") {
      return received_count.load();
    }
    if (key == "image") {
      std::lock_guard lock(image_mtx);
      if (current_image) {
        return current_image;
      }
    }
    return std::nullopt;
  }

  virtual void process([[maybe_unused]] std::string input_name,
                       coalsack::graph_message_ptr message) override {
    const auto result_msg = std::dynamic_pointer_cast<reconstruction_result_message>(message);
    if (!result_msg) {
      output->send(message);
      return;
    }

    if (const auto heatmap_it = result_msg->get_result().heatmaps.find(camera_name);
        heatmap_it != result_msg->get_result().heatmaps.end()) {
      const auto& heatmap_img = heatmap_it->second;

      int output_width = width;
      int output_height = height;
      const auto& cameras = result_msg->get_cameras();
      if (const auto camera_it = cameras.find(camera_name); camera_it != cameras.end()) {
        if (output_width <= 0) output_width = static_cast<int>(camera_it->second.width);
        if (output_height <= 0) output_height = static_cast<int>(camera_it->second.height);
      }
      if (output_width <= 0) output_width = 960;
      if (output_height <= 0) output_height = 540;

      cv::Mat gray(static_cast<int>(heatmap_img.get_height()),
                   static_cast<int>(heatmap_img.get_width()), CV_8UC1,
                   const_cast<uint8_t*>(heatmap_img.get_data()),
                   static_cast<size_t>(heatmap_img.get_stride()));
      cv::Mat bgr;
      cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
      cv::resize(bgr, bgr, cv::Size(output_width, output_height));

      coalsack::image rendered(static_cast<std::uint32_t>(bgr.cols),
                               static_cast<std::uint32_t>(bgr.rows),
                               static_cast<std::uint32_t>(bgr.elemSize()),
                               static_cast<std::uint32_t>(bgr.step), bgr.data);
      rendered.set_format(coalsack::image_format::B8G8R8_UINT);
      {
        std::lock_guard lock(image_mtx);
        current_image = std::make_shared<coalsack::image>(rendered);
      }
      received_count.fetch_add(1);
      output->send(message);
      return;
    }

    int image_width = width;
    int image_height = height;

    const auto& cameras = result_msg->get_cameras();
    if (const auto camera_it = cameras.find(camera_name); camera_it != cameras.end()) {
      if (image_width <= 0) {
        image_width = static_cast<int>(camera_it->second.width);
      }
      if (image_height <= 0) {
        image_height = static_cast<int>(camera_it->second.height);
      }
    }
    if (image_width <= 0) {
      image_width = 960;
    }
    if (image_height <= 0) {
      image_height = 540;
    }

    cv::Mat rendered(image_height, image_width, CV_8UC3, cv::Scalar::all(0));

    const auto& result = result_msg->get_result();
    const auto view_it = std::find_if(result.views.begin(), result.views.end(),
                                      [&](const auto& view) { return view.name == camera_name; });
    if (view_it != result.views.end()) {
      for (size_t detection_index = 0; detection_index < view_it->detections.size();
           ++detection_index) {
        const auto& detection = view_it->detections[detection_index];
        const auto color = color_from_index(detection_index);

        cv::rectangle(rendered,
                      cv::Point(static_cast<int>(std::round(detection.bbox.left)),
                                static_cast<int>(std::round(detection.bbox.top))),
                      cv::Point(static_cast<int>(std::round(detection.bbox.right)),
                                static_cast<int>(std::round(detection.bbox.bottom))),
                      color, 2);

        for (size_t keypoint_index = 0; keypoint_index < detection.keypoints.size();
             ++keypoint_index) {
          const auto& keypoint = detection.keypoints[keypoint_index];
          cv::circle(rendered,
                     cv::Point(static_cast<int>(std::round(keypoint.x)),
                               static_cast<int>(std::round(keypoint.y))),
                     4, color, cv::FILLED);
        }
      }
    }

    coalsack::image image(static_cast<std::uint32_t>(rendered.cols),
                          static_cast<std::uint32_t>(rendered.rows),
                          static_cast<std::uint32_t>(rendered.elemSize()),
                          static_cast<std::uint32_t>(rendered.step), rendered.data);
    image.set_format(coalsack::image_format::B8G8R8_UINT);

    {
      std::lock_guard lock(image_mtx);
      current_image = std::make_shared<coalsack::image>(image);
    }
    received_count.fetch_add(1);

    output->send(message);
  }
};

}  // namespace stargazer

COALSACK_REGISTER_NODE(stargazer::feature_render_node, coalsack::graph_node)