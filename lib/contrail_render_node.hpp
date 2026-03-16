#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "coalsack/core/graph_proc_registry.h"
#include "coalsack/image/image.h"
#include "messages.hpp"

namespace stargazer {

class contrail_render_node : public coalsack::graph_node {
  coalsack::graph_edge_ptr output;
  std::string camera_name;
  int width = 0;
  int height = 0;

  mutable std::mutex image_mtx;
  std::shared_ptr<coalsack::image> current_image;
  std::atomic<std::int64_t> received_count;

  mutable std::mutex points_mtx;
  std::vector<float2> accumulated_points;

 public:
  contrail_render_node()
      : graph_node(),
        output(std::make_shared<coalsack::graph_edge>(this)),
        camera_name(),
        width(0),
        height(0),
        image_mtx(),
        current_image(),
        received_count(0),
        points_mtx(),
        accumulated_points() {
    set_output(output);
  }

  void set_camera_name(std::string value) { camera_name = std::move(value); }
  void set_width(int value) { width = value; }
  void set_height(int value) { height = value; }

  virtual std::string get_proc_name() const override { return "contrail_render"; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(camera_name, width, height);
  }

  virtual std::optional<coalsack::property_value> get_property(const std::string& key) const override {
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
    if (const auto frame_msg = std::dynamic_pointer_cast<float2_list_message>(message)) {
      int img_width = width > 0 ? width : 820;
      int img_height = height > 0 ? height : 616;

      {
        std::lock_guard lock(points_mtx);
        for (const auto& pt : frame_msg->get_data()) {
          accumulated_points.push_back(pt);
        }
      }

      cv::Mat rendered(img_height, img_width, CV_8UC3, cv::Scalar::all(0));
      {
        std::lock_guard lock(points_mtx);
        for (const auto& pt : accumulated_points) {
          const int x = static_cast<int>(std::round(pt.x));
          const int y = static_cast<int>(std::round(pt.y));
          if (x >= 0 && x < img_width && y >= 0 && y < img_height) {
            cv::circle(rendered, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), cv::FILLED);  // BGR → RGB: red
          }
        }
      }

      coalsack::image img(
          static_cast<std::uint32_t>(rendered.cols),
          static_cast<std::uint32_t>(rendered.rows),
          static_cast<std::uint32_t>(rendered.elemSize()),
          static_cast<std::uint32_t>(rendered.step),
          rendered.data);
      img.set_format(coalsack::image_format::B8G8R8_UINT);

      {
        std::lock_guard lock(image_mtx);
        current_image = std::make_shared<coalsack::image>(img);
      }
      received_count.fetch_add(1);
    }

    output->send(message);
  }
};

}  // namespace stargazer

COALSACK_REGISTER_NODE(stargazer::contrail_render_node, coalsack::graph_node)
