#pragma once

#include <memory>
#include <vector>

#include "calibration_target.hpp"
#include "graph_proc.h"
#include "messages.hpp"
#include "point_data.hpp"

namespace stargazer {

using namespace coalsack;

class three_point_bar_calibration_target_detector_node : public graph_node {
  std::unique_ptr<three_point_bar_calibration_target> detector;
  graph_edge_ptr output;

 public:
  three_point_bar_calibration_target_detector_node()
      : graph_node(), detector(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override {
    return "three_point_bar_calibration_target_detector";
  }

  template <typename Archive>
  void serialize(Archive& archive) {}

  virtual void run() override { detector = std::make_unique<three_point_bar_calibration_target>(); }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (const auto frame_msg = std::dynamic_pointer_cast<float2_list_message>(message)) {
      if (detector) {
        std::vector<point_data> markers;
        for (const auto& pt : frame_msg->get_data()) {
          markers.push_back({{pt.x, pt.y}, 0.0, 0.0});
        }

        const auto points = detector->detect_points(markers);

        std::vector<float2> float2_data;
        for (const auto& pt : points) {
          float2_data.push_back({pt.x, pt.y});
        }

        auto msg = std::make_shared<float2_list_message>();
        msg->set_data(float2_data);
        msg->set_frame_number(
            std::dynamic_pointer_cast<frame_message_base>(message)->get_frame_number());
        msg->set_timestamp(std::dynamic_pointer_cast<frame_message_base>(message)->get_timestamp());

        output->send(msg);
      }
    }
  }
};

}  // namespace stargazer

COALSACK_REGISTER_NODE(stargazer::three_point_bar_calibration_target_detector_node,
                       coalsack::graph_node)
