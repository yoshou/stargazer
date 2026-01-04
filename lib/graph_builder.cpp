#include "graph_builder.hpp"

#include <iostream>

#include "axis_calibration_node.hpp"
#include "calibration_node.hpp"
#include "callback_node.hpp"
#include "dump_blob_node.hpp"
#include "dump_keypoint_node.hpp"
#include "dump_reconstruction_node.hpp"
#include "dump_se3_node.hpp"
#include "epipolar_reconstruct_node.hpp"
#include "ext/graph_proc_cv_ext.h"
#include "ext/graph_proc_depthai.h"
#include "ext/graph_proc_jpeg.h"
#include "ext/graph_proc_libcamera.h"
#include "ext/graph_proc_rs_d435.h"
#include "glm_serialize.hpp"
#include "graph_proc.h"
#include "graph_proc_cv.h"
#include "graph_proc_img.h"
#include "grpc_server_node.hpp"
#include "image_reconstruct_node.hpp"
#include "intrinsic_calibration_node.hpp"
#include "load_blob_node.hpp"
#include "load_marker_node.hpp"
#include "load_panoptic_node.hpp"
#include "mvp_reconstruct_node.hpp"
#include "mvpose_reconstruct_node.hpp"
#include "object_map_node.hpp"
#include "object_mux_node.hpp"
#include "pattern_board_calibration_target_detector_node.hpp"
#include "three_point_bar_calibration_target_detector_node.hpp"
#include "voxelpose_reconstruct_node.hpp"

using namespace coalsack;
using namespace stargazer;

namespace stargazer {

void build_graph_from_json(const std::vector<node_def>& nodes,
                           std::map<std::string, std::shared_ptr<subgraph>>& subgraphs,
                           std::unordered_map<std::string, std::shared_ptr<graph_node>>& node_map) {
  // Create nodes
  for (const auto& node : nodes) {
    // Skip if node already exists in the global map
    if (node_map.find(node.name) != node_map.end()) {
      continue;
    }

    graph_node_ptr graph_node;

    switch (node.get_type()) {
      case node_type::libcamera_capture: {
        auto n = std::make_shared<libcamera_capture_node>();
        if (node.contains_param("stream")) {
          const auto stream_str = node.get_param<std::string>("stream");
          if (stream_str == "COLOR") {
            n->set_stream(stream_type::COLOR);
          } else if (stream_str == "INFRARED") {
            n->set_stream(stream_type::INFRARED);
          }
        }
        if (node.contains_param("fps")) {
          n->set_fps(static_cast<int>(node.get_param<std::int64_t>("fps")));
        }
        if (node.contains_param("width")) {
          n->set_width(static_cast<int>(node.get_param<std::int64_t>("width")));
        }
        if (node.contains_param("height")) {
          n->set_height(static_cast<int>(node.get_param<std::int64_t>("height")));
        }
        if (node.contains_param("format")) {
          const auto format_str = node.get_param<std::string>("format");
          if (format_str == "Y8_UINT") {
            n->set_format(image_format::Y8_UINT);
          } else if (format_str == "R8G8B8_UINT") {
            n->set_format(image_format::R8G8B8_UINT);
          }
        }
        if (node.contains_param("exposure")) {
          n->set_option(libcamera_capture_node::option::exposure,
                        static_cast<int>(node.get_param<std::int64_t>("exposure")));
        }
        if (node.contains_param("gain")) {
          n->set_option(libcamera_capture_node::option::gain,
                        static_cast<int>(node.get_param<std::int64_t>("gain")));
        }
        if (node.contains_param("emitter_enabled")) {
          n->set_emitter_enabled(node.get_param<bool>("emitter_enabled"));
        }
        graph_node = n;
        break;
      }
      case node_type::timestamp: {
        auto n = std::make_shared<timestamp_node>();
        graph_node = n;
        break;
      }
      case node_type::broadcast_talker: {
        auto n = std::make_shared<broadcast_talker_node>();
        if (node.contains_param("address") && node.contains_param("port")) {
          const auto address = node.get_param<std::string>("address");
          const auto port = static_cast<uint16_t>(node.get_param<std::int64_t>("port"));
          n->set_endpoint(address, port);
        }
        graph_node = n;
        break;
      }
      case node_type::broadcast_listener: {
        auto n = std::make_shared<broadcast_listener_node>();
        if (node.contains_param("address") && node.contains_param("port")) {
          const auto address = node.get_param<std::string>("address");
          const auto port = static_cast<uint16_t>(node.get_param<std::int64_t>("port"));
          n->set_endpoint(address, port);
        }
        graph_node = n;
        break;
      }
      case node_type::encode_jpeg: {
        auto n = std::make_shared<encode_jpeg_node>();
        graph_node = n;
        break;
      }
      case node_type::decode_jpeg: {
        auto n = std::make_shared<decode_jpeg_node>();
        graph_node = n;
        break;
      }
      case node_type::scale: {
        auto n = std::make_shared<scale_node>();
        if (node.contains_param("alpha")) {
          n->set_alpha(node.get_param<double>("alpha"));
        }
        if (node.contains_param("beta")) {
          n->set_beta(node.get_param<double>("beta"));
        }
        graph_node = n;
        break;
      }
      case node_type::resize: {
        auto n = std::make_shared<resize_node>();
        if (node.contains_param("width")) {
          n->set_width(static_cast<int>(node.get_param<std::int64_t>("width")));
        }
        if (node.contains_param("height")) {
          n->set_height(static_cast<int>(node.get_param<std::int64_t>("height")));
        }
        graph_node = n;
        break;
      }
      case node_type::gaussian_blur: {
        auto n = std::make_shared<gaussian_blur_node>();
        if (node.contains_param("kernel_width")) {
          n->set_kernel_width(static_cast<int>(node.get_param<std::int64_t>("kernel_width")));
        }
        if (node.contains_param("kernel_height")) {
          n->set_kernel_height(static_cast<int>(node.get_param<std::int64_t>("kernel_height")));
        }
        if (node.contains_param("sigma_x")) {
          n->set_sigma_x(node.get_param<double>("sigma_x"));
        }
        if (node.contains_param("sigma_y")) {
          n->set_sigma_y(node.get_param<double>("sigma_y"));
        }
        graph_node = n;
        break;
      }
      case node_type::mask: {
        auto n = std::make_shared<mask_node>();
        graph_node = n;
        break;
      }
      case node_type::p2p_tcp_talker: {
        auto n = std::make_shared<p2p_tcp_talker_node>();
        graph_node = n;
        break;
      }
      case node_type::p2p_tcp_listener: {
        auto n = std::make_shared<p2p_tcp_listener_node>();
        if (node.contains_param("endpoint_address")) {
          const auto address = node.get_param<std::string>("endpoint_address");
          const auto port =
              node.contains_param("endpoint_port")
                  ? static_cast<uint16_t>(node.get_param<std::int64_t>("endpoint_port"))
                  : 0;
          n->set_endpoint(address, port);
        } else {
          n->set_endpoint("", 0);
        }
        graph_node = n;
        break;
      }
      case node_type::fifo: {
        auto n = std::make_shared<fifo_node>();
        if (node.contains_param("max_size")) {
          n->set_max_size(static_cast<size_t>(node.get_param<std::int64_t>("max_size")));
        }
        graph_node = n;
        break;
      }
      case node_type::video_time_sync_control: {
        auto n = std::make_shared<video_time_sync_control_node>();
        if (node.contains_param("gain")) {
          n->set_gain(node.get_param<double>("gain"));
        }
        if (node.contains_param("interval")) {
          n->set_interval(node.get_param<double>("interval"));
        }
        if (node.contains_param("max_interval")) {
          n->set_max_interval(node.get_param<double>("max_interval"));
        }
        if (node.contains_param("min_interval")) {
          n->set_min_interval(node.get_param<double>("min_interval"));
        }
        graph_node = n;
        break;
      }
      case node_type::fast_blob_detector: {
        auto n = std::make_shared<fast_blob_detector_node>();
        auto params = n->get_parameters();
        if (node.contains_param("min_dist_between_blobs")) {
          params.min_dist_between_blobs =
              static_cast<float>(node.get_param<double>("min_dist_between_blobs"));
        }
        if (node.contains_param("step_threshold")) {
          params.step_threshold = node.get_param<double>("step_threshold");
        }
        if (node.contains_param("min_threshold")) {
          params.min_threshold = node.get_param<double>("min_threshold");
        }
        if (node.contains_param("max_threshold")) {
          params.max_threshold = node.get_param<double>("max_threshold");
        }
        if (node.contains_param("min_area")) {
          params.min_area = node.get_param<double>("min_area");
        }
        if (node.contains_param("max_area")) {
          params.max_area = node.get_param<double>("max_area");
        }
        if (node.contains_param("min_circularity")) {
          params.min_circularity = node.get_param<double>("min_circularity");
        }
        if (node.contains_param("max_circularity")) {
          params.max_circularity = node.get_param<double>("max_circularity");
        }
        if (node.contains_param("min_repeatability")) {
          params.min_repeatability =
              static_cast<std::int32_t>(node.get_param<std::int64_t>("min_repeatability"));
        }
        n->set_parameters(params);
        graph_node = n;
        break;
      }
      case node_type::detect_circle_grid: {
        auto n = std::make_shared<detect_circle_grid_node>();
        auto& params = n->get_parameters();
        if (node.contains_param("min_dist_between_blobs")) {
          params.min_dist_between_blobs =
              static_cast<float>(node.get_param<double>("min_dist_between_blobs"));
        }
        if (node.contains_param("threshold_step")) {
          params.threshold_step = static_cast<float>(node.get_param<double>("threshold_step"));
        }
        if (node.contains_param("min_threshold")) {
          params.min_threshold = static_cast<float>(node.get_param<double>("min_threshold"));
        }
        if (node.contains_param("max_threshold")) {
          params.max_threshold = static_cast<float>(node.get_param<double>("max_threshold"));
        }
        if (node.contains_param("min_area")) {
          params.min_area = static_cast<float>(node.get_param<double>("min_area"));
        }
        if (node.contains_param("max_area")) {
          params.max_area = static_cast<float>(node.get_param<double>("max_area"));
        }
        if (node.contains_param("min_circularity")) {
          params.min_circularity = static_cast<float>(node.get_param<double>("min_circularity"));
        }
        if (node.contains_param("max_circularity")) {
          params.max_circularity = static_cast<float>(node.get_param<double>("max_circularity"));
        }
        if (node.contains_param("filter_by_area")) {
          params.filter_by_area = node.get_param<bool>("filter_by_area");
        }
        if (node.contains_param("filter_by_circularity")) {
          params.filter_by_circularity = node.get_param<bool>("filter_by_circularity");
        }
        if (node.contains_param("filter_by_inertia")) {
          params.filter_by_inertia = node.get_param<bool>("filter_by_inertia");
        }
        if (node.contains_param("filter_by_convexity")) {
          params.filter_by_convexity = node.get_param<bool>("filter_by_convexity");
        }
        if (node.contains_param("filter_by_color")) {
          params.filter_by_color = node.get_param<bool>("filter_by_color");
        }
        if (node.contains_param("blob_color")) {
          params.blob_color = static_cast<uint8_t>(node.get_param<std::int64_t>("blob_color"));
        }
        if (node.contains_param("min_repeatability")) {
          params.min_repeatability =
              static_cast<size_t>(node.get_param<std::int64_t>("min_repeatability"));
        }
        if (node.contains_param("min_inertia_ratio")) {
          params.min_inertia_ratio =
              static_cast<float>(node.get_param<double>("min_inertia_ratio"));
        }
        if (node.contains_param("max_inertia_ratio")) {
          params.max_inertia_ratio =
              static_cast<float>(node.get_param<double>("max_inertia_ratio"));
        }
        if (node.contains_param("min_convexity")) {
          params.min_convexity = static_cast<float>(node.get_param<double>("min_convexity"));
        }
        if (node.contains_param("max_convexity")) {
          params.max_convexity = static_cast<float>(node.get_param<double>("max_convexity"));
        }
        graph_node = n;
        break;
      }
      case node_type::load_blob: {
        auto n = std::make_shared<load_blob_node>();
        if (node.contains_param("db_path")) {
          n->set_db_path(node.get_param<std::string>("db_path"));
        }
        if (node.contains_param("topic_name")) {
          n->set_topic_name(node.get_param<std::string>("topic_name"));
        }
        graph_node = n;
        break;
      }
      case node_type::load_marker: {
        auto n = std::make_shared<load_marker_node>();
        if (node.contains_param("db_path")) {
          n->set_db_path(node.get_param<std::string>("db_path"));
        }
        if (node.contains_param("topic_name")) {
          n->set_topic_name(node.get_param<std::string>("topic_name"));
        }
        graph_node = n;
        break;
      }
      case node_type::load_panoptic: {
        auto n = std::make_shared<load_panoptic_node>();
        if (node.contains_param("db_path")) {
          n->set_db_path(node.get_param<std::string>("db_path"));
        }
        if (node.contains_param("topic_name")) {
          n->set_topic_name(node.get_param<std::string>("topic_name"));
        }
        if (node.contains_param("fps")) {
          n->set_fps(static_cast<int>(node.get_param<double>("fps")));
        }
        graph_node = n;
        break;
      }
      case node_type::approximate_time_sync: {
        auto n = std::make_shared<approximate_time_sync_node>();
        if (node.contains_param("interval")) {
          n->get_config().set_interval(node.get_param<double>("interval"));
        }
        graph_node = n;
        break;
      }
      case node_type::frame_number_numbering: {
        auto n = std::make_shared<frame_number_numbering_node>();
        graph_node = n;
        break;
      }
      case node_type::parallel_queue: {
        auto n = std::make_shared<parallel_queue_node>();
        if (node.contains_param("num_threads")) {
          n->set_num_threads(static_cast<size_t>(node.get_param<std::int64_t>("num_threads")));
        }
        graph_node = n;
        break;
      }
      case node_type::frame_number_ordering: {
        auto n = std::make_shared<frame_number_ordering_node>();
        graph_node = n;
        break;
      }
      case node_type::callback: {
        auto n = std::make_shared<callback_node>();

        if (!node.contains_param("callback_name")) {
          throw std::runtime_error("callback_name parameter is required for callback node: " +
                                   node.name);
        }
        n->set_callback_name(node.get_param<std::string>("callback_name"));

        if (node.contains_param("camera_name")) {
          n->set_camera_name(node.get_param<std::string>("camera_name"));
        }

        if (node.contains_param("callback_type")) {
          const auto type_str = node.get_param<std::string>("callback_type");
          callback_node::callback_type cb_type = callback_node::callback_type::unknown;
          if (type_str == "image") {
            cb_type = callback_node::callback_type::image;
          } else if (type_str == "marker") {
            cb_type = callback_node::callback_type::marker;
          } else if (type_str == "object") {
            cb_type = callback_node::callback_type::object;
          } else {
            throw std::runtime_error("Unknown callback_type: " + type_str);
          }
          n->set_callback_type(cb_type);
        }
        graph_node = n;
        break;
      }
      case node_type::charuco_detector: {
        auto n = std::make_shared<charuco_detector_node>();
        graph_node = n;
        break;
      }
      case node_type::depthai_color_camera: {
        auto n = std::make_shared<depthai_color_camera_node>();
        if (node.contains_param("fps")) {
          n->set_fps(static_cast<int>(node.get_param<std::int64_t>("fps")));
        }
        if (node.contains_param("width")) {
          n->set_width(static_cast<int>(node.get_param<std::int64_t>("width")));
        }
        if (node.contains_param("height")) {
          n->set_height(static_cast<int>(node.get_param<std::int64_t>("height")));
        }
        graph_node = n;
        break;
      }
      case node_type::rs_d435: {
        auto n = std::make_shared<rs_d435_node>();

        if (node.contains_param("global_time_enabled")) {
          n->set_option(INFRA1, rs2_option_type::GLOBAL_TIME_ENABLED,
                        node.get_param<bool>("global_time_enabled") ? 1.0f : 0.0f);
        }
        if (node.contains_param("exposure")) {
          n->set_option(INFRA1, rs2_option_type::EXPOSURE,
                        static_cast<float>(node.get_param<std::int64_t>("exposure")));
        }
        if (node.contains_param("gain")) {
          n->set_option(INFRA1, rs2_option_type::GAIN,
                        static_cast<float>(node.get_param<std::int64_t>("gain")));
        }
        if (node.contains_param("laser_power")) {
          n->set_option(INFRA1, rs2_option_type::LASER_POWER,
                        static_cast<float>(node.get_param<std::int64_t>("laser_power")));
        }
        if (node.contains_param("emitter_enabled")) {
          n->set_option(INFRA1, rs2_option_type::EMITTER_ENABLED,
                        node.get_param<bool>("emitter_enabled") ? 1.0f : 0.0f);
        }

        if (node.contains_param("stream_type") && node.contains_param("width") &&
            node.contains_param("height") && node.contains_param("fps")) {
          const auto stream_type_str = node.get_param<std::string>("stream_type");
          stream_index_pair stream_type = INFRA1;
          if (stream_type_str == "INFRA1") {
            stream_type = INFRA1;
          } else if (stream_type_str == "COLOR") {
            stream_type = COLOR;
          }

          const int width = static_cast<int>(node.get_param<std::int64_t>("width"));
          const int height = static_cast<int>(node.get_param<std::int64_t>("height"));
          const int fps = static_cast<int>(node.get_param<std::int64_t>("fps"));

          rs2_format_type format = rs2_format_type::Y8;
          if (node.contains_param("format")) {
            const auto format_str = node.get_param<std::string>("format");
            if (format_str == "Y8") {
              format = rs2_format_type::Y8;
            } else if (format_str == "BGR8") {
              format = rs2_format_type::BGR8;
            }
          }

          n->add_output(stream_type, width, height, format, fps);
        }

        graph_node = n;
        break;
      }
      case node_type::object_map: {
        auto n = std::make_shared<object_map_node>();
        for (const auto& output_name : node.outputs) {
          n->add_output(output_name);
        }
        graph_node = n;
        break;
      }
      case node_type::object_mux: {
        auto n = std::make_shared<object_mux_node>();
        graph_node = n;
        break;
      }
      case node_type::pattern_board_calibration_target_detector: {
        auto n = std::make_shared<pattern_board_calibration_target_detector_node>();
        graph_node = n;
        break;
      }
      case node_type::three_point_bar_calibration_target_detector: {
        auto n = std::make_shared<three_point_bar_calibration_target_detector_node>();
        graph_node = n;
        break;
      }
      case node_type::calibration: {
        auto n = std::make_shared<calibration_node>();
        if (node.contains_param("only_extrinsic")) {
          n->set_only_extrinsic(node.get_param<bool>("only_extrinsic"));
        }
        if (node.contains_param("robust")) {
          n->set_robust(node.get_param<bool>("robust"));
        }
        graph_node = n;
        break;
      }
      case node_type::intrinsic_calibration: {
        auto n = std::make_shared<intrinsic_calibration_node>();
        graph_node = n;
        break;
      }
      case node_type::axis_calibration: {
        auto n = std::make_shared<axis_calibration_node>();
        graph_node = n;
        break;
      }
      case node_type::grpc_server: {
        auto n = std::make_shared<grpc_server_node>();
        if (node.contains_param("address")) {
          n->set_address(node.get_param<std::string>("address"));
        }
        graph_node = n;
        break;
      }
      case node_type::frame_demux: {
        auto n = std::make_shared<frame_demux_node>();
        for (const auto& output_name : node.outputs) {
          n->add_output(output_name);
        }
        graph_node = n;
        break;
      }
      case node_type::dump_se3: {
        auto n = std::make_shared<dump_se3_node>();
        if (node.contains_param("db_path")) {
          n->set_db_path(node.get_param<std::string>("db_path"));
        }
        if (node.contains_param("topic_name")) {
          n->set_name(node.get_param<std::string>("topic_name"));
        }
        graph_node = n;
        break;
      }
      case node_type::dump_reconstruction: {
        auto n = std::make_shared<dump_reconstruction_node>();
        if (node.contains_param("db_path")) {
          n->set_db_path(node.get_param<std::string>("db_path"));
        }
        if (node.contains_param("topic_name")) {
          n->set_name(node.get_param<std::string>("topic_name"));
        }
        graph_node = n;
        break;
      }
      case node_type::voxelpose_reconstruction: {
        auto n = std::make_shared<voxelpose_reconstruct_node>();
        graph_node = n;
        break;
      }
      case node_type::mvpose_reconstruction: {
        auto n = std::make_shared<mvpose_reconstruct_node>();
        graph_node = n;
        break;
      }
      case node_type::mvp_reconstruction: {
        auto n = std::make_shared<mvp_reconstruct_node>();
        graph_node = n;
        break;
      }
      case node_type::epipolar_reconstruction: {
        auto n = std::make_shared<epipolar_reconstruct_node>();
        graph_node = n;
        break;
      }
      default:
        throw std::runtime_error("Unknown node type: " + node.name);
    }

    node_map[node.name] = graph_node;

    // Add node to the appropriate subgraph
    if (subgraphs.find(node.subgraph_instance) != subgraphs.end()) {
      subgraphs[node.subgraph_instance]->add_node(graph_node);
    }
  }

  // Connect nodes
  for (const auto& node : nodes) {
    if (node.inputs.empty()) {
      continue;
    }

    auto target_node = node_map.at(node.name);

    for (const auto& [input_name, source_name] : node.inputs) {
      size_t pos = source_name.find(':');
      if (pos != std::string::npos) {
        auto node_name = source_name.substr(0, pos);
        auto output_name = source_name.substr(pos + 1);
        auto source_node = node_map.at(node_name);
        target_node->set_input(source_node->get_output(output_name), input_name);
      } else {
        auto source_node = node_map.at(source_name);
        target_node->set_input(source_node->get_output(), input_name);
      }
    }
  }
}

}  // namespace stargazer
