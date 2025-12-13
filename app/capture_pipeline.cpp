#include "capture_pipeline.hpp"

#include <sqlite3.h>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <regex>

#include "callback_node.hpp"
#include "dump_blob_node.hpp"
#include "dump_keypoint_node.hpp"
#include "ext/graph_proc_cv_ext.h"
#include "ext/graph_proc_depthai.h"
#include "ext/graph_proc_jpeg.h"
#include "ext/graph_proc_libcamera.h"
#include "ext/graph_proc_rs_d435.h"
#include "graph_proc.h"
#include "graph_proc_cv.h"
#include "load_blob_node.hpp"
#include "load_marker_node.hpp"
#include "load_panoptic_node.hpp"

using namespace coalsack;
using namespace stargazer;

class local_server {
  asio::io_context io_context;
  std::shared_ptr<resource_list> resources;
  std::shared_ptr<graph_proc_server> server;
  std::shared_ptr<std::thread> th;
  std::atomic_bool running;

 public:
  local_server(uint16_t port = 0)
      : io_context(),
        resources(std::make_shared<resource_list>()),
        server(std::make_shared<graph_proc_server>(io_context, "0.0.0.0", port, resources)),
        th(),
        running(false) {}

  uint16_t get_port() const { return server->get_port(); }

  void run() {
    running = true;
    th.reset(new std::thread([this] { io_context.run(); }));
  }

  void stop() {
    if (running.load()) {
      running.store(false);
      io_context.stop();
      if (th && th->joinable()) {
        th->join();
      }
    }
  }

  ~local_server() { stop(); }

  void add_resource(std::shared_ptr<resource_base> resource) { resources->add(resource); }
};

static void build_graph_from_json(
    const std::vector<node_info>& node_infos,
    std::map<std::string, std::shared_ptr<subgraph>>& subgraphs,
    std::unordered_map<std::string, std::shared_ptr<graph_node>>& node_map) {
  // Create nodes
  for (const auto& info : node_infos) {
    // Skip if node already exists in the global map
    if (node_map.find(info.name) != node_map.end()) {
      continue;
    }

    graph_node_ptr node;

    switch (info.get_type()) {
      case node_type::libcamera_capture: {
        auto n = std::make_shared<libcamera_capture_node>();
        if (info.contains_param("stream")) {
          const auto stream_str = info.get_param<std::string>("stream");
          if (stream_str == "COLOR") {
            n->set_stream(stream_type::COLOR);
          } else if (stream_str == "INFRARED") {
            n->set_stream(stream_type::INFRARED);
          }
        }
        if (info.contains_param("fps")) {
          n->set_fps(static_cast<int>(info.get_param<std::int64_t>("fps")));
        }
        if (info.contains_param("width")) {
          n->set_width(static_cast<int>(info.get_param<std::int64_t>("width")));
        }
        if (info.contains_param("height")) {
          n->set_height(static_cast<int>(info.get_param<std::int64_t>("height")));
        }
        if (info.contains_param("format")) {
          const auto format_str = info.get_param<std::string>("format");
          if (format_str == "Y8_UINT") {
            n->set_format(image_format::Y8_UINT);
          } else if (format_str == "R8G8B8_UINT") {
            n->set_format(image_format::R8G8B8_UINT);
          }
        }
        if (info.contains_param("exposure")) {
          n->set_option(libcamera_capture_node::option::exposure,
                        static_cast<int>(info.get_param<std::int64_t>("exposure")));
        }
        if (info.contains_param("gain")) {
          n->set_option(libcamera_capture_node::option::gain,
                        static_cast<int>(info.get_param<std::int64_t>("gain")));
        }
        if (info.contains_param("emitter_enabled")) {
          n->set_emitter_enabled(info.get_param<bool>("emitter_enabled"));
        }
        node = n;
        break;
      }
      case node_type::timestamp: {
        auto n = std::make_shared<timestamp_node>();
        node = n;
        break;
      }
      case node_type::broadcast_talker: {
        auto n = std::make_shared<broadcast_talker_node>();
        if (info.contains_param("address") && info.contains_param("port")) {
          const auto address = info.get_param<std::string>("address");
          const auto port = static_cast<uint16_t>(info.get_param<std::int64_t>("port"));
          n->set_endpoint(address, port);
        }
        node = n;
        break;
      }
      case node_type::broadcast_listener: {
        auto n = std::make_shared<broadcast_listener_node>();
        if (info.contains_param("address") && info.contains_param("port")) {
          const auto address = info.get_param<std::string>("address");
          const auto port = static_cast<uint16_t>(info.get_param<std::int64_t>("port"));
          n->set_endpoint(address, port);
        }
        node = n;
        break;
      }
      case node_type::encode_jpeg: {
        auto n = std::make_shared<encode_jpeg_node>();
        node = n;
        break;
      }
      case node_type::decode_jpeg: {
        auto n = std::make_shared<decode_jpeg_node>();
        node = n;
        break;
      }
      case node_type::scale: {
        auto n = std::make_shared<scale_node>();
        if (info.contains_param("alpha")) {
          n->set_alpha(info.get_param<double>("alpha"));
        }
        if (info.contains_param("beta")) {
          n->set_beta(info.get_param<double>("beta"));
        }
        node = n;
        break;
      }
      case node_type::resize: {
        auto n = std::make_shared<resize_node>();
        if (info.contains_param("width")) {
          n->set_width(static_cast<int>(info.get_param<std::int64_t>("width")));
        }
        if (info.contains_param("height")) {
          n->set_height(static_cast<int>(info.get_param<std::int64_t>("height")));
        }
        node = n;
        break;
      }
      case node_type::gaussian_blur: {
        auto n = std::make_shared<gaussian_blur_node>();
        if (info.contains_param("kernel_width")) {
          n->set_kernel_width(static_cast<int>(info.get_param<std::int64_t>("kernel_width")));
        }
        if (info.contains_param("kernel_height")) {
          n->set_kernel_height(static_cast<int>(info.get_param<std::int64_t>("kernel_height")));
        }
        if (info.contains_param("sigma_x")) {
          n->set_sigma_x(info.get_param<double>("sigma_x"));
        }
        if (info.contains_param("sigma_y")) {
          n->set_sigma_y(info.get_param<double>("sigma_y"));
        }
        node = n;
        break;
      }
      case node_type::mask: {
        auto n = std::make_shared<mask_node>();
        node = n;
        break;
      }
      case node_type::p2p_tcp_talker: {
        auto n = std::make_shared<p2p_tcp_talker_node>();
        node = n;
        break;
      }
      case node_type::p2p_tcp_listener: {
        auto n = std::make_shared<p2p_tcp_listener_node>();
        // Use endpoint_address/endpoint_port parameters
        if (info.contains_param("endpoint_address")) {
          const auto address = info.get_param<std::string>("endpoint_address");
          const auto port =
              info.contains_param("endpoint_port")
                  ? static_cast<uint16_t>(info.get_param<std::int64_t>("endpoint_port"))
                  : 0;  // Port 0 for dynamic allocation
          n->set_endpoint(address, port);
        } else {
          // Default: bind to any interface with dynamic port
          n->set_endpoint("", 0);
        }
        node = n;
        break;
      }
      case node_type::fifo: {
        auto n = std::make_shared<fifo_node>();
        if (info.contains_param("max_size")) {
          n->set_max_size(static_cast<size_t>(info.get_param<std::int64_t>("max_size")));
        }
        node = n;
        break;
      }
      case node_type::video_time_sync_control: {
        auto n = std::make_shared<video_time_sync_control_node>();
        if (info.contains_param("gain")) {
          n->set_gain(info.get_param<double>("gain"));
        }
        if (info.contains_param("interval")) {
          n->set_interval(info.get_param<double>("interval"));
        }
        if (info.contains_param("max_interval")) {
          n->set_max_interval(info.get_param<double>("max_interval"));
        }
        if (info.contains_param("min_interval")) {
          n->set_min_interval(info.get_param<double>("min_interval"));
        }
        node = n;
        break;
      }
      case node_type::fast_blob_detector: {
        auto n = std::make_shared<fast_blob_detector_node>();
        auto params = n->get_parameters();
        if (info.contains_param("min_dist_between_blobs")) {
          params.min_dist_between_blobs =
              static_cast<float>(info.get_param<double>("min_dist_between_blobs"));
        }
        if (info.contains_param("step_threshold")) {
          params.step_threshold = info.get_param<double>("step_threshold");
        }
        if (info.contains_param("min_threshold")) {
          params.min_threshold = info.get_param<double>("min_threshold");
        }
        if (info.contains_param("max_threshold")) {
          params.max_threshold = info.get_param<double>("max_threshold");
        }
        if (info.contains_param("min_area")) {
          params.min_area = info.get_param<double>("min_area");
        }
        if (info.contains_param("max_area")) {
          params.max_area = info.get_param<double>("max_area");
        }
        if (info.contains_param("min_circularity")) {
          params.min_circularity = info.get_param<double>("min_circularity");
        }
        if (info.contains_param("max_circularity")) {
          params.max_circularity = info.get_param<double>("max_circularity");
        }
        if (info.contains_param("min_repeatability")) {
          params.min_repeatability =
              static_cast<std::int32_t>(info.get_param<std::int64_t>("min_repeatability"));
        }
        n->set_parameters(params);
        node = n;
        break;
      }
      case node_type::detect_circle_grid: {
        auto n = std::make_shared<detect_circle_grid_node>();
        auto& params = n->get_parameters();
        if (info.contains_param("min_dist_between_blobs")) {
          params.min_dist_between_blobs =
              static_cast<float>(info.get_param<double>("min_dist_between_blobs"));
        }
        if (info.contains_param("threshold_step")) {
          params.threshold_step = static_cast<float>(info.get_param<double>("threshold_step"));
        }
        if (info.contains_param("min_threshold")) {
          params.min_threshold = static_cast<float>(info.get_param<double>("min_threshold"));
        }
        if (info.contains_param("max_threshold")) {
          params.max_threshold = static_cast<float>(info.get_param<double>("max_threshold"));
        }
        if (info.contains_param("min_area")) {
          params.min_area = static_cast<float>(info.get_param<double>("min_area"));
        }
        if (info.contains_param("max_area")) {
          params.max_area = static_cast<float>(info.get_param<double>("max_area"));
        }
        if (info.contains_param("min_circularity")) {
          params.min_circularity = static_cast<float>(info.get_param<double>("min_circularity"));
        }
        if (info.contains_param("max_circularity")) {
          params.max_circularity = static_cast<float>(info.get_param<double>("max_circularity"));
        }
        if (info.contains_param("filter_by_area")) {
          params.filter_by_area = info.get_param<bool>("filter_by_area");
        }
        if (info.contains_param("filter_by_circularity")) {
          params.filter_by_circularity = info.get_param<bool>("filter_by_circularity");
        }
        if (info.contains_param("filter_by_inertia")) {
          params.filter_by_inertia = info.get_param<bool>("filter_by_inertia");
        }
        if (info.contains_param("filter_by_convexity")) {
          params.filter_by_convexity = info.get_param<bool>("filter_by_convexity");
        }
        if (info.contains_param("filter_by_color")) {
          params.filter_by_color = info.get_param<bool>("filter_by_color");
        }
        if (info.contains_param("blob_color")) {
          params.blob_color = static_cast<uint8_t>(info.get_param<std::int64_t>("blob_color"));
        }
        if (info.contains_param("min_repeatability")) {
          params.min_repeatability =
              static_cast<size_t>(info.get_param<std::int64_t>("min_repeatability"));
        }
        if (info.contains_param("min_inertia_ratio")) {
          params.min_inertia_ratio =
              static_cast<float>(info.get_param<double>("min_inertia_ratio"));
        }
        if (info.contains_param("max_inertia_ratio")) {
          params.max_inertia_ratio =
              static_cast<float>(info.get_param<double>("max_inertia_ratio"));
        }
        if (info.contains_param("min_convexity")) {
          params.min_convexity = static_cast<float>(info.get_param<double>("min_convexity"));
        }
        if (info.contains_param("max_convexity")) {
          params.max_convexity = static_cast<float>(info.get_param<double>("max_convexity"));
        }
        node = n;
        break;
      }
      case node_type::load_blob: {
        auto n = std::make_shared<load_blob_node>();
        if (info.contains_param("db_path")) {
          n->set_db_path(info.get_param<std::string>("db_path"));
        }
        if (info.contains_param("topic_name")) {
          n->set_topic_name(info.get_param<std::string>("topic_name"));
        }
        node = n;
        break;
      }
      case node_type::load_marker: {
        auto n = std::make_shared<load_marker_node>();
        if (info.contains_param("db_path")) {
          n->set_db_path(info.get_param<std::string>("db_path"));
        }
        if (info.contains_param("topic_name")) {
          n->set_topic_name(info.get_param<std::string>("topic_name"));
        }
        node = n;
        break;
      }
      case node_type::load_panoptic: {
        auto n = std::make_shared<load_panoptic_node>();
        if (info.contains_param("db_path")) {
          n->set_db_path(info.get_param<std::string>("db_path"));
        }
        if (info.contains_param("topic_name")) {
          n->set_topic_name(info.get_param<std::string>("topic_name"));
        }
        if (info.contains_param("fps")) {
          n->set_fps(static_cast<int>(info.get_param<double>("fps")));
        }
        node = n;
        break;
      }
      case node_type::approximate_time_sync: {
        auto n = std::make_shared<approximate_time_sync_node>();
        if (info.contains_param("interval")) {
          n->get_config().set_interval(info.get_param<double>("interval"));
        }
        node = n;
        break;
      }
      case node_type::callback: {
        auto n = std::make_shared<callback_node>();

        // Get callback_name from parameter (required)
        if (!info.contains_param("callback_name")) {
          throw std::runtime_error("callback_name parameter is required for callback node: " +
                                   info.name);
        }
        n->set_callback_name(info.get_param<std::string>("callback_name"));

        if (info.contains_param("camera_name")) {
          n->set_camera_name(info.get_param<std::string>("camera_name"));
        }

        // Get callback type from parameter (required)
        if (!info.contains_param("callback_type")) {
          throw std::runtime_error("callback_type parameter is required for callback node: " +
                                   info.name);
        }
        const auto type_str = info.get_param<std::string>("callback_type");
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
        node = n;
        break;
      }
      case node_type::charuco_detector: {
        auto n = std::make_shared<charuco_detector_node>();
        node = n;
        break;
      }
      case node_type::depthai_color_camera: {
        auto n = std::make_shared<depthai_color_camera_node>();
        if (info.contains_param("fps")) {
          n->set_fps(static_cast<int>(info.get_param<std::int64_t>("fps")));
        }
        if (info.contains_param("width")) {
          n->set_width(static_cast<int>(info.get_param<std::int64_t>("width")));
        }
        if (info.contains_param("height")) {
          n->set_height(static_cast<int>(info.get_param<std::int64_t>("height")));
        }
        node = n;
        break;
      }
      case node_type::rs_d435: {
        auto n = std::make_shared<rs_d435_node>();

        // Set options if provided
        if (info.contains_param("global_time_enabled")) {
          n->set_option(INFRA1, rs2_option_type::GLOBAL_TIME_ENABLED,
                        info.get_param<bool>("global_time_enabled") ? 1.0f : 0.0f);
        }
        if (info.contains_param("exposure")) {
          n->set_option(INFRA1, rs2_option_type::EXPOSURE,
                        static_cast<float>(info.get_param<std::int64_t>("exposure")));
        }
        if (info.contains_param("gain")) {
          n->set_option(INFRA1, rs2_option_type::GAIN,
                        static_cast<float>(info.get_param<std::int64_t>("gain")));
        }
        if (info.contains_param("laser_power")) {
          n->set_option(INFRA1, rs2_option_type::LASER_POWER,
                        static_cast<float>(info.get_param<std::int64_t>("laser_power")));
        }
        if (info.contains_param("emitter_enabled")) {
          n->set_option(INFRA1, rs2_option_type::EMITTER_ENABLED,
                        info.get_param<bool>("emitter_enabled") ? 1.0f : 0.0f);
        }

        // Add output stream if parameters are provided
        if (info.contains_param("stream_type") && info.contains_param("width") &&
            info.contains_param("height") && info.contains_param("fps")) {
          const auto stream_type_str = info.get_param<std::string>("stream_type");
          stream_index_pair stream_type = INFRA1;
          if (stream_type_str == "INFRA1") {
            stream_type = INFRA1;
          } else if (stream_type_str == "COLOR") {
            stream_type = COLOR;
          }

          const int width = static_cast<int>(info.get_param<std::int64_t>("width"));
          const int height = static_cast<int>(info.get_param<std::int64_t>("height"));
          const int fps = static_cast<int>(info.get_param<std::int64_t>("fps"));

          rs2_format_type format = rs2_format_type::Y8;
          if (info.contains_param("format")) {
            const auto format_str = info.get_param<std::string>("format");
            if (format_str == "Y8") {
              format = rs2_format_type::Y8;
            } else if (format_str == "BGR8") {
              format = rs2_format_type::BGR8;
            }
          }

          n->add_output(stream_type, width, height, format, fps);
        }

        node = n;
        break;
      }
      default:
        throw std::runtime_error("Unknown node type: " + info.name);
    }

    node_map[info.name] = node;

    // Add node to the appropriate subgraph
    if (subgraphs.find(info.subgraph_instance) != subgraphs.end()) {
      subgraphs[info.subgraph_instance]->add_node(node);
    }
  }

  // Connect nodes
  for (const auto& info : node_infos) {
    if (info.inputs.empty()) {
      continue;
    }

    auto target_node = node_map.at(info.name);

    for (const auto& [input_name, source_name] : info.inputs) {
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

  // Log subgraphs and their nodes
  std::cout << "=== Capture Pipeline Graph Structure ===" << std::endl;
  for (const auto& [subgraph_name, subgraph_ptr] : subgraphs) {
    std::cout << "Subgraph: " << subgraph_name << std::endl;

    // Collect nodes belonging to this subgraph
    std::vector<std::string> nodes_in_subgraph;
    for (const auto& info : node_infos) {
      if (info.subgraph_instance == subgraph_name) {
        nodes_in_subgraph.push_back(info.name);
      }
    }

    std::cout << "  Nodes (" << nodes_in_subgraph.size() << "):" << std::endl;
    for (const auto& node_name : nodes_in_subgraph) {
      const auto& info = *std::find_if(
          node_infos.begin(), node_infos.end(),
          [&node_name](const stargazer::node_info& i) { return i.name == node_name; });
      std::cout << "    - " << node_name << " (type: ";

      // Print node type name
      switch (info.get_type()) {
        case node_type::libcamera_capture:
          std::cout << "libcamera_capture";
          break;
        case node_type::timestamp:
          std::cout << "timestamp";
          break;
        case node_type::broadcast_talker:
          std::cout << "broadcast_talker";
          break;
        case node_type::broadcast_listener:
          std::cout << "broadcast_listener";
          break;
        case node_type::encode_jpeg:
          std::cout << "encode_jpeg";
          break;
        case node_type::decode_jpeg:
          std::cout << "decode_jpeg";
          break;
        case node_type::scale:
          std::cout << "scale";
          break;
        case node_type::resize:
          std::cout << "resize";
          break;
        case node_type::gaussian_blur:
          std::cout << "gaussian_blur";
          break;
        case node_type::mask:
          std::cout << "mask";
          break;
        case node_type::p2p_tcp_talker:
          std::cout << "p2p_tcp_talker";
          break;
        case node_type::p2p_tcp_listener:
          std::cout << "p2p_tcp_listener";
          break;
        case node_type::fifo:
          std::cout << "fifo";
          break;
        case node_type::video_time_sync_control:
          std::cout << "video_time_sync_control";
          break;
        case node_type::fast_blob_detector:
          std::cout << "fast_blob_detector";
          break;
        case node_type::detect_circle_grid:
          std::cout << "detect_circle_grid";
          break;
        case node_type::load_blob:
          std::cout << "load_blob";
          break;
        case node_type::load_marker:
          std::cout << "load_marker";
          break;
        case node_type::load_panoptic:
          std::cout << "load_panoptic";
          break;
        case node_type::approximate_time_sync:
          std::cout << "approximate_time_sync";
          break;
        case node_type::callback:
          std::cout << "callback";
          break;
        default:
          std::cout << "unknown";
          break;
      }

      std::cout << ")" << std::endl;

      // Print node parameters
      if (info.contains_param("fps")) {
        std::cout << "      fps: " << info.get_param<double>("fps") << std::endl;
      }
      if (info.contains_param("interval")) {
        std::cout << "      interval: " << info.get_param<double>("interval") << std::endl;
      }
      if (info.contains_param("num_threads")) {
        std::cout << "      num_threads: " << info.get_param<std::int64_t>("num_threads")
                  << std::endl;
      }

      // Print callback node details
      if (info.get_type() == node_type::callback) {
        auto node_it = node_map.find(node_name);
        if (node_it != node_map.end()) {
          auto callback_node_ptr = std::dynamic_pointer_cast<callback_node>(node_it->second);
          if (callback_node_ptr) {
            std::cout << "      callback_name: " << callback_node_ptr->get_callback_name()
                      << std::endl;
            if (!callback_node_ptr->get_camera_name().empty()) {
              std::cout << "      camera_name: " << callback_node_ptr->get_camera_name()
                        << std::endl;
            }
            std::cout << "      callback_type: ";
            switch (callback_node_ptr->get_callback_type()) {
              case callback_node::callback_type::image:
                std::cout << "image";
                break;
              case callback_node::callback_type::marker:
                std::cout << "marker";
                break;
              case callback_node::callback_type::object:
                std::cout << "object";
                break;
              default:
                std::cout << "unknown";
                break;
            }
            std::cout << std::endl;
          }
        }
      }

      // Print inputs if any
      if (!info.inputs.empty()) {
        std::cout << "      inputs:" << std::endl;
        for (const auto& [input_name, source_name] : info.inputs) {
          std::cout << "        " << input_name << " <- " << source_name << std::endl;
        }
      }
    }
    std::cout << std::endl;
  }
  std::cout << "========================================" << std::endl;
}

class multiview_capture_pipeline::impl {
  local_server server;
  asio::io_context io_context;
  graph_proc_client client;
  std::unique_ptr<std::thread> io_thread;

  mutable std::mutex frames_mtx;
  std::map<std::string, cv::Mat> frames;

  mutable std::mutex image_received_mtx;
  std::vector<std::function<void(const std::map<std::string, cv::Mat>&)>> image_received;

  std::map<std::string, std::shared_ptr<mask_node>> mask_nodes;
  std::map<std::string, cv::Mat> masks;

  mutable std::mutex marker_collecting_clusters_mtx;
  std::unordered_set<std::string> marker_collecting_clusters;

  mutable std::mutex frame_received_mtx;
  std::vector<std::function<void(const std::map<std::string, marker_frame_data>&)>> marker_received;

 public:
  void add_marker_received(std::function<void(const std::map<std::string, marker_frame_data>&)> f) {
    std::lock_guard lock(frame_received_mtx);
    marker_received.push_back(f);
  }

  void clear_marker_received() {
    std::lock_guard lock(frame_received_mtx);
    marker_received.clear();
  }

  void add_image_received(std::function<void(const std::map<std::string, cv::Mat>&)> f) {
    std::lock_guard lock(image_received_mtx);
    image_received.push_back(f);
  }

  void clear_image_received() {
    std::lock_guard lock(image_received_mtx);
    image_received.clear();
  }

  impl(const std::map<std::string, cv::Mat>& masks) : server(0), masks(masks) {}

  void run(const std::vector<node_info>& infos) {
    std::vector<node_info> node_infos = infos;

    // Group nodes by subgraph instance
    std::map<std::string, std::vector<node_info>> nodes_by_subgraph;
    for (const auto& info : node_infos) {
      nodes_by_subgraph[info.subgraph_instance].push_back(info);
    }

    // Create a global node map shared across all subgraphs
    std::unordered_map<std::string, std::shared_ptr<graph_node>> global_node_map;

    // Create empty subgraphs first
    std::map<std::string, std::shared_ptr<subgraph>> subgraphs;
    for (const auto& [subgraph_name, nodes] : nodes_by_subgraph) {
      subgraphs[subgraph_name] = std::make_shared<subgraph>();
    }

    // Build all subgraphs in one pass
    build_graph_from_json(node_infos, subgraphs, global_node_map);

    const auto callbacks = std::make_shared<callback_list>();

    callbacks->add([this](const callback_node* node, std::string input_name,
                          graph_message_ptr message) {
      const auto& callback_name = node->get_callback_name();
      const auto& camera_name = node->get_camera_name();
      const auto callback_type = node->get_callback_type();

      // Handle individual image callback
      if (callback_name == "image") {
        if (auto image_msg = std::dynamic_pointer_cast<frame_message<image>>(message)) {
          const auto& img = image_msg->get_data();

          int type = -1;
          if (image_msg->get_profile()) {
            auto format = image_msg->get_profile()->get_format();
            type = stream_format_to_cv_type(format);
          }

          if (type < 0) {
            throw std::logic_error("Unknown image format");
          }

          cv::Mat frame = cv::Mat(img.get_height(), img.get_width(), type, (uchar*)img.get_data(),
                                  img.get_stride())
                              .clone();

          {
            std::lock_guard lock(frames_mtx);
            this->frames[camera_name] = frame;
          }
        }
      } else if (callback_name == "marker") {
        {
          std::lock_guard lock(marker_collecting_clusters_mtx);
          if (marker_collecting_clusters.empty() ||
              marker_collecting_clusters.find(camera_name) == marker_collecting_clusters.end()) {
            return;
          }
        }

        if (const auto keypoints_msg = std::dynamic_pointer_cast<keypoint_frame_message>(message)) {
          const auto& keypoints = keypoints_msg->get_data();

          marker_frame_data frame_data;
          for (const auto& keypoint : keypoints) {
            marker_data kp;
            kp.x = keypoint.pt_x;
            kp.y = keypoint.pt_y;
            kp.r = keypoint.size;
            frame_data.markers.push_back(kp);
          }

          frame_data.timestamp = keypoints_msg->get_timestamp();
          frame_data.frame_number = keypoints_msg->get_frame_number();

          std::vector<std::function<void(const std::map<std::string, marker_frame_data>&)>>
              marker_received;
          {
            std::lock_guard lock(frame_received_mtx);
            marker_received = this->marker_received;
          }

          for (const auto& f : marker_received) {
            std::map<std::string, marker_frame_data> frames_map;
            frames_map[camera_name] = frame_data;
            f(frames_map);
          }
        }
      }

      // Keep old format support for backward compatibility
      if (callback_name == "images") {
        if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message)) {
          // Use local variable for aggregated images, don't update this->frames
          std::map<std::string, cv::Mat> frames;
          for (const auto& [name, field] : obj_msg->get_fields()) {
            if (auto image_msg = std::dynamic_pointer_cast<frame_message<image>>(field)) {
              const auto& image = image_msg->get_data();

              int type = -1;
              if (image_msg->get_profile()) {
                auto format = image_msg->get_profile()->get_format();
                type = stream_format_to_cv_type(format);
              }

              if (type < 0) {
                throw std::logic_error("Unknown image format");
              }

              frames.insert(
                  std::make_pair(name, cv::Mat(image.get_height(), image.get_width(), type,
                                               (uchar*)image.get_data(), image.get_stride())
                                           .clone()));
            }
          }

          // Don't update this->frames here - it's for individual image display only
          std::vector<std::function<void(const std::map<std::string, cv::Mat>&)>> image_received;
          {
            std::lock_guard lock(image_received_mtx);
            image_received = this->image_received;
          }

          for (const auto& f : image_received) {
            f(frames);
          }
        }
      } else if (node->get_callback_name() == "markers") {
        {
          std::lock_guard lock(marker_collecting_clusters_mtx);
          if (marker_collecting_clusters.empty()) {
            return;
          }
        }
        if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message)) {
          // Use local variable for aggregated markers
          std::map<std::string, marker_frame_data> frames;
          for (const auto& [name, field] : obj_msg->get_fields()) {
            {
              std::lock_guard lock(marker_collecting_clusters_mtx);
              if (marker_collecting_clusters.find(name) == marker_collecting_clusters.end()) {
                continue;
              }
            }
            if (const auto keypoints_msg =
                    std::dynamic_pointer_cast<keypoint_frame_message>(field)) {
              const auto& keypoints = keypoints_msg->get_data();

              marker_frame_data frame;
              for (const auto& keypoint : keypoints) {
                marker_data kp;
                kp.x = keypoint.pt_x;
                kp.y = keypoint.pt_y;
                kp.r = keypoint.size;
                frame.markers.push_back(kp);
              }

              frame.timestamp = keypoints_msg->get_timestamp();
              frame.frame_number = keypoints_msg->get_frame_number();

              frames.insert(std::make_pair(name, frame));
            }
          }

          std::vector<std::function<void(const std::map<std::string, marker_frame_data>&)>>
              marker_received;
          {
            std::lock_guard lock(frame_received_mtx);
            marker_received = this->marker_received;
          }

          for (const auto& f : marker_received) {
            f(frames);
          }
        }
      }
    });

    server.add_resource(callbacks);
    server.run();

    // Build dependency graph for subgraphs based on p2p connections
    // subgraph_dependencies[target] = {source1, source2, ...}
    std::map<std::string, std::set<std::string>> subgraph_dependencies;

    for (const auto& info : node_infos) {
      const auto& target_subgraph = info.subgraph_instance;

      for (const auto& [input_name, source_name] : info.inputs) {
        // Extract node name from source (might be "node" or "node:output")
        size_t pos = source_name.find(':');
        std::string source_node_name =
            (pos != std::string::npos) ? source_name.substr(0, pos) : source_name;

        // Find which subgraph the source node belongs to
        for (const auto& source_info : node_infos) {
          if (source_info.name == source_node_name) {
            const auto& source_subgraph = source_info.subgraph_instance;
            if (source_subgraph != target_subgraph) {
              // Cross-subgraph dependency found
              subgraph_dependencies[target_subgraph].insert(source_subgraph);
            }
            break;
          }
        }
      }
    }

    // Step 1: Get deploy address for each subgraph
    std::map<std::string, std::pair<std::string, uint16_t>> subgraph_deploy_info;

    for (const auto& [subgraph_name, graph] : subgraphs) {
      std::string deploy_address = "127.0.0.1";
      uint16_t deploy_port = server.get_port();

      for (const auto& info : nodes_by_subgraph[subgraph_name]) {
        if (info.contains_param("address")) {
          deploy_address = info.get_param<std::string>("address");
        }
        if (info.contains_param("deploy_port")) {
          deploy_port = static_cast<uint16_t>(info.get_param<std::int64_t>("deploy_port"));
        }
      }

      subgraph_deploy_info[subgraph_name] = std::make_pair(deploy_address, deploy_port);
    }

    // Step 2: Group subgraphs by deploy address and create merged subgraphs
    std::map<std::pair<std::string, uint16_t>, std::vector<std::string>> address_groups;

    for (const auto& [subgraph_name, deploy_key] : subgraph_deploy_info) {
      address_groups[deploy_key].push_back(subgraph_name);
    }

    // Create merged subgraphs and mapping from original to merged group name
    std::map<std::string, std::shared_ptr<subgraph>> deploy_subgraphs;
    std::map<std::string, std::string> original_to_merged;
    std::map<std::string, std::pair<std::string, uint16_t>> merged_deploy_info;
    std::map<std::string, std::vector<std::string>> merged_subgraph_members;

    for (const auto& [deploy_key, subgraph_names] : address_groups) {
      if (subgraph_names.size() == 1) {
        // Single subgraph - no merge needed
        const auto& name = subgraph_names[0];
        deploy_subgraphs[name] = subgraphs[name];
        original_to_merged[name] = name;
        merged_deploy_info[name] = deploy_key;
        merged_subgraph_members[name] = {name};
      } else {
        // Multiple subgraphs - merge them
        auto merged = std::make_shared<subgraph>();
        std::string merged_name = "merged_" + std::to_string(deploy_key.second);

        for (const auto& name : subgraph_names) {
          merged->merge(*subgraphs[name]);
          original_to_merged[name] = merged_name;
        }

        deploy_subgraphs[merged_name] = merged;
        merged_deploy_info[merged_name] = deploy_key;
        merged_subgraph_members[merged_name] = subgraph_names;
      }
    }

    // Step 3: Build dependency graph based on merged subgraphs
    std::map<std::string, std::set<std::string>> merged_dependencies;

    for (const auto& info : node_infos) {
      const auto& target_subgraph = info.subgraph_instance;
      const auto& target_merged = original_to_merged[target_subgraph];

      for (const auto& [input_name, source_name] : info.inputs) {
        size_t pos = source_name.find(':');
        std::string source_node_name =
            (pos != std::string::npos) ? source_name.substr(0, pos) : source_name;

        for (const auto& source_info : node_infos) {
          if (source_info.name == source_node_name) {
            const auto& source_subgraph = source_info.subgraph_instance;
            const auto& source_merged = original_to_merged[source_subgraph];

            if (source_merged != target_merged) {
              merged_dependencies[target_merged].insert(source_merged);
            }
            break;
          }
        }
      }
    }

    // Step 4: Topological sort on merged subgraphs
    std::vector<std::string> deploy_order;
    std::set<std::string> deployed;
    std::set<std::string> visiting;

    std::function<void(const std::string&)> visit = [&](const std::string& merged_name) {
      if (deployed.count(merged_name)) return;
      if (visiting.count(merged_name)) {
        throw std::runtime_error("Circular dependency detected in subgraph dependencies");
      }

      visiting.insert(merged_name);

      if (merged_dependencies.count(merged_name)) {
        for (const auto& dep : merged_dependencies[merged_name]) {
          visit(dep);
        }
      }

      visiting.erase(merged_name);
      deployed.insert(merged_name);
      deploy_order.push_back(merged_name);
    };

    for (const auto& [merged_name, graph] : deploy_subgraphs) {
      visit(merged_name);
    }

    // Step 5: Deploy in topological order
    std::cout << "=== Deploying Subgraphs ===" << std::endl;

    for (const auto& merged_name : deploy_order) {
      const auto& [deploy_address, deploy_port] = merged_deploy_info[merged_name];
      const auto& members = merged_subgraph_members[merged_name];

      if (members.size() == 1) {
        std::cout << "Subgraph: " << members[0] << std::endl;
      } else {
        std::cout << "Merged subgraph (";
        for (size_t i = 0; i < members.size(); ++i) {
          if (i > 0) std::cout << " + ";
          std::cout << members[i];
        }
        std::cout << ")" << std::endl;
      }

      std::cout << "  Deploy address: " << deploy_address << ":" << deploy_port << std::endl;

      if (merged_dependencies.count(merged_name) && !merged_dependencies[merged_name].empty()) {
        std::cout << "  Dependencies: ";
        bool first = true;
        for (const auto& dep : merged_dependencies[merged_name]) {
          if (!first) std::cout << ", ";
          const auto& dep_members = merged_subgraph_members[dep];
          if (dep_members.size() == 1) {
            std::cout << dep_members[0];
          } else {
            std::cout << dep;
          }
          first = false;
        }
        std::cout << std::endl;
      }

      client.deploy(io_context, deploy_address, deploy_port, deploy_subgraphs[merged_name]);
    }

    std::cout << "===========================" << std::endl;

    io_thread.reset(new std::thread([this] { io_context.run(); }));

    client.run();
  }

  void stop() {
    client.stop();
    server.stop();
    io_context.stop();
    if (io_thread && io_thread->joinable()) {
      io_thread->join();
    }
    io_thread.reset();
  }

  std::map<std::string, cv::Mat> get_frames() const {
    std::map<std::string, cv::Mat> result;

    {
      std::lock_guard lock(frames_mtx);
      result = this->frames;
    }

    return result;
  }
  void gen_mask() {
    const auto frames = get_frames();
    for (const auto& [name, mask_node] : mask_nodes) {
      if (frames.find(name) == frames.end()) {
        continue;
      }
      const auto frame_img = frames.at(name);
      if (frame_img.empty()) {
        continue;
      }

      cv::Mat mask_img;
      {
        cv::threshold(frame_img, mask_img, 128, 255, cv::THRESH_BINARY);
        cv::morphologyEx(mask_img, mask_img, cv::MORPH_OPEN,
                         cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2)));
        cv::dilate(mask_img, mask_img,
                   cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10)));
        cv::bitwise_not(mask_img, mask_img);
      }

      image mask(mask_img.cols, mask_img.rows, CV_8UC1, mask_img.step,
                 (const uint8_t*)mask_img.data);
      const auto mask_msg = std::make_shared<image_message>();
      mask_msg->set_image(mask);
      client.process(mask_node.get(), "mask", mask_msg);

      masks[name] = mask_img;
    }
  }
  void clear_mask() {
    for (const auto& [name, mask_node] : mask_nodes) {
      const int width = 820;
      const int height = 616;
      cv::Mat mask_img(height, width, CV_8UC1, cv::Scalar(255));
      image mask(mask_img.cols, mask_img.rows, CV_8UC1, mask_img.step,
                 (const uint8_t*)mask_img.data);
      const auto image_msg = std::make_shared<image_message>();
      image_msg->set_image(mask);
      client.process(mask_node.get(), "mask", image_msg);

      masks[name] = mask_img;
    }
  }
  std::map<std::string, cv::Mat> get_masks() const { return masks; }

  void enable_marker_collecting(std::string name) {
    std::lock_guard lock(marker_collecting_clusters_mtx);
    marker_collecting_clusters.insert(name);
  }
  void disable_marker_collecting(std::string name) {
    std::lock_guard lock(marker_collecting_clusters_mtx);
    const auto found = marker_collecting_clusters.find(name);
    if (found != marker_collecting_clusters.end()) {
      marker_collecting_clusters.erase(found);
    }
  }
};

multiview_capture_pipeline::multiview_capture_pipeline()
    : pimpl(new impl(std::map<std::string, cv::Mat>())) {}
multiview_capture_pipeline::multiview_capture_pipeline(const std::map<std::string, cv::Mat>& masks)
    : pimpl(new impl(masks)) {}
multiview_capture_pipeline::~multiview_capture_pipeline() = default;

void multiview_capture_pipeline::run(const std::vector<node_info>& infos) { pimpl->run(infos); }

void multiview_capture_pipeline::stop() { pimpl->stop(); }
std::map<std::string, cv::Mat> multiview_capture_pipeline::get_frames() const {
  return pimpl->get_frames();
}
void multiview_capture_pipeline::gen_mask() { pimpl->gen_mask(); }
void multiview_capture_pipeline::clear_mask() { pimpl->clear_mask(); }
std::map<std::string, cv::Mat> multiview_capture_pipeline::get_masks() const {
  return pimpl->get_masks();
}

void multiview_capture_pipeline::enable_marker_collecting(std::string name) {
  pimpl->enable_marker_collecting(name);
}
void multiview_capture_pipeline::disable_marker_collecting(std::string name) {
  pimpl->disable_marker_collecting(name);
}
void multiview_capture_pipeline::add_marker_received(
    std::function<void(const std::map<std::string, marker_frame_data>&)> f) {
  pimpl->add_marker_received(f);
}
void multiview_capture_pipeline::clear_marker_received() { pimpl->clear_marker_received(); }
void multiview_capture_pipeline::add_image_received(
    std::function<void(const std::map<std::string, cv::Mat>&)> f) {
  pimpl->add_image_received(f);
}
void multiview_capture_pipeline::clear_image_received() { pimpl->clear_image_received(); }
