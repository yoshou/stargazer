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
#include "graph_builder.hpp"
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

class capture_pipeline::impl {
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

  void run(const std::vector<node_def>& nodes) {
    // Group nodes by subgraph instance
    std::map<std::string, std::vector<node_def>> nodes_by_subgraph;
    for (const auto& node : nodes) {
      nodes_by_subgraph[node.subgraph_instance].push_back(node);
    }

    // Create a global node map shared across all subgraphs
    std::unordered_map<std::string, std::shared_ptr<graph_node>> global_node_map;

    // Create empty subgraphs first
    std::map<std::string, std::shared_ptr<subgraph>> subgraphs;
    for (const auto& [subgraph_name, nodes] : nodes_by_subgraph) {
      subgraphs[subgraph_name] = std::make_shared<subgraph>();
    }

    // Build all subgraphs in one pass
    stargazer::build_graph_from_json(nodes, subgraphs, global_node_map);

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

    for (const auto& node : nodes) {
      const auto& target_subgraph = node.subgraph_instance;

      for (const auto& [input_name, source_name] : node.inputs) {
        // Extract node name from source (might be "node" or "node:output")
        size_t pos = source_name.find(':');
        std::string source_node_name =
            (pos != std::string::npos) ? source_name.substr(0, pos) : source_name;

        // Find which subgraph the source node belongs to
        for (const auto& source_node : nodes) {
          if (source_node.name == source_node_name) {
            const auto& source_subgraph = source_node.subgraph_instance;
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

      for (const auto& node : nodes_by_subgraph[subgraph_name]) {
        if (node.contains_param("address")) {
          deploy_address = node.get_param<std::string>("address");
        }
        if (node.contains_param("deploy_port")) {
          deploy_port = static_cast<uint16_t>(node.get_param<std::int64_t>("deploy_port"));
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

    for (const auto& node : nodes) {
      const auto& target_subgraph = node.subgraph_instance;
      const auto& target_merged = original_to_merged[target_subgraph];

      for (const auto& [input_name, source_name] : node.inputs) {
        size_t pos = source_name.find(':');
        std::string source_node_name =
            (pos != std::string::npos) ? source_name.substr(0, pos) : source_name;

        for (const auto& source_node : nodes) {
          if (source_node.name == source_node_name) {
            const auto& source_subgraph = source_node.subgraph_instance;
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

capture_pipeline::capture_pipeline() : pimpl(new impl(std::map<std::string, cv::Mat>())) {}
capture_pipeline::capture_pipeline(const std::map<std::string, cv::Mat>& masks)
    : pimpl(new impl(masks)) {}
capture_pipeline::~capture_pipeline() = default;

void capture_pipeline::run(const std::vector<node_def>& nodes) { pimpl->run(nodes); }

void capture_pipeline::stop() { pimpl->stop(); }
std::map<std::string, cv::Mat> capture_pipeline::get_frames() const { return pimpl->get_frames(); }
void capture_pipeline::gen_mask() { pimpl->gen_mask(); }
void capture_pipeline::clear_mask() { pimpl->clear_mask(); }
std::map<std::string, cv::Mat> capture_pipeline::get_masks() const { return pimpl->get_masks(); }

void capture_pipeline::enable_marker_collecting(std::string name) {
  pimpl->enable_marker_collecting(name);
}
void capture_pipeline::disable_marker_collecting(std::string name) {
  pimpl->disable_marker_collecting(name);
}
void capture_pipeline::add_marker_received(
    std::function<void(const std::map<std::string, marker_frame_data>&)> f) {
  pimpl->add_marker_received(f);
}
void capture_pipeline::clear_marker_received() { pimpl->clear_marker_received(); }
void capture_pipeline::add_image_received(
    std::function<void(const std::map<std::string, cv::Mat>&)> f) {
  pimpl->add_image_received(f);
}
void capture_pipeline::clear_image_received() { pimpl->clear_image_received(); }
