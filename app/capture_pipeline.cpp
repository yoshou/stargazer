#include "capture_pipeline.hpp"

#include <sqlite3.h>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <regex>
#include <set>

#include "callback_node.hpp"
#include "coalsack/core/graph_proc.h"
#include "coalsack/core/graph_proc_client.h"
#include "coalsack/core/graph_proc_server.h"
#include "coalsack/ext/graph_proc_cv_ext.h"
#include "coalsack/ext/graph_proc_depthai.h"
#include "coalsack/ext/graph_proc_jpeg.h"
#include "coalsack/ext/graph_proc_libcamera.h"
#include "coalsack/ext/graph_proc_rs_d435.h"
#include "coalsack/image/graph_proc_cv.h"
#include "dump_blob_node.hpp"
#include "dump_keypoint_node.hpp"
#include "graph_builder.hpp"
#include "load_blob_node.hpp"
#include "load_marker_node.hpp"
#include "load_panoptic_node.hpp"

using namespace coalsack;
using namespace stargazer;

class capture_pipeline::impl {
  std::unordered_map<std::string, std::shared_ptr<graph_node>> node_map;
  graph_proc local_graph;
  std::shared_ptr<subgraph> local_subgraph;
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

  bool has_remote_subgraphs = false;

  bool is_local_node(const graph_node* node) const {
    return node && local_subgraph && node->get_parent() == local_subgraph.get();
  }

  void process_node(const graph_node* node, const std::string& input_name,
                    const graph_message_ptr& message) {
    if (!node) {
      return;
    }
    if (is_local_node(node)) {
      local_graph.process(node, input_name, message);
      return;
    }
    client.process(node, input_name, message);
  }

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

  impl(const std::map<std::string, cv::Mat>& masks)
      : node_map(),
        local_graph(),
        local_subgraph(),
        io_context(),
        client(),
        io_thread(),
        frames_mtx(),
        frames(),
        image_received_mtx(),
        image_received(),
        mask_nodes(),
        masks(masks),
        marker_collecting_clusters_mtx(),
        marker_collecting_clusters(),
        frame_received_mtx(),
        marker_received(),
        has_remote_subgraphs(false) {}

  void run(const std::vector<node_def>& nodes) {
    node_map.clear();
    local_graph = graph_proc();
    local_subgraph.reset();
    client = graph_proc_client();
    io_context.restart();
    has_remote_subgraphs = false;

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
    node_map = global_node_map;

    const auto callbacks = std::make_shared<callback_list>();

    callbacks->add([this](const callback_node* node, std::string input_name,
                          graph_message_ptr message) {
      (void)input_name;
      const auto& callback_name = node->get_callback_name();
      const auto& camera_name = node->get_camera_name();
      const auto callback_type = node->get_callback_type();
      (void)callback_type;

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

          std::vector<std::function<void(const std::map<std::string, cv::Mat>&)>> image_received;
          {
            std::lock_guard lock(image_received_mtx);
            image_received = this->image_received;
          }

          for (const auto& f : image_received) {
            std::map<std::string, cv::Mat> frames_map;
            frames_map[camera_name] = frame;
            f(frames_map);
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

    local_graph.get_resources()->add(callbacks);

    std::map<std::string, std::pair<std::string, uint16_t>> remote_subgraph_deploy_info;
    std::vector<std::string> local_subgraph_names;

    for (const auto& [subgraph_name, subgraph_nodes] : nodes_by_subgraph) {
      bool is_remote = false;
      std::string deploy_address = "127.0.0.1";
      uint16_t deploy_port = 0;

      for (const auto& node : subgraph_nodes) {
        if (node.contains_param("deploy_port")) {
          is_remote = true;
          deploy_port = static_cast<uint16_t>(node.get_param<std::int64_t>("deploy_port"));
        }
        if (node.contains_param("address")) {
          deploy_address = node.get_param<std::string>("address");
        }
      }

      if (is_remote) {
        remote_subgraph_deploy_info[subgraph_name] = std::make_pair(deploy_address, deploy_port);
      } else {
        local_subgraph_names.push_back(subgraph_name);
      }
    }

    if (!local_subgraph_names.empty()) {
      local_subgraph = std::make_shared<subgraph>();
      for (const auto& subgraph_name : local_subgraph_names) {
        local_subgraph->merge(*subgraphs.at(subgraph_name));
      }
    }

    // Group remote subgraphs by deploy address and create merged subgraphs
    std::map<std::pair<std::string, uint16_t>, std::vector<std::string>> address_groups;

    for (const auto& [subgraph_name, deploy_key] : remote_subgraph_deploy_info) {
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

    // Build dependency graph based on merged remote subgraphs
    std::map<std::string, std::set<std::string>> merged_dependencies;

    for (const auto& node : nodes) {
      const auto& target_subgraph = node.subgraph_instance;
      if (remote_subgraph_deploy_info.find(target_subgraph) == remote_subgraph_deploy_info.end()) {
        continue;
      }
      const auto& target_merged = original_to_merged[target_subgraph];

      for (const auto& [input_name, source_name] : node.inputs) {
        size_t pos = source_name.find(':');
        std::string source_node_name =
            (pos != std::string::npos) ? source_name.substr(0, pos) : source_name;

        for (const auto& source_node : nodes) {
          if (source_node.name == source_node_name) {
            const auto& source_subgraph = source_node.subgraph_instance;
            if (remote_subgraph_deploy_info.find(source_subgraph) ==
                remote_subgraph_deploy_info.end()) {
              break;
            }
            const auto& source_merged = original_to_merged[source_subgraph];

            if (source_merged != target_merged) {
              merged_dependencies[target_merged].insert(source_merged);
            }
            break;
          }
        }
      }
    }

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

    if (local_subgraph && local_subgraph->get_node_count() > 0) {
      local_graph.deploy(local_subgraph);
      local_graph.run();
    }

    if (!deploy_order.empty()) {
      has_remote_subgraphs = true;
      for (const auto& merged_name : deploy_order) {
        const auto& [deploy_address, deploy_port] = merged_deploy_info[merged_name];
        client.deploy(io_context, deploy_address, deploy_port, deploy_subgraphs[merged_name]);
      }

      io_thread.reset(new std::thread([this] { io_context.run(); }));
      client.run();
    }
  }

  void stop() {
    if (has_remote_subgraphs) {
      client.stop();
      has_remote_subgraphs = false;
    }
    if (local_subgraph && local_subgraph->get_node_count() > 0) {
      local_graph.stop();
    }
    io_context.stop();
    if (io_thread && io_thread->joinable()) {
      io_thread->join();
    }
    io_thread.reset();
    local_subgraph.reset();
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
      process_node(mask_node.get(), "mask", mask_msg);

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
      process_node(mask_node.get(), "mask", image_msg);

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

  std::optional<property_value> get_node_property(const std::string& node_name,
                                                  const std::string& key) const {
    const auto found = node_map.find(node_name);
    if (found == node_map.end() || !found->second || !is_local_node(found->second.get())) {
      return std::nullopt;
    }
    return found->second->get_property(key);
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

std::optional<property_value> capture_pipeline::get_node_property(const std::string& node_name,
                                                                  const std::string& key) const {
  return pimpl->get_node_property(node_name, key);
}
