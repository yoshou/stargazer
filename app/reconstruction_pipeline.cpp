#include "reconstruction_pipeline.hpp"

#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <spdlog/spdlog.h>
#include <sqlite3.h>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <regex>

#include "callback_node.hpp"
#include "capture_pipeline.hpp"
#include "correspondance.hpp"
#include "dump_se3_node.hpp"
#include "epipolar_reconstruct_node.hpp"
#include "glm_json.hpp"
#include "glm_serialize.hpp"
#include "graph_proc.h"
#include "graph_proc_cv.h"
#include "graph_proc_img.h"
#include "graph_proc_tensor.h"
#include "grpc_server_node.hpp"
#include "messages.hpp"
#include "mvp.hpp"
#include "mvpose.hpp"
#include "parameters.hpp"
#include "point_data.hpp"
#include "reconstruction.hpp"
#include "sensor.grpc.pb.h"
#include "triangulation.hpp"
#include "utils.hpp"
#include "voxelpose.hpp"

using namespace coalsack;
using namespace stargazer;

class multiview_point_reconstruction_pipeline::impl {
  graph_proc graph;

  std::atomic_bool running;

  mutable std::mutex markers_mtx;
  std::vector<glm::vec3> markers;
  std::vector<std::function<void(const std::vector<glm::vec3>&)>> markers_received;

  std::shared_ptr<epipolar_reconstruct_node> reconstruct_node;
  std::shared_ptr<graph_node> input_node;

 public:
  void add_markers_received(std::function<void(const std::vector<glm::vec3>&)> f) {
    std::lock_guard lock(markers_mtx);
    markers_received.push_back(f);
  }

  void clear_markers_received() {
    std::lock_guard lock(markers_mtx);
    markers_received.clear();
  }

  impl()
      : graph(), running(false), markers(), markers_received(), reconstruct_node(), input_node() {}

  void set_camera(const std::string& name, const camera_t& camera) {
    auto camera_msg = std::make_shared<camera_message>(camera);
    camera_msg->set_camera(camera);

    auto obj_msg = std::make_shared<object_message>();
    obj_msg->add_field(name, camera_msg);

    if (reconstruct_node) {
      graph.process(reconstruct_node.get(), "cameras", obj_msg);
    }
  }

  void set_axis(const glm::mat4& axis) {
    auto mat4_msg = std::make_shared<mat4_message>();
    mat4_msg->set_data(axis);

    if (reconstruct_node) {
      graph.process(reconstruct_node.get(), "axis", mat4_msg);
    }
  }

  using frame_type = std::map<std::string, std::vector<point_data>>;

  void push_frame(const frame_type& frame) {
    if (!running) {
      return;
    }

    auto msg = std::make_shared<object_message>();
    for (const auto& [name, field] : frame) {
      auto float2_msg = std::make_shared<float2_list_message>();
      std::vector<float2> float2_data;
      for (const auto& pt : field) {
        float2_data.push_back({pt.point.x, pt.point.y});
      }
      float2_msg->set_data(float2_data);
      msg->add_field(name, float2_msg);
    }

    auto frame_msg = std::make_shared<frame_message<object_message>>();
    frame_msg->set_data(*msg);

    if (input_node) {
      graph.process(input_node.get(), frame_msg);
    }
  }

  void run() {
    std::shared_ptr<subgraph> g(new subgraph());

    std::shared_ptr<frame_number_numbering_node> n4(new frame_number_numbering_node());
    g->add_node(n4);

    input_node = n4;

    std::shared_ptr<parallel_queue_node> n6(new parallel_queue_node());
    n6->set_input(n4->get_output());
    g->add_node(n6);

    std::shared_ptr<epipolar_reconstruct_node> n1(new epipolar_reconstruct_node());
    n1->set_input(n6->get_output());
    g->add_node(n1);

    reconstruct_node = n1;

    std::shared_ptr<frame_number_ordering_node> n5(new frame_number_ordering_node());
    n5->set_input(n1->get_output());
    g->add_node(n5);

    std::shared_ptr<callback_node> n2(new callback_node());
    n2->set_input(n5->get_output());
    g->add_node(n2);

    n2->set_callback_name("markers");

    std::shared_ptr<grpc_server_node> n3(new grpc_server_node());
    n3->set_input(n5->get_output(), "sphere");
    g->add_node(n3);

    const auto callbacks = std::make_shared<callback_list>();

    callbacks->add(
        [this](const callback_node* node, std::string input_name, graph_message_ptr message) {
          if (node->get_callback_name() == "markers") {
            if (const auto markers_msg = std::dynamic_pointer_cast<float3_list_message>(message)) {
              std::vector<glm::vec3> markers;
              for (const auto& marker : markers_msg->get_data()) {
                markers.push_back(glm::vec3(marker.x, marker.y, marker.z));
              }

              {
                std::lock_guard lock(markers_mtx);
                this->markers = markers;
              }

              for (const auto& f : markers_received) {
                f(markers);
              }
            }
          }
        });

    graph.deploy(g);
    graph.get_resources()->add(callbacks);
    graph.run();

    running = true;
  }

  void stop() {
    running.store(false);
    graph.stop();
  }

  std::vector<glm::vec3> get_markers() const {
    std::vector<glm::vec3> result;

    {
      std::lock_guard lock(markers_mtx);
      result = this->markers;
    }

    return result;
  }
};

multiview_point_reconstruction_pipeline::multiview_point_reconstruction_pipeline()
    : pimpl(new impl()) {}
multiview_point_reconstruction_pipeline::~multiview_point_reconstruction_pipeline() = default;

void multiview_point_reconstruction_pipeline::push_frame(const frame_type& frame) {
  pimpl->push_frame(frame);
}

void multiview_point_reconstruction_pipeline::run() { pimpl->run(); }

void multiview_point_reconstruction_pipeline::stop() { pimpl->stop(); }

std::vector<glm::vec3> multiview_point_reconstruction_pipeline::get_markers() const {
  return pimpl->get_markers();
}

void multiview_point_reconstruction_pipeline::set_camera(const std::string& name,
                                                         const camera_t& camera) {
  cameras[name] = camera;
  pimpl->set_camera(name, camera);
}
void multiview_point_reconstruction_pipeline::set_axis(const glm::mat4& axis) {
  this->axis = axis;
  pimpl->set_axis(axis);
}

#define PANOPTIC

class image_reconstruct_node : public graph_node {
 public:
  virtual std::map<std::string, cv::Mat> get_features() const = 0;
};

COALSACK_REGISTER_NODE(image_reconstruct_node, graph_node)

class voxelpose_reconstruct_node : public image_reconstruct_node {
  mutable std::mutex cameras_mtx;
  std::map<std::string, camera_t> cameras;
  glm::mat4 axis;
  graph_edge_ptr output;

  std::vector<std::string> names;
  coalsack::tensor<float, 4> features;
  mutable std::mutex features_mtx;

  stargazer::voxelpose::voxelpose pose_estimator;

 public:
  voxelpose_reconstruct_node()
      : image_reconstruct_node(), cameras(), axis(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "voxelpose_reconstruct"; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cameras, axis);
  }

  std::vector<glm::vec3> reconstruct(const std::map<std::string, camera_t>& cameras,
                                     const std::map<std::string, cv::Mat>& frame,
                                     const glm::mat4& axis) {
    using namespace stargazer::voxelpose;

    std::vector<std::string> names;
    std::vector<cv::Mat> images_list;
    std::vector<camera_data> cameras_list;

    if (frame.size() <= 1) {
      return std::vector<glm::vec3>();
    }

    for (const auto& [camera_name, image] : frame) {
      names.push_back(camera_name);
    }

    for (size_t i = 0; i < frame.size(); i++) {
      const auto name = names[i];
      camera_data camera;

      const auto& src_camera = cameras.at(name);

      camera.fx = src_camera.intrin.fx;
      camera.fy = src_camera.intrin.fy;
      camera.cx = src_camera.intrin.cx;
      camera.cy = src_camera.intrin.cy;
      camera.k[0] = src_camera.intrin.coeffs[0];
      camera.k[1] = src_camera.intrin.coeffs[1];
      camera.k[2] = src_camera.intrin.coeffs[4];
      camera.p[0] = src_camera.intrin.coeffs[2];
      camera.p[1] = src_camera.intrin.coeffs[3];

      glm::mat4 gl_to_cv(1.f);
      gl_to_cv[0] = glm::vec4(1.f, 0.f, 0.f, 0.f);
      gl_to_cv[1] = glm::vec4(0.f, -1.f, 0.f, 0.f);
      gl_to_cv[2] = glm::vec4(0.f, 0.f, -1.f, 0.f);

      glm::mat4 m(1.f);
      m[0] = glm::vec4(1.f, 0.f, 0.f, 0.f);
      m[1] = glm::vec4(0.f, 0.f, 1.f, 0.f);
      m[2] = glm::vec4(0.f, -1.f, 0.f, 0.f);

      const auto camera_pose =
          axis * glm::inverse(src_camera.extrin.transform_matrix() * gl_to_cv * m);

      for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
          camera.rotation[i][j] = camera_pose[i][j];
        }
        camera.translation[i] = camera_pose[3][i] * 1000.0;
      }

      cameras_list.push_back(camera);
      images_list.push_back(frame.at(name));
    }

#ifdef PANOPTIC
    std::array<float, 3> grid_center = {0.0f, -500.0f, 800.0f};
#else
    std::array<float, 3> grid_center = {0.0f, 0.0f, 0.0f};
#endif

    pose_estimator.set_grid_center(grid_center);

    const auto points = pose_estimator.inference(images_list, cameras_list);

    coalsack::tensor<float, 4> heatmaps(
        {pose_estimator.get_heatmap_width(), pose_estimator.get_heatmap_height(),
         pose_estimator.get_num_joints(), (uint32_t)images_list.size()});
    pose_estimator.copy_heatmap_to(images_list.size(), heatmaps.get_data());

    {
      std::lock_guard lock(features_mtx);
      this->names = names;
      this->features = std::move(heatmaps);
    }

    return points;
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (input_name == "cameras") {
      if (auto camera_msg = std::dynamic_pointer_cast<object_message>(message)) {
        for (const auto& [name, field] : camera_msg->get_fields()) {
          if (auto camera_msg = std::dynamic_pointer_cast<camera_message>(field)) {
            std::lock_guard lock(cameras_mtx);
            cameras[name] = camera_msg->get_camera();
          }
        }
      }

      return;
    }
    if (input_name == "axis") {
      if (auto mat4_msg = std::dynamic_pointer_cast<mat4_message>(message)) {
        axis = mat4_msg->get_data();
      }

      return;
    }

    if (auto frame_msg = std::dynamic_pointer_cast<frame_message<object_message>>(message)) {
      const auto obj_msg = frame_msg->get_data();

      std::map<std::string, camera_t> cameras;
      {
        std::lock_guard lock(cameras_mtx);
        cameras = this->cameras;
      }

      std::map<std::string, cv::Mat> images;

      for (const auto& [name, field] : obj_msg.get_fields()) {
        if (auto img_msg = std::dynamic_pointer_cast<image_message>(field)) {
          if (cameras.find(name) == cameras.end()) {
            continue;
          }
          const auto& image = img_msg->get_image();
          cv::Mat img(image.get_height(), image.get_width(), convert_to_cv_type(image.get_format()),
                      (void*)image.get_data(), image.get_stride());
          images[name] = img;
        }
      }

      const auto markers = reconstruct(cameras, images, axis);

      auto marker_msg = std::make_shared<float3_list_message>();
      std::vector<float3> marker_data;
      for (const auto& marker : markers) {
        marker_data.push_back({marker.x, marker.y, marker.z});
      }
      marker_msg->set_data(marker_data);
      marker_msg->set_frame_number(frame_msg->get_frame_number());

      output->send(marker_msg);
    }
  }

  std::map<std::string, cv::Mat> get_features() const override {
    coalsack::tensor<float, 4> features;
    std::vector<std::string> names;
    {
      std::lock_guard lock(features_mtx);
      features = this->features;
      names = this->names;
    }
    std::map<std::string, cv::Mat> result;
    if (features.get_size() == 0) {
      return result;
    }
    for (size_t i = 0; i < names.size(); i++) {
      const auto name = names[i];
      const auto heatmap =
          features
              .view<3>({features.shape[0], features.shape[1], features.shape[2], 0},
                       {0, 0, 0, static_cast<uint32_t>(i)})
              .contiguous()
              .sum<1>({2});

      cv::Mat heatmap_mat;
      cv::Mat(heatmap.shape[1], heatmap.shape[0], CV_32FC1, (float*)heatmap.get_data())
          .clone()
          .convertTo(heatmap_mat, CV_8U, 255);
      cv::resize(heatmap_mat, heatmap_mat, cv::Size(960, 540));
      cv::cvtColor(heatmap_mat, heatmap_mat, cv::COLOR_GRAY2BGR);

      result[name] = heatmap_mat;
    }
    return result;
  }
};

COALSACK_REGISTER_NODE(voxelpose_reconstruct_node, image_reconstruct_node)

class mvpose_reconstruct_node : public image_reconstruct_node {
  mutable std::mutex cameras_mtx;
  std::map<std::string, camera_t> cameras;
  glm::mat4 axis;
  graph_edge_ptr output;

  std::vector<std::string> names;
  coalsack::tensor<float, 4> features;
  mutable std::mutex features_mtx;

  stargazer::mvpose::mvpose pose_estimator;

 public:
  mvpose_reconstruct_node()
      : image_reconstruct_node(), cameras(), axis(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "mvpose_reconstruct"; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cameras, axis);
  }

  std::vector<glm::vec3> reconstruct(const std::map<std::string, camera_t>& cameras,
                                     const std::map<std::string, cv::Mat>& frame,
                                     const glm::mat4& axis) {
    using namespace stargazer::mvpose;

    std::vector<std::string> names;
    coalsack::tensor<float, 4> heatmaps;
    std::vector<cv::Mat> images_list;
    std::vector<camera_t> cameras_list;

    if (frame.size() <= 1) {
      return std::vector<glm::vec3>();
    }

    for (const auto& [camera_name, image] : frame) {
      names.push_back(camera_name);
    }

    for (size_t i = 0; i < frame.size(); i++) {
      const auto name = names[i];

      camera_t camera;

      const auto& src_camera = cameras.at(name);

      camera.intrin.fx = src_camera.intrin.fx;
      camera.intrin.fy = src_camera.intrin.fy;
      camera.intrin.cx = src_camera.intrin.cx;
      camera.intrin.cy = src_camera.intrin.cy;
      camera.intrin.coeffs[0] = src_camera.intrin.coeffs[0];
      camera.intrin.coeffs[1] = src_camera.intrin.coeffs[1];
      camera.intrin.coeffs[2] = src_camera.intrin.coeffs[2];
      camera.intrin.coeffs[3] = src_camera.intrin.coeffs[3];
      camera.intrin.coeffs[4] = src_camera.intrin.coeffs[4];

      const auto camera_pose = src_camera.extrin.transform_matrix() * glm::inverse(axis);

      for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
          camera.extrin.rotation[i][j] = camera_pose[i][j];
        }
        camera.extrin.translation[i] = camera_pose[3][i];
      }

      cameras_list.push_back(camera);
      images_list.push_back(frame.at(name));
    }

    const auto points = pose_estimator.inference(images_list, cameras_list);

    {
      std::lock_guard lock(features_mtx);
      this->names = names;
      this->features = std::move(heatmaps);
    }

    return points;
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (input_name == "cameras") {
      if (auto camera_msg = std::dynamic_pointer_cast<object_message>(message)) {
        for (const auto& [name, field] : camera_msg->get_fields()) {
          if (auto camera_msg = std::dynamic_pointer_cast<camera_message>(field)) {
            std::lock_guard lock(cameras_mtx);
            cameras[name] = camera_msg->get_camera();
          }
        }
      }

      return;
    }
    if (input_name == "axis") {
      if (auto mat4_msg = std::dynamic_pointer_cast<mat4_message>(message)) {
        axis = mat4_msg->get_data();
      }

      return;
    }

    if (auto frame_msg = std::dynamic_pointer_cast<frame_message<object_message>>(message)) {
      const auto obj_msg = frame_msg->get_data();

      std::map<std::string, camera_t> cameras;
      {
        std::lock_guard lock(cameras_mtx);
        cameras = this->cameras;
      }

      std::map<std::string, cv::Mat> images;

      for (const auto& [name, field] : obj_msg.get_fields()) {
        if (auto img_msg = std::dynamic_pointer_cast<image_message>(field)) {
          if (cameras.find(name) == cameras.end()) {
            continue;
          }
          const auto& image = img_msg->get_image();
          cv::Mat img(image.get_height(), image.get_width(), convert_to_cv_type(image.get_format()),
                      (void*)image.get_data(), image.get_stride());
          images[name] = img;
        }
      }

      const auto markers = reconstruct(cameras, images, axis);

      auto marker_msg = std::make_shared<float3_list_message>();
      std::vector<float3> marker_data;
      for (const auto& marker : markers) {
        marker_data.push_back({marker.x, marker.y, marker.z});
      }
      marker_msg->set_data(marker_data);
      marker_msg->set_frame_number(frame_msg->get_frame_number());

      output->send(marker_msg);
    }
  }

  std::map<std::string, cv::Mat> get_features() const override {
    coalsack::tensor<float, 4> features;
    std::vector<std::string> names;
    {
      std::lock_guard lock(features_mtx);
      features = this->features;
      names = this->names;
    }
    std::map<std::string, cv::Mat> result;
    if (features.get_size() == 0) {
      return result;
    }
    for (size_t i = 0; i < names.size(); i++) {
      const auto name = names[i];
      const auto heatmap =
          features
              .view<3>({features.shape[0], features.shape[1], features.shape[2], 0},
                       {0, 0, 0, static_cast<uint32_t>(i)})
              .contiguous()
              .sum<1>({2});

      cv::Mat heatmap_mat;
      cv::Mat(heatmap.shape[1], heatmap.shape[0], CV_32FC1, (float*)heatmap.get_data())
          .clone()
          .convertTo(heatmap_mat, CV_8U, 255);
      cv::resize(heatmap_mat, heatmap_mat, cv::Size(960, 540));
      cv::cvtColor(heatmap_mat, heatmap_mat, cv::COLOR_GRAY2BGR);

      result[name] = heatmap_mat;
    }
    return result;
  }
};

COALSACK_REGISTER_NODE(mvpose_reconstruct_node, image_reconstruct_node)

class mvp_reconstruct_node : public image_reconstruct_node {
  mutable std::mutex cameras_mtx;
  std::map<std::string, camera_t> cameras;
  glm::mat4 axis;
  graph_edge_ptr output;

  std::vector<std::string> names;
  mutable std::mutex features_mtx;

  stargazer::mvp::mvp pose_estimator;

 public:
  mvp_reconstruct_node()
      : image_reconstruct_node(), cameras(), axis(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "mvp_reconstruct"; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cameras, axis);
  }

  std::vector<glm::vec3> reconstruct(const std::map<std::string, camera_t>& cameras,
                                     const std::map<std::string, cv::Mat>& frame,
                                     const glm::mat4& axis) {
    using namespace stargazer::mvp;

    std::vector<std::string> names;
    std::vector<cv::Mat> images_list;
    std::vector<camera_data> cameras_list;

    // MVP requires exactly 5 views
    if (frame.size() != 5) {
      return std::vector<glm::vec3>();
    }

    for (const auto& [camera_name, image] : frame) {
      names.push_back(camera_name);
    }

    for (size_t i = 0; i < frame.size(); i++) {
      const auto name = names[i];

      if (cameras.find(name) == cameras.end()) {
        spdlog::warn("Camera {} not found", name);
        return std::vector<glm::vec3>();
      }

      const auto& cam = cameras.at(name);

      // Convert camera_t to mvp::camera_data
      // Convert back to original Panoptic coordinate system for MVP model
      camera_data mvp_cam;
      mvp_cam.fx = cam.intrin.fx;
      mvp_cam.fy = cam.intrin.fy;
      mvp_cam.cx = cam.intrin.cx;
      mvp_cam.cy = cam.intrin.cy;

      for (int j = 0; j < 5; j++) {
        mvp_cam.dist_coeff[j] = cam.intrin.coeffs[j];
      }

      // cam.extrin.rotation is glm::mat3 (column-major)
      // Convert to row-major and apply flip_yz inverse transform
      double R_stored[3][3];
      for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
          R_stored[row][col] = cam.extrin.rotation[col][row];
        }
      }

      // Apply flip_yz = [[1,0,0],[0,-1,0],[0,0,-1]] to restore original rotation
      for (int row = 0; row < 3; row++) {
        mvp_cam.rotation[row][0] = R_stored[row][0];
        mvp_cam.rotation[row][1] = -R_stored[row][1];
        mvp_cam.rotation[row][2] = -R_stored[row][2];
      }

      // Translation in mm
      mvp_cam.translation[0] = cam.extrin.translation.x * 1000.0;  // m to mm
      mvp_cam.translation[1] = cam.extrin.translation.y * 1000.0;
      mvp_cam.translation[2] = cam.extrin.translation.z * 1000.0;

      cameras_list.push_back(mvp_cam);
      images_list.push_back(frame.at(name));
    }

    std::array<float, 3> grid_center = {0.0f, -500.0f, 800.0f};
    pose_estimator.set_grid_center(grid_center);

    const auto points = pose_estimator.inference(images_list, cameras_list);

    {
      std::lock_guard lock(features_mtx);
      this->names = names;
    }

    return points;
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (input_name == "cameras") {
      if (auto camera_msg = std::dynamic_pointer_cast<object_message>(message)) {
        for (const auto& [name, field] : camera_msg->get_fields()) {
          if (auto camera_msg = std::dynamic_pointer_cast<camera_message>(field)) {
            std::lock_guard lock(cameras_mtx);
            cameras[name] = camera_msg->get_camera();
          }
        }
      }

      return;
    }
    if (input_name == "axis") {
      if (auto mat4_msg = std::dynamic_pointer_cast<mat4_message>(message)) {
        axis = mat4_msg->get_data();
      }

      return;
    }

    if (auto frame_msg = std::dynamic_pointer_cast<frame_message<object_message>>(message)) {
      const auto obj_msg = frame_msg->get_data();

      std::map<std::string, camera_t> cameras;
      {
        std::lock_guard lock(cameras_mtx);
        cameras = this->cameras;
      }

      std::map<std::string, cv::Mat> images;

      for (const auto& [name, field] : obj_msg.get_fields()) {
        if (auto img_msg = std::dynamic_pointer_cast<image_message>(field)) {
          if (cameras.find(name) == cameras.end()) {
            continue;
          }
          const auto& image = img_msg->get_image();
          cv::Mat img(image.get_height(), image.get_width(), convert_to_cv_type(image.get_format()),
                      (void*)image.get_data(), image.get_stride());
          images[name] = img;
        }
      }

      const auto markers = reconstruct(cameras, images, axis);

      auto marker_msg = std::make_shared<float3_list_message>();
      std::vector<float3> marker_data;
      for (const auto& marker : markers) {
        marker_data.push_back({marker.x, marker.y, marker.z});
      }
      marker_msg->set_data(marker_data);
      marker_msg->set_frame_number(frame_msg->get_frame_number());

      output->send(marker_msg);
    }
  }

  std::map<std::string, cv::Mat> get_features() const override {
    // MVP does not expose intermediate features like heatmaps
    return std::map<std::string, cv::Mat>();
  }
};

COALSACK_REGISTER_NODE(mvp_reconstruct_node, image_reconstruct_node)

class multiview_image_reconstruction_pipeline::impl {
  graph_proc graph;

  std::atomic_bool running;

  mutable std::mutex markers_mtx;
  std::vector<glm::vec3> markers;
  std::vector<std::function<void(const std::vector<glm::vec3>&)>> markers_received;

  std::shared_ptr<image_reconstruct_node> reconstruct_node;
  std::shared_ptr<graph_node> input_node;

 public:
  void add_markers_received(std::function<void(const std::vector<glm::vec3>&)> f) {
    std::lock_guard lock(markers_mtx);
    markers_received.push_back(f);
  }

  void clear_markers_received() {
    std::lock_guard lock(markers_mtx);
    markers_received.clear();
  }

  impl()
      : graph(), running(false), markers(), markers_received(), reconstruct_node(), input_node() {}

  void set_camera(const std::string& name, const camera_t& camera) {
    auto camera_msg = std::make_shared<camera_message>(camera);
    camera_msg->set_camera(camera);

    auto obj_msg = std::make_shared<object_message>();
    obj_msg->add_field(name, camera_msg);

    if (reconstruct_node) {
      graph.process(reconstruct_node.get(), "cameras", obj_msg);
    }
  }

  void set_axis(const glm::mat4& axis) {
    auto mat4_msg = std::make_shared<mat4_message>();
    mat4_msg->set_data(axis);

    if (reconstruct_node) {
      graph.process(reconstruct_node.get(), "axis", mat4_msg);
    }
  }

  using frame_type = std::map<std::string, cv::Mat>;

  static image_format convert_to_image_format(int type) {
    switch (type) {
      case CV_8UC1:
        return image_format::Y8_UINT;
      case CV_8UC3:
        return image_format::B8G8R8_UINT;
      case CV_8UC4:
        return image_format::B8G8R8A8_UINT;
      default:
        throw std::runtime_error("Invalid image format");
    }
  }

  void push_frame(const frame_type& frame) {
    if (!running) {
      return;
    }

    auto msg = std::make_shared<object_message>();
    for (const auto& [name, field] : frame) {
      auto img_msg = std::make_shared<image_message>();

      image img(static_cast<std::uint32_t>(field.size().width),
                static_cast<std::uint32_t>(field.size().height),
                static_cast<std::uint32_t>(field.elemSize()),
                static_cast<std::uint32_t>(field.step), (const uint8_t*)field.data);
      img.set_format(convert_to_image_format(field.type()));

      img_msg->set_image(std::move(img));
      msg->add_field(name, img_msg);
    }

    auto frame_msg = std::make_shared<frame_message<object_message>>();
    frame_msg->set_data(*msg);

    if (input_node) {
      graph.process(input_node.get(), frame_msg);
    }
  }

  void run(const std::vector<node_info>& infos) {
    std::cout << "=== Reconstruction Pipeline Graph Structure ===" << std::endl;
    for (const auto& info : infos) {
      std::cout << "Node: " << info.name << " (type: " << static_cast<int>(info.get_type()) << ")"
                << std::endl;

      // Print node parameters
      if (info.contains_param("fps")) {
        std::cout << "  fps: " << info.get_param<double>("fps") << std::endl;
      }
      if (info.contains_param("interval")) {
        std::cout << "  interval: " << info.get_param<double>("interval") << std::endl;
      }
      if (info.contains_param("num_threads")) {
        std::cout << "  num_threads: " << info.get_param<std::int64_t>("num_threads") << std::endl;
      }
      if (info.contains_param("callback_name")) {
        std::cout << "  callback_name: " << info.get_param<std::string>("callback_name")
                  << std::endl;
      }

      for (const auto& [input_name, source_name] : info.inputs) {
        std::cout << "  Input '" << input_name << "' <- '" << source_name << "'" << std::endl;
      }
      for (const auto& output_name : info.outputs) {
        std::cout << "  Output: '" << output_name << "'" << std::endl;
      }
    }
    std::cout << "==============================================" << std::endl;

    std::shared_ptr<subgraph> g(new subgraph());

    std::unordered_map<std::string, graph_node_ptr> node_map;

    for (const auto& info : infos) {
      graph_node_ptr node;

      switch (info.get_type()) {
        case node_type::frame_number_numbering: {
          auto n = std::make_shared<frame_number_numbering_node>();
          node = n;
          input_node = n;
          break;
        }
        case node_type::parallel_queue: {
          auto n = std::make_shared<parallel_queue_node>();
          if (info.contains_param("num_threads")) {
            n->set_num_threads(static_cast<size_t>(info.get_param<std::int64_t>("num_threads")));
          }
          node = n;
          break;
        }
        case node_type::voxelpose_reconstruction: {
          auto n = std::make_shared<voxelpose_reconstruct_node>();
          node = n;
          reconstruct_node = n;
          break;
        }
        case node_type::mvpose_reconstruction: {
          auto n = std::make_shared<mvpose_reconstruct_node>();
          node = n;
          reconstruct_node = n;
          break;
        }
        case node_type::mvp_reconstruction: {
          auto n = std::make_shared<mvp_reconstruct_node>();
          node = n;
          reconstruct_node = n;
          break;
        }
        case node_type::frame_number_ordering: {
          auto n = std::make_shared<frame_number_ordering_node>();
          node = n;
          break;
        }
        case node_type::callback: {
          auto n = std::make_shared<callback_node>();
          if (info.contains_param("callback_name")) {
            n->set_callback_name(info.get_param<std::string>("callback_name"));
          }
          node = n;
          break;
        }
        case node_type::grpc_server: {
          auto n = std::make_shared<grpc_server_node>();
          if (info.contains_param("address")) {
            n->set_address(info.get_param<std::string>("address"));
          }
          node = n;
          break;
        }
        case node_type::frame_demux: {
          auto n = std::make_shared<frame_demux_node>();
          for (const auto& output_name : info.outputs) {
            n->add_output(output_name);
          }
          node = n;
          break;
        }
        case node_type::dump_se3: {
          auto n = std::make_shared<dump_se3_node>();
          if (info.contains_param("db_path")) {
            n->set_db_path(info.get_param<std::string>("db_path"));
          }
          if (info.contains_param("topic_name")) {
            n->set_name(info.get_param<std::string>("topic_name"));
          }
          node = n;
          break;
        }
        default:
          throw std::runtime_error("Unknown node type: " + info.name);
      }

      node_map[info.name] = node;
      g->add_node(node);
    }

    for (const auto& info : infos) {
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

    const auto callbacks = std::make_shared<callback_list>();

    callbacks->add(
        [this](const callback_node* node, std::string input_name, graph_message_ptr message) {
          if (node->get_callback_name() == "markers") {
            if (const auto markers_msg = std::dynamic_pointer_cast<float3_list_message>(message)) {
              std::vector<glm::vec3> markers;
              for (const auto& marker : markers_msg->get_data()) {
                markers.push_back(glm::vec3(marker.x, marker.y, marker.z));
              }

              {
                std::lock_guard lock(markers_mtx);
                this->markers = markers;
              }

              for (const auto& f : markers_received) {
                f(markers);
              }
            }
          }
        });

    graph.deploy(g);
    graph.get_resources()->add(callbacks);
    graph.run();

    running = true;
  }

  void stop() {
    running.store(false);
    graph.stop();
  }

  std::vector<glm::vec3> get_markers() const {
    std::vector<glm::vec3> result;

    {
      std::lock_guard lock(markers_mtx);
      result = this->markers;
    }

    return result;
  }

  std::map<std::string, cv::Mat> get_features() const { return reconstruct_node->get_features(); }
};

multiview_image_reconstruction_pipeline::multiview_image_reconstruction_pipeline()
    : pimpl(new impl()) {}
multiview_image_reconstruction_pipeline::~multiview_image_reconstruction_pipeline() = default;

void multiview_image_reconstruction_pipeline::push_frame(const frame_type& frame) {
  pimpl->push_frame(frame);
}

void multiview_image_reconstruction_pipeline::run(const std::vector<node_info>& infos) {
  pimpl->run(infos);
}

void multiview_image_reconstruction_pipeline::stop() { pimpl->stop(); }

std::map<std::string, cv::Mat> multiview_image_reconstruction_pipeline::get_features() const {
  return pimpl->get_features();
}

std::vector<glm::vec3> multiview_image_reconstruction_pipeline::get_markers() const {
  return pimpl->get_markers();
}

void multiview_image_reconstruction_pipeline::set_camera(const std::string& name,
                                                         const camera_t& camera) {
  pimpl->set_camera(name, camera);
}

void multiview_image_reconstruction_pipeline::set_axis(const glm::mat4& axis) {
  pimpl->set_axis(axis);
}