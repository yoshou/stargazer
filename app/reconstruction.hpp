#pragma once

#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <mutex>
#include <random>
#include <thread>

#include "capture.hpp"
#include "graph_proc.h"
#include "graph_proc_cv.h"
#include "graph_proc_img.h"
#include "graph_proc_tensor.h"
#include "mvpose.hpp"
#include "parameters.hpp"
#include "point_data.hpp"
#include "sensor.grpc.pb.h"
#include "task_queue.hpp"
#include "voxelpose.hpp"

class ServiceImpl;

class SensorServiceImpl;

struct se3 {
  glm::vec3 position;
  glm::quat rotation;

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(position, rotation);
  }
};

std::vector<glm::vec3> reconstruct(
    const std::map<std::string, stargazer::camera_t> &cameras,
    const std::map<std::string, std::vector<stargazer::point_data>> &frame,
    glm::mat4 axis = glm::mat4(1.0f));

class multiview_point_reconstruction {
  std::map<std::string, stargazer::camera_t> cameras;
  glm::mat4 axis;

 public:
  using frame_type = std::map<std::string, std::vector<stargazer::point_data>>;

  multiview_point_reconstruction() = default;
  virtual ~multiview_point_reconstruction() = default;

  virtual void push_frame(const frame_type &frame) = 0;
  virtual void run() = 0;
  virtual void stop() = 0;

  virtual std::vector<glm::vec3> get_markers() const = 0;

  virtual std::map<std::string, stargazer::camera_t> get_cameras() const { return cameras; }
  virtual void set_camera(const std::string &name, const stargazer::camera_t &camera) {
    cameras[name] = camera;
  }
  virtual const stargazer::camera_t &get_camera(const std::string &name) const {
    return cameras.at(name);
  }
  virtual stargazer::camera_t &get_camera(const std::string &name) { return cameras.at(name); }
  virtual void set_axis(const glm::mat4 &axis) { this->axis = axis; }
  virtual glm::mat4 get_axis() const { return axis; }
};

class epipolar_reconstruction : public multiview_point_reconstruction {
  class impl;
  std::unique_ptr<impl> pimpl;

 public:
  epipolar_reconstruction();
  virtual ~epipolar_reconstruction();

  void push_frame(const frame_type &frame);
  void run();
  void stop();

  std::vector<glm::vec3> get_markers() const;
  void set_camera(const std::string &name, const stargazer::camera_t &camera) override;
  void set_axis(const glm::mat4 &axis) override;
};

class multiview_image_reconstruction {
  class impl;
  std::unique_ptr<impl> pimpl;

  using frame_type = std::map<std::string, cv::Mat>;

 public:
  multiview_image_reconstruction();
  virtual ~multiview_image_reconstruction();

  void push_frame(const frame_type &frame);
  void run(const std::vector<node_info> &infos);
  void stop();

  std::vector<glm::vec3> get_markers() const;
  std::map<std::string, cv::Mat> get_features() const;
  void set_camera(const std::string &name, const stargazer::camera_t &camera);
  void set_axis(const glm::mat4 &axis);
};

class grpc_server {
  std::string server_address;
  std::atomic_bool running;
  std::shared_ptr<std::thread> server_th;
  std::unique_ptr<grpc::Server> server;
  std::unique_ptr<SensorServiceImpl> service;

 public:
  grpc_server(const std::string &server_address);
  ~grpc_server();

  void run();
  void stop();

  void notify_sphere(const std::string &name, int64_t timestamp,
                     const std::vector<glm::vec3> &spheres);
  void receive_se3(std::function<void(const std::string &, int64_t, const std::vector<se3> &)> f);
};