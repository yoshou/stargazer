#pragma once

#include "camera_info.hpp"
#include "capture.hpp"
#include <atomic>
#include <deque>
#include <mutex>
#include <thread>
#include <memory>
#include <condition_variable>
#include <random>

#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>

#include "sensor.grpc.pb.h"
#include "multiview_point_data.hpp"
#include "task_queue.hpp"

#include "graph_proc.h"
#include "graph_proc_img.h"
#include "graph_proc_cv.h"
#include "graph_proc_tensor.h"

#include "voxelpose.hpp"

class ServiceImpl;

class SensorServiceImpl;

std::vector<glm::vec3> reconstruct(const std::map<std::string, stargazer::camera_t> &cameras, const std::map<std::string, std::vector<stargazer::point_data>> &frame, glm::mat4 axis = glm::mat4(1.0f));

class multiview_point_reconstruction
{
public:
    using frame_type = std::map<std::string, std::vector<stargazer::point_data>>;

    std::map<std::string, stargazer::camera_t> cameras;
    glm::mat4 axis;

    multiview_point_reconstruction() = default;
    virtual ~multiview_point_reconstruction() = default;

    virtual void push_frame(const frame_type &frame) = 0;
    virtual void run() = 0;
    virtual void stop() = 0;

    virtual std::vector<glm::vec3> get_markers() const = 0;
};

class epipolar_reconstruction : public multiview_point_reconstruction
{
    std::atomic_bool running;
    std::shared_ptr<task_queue<std::function<void()>>> reconstruction_workers;
    std::shared_ptr<std::thread> server_th;
    std::unique_ptr<grpc::Server> server;
    std::unique_ptr<SensorServiceImpl> service;
    std::deque<uint32_t> reconstruction_task_wait_queue;
    mutable std::mutex reconstruction_task_wait_queue_mtx;
    std::condition_variable reconstruction_task_wait_queue_cv;
    std::mt19937 task_id_gen;

    std::vector<glm::vec3> markers;
    mutable std::mutex markers_mtx;

public:

    epipolar_reconstruction();
    virtual ~epipolar_reconstruction();

    void push_frame(const frame_type &frame);
    void run();
    void stop();

    std::vector<glm::vec3> get_markers() const;
};

class multiview_image_reconstruction
{
public:
    using frame_type = std::map<std::string, cv::Mat>;

    std::map<std::string, stargazer::camera_t> cameras;
    glm::mat4 axis;

    multiview_image_reconstruction() = default;
    virtual ~multiview_image_reconstruction() = default;

    virtual void push_frame(const frame_type &frame) = 0;
    virtual void run() = 0;
    virtual void stop() = 0;

    virtual std::vector<glm::vec3> get_markers() const = 0;
    virtual std::map<std::string, cv::Mat> get_features() const = 0;
};

class voxelpose_reconstruction : public multiview_image_reconstruction
{
    std::vector<glm::vec3> dnn_reconstruct(const std::map<std::string, stargazer::camera_t> &cameras, const std::map<std::string, cv::Mat> &frame, glm::mat4 axis);

    std::atomic_bool running;
    std::shared_ptr<task_queue<std::function<void()>>> reconstruction_workers;
    std::shared_ptr<std::thread> server_th;
    std::unique_ptr<grpc::Server> server;
    std::unique_ptr<SensorServiceImpl> service;
    std::deque<uint32_t> reconstruction_task_wait_queue;
    mutable std::mutex reconstruction_task_wait_queue_mtx;
    std::condition_variable reconstruction_task_wait_queue_cv;
    std::mt19937 task_id_gen;

    std::vector<glm::vec3> markers;
    mutable std::mutex markers_mtx;

    std::vector<std::string> names;
    coalsack::tensor<float, 4> features;
    mutable std::mutex features_mtx;

    voxelpose pose_estimator;

public:

    voxelpose_reconstruction();
    virtual ~voxelpose_reconstruction();

    void push_frame(const frame_type &frame);
    void run();
    void stop();

    std::vector<glm::vec3> get_markers() const;

    std::map<std::string, cv::Mat> get_features() const;
};
