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

class ServiceImpl;

class SensorServiceImpl;

std::vector<glm::vec3> reconstruct(const std::map<std::string, stargazer::camera_t> &cameras, const std::map<std::string, std::vector<stargazer::point_data>> &frame, glm::mat4 axis = glm::mat4(1.0f));

class marker_stream_server
{
    using frame_type = std::map<std::string, std::vector<stargazer::point_data>>;
    std::atomic_bool running;
    std::mutex frames_mtx;
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

    marker_stream_server();
    virtual ~marker_stream_server();

    std::map<std::string, stargazer::camera_t> cameras;
    glm::mat4 axis;

    void push_frame(const frame_type &frame);
    void run();
    void stop();

    std::vector<glm::vec3> get_markers() const
    {
        std::vector<glm::vec3> result;
        {
            std::lock_guard lock(markers_mtx);
            result = markers;
        }
        return result;
    }
};
