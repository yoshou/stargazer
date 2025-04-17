#pragma once

#include "parameters.hpp"
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
#include "point_data.hpp"
#include "task_queue.hpp"

#include "graph_proc.h"
#include "graph_proc_img.h"
#include "graph_proc_cv.h"
#include "graph_proc_tensor.h"

#include "voxelpose.hpp"
#include "mvpose.hpp"

class ServiceImpl;

class SensorServiceImpl;

std::vector<glm::vec3> reconstruct(const std::map<std::string, stargazer::camera_t> &cameras, const std::map<std::string, std::vector<stargazer::point_data>> &frame, glm::mat4 axis = glm::mat4(1.0f));

class multiview_point_reconstruction
{
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

    virtual std::map<std::string, stargazer::camera_t> get_cameras() const
    {
        return cameras;
    }
    virtual void set_camera(const std::string &name, const stargazer::camera_t &camera)
    {
        cameras[name] = camera;
    }
    virtual const stargazer::camera_t& get_camera(const std::string &name) const
    {
        return cameras.at(name);
    }
    virtual stargazer::camera_t& get_camera(const std::string &name)
    {
        return cameras.at(name);
    }
    virtual void set_axis(const glm::mat4 &axis)
    {
        this->axis = axis;
    }
    virtual glm::mat4 get_axis() const
    {
        return axis;
    }
};

class epipolar_reconstruction : public multiview_point_reconstruction
{
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

    stargazer::voxelpose::voxelpose pose_estimator;

public:

    voxelpose_reconstruction();
    virtual ~voxelpose_reconstruction();

    void push_frame(const frame_type &frame);
    void run();
    void stop();

    std::vector<glm::vec3> get_markers() const;

    std::map<std::string, cv::Mat> get_features() const;
};

#define USE_THREAD_POOL 1

template <typename T>
class task_queue_processor
{
    std::deque<std::future<T>> results;
    mutable std::mutex results_mtx;
    std::condition_variable results_cv;
    task_queue<std::function<void()>> workers;
    std::atomic_bool running;
    std::shared_ptr<std::thread> th;

    template<typename Func>
    struct function_wrapper
    {
        Func func;
        function_wrapper(Func &&func)
            : func(std::move(func))
        {}

        auto operator()() {
            return func();
        }
    };

public:
    task_queue_processor(size_t thread_count = 4)
        : workers(thread_count)
    {
    }

    template <typename Task>
    void push(Task task)
    {
        if (!running)
        {
            return;
        }
        {
            std::lock_guard lock(results_mtx);
#if USE_THREAD_POOL
            if (workers.size() > 10)
            {
                return;
            }

            std::promise<T> p;
            auto f = p.get_future();

            auto callback = [p = std::move(p), task]() mutable {
                p.set_value(task());
            };

            auto copyable_callback = std::make_shared<function_wrapper<decltype(callback)>>(std::move(callback));

            workers.push_task([copyable_callback]()
                              { (*copyable_callback)(); });
#else
            auto f = std::async(std::launch::async, task);
#endif
            results.emplace_back(std::move(f));
        }
        results_cv.notify_one();
    }

    bool pop(T &result)
    {
        std::unique_lock<std::mutex> lock(results_mtx);
        results_cv.wait(lock, [this]
                        { return !running || results.size() > 0; });

        if (results.size() == 0)
        {
            return false;
        }

        auto&& f = results.front();
        result = f.get();

        results.pop_front();
        return true;
    }

    template<typename F>
    void run(F callback)
    {
        running = true;
        th.reset(new std::thread([this, callback]()
                                 {
            while (running)
            {
                T result;
                if (pop(result))
                {
                    callback(result);
                }
            } }));
    }

    void stop()
    {
        if (!running)
        {
            return;
        }

        {
            std::lock_guard lock(results_mtx);
            running.store(false);
        }
        results_cv.notify_one();
        if (th && th->joinable())
        {
            th->join();
        }
    }
};

class grpc_server
{
    std::string server_address;
    std::atomic_bool running;
    std::shared_ptr<std::thread> server_th;
    std::unique_ptr<grpc::Server> server;
    std::unique_ptr<SensorServiceImpl> service;

public:
    grpc_server(const std::string& server_address);
    ~grpc_server();

    void run();
    void stop();

    void notify_sphere(const std::vector<glm::vec3> &spheres);
};

class mvpose_reconstruction : public multiview_image_reconstruction
{
    std::tuple<std::vector<std::string>, coalsack::tensor<float, 4>, std::vector<glm::vec3>> mvpose_reconstruct(const std::map<std::string, stargazer::camera_t> &cameras, const std::map<std::string, cv::Mat> &frame, glm::mat4 axis);

    std::vector<glm::vec3> markers;
    mutable std::mutex markers_mtx;

    std::vector<std::string> names;
    coalsack::tensor<float, 4> features;
    mutable std::mutex features_mtx;

    stargazer_mvpose::mvpose pose_estimator;

    struct task_result
    {
        std::vector<std::string> camera_names;
        coalsack::tensor<float, 4> features;
        std::vector<glm::vec3> markers;
    };

    task_queue_processor<task_result> processor;
    grpc_server output;

public:
    mvpose_reconstruction();
    virtual ~mvpose_reconstruction();

    void push_frame(const frame_type &frame);
    void run();
    void stop();

    std::vector<glm::vec3> get_markers() const;

    std::map<std::string, cv::Mat> get_features() const;
};
