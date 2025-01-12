#pragma once

#include <map>
#include <memory>
#include <iostream>
#include <random>

#include <glm/glm.hpp>

#include "multiview_point_data.hpp"
#include "bundle_adjust_data.hpp"
#include "camera_info.hpp"
#include "task_queue.hpp"
#include "graph_proc.h"

class calibration_target
{
public:
    virtual std::vector<glm::vec2> detect_points(const std::vector<stargazer::point_data> &markers) = 0;
    virtual ~calibration_target() = default;
};

template <class T, class F>
void combination(const std::vector<T> &seed, int target_size, F callback)
{
    std::vector<int> indices(target_size);
    const int seed_size = seed.size();
    int start_index = 0;
    int size = 0;

    while (size >= 0)
    {
        for (int i = start_index; i < seed_size; ++i)
        {
            indices[size++] = i;
            if (size == target_size)
            {
                std::vector<T> comb(target_size);
                for (int x = 0; x < target_size; ++x)
                {
                    comb[x] = seed[indices[x]];
                }
                if (callback(comb))
                    return;
                break;
            }
        }
        --size;
        if (size < 0)
            break;
        start_index = indices[size] + 1;
    }
}

struct observed_points_t
{
    size_t camera_idx;
    std::vector<glm::vec2> points;
};

class calibration
{
    std::unordered_map<std::string, stargazer::camera_t> cameras;

    class impl;
    std::unique_ptr<impl> pimpl;

public:
    void add_calibrated(std::function<void(const std::unordered_map<std::string, stargazer::camera_t> &)> f);
    void clear_calibrated();

    void set_camera(const std::string &name, const stargazer::camera_t &camera)
    {
        cameras[name] = camera;
    }

    size_t get_camera_size() const
    {
        return cameras.size();
    }

    const std::unordered_map<std::string, stargazer::camera_t>& get_cameras() const
    {
        return cameras;
    }

    std::unordered_map<std::string, stargazer::camera_t> &get_cameras()
    {
        return cameras;
    }

    const std::unordered_map<std::string, stargazer::camera_t>& get_calibrated_cameras() const;

    calibration();
    virtual ~calibration();

    size_t get_num_frames(std::string name) const;
    const std::vector<observed_points_t> &get_observed_points(std::string name) const;

    void push_frame(const std::map<std::string, std::vector<stargazer::point_data>> &frame);
    void run();
    void stop();

    void calibrate();
};

enum class calibration_pattern
{
    CHESSBOARD,
    CIRCLES_GRID,
    ASYMMETRIC_CIRCLES_GRID,
};

class intrinsic_calibration
{
    std::vector<std::vector<stargazer::point_data>> frames;

public:
    stargazer::camera_t calibrated_camera;
    double rms = 0.0;
    int image_width = 0;
    int image_height = 0;

    size_t get_num_frames() const
    {
        return frames.size();
    }

    void add_frame(const std::vector<stargazer::point_data> &frame);

    void calibrate();
};

void calc_board_corner_positions(cv::Size board_size, cv::Size2f square_size, std::vector<cv::Point3f> &corners, const calibration_pattern pattern_type = calibration_pattern::CHESSBOARD);

bool detect_calibration_board(cv::Mat frame, std::vector<cv::Point2f> &points, const calibration_pattern pattern_type = calibration_pattern::CHESSBOARD);

void detect_aruco_marker(cv::Mat image, std::vector<std::vector<cv::Point2f>> &points, std::vector<int> &id);

bool detect_charuco_board(cv::Mat image, std::vector<cv::Point2f> &points, std::vector<int>& ids);

class extrinsic_calibration
{
    using frame_type = std::unordered_map<std::string, cv::Mat>;
    calibration_pattern pattern;

    std::shared_ptr<task_queue<std::function<void()>>> workers;
    std::deque<uint32_t> task_wait_queue;
    mutable std::mutex task_wait_queue_mtx;
    std::condition_variable task_wait_queue_cv;
    std::mt19937 task_id_gen;

    std::unordered_map<std::string, size_t> camera_name_to_index;
    std::unordered_map<std::string, std::vector<observed_points_t>> observed_frames;
    std::unordered_map<std::string, size_t> num_frames;

    mutable std::mutex observed_frames_mtx;
    
    std::vector<std::unordered_map<std::string, observed_points_t>> detect_pattern(const frame_type &frame);

public:
    glm::mat4 axis;
    std::unordered_map<std::string, stargazer::camera_t> cameras;
    std::unordered_map<std::string, stargazer::camera_t> calibrated_cameras;

    extrinsic_calibration(calibration_pattern pattern = calibration_pattern::CHESSBOARD);

    void add_frame(const frame_type &frame);

    void calibrate();

    size_t get_num_frames(std::string name) const
    {
        if (num_frames.find(name) == num_frames.end())
        {
            return 0;
        }
        return num_frames.at(name);
    }
    const std::vector<observed_points_t> &get_observed_points(std::string name) const
    {
        static std::vector<observed_points_t> empty;
        if (observed_frames.find(name) == observed_frames.end())
        {
            return empty;
        }
        return observed_frames.at(name);
    }
};
