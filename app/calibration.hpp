#pragma once

#include <map>
#include <memory>
#include <iostream>
#include <random>

#include <glm/glm.hpp>

#include "multiview_point_data.hpp"
#include "bundle_adjust_data.hpp"
#include "parameters.hpp"
#include "task_queue.hpp"
#include "node_info.hpp"

struct observed_points_t
{
    size_t camera_idx;
    std::vector<glm::vec2> points;
};

class calibration
{
    class impl;
    std::unique_ptr<impl> pimpl;

public:
    void add_calibrated(std::function<void(const std::unordered_map<std::string, stargazer::camera_t> &)> f);
    void clear_calibrated();

    void set_camera(const std::string &name, const stargazer::camera_t &camera);

    size_t get_camera_size() const;

    const std::unordered_map<std::string, stargazer::camera_t>& get_cameras() const;

    std::unordered_map<std::string, stargazer::camera_t> &get_cameras();

    const std::unordered_map<std::string, stargazer::camera_t>& get_calibrated_cameras() const;

    calibration();
    virtual ~calibration();

    size_t get_num_frames(std::string name) const;
    const std::vector<observed_points_t> get_observed_points(std::string name) const;

    void push_frame(const std::map<std::string, std::vector<stargazer::point_data>> &frame);
    void run(const std::vector<node_info> &infos);
    void stop();

    void calibrate();
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
    void add_frame(const cv::Mat &frame);

    void calibrate();
};
