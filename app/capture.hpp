#pragma once

#include <memory>
#include <string>
#include <opencv2/opencv.hpp>
#include <glm/glm.hpp>

#include "device_info.hpp"

class capture_pipeline
{
    class impl;

    std::unique_ptr<impl> pimpl;

public:
    capture_pipeline();
    virtual ~capture_pipeline();

    void run(const device_info& info);
    void stop();

    void set_mask(cv::Mat mask);

    cv::Mat get_frame() const;
    std::unordered_map<int, cv::Point2f> get_markers() const;
};

struct marker_data
{  
    float x, y, r;
};

struct marker_frame_data
{
    std::vector<marker_data> markers;
    double timestamp;
    uint64_t frame_number;
};

class multiview_capture_pipeline
{
    class impl;

    std::unique_ptr<impl> pimpl;

public:
    multiview_capture_pipeline();
    multiview_capture_pipeline(const std::map<std::string, cv::Mat> &masks);
    virtual ~multiview_capture_pipeline();

    void run(const std::vector<device_info> &infos);
    void stop();

    std::map<std::string, cv::Mat> get_frames() const;
    std::map<std::string, cv::Mat> get_masks() const;

    void gen_mask();
    void clear_mask();

    void enable_marker_collecting(std::string name);
    void disable_marker_collecting(std::string name);

    std::vector<std::map<std::string, marker_frame_data>> pop_marker_frames();

    void add_marker_received(std::function<void(const std::map<std::string, marker_frame_data> &)> f);
    void clear_marker_received();
    void add_image_received(std::function<void(const std::map<std::string, cv::Mat> &)> f);
    void clear_image_received();
};
