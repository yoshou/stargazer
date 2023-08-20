#pragma once

#include <memory>
#include <string>
#include <opencv2/opencv.hpp>
#include <glm/glm.hpp>

enum class cluster_type
{
    rs_d435,
    raspi,
    raspi_color,
    depthai_color,
};

struct cluster_info
{
    std::string name;
    cluster_type type;
    std::string id;
    std::string address;
    std::string endpoint;
};

class capture_controller
{
    class impl;

    std::unique_ptr<impl> pimpl;

public:
    capture_controller();
    virtual ~capture_controller();

    void run(const cluster_info& info);
    void stop();

    void set_mask(cv::Mat mask);

    cv::Mat get_frame() const;
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

class sync_capture_controller
{
    class impl;

    std::unique_ptr<impl> pimpl;

public:
    sync_capture_controller();
    sync_capture_controller(const std::map<std::string, cv::Mat> &masks);
    virtual ~sync_capture_controller();

    void run(const std::vector<cluster_info> &infos);
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
};
