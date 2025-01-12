#include "capture.hpp"

#include "graph_proc.h"
#include "graph_proc_cv.h"
#include "ext/graph_proc_jpeg.h"
#include "ext/graph_proc_cv.h"
#include "ext/graph_proc_libcamera.h"
#include "ext/graph_proc_depthai.h"
#include "ext/graph_proc_rs_d435.h"

#include <nlohmann/json.hpp>
#include <sqlite3.h>

using namespace coalsack;

using encode_image_node = encode_jpeg_node;
using decode_image_node = decode_jpeg_node;

class remote_cluster
{
public:
    std::shared_ptr<subgraph> g;
    graph_edge_ptr infra1_output;
    graph_edge_ptr infra1_marker_output;
    std::shared_ptr<mask_node> mask_node_;
};

class remote_cluster_raspi : public remote_cluster
{
public:
    explicit remote_cluster_raspi(int fps, const image *mask, bool is_master = false, bool emitter_enabled = true)
    {
        constexpr bool with_marker = true;

        g.reset(new subgraph());

        // Camera Module v2
        const int width = 820;
        const int height = 616;
        // Camera Module 3 Wide
        // const int width = 1536;
        // const int height = 864;

        std::shared_ptr<video_time_sync_control_node> n13(new video_time_sync_control_node());

        std::shared_ptr<libcamera_capture_node> n1(new libcamera_capture_node());
        n1->set_stream(stream_type::INFRARED);
        n1->set_option(libcamera_capture_node::option::exposure, 5000);
        n1->set_option(libcamera_capture_node::option::gain, 10);

        n1->set_fps(fps);
        n1->set_width(width);
        n1->set_height(height);
        n1->set_format(image_format::Y8_UINT);
        n1->set_emitter_enabled(emitter_enabled);

        if (!is_master)
        {
            n1->set_input(n13->get_output(), "interval");
        }

        g->add_node(n1);

        std::shared_ptr<fifo_node> n7(new fifo_node());
        n7->set_input(n1->get_output());
        g->add_node(n7);

        if (is_master)
        {
            std::shared_ptr<timestamp_node> n11(new timestamp_node());
            n11->set_input(n7->get_output());
            g->add_node(n11);

            std::shared_ptr<broadcast_talker_node> n12(new broadcast_talker_node());
            n12->set_input(n11->get_output());
            n12->set_endpoint("192.168.0.255", 40000);
            g->add_node(n12);
        }
        else
        {
            std::shared_ptr<timestamp_node> n11(new timestamp_node());
            n11->set_input(n7->get_output());
            g->add_node(n11);

            std::shared_ptr<broadcast_listener_node> n12(new broadcast_listener_node());
            n12->set_endpoint("192.168.0.1", 40000);
            g->add_node(n12);

            n13->set_input(n11->get_output());
            n13->set_input(n12->get_output(), "ref");
            n13->set_gain(0.1);
            n13->set_interval(1000.0 / fps);
            g->add_node(n13);
        }

        std::shared_ptr<scale_node> n10(new scale_node());
        n10->set_input(n7->get_output());

        // constexpr auto scale = 3.5;
        // constexpr auto bias = -50.0;
        constexpr auto scale = 1.1;
        constexpr auto bias = 0;

        n10->set_alpha(scale);
        n10->set_beta(bias);
        g->add_node(n10);

        std::shared_ptr<gaussian_blur_node> n8(new gaussian_blur_node());
        n8->set_input(n10->get_output());
        n8->set_kernel_width(3);
        n8->set_kernel_height(3);
        n8->set_sigma_x(1.5);
        n8->set_sigma_y(1.5);
        g->add_node(n8);

        coalsack::graph_edge_ptr infra1 = n8->get_output();
        {
            mask_node_.reset(new coalsack::mask_node());
            mask_node_->set_input(infra1);
            if (mask)
            {
                mask_node_->set_mask(*mask);
            }
            else
            {
                cv::Mat img(height, width, CV_8UC1, cv::Scalar(255));
                image mask_img(img.cols, img.rows, CV_8UC1, img.step, (const uint8_t *)img.data);
                mask_node_->set_mask(mask_img);
            }
            g->add_node(mask_node_);

            infra1 = mask_node_->get_output();
        }

        {
            std::shared_ptr<fifo_node> n4(new fifo_node());
            n4->set_input(infra1);
            g->add_node(n4);

            std::shared_ptr<encode_image_node> n2(new encode_image_node());
            n2->set_input(n4->get_output());
            g->add_node(n2);

            std::shared_ptr<p2p_talker_node> n3(new p2p_talker_node());
            n3->set_input(n2->get_output());
            g->add_node(n3);

            infra1_output = n3->get_output();
        }

        {
            std::shared_ptr<fifo_node> n9(new fifo_node());
            n9->set_input(infra1);
            g->add_node(n9);

            std::shared_ptr<fast_blob_detector_node> n4(new fast_blob_detector_node());
            n4->set_input(n9->get_output());

            auto params = fast_blob_detector_node::blob_detector_params();
            {
                params.min_dist_between_blobs = 1;
                params.step_threshold = 5;
                params.min_threshold = 50;
                params.max_threshold = params.min_threshold + 50;

                params.min_area = 6;
                params.max_area = 1000;

                params.min_circularity = 0.8;
            }

            n4->set_parameters(params);
            g->add_node(n4);

            std::shared_ptr<p2p_talker_node> n5(new p2p_talker_node());
            n5->set_input(n4->get_output());
            g->add_node(n5);

            infra1_marker_output = n5->get_output();
        }
    }
};

class remote_cluster_raspi_color : public remote_cluster
{
public:
    explicit remote_cluster_raspi_color(int fps, bool is_master = false)
    {
        constexpr bool with_image = true;
        constexpr bool with_marker = false;

        g.reset(new subgraph());

        // Camera Module v2
        // const int width = 820;
        // const int height = 616;
        // Camera Module 3 Wide
        const int width = 2304;
        const int height = 1296;

        std::shared_ptr<video_time_sync_control_node> n13(new video_time_sync_control_node());

        std::shared_ptr<libcamera_capture_node> n1(new libcamera_capture_node());
        n1->set_stream(stream_type::COLOR);
        n1->set_emitter_enabled(false);
        n1->set_option(libcamera_capture_node::option::exposure, 5000);
        n1->set_option(libcamera_capture_node::option::gain, 10);

        n1->set_fps(fps);
        n1->set_width(width);
        n1->set_height(height);
        n1->set_format(image_format::R8G8B8_UINT);

        if (!is_master)
        {
            n1->set_input(n13->get_output(), "interval");
        }

        g->add_node(n1);

        std::shared_ptr<fifo_node> n7(new fifo_node());
        n7->set_input(n1->get_output());
        g->add_node(n7);

        if (is_master)
        {
            std::shared_ptr<timestamp_node> n11(new timestamp_node());
            n11->set_input(n7->get_output());
            g->add_node(n11);

            std::shared_ptr<broadcast_talker_node> n12(new broadcast_talker_node());
            n12->set_input(n11->get_output());
            n12->set_endpoint("192.168.0.255", 40000);
            g->add_node(n12);
        }
        else
        {
            std::shared_ptr<timestamp_node> n11(new timestamp_node());
            n11->set_input(n7->get_output());
            g->add_node(n11);

            std::shared_ptr<broadcast_listener_node> n12(new broadcast_listener_node());
            n12->set_endpoint("192.168.0.1", 40000);
            g->add_node(n12);

            n13->set_input(n11->get_output());
            n13->set_input(n12->get_output(), "ref");
            n13->set_gain(0.1);
            n13->set_interval(1000.0 / fps);
            g->add_node(n13);
        }

        std::shared_ptr<resize_node> n10(new resize_node());
        n10->set_input(n7->get_output());
        n10->set_width(960);
        n10->set_height(540);
        g->add_node(n10);

        auto infra1 = n10->get_output();

        if (with_image)
        {
            std::shared_ptr<fifo_node> n4(new fifo_node());
            n4->set_input(n10->get_output());
            g->add_node(n4);

            std::shared_ptr<encode_image_node> n2(new encode_image_node());
            n2->set_input(n4->get_output());
            g->add_node(n2);

            std::shared_ptr<p2p_talker_node> n3(new p2p_talker_node());
            n3->set_input(n2->get_output());
            g->add_node(n3);

            infra1_output = n3->get_output();
        }

        if (with_marker)
        {
            std::shared_ptr<fifo_node> n9(new fifo_node());
            n9->set_input(infra1);
            g->add_node(n9);

            std::shared_ptr<charuco_detector_node> n4(new charuco_detector_node());
            n4->set_input(n9->get_output());
            g->add_node(n4);

            std::shared_ptr<p2p_talker_node> n5(new p2p_talker_node());
            n5->set_input(n4->get_output());
            g->add_node(n5);

            infra1_marker_output = n5->get_output();
        }
    }
};

class remote_cluster_depthai_color : public remote_cluster
{
public:
    explicit remote_cluster_depthai_color(int fps)
    {
        g.reset(new subgraph());

        constexpr int width = 1920;
        constexpr int height = 1080;

        std::shared_ptr<depthai_color_camera_node> n1(new depthai_color_camera_node());

        n1->set_fps(fps);
        n1->set_width(width);
        n1->set_height(height);

        g->add_node(n1);

        std::shared_ptr<fifo_node> n7(new fifo_node());
        n7->set_input(n1->get_output());
        g->add_node(n7);

        std::shared_ptr<resize_node> n10(new resize_node());
        n10->set_input(n7->get_output());
        n10->set_width(960);
        n10->set_height(540);
        g->add_node(n10);

        {
            std::shared_ptr<fifo_node> n4(new fifo_node());
            n4->set_input(n10->get_output());
            g->add_node(n4);

            std::shared_ptr<encode_image_node> n2(new encode_image_node());
            n2->set_input(n4->get_output());
            g->add_node(n2);

            std::shared_ptr<p2p_talker_node> n3(new p2p_talker_node());
            n3->set_input(n2->get_output());
            g->add_node(n3);

            infra1_output = n3->get_output();
        }
    }
};

class remote_cluster_rs_d435 : public remote_cluster
{
public:
    explicit remote_cluster_rs_d435(int fps, int exposure, int gain, int laser_power, bool with_image, bool emitter_enabled = true)
    {
        constexpr bool with_marker = false;

        g.reset(new subgraph());

        const int width = 640;
        const int height = 480;

        std::shared_ptr<rs_d435_node> n1(new rs_d435_node());
        g->add_node(n1);

        std::map<rs2_option_type, float> options = {
            std::make_pair(rs2_option_type::GLOBAL_TIME_ENABLED, (float)true),
            std::make_pair(rs2_option_type::EXPOSURE, (float)exposure),
            std::make_pair(rs2_option_type::GAIN, (float)gain),
            std::make_pair(rs2_option_type::LASER_POWER, (float)laser_power),
            std::make_pair(rs2_option_type::EMITTER_ENABLED, (float)emitter_enabled),
        };

        for (const auto [option, value] : options)
        {
            n1->set_option(INFRA1, option, value);
        }

#if 1
        std::shared_ptr<fifo_node> n8(new fifo_node());
        n8->set_input(n1->add_output(INFRA1, width, height, rs2_format_type::Y8, fps));
        g->add_node(n8);

        auto infra1 = n8->get_output();
#else

        auto infra1 = n1->add_output(INFRA1, 640, 480, rs2_format_type::Y8, fps);
#endif

        if (with_image)
        {
            std::shared_ptr<encode_image_node> n2(new encode_image_node());
            n2->set_input(infra1);
            g->add_node(n2);

            std::shared_ptr<p2p_talker_node> n3(new p2p_talker_node());
            n3->set_input(n2->get_output());
            g->add_node(n3);

            infra1_output = n3->get_output();
        }

        if (with_marker)
        {
            auto params = fast_blob_detector_node::blob_detector_params();
            {
                params.min_dist_between_blobs = 2;
                params.step_threshold = 20;
                params.min_threshold = 80;
                params.max_threshold = 250;

                params.min_area = 1;
                params.max_area = 100;

                params.min_circularity = 0.5;
            }

            std::shared_ptr<fifo_node> n9(new fifo_node());
            n9->set_input(infra1);
            g->add_node(n9);

            std::shared_ptr<fast_blob_detector_node> n4(new fast_blob_detector_node());
            n4->set_input(n9->get_output());
            n4->set_parameters(params);
            g->add_node(n4);

            std::shared_ptr<p2p_talker_node> n5(new p2p_talker_node());
            n5->set_input(n4->get_output());
            g->add_node(n5);

            infra1_marker_output = n5->get_output();
        }
    }
};

class remote_cluster_rs_d435_color : public remote_cluster
{
public:
    explicit remote_cluster_rs_d435_color(int fps)
    {
        constexpr bool with_image = true;
        constexpr bool with_marker = false;

        g.reset(new subgraph());

        const int width = 1920;
        const int height = 1080;

        std::shared_ptr<rs_d435_node> n1(new rs_d435_node());
        g->add_node(n1);

        std::shared_ptr<fifo_node> n7(new fifo_node());
        n7->set_input(n1->add_output(COLOR, width, height, rs2_format_type::BGR8, fps));
        g->add_node(n7);

        std::shared_ptr<resize_node> n10(new resize_node());
        n10->set_input(n7->get_output());
        n10->set_width(960);
        n10->set_height(540);
        g->add_node(n10);

        auto infra1 = n10->get_output();

        if (with_image)
        {
            std::shared_ptr<fifo_node> n4(new fifo_node());
            n4->set_input(n10->get_output());
            g->add_node(n4);

            std::shared_ptr<encode_image_node> n2(new encode_image_node());
            n2->set_input(n4->get_output());
            g->add_node(n2);

            std::shared_ptr<p2p_talker_node> n3(new p2p_talker_node());
            n3->set_input(n2->get_output());
            g->add_node(n3);

            infra1_output = n3->get_output();
        }

        if (with_marker)
        {
            std::shared_ptr<fifo_node> n9(new fifo_node());
            n9->set_input(infra1);
            g->add_node(n9);

            std::shared_ptr<charuco_detector_node> n4(new charuco_detector_node());
            n4->set_input(n9->get_output());
            g->add_node(n4);

            std::shared_ptr<p2p_talker_node> n5(new p2p_talker_node());
            n5->set_input(n4->get_output());
            g->add_node(n5);

            infra1_marker_output = n5->get_output();
        }
    }
};

class local_server
{
    asio::io_service io_service;
    std::shared_ptr<resource_list> resources;
    std::shared_ptr<graph_proc_server> server;
    std::shared_ptr<std::thread> th;
    std::atomic_bool running;

public:
    local_server(uint16_t port = 0)
        : io_service(), resources(std::make_shared<resource_list>()), server(std::make_shared<graph_proc_server>(io_service, "0.0.0.0", port, resources)), th(), running(false)
    {
    }

    uint16_t get_port() const
    {
        return server->get_port();
    }

    void run()
    {
        running = true;
        th.reset(new std::thread([this]
                                 { io_service.run(); }));
    }

    void stop()
    {
        if (running.load())
        {
            running.store(false);
            io_service.stop();
            if (th && th->joinable())
            {
                th->join();
            }
        }
    }

    ~local_server()
    {
        stop();
    }

    void add_resource(std::shared_ptr<resource_base> resource)
    {
        resources->add(resource);
    }
};

class callback_node;

class callback_list : public resource_base
{
    using callback_func = std::function<void(const callback_node *, std::string, graph_message_ptr)>;
    std::vector<callback_func> callbacks;

public:
    virtual std::string get_name() const
    {
        return "callback_list";
    }

    void add(callback_func callback)
    {
        callbacks.push_back(callback);
    }

    void invoke(const callback_node *node, std::string input_name, graph_message_ptr message) const
    {
        for (auto &callback : callbacks)
        {
            callback(node, input_name, message);
        }
    }
};

class callback_node : public graph_node
{
    std::string name;

public:
    callback_node()
        : graph_node()
    {
    }

    virtual std::string get_proc_name() const override
    {
        return "callback_node";
    }

    void set_name(const std::string &value)
    {
        name = value;
    }
    std::string get_name() const
    {
        return name;
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(name);
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (const auto resource = resources->get("callback_list"))
        {
            if (const auto callbacks = std::dynamic_pointer_cast<callback_list>(resource))
            {
                callbacks->invoke(this, input_name, message);
            }
        }
    }
};

CEREAL_REGISTER_TYPE(callback_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, callback_node)

class capture_pipeline::impl
{
    local_server server;
    asio::io_service io_service;
    graph_proc_client client;
    std::unique_ptr<std::thread> io_thread;

    mutable std::mutex frame_mtx;
    std::shared_ptr<frame_message<image>> frame;
    std::vector<keypoint> markers;

public:
    impl() : server(0)
    {
    }

    void run(const device_info &info)
    {
        std::vector<device_info> device_infos = {info};

        bool is_master = true;

        std::vector<std::unique_ptr<remote_cluster>> clusters;
        for (std::size_t i = 0; i < device_infos.size(); i++)
        {
            if (device_infos[i].type == device_type::raspi)
            {
                constexpr int fps = 90;
                clusters.emplace_back(std::make_unique<remote_cluster_raspi>(fps, nullptr, is_master));
                is_master = false;
            }
            else if (device_infos[i].type == device_type::raspi_color)
            {
                constexpr int fps = 30;
                clusters.emplace_back(std::make_unique<remote_cluster_raspi_color>(fps, is_master));
                is_master = false;
            }
            else if (device_infos[i].type == device_type::depthai_color)
            {
                constexpr int fps = 30;
                clusters.emplace_back(std::make_unique<remote_cluster_depthai_color>(fps));
            }
            else if (device_infos[i].type == device_type::rs_d435)
            {
                constexpr int fps = 90;
                constexpr int exposure = 5715;
                constexpr int gain = 248;
                constexpr int laser_power = 150;
                constexpr bool with_image = true;
                constexpr bool emitter_enabled = false;
                clusters.emplace_back(std::make_unique<remote_cluster_rs_d435>(fps, exposure, gain, laser_power, with_image, emitter_enabled));
            }
            else if (device_infos[i].type == device_type::rs_d435_color)
            {
                constexpr int fps = 30;
                clusters.emplace_back(std::make_unique<remote_cluster_rs_d435_color>(fps));
            }
        }

        std::shared_ptr<subgraph> g(new subgraph());

        std::vector<std::shared_ptr<graph_node>> rcv_nodes;
        std::vector<std::shared_ptr<p2p_listener_node>> rcv_marker_nodes;
        for (std::size_t i = 0; i < clusters.size(); i++)
        {
            auto &cluster = clusters[i];

            if (cluster->infra1_output)
            {
                std::shared_ptr<p2p_listener_node> n1(new p2p_listener_node());
                n1->set_input(cluster->infra1_output);
                n1->set_endpoint(device_infos[i].endpoint, 0);
                g->add_node(n1);

                std::shared_ptr<decode_image_node> n7(new decode_image_node());
                n7->set_input(n1->get_output());
                g->add_node(n7);

                rcv_nodes.push_back(n7);

                std::shared_ptr<callback_node> n8(new callback_node());
                n8->set_input(n7->get_output());
                g->add_node(n8);

                n8->set_name("image#" + device_infos[i].name);
            }

            if (cluster->infra1_marker_output)
            {
                std::shared_ptr<p2p_listener_node> n2(new p2p_listener_node());
                n2->set_input(cluster->infra1_marker_output);
                n2->set_endpoint(device_infos[i].endpoint, 0);
                g->add_node(n2);

                rcv_marker_nodes.push_back(n2);

                std::shared_ptr<callback_node> n8(new callback_node());
                n8->set_input(n2->get_output());
                g->add_node(n8);

                n8->set_name("marker#" + device_infos[i].name);
            }
        }

        const auto callbacks = std::make_shared<callback_list>();

        callbacks->add([this](const callback_node *node, std::string input_name, graph_message_ptr message)
                       {
            if (node->get_name().find_first_of("image#") == 0)
            {
                const auto camera_name = node->get_name().substr(6);

                if (auto frame_msg = std::dynamic_pointer_cast<frame_message<image>>(message))
                {
                    std::lock_guard lock(frame_mtx);
                    frame = frame_msg;
                }
            }
            if (node->get_name().find_first_of("marker#") == 0)
            {
                const auto camera_name = node->get_name().substr(6);

                if (auto frame_msg = std::dynamic_pointer_cast<keypoint_frame_message>(message))
                {
                    std::lock_guard lock(frame_mtx);
                    markers = frame_msg->get_data();
                }
            } });

        server.add_resource(callbacks);
        server.run();

        for (std::size_t i = 0; i < clusters.size(); i++)
        {
            client.deploy(io_service, device_infos[i].address, 31400, clusters[i]->g);
        }
        client.deploy(io_service, "127.0.0.1", server.get_port(), g);

        io_thread.reset(new std::thread([this]
                                        { io_service.run(); }));

        client.run();
    }

    void stop()
    {
        client.stop();
        server.stop();
        io_service.stop();
        if (io_thread && io_thread->joinable())
        {
            io_thread->join();
        }
        io_thread.reset();
    }

    cv::Mat get_frame() const
    {
        std::shared_ptr<frame_message<image>> frame;
        {
            std::lock_guard lock(frame_mtx);
            frame = this->frame;
        }

        if (!frame)
        {
            return cv::Mat();
        }

        const auto &image = frame->get_data();

        int type = -1;
        if (frame->get_profile())
        {
            auto format = frame->get_profile()->get_format();
            type = stream_format_to_cv_type(format);
        }

        if (type < 0)
        {
            throw std::logic_error("Unknown image format");
        }

        return cv::Mat(image.get_height(), image.get_width(), type, (uchar *)image.get_data(), image.get_stride()).clone();
    }

    std::unordered_map<int, cv::Point2f> get_markers() const
    {
        std::unordered_map<int, cv::Point2f> result;
        for (const auto& marker: markers)
        {
            result[marker.class_id] = cv::Point2f(marker.pt_x, marker.pt_y);
        }
        return result;
    }
};

capture_pipeline::capture_pipeline()
    : pimpl(new impl())
{
}
capture_pipeline::~capture_pipeline() = default;

void capture_pipeline::run(const device_info &info)
{
    pimpl->run(info);
}

void capture_pipeline::stop()
{
    pimpl->stop();
}
cv::Mat capture_pipeline::get_frame() const
{
    return pimpl->get_frame();
}
std::unordered_map<int, cv::Point2f> capture_pipeline::get_markers() const
{
    return pimpl->get_markers();
}

void capture_pipeline::set_mask(cv::Mat mask)
{
}

#include <fstream>

class dump_blob_node : public graph_node
{
    std::string data_dir;
    std::string ext;

public:
    dump_blob_node()
        : graph_node()
    {
    }

    void set_data_dir(std::string value)
    {
        data_dir = value;
    }

    void set_ext(std::string value)
    {
        ext = value;
    }

    virtual std::string get_proc_name() const override
    {
        return "dump_blob_node";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(data_dir);
        archive(ext);
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto frame_msg = std::dynamic_pointer_cast<frame_message<blob>>(message))
        {
            const auto &data = frame_msg->get_data();

            const auto path = data_dir + "/" + input_name + "_" + std::to_string((uint64_t)frame_msg->get_timestamp()) + ext;
            std::ofstream ofs;
            ofs.open(path, std::ios::binary | std::ios::out);
            ofs.write((const char *)data.data(), data.size());

            spdlog::debug("Saved image to '{0}'", path);
        }
    }
};

#include <regex>
#include <filesystem>

class load_blob_node : public graph_node
{
    std::string db_path;
    std::string name;
    stream_type stream;
    stream_format format;
    int fps;

    std::vector<uint64_t> timestamps;
    std::shared_ptr<std::thread> th;
    std::atomic_bool playing;
    graph_edge_ptr output;

    uint64_t start_timestamp;
    uint64_t id;

public:
    load_blob_node()
        : graph_node(), stream(stream_type::COLOR), format(stream_format::BGR8), fps(90), timestamps(), output(std::make_shared<graph_edge>(this)), start_timestamp(0)
    {
        set_output(output);
    }

    void set_db_path(std::string value)
    {
        db_path = value;
    }

    void set_name(std::string value)
    {
        name = value;
    }

    void set_stream(stream_type value)
    {
        stream = value;
    }

    void set_format(stream_format value)
    {
        format = value;
    }

    void set_fps(int value)
    {
        fps = value;
    }

    virtual std::string get_proc_name() const override
    {
        return "load_blob_node";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(db_path);
        archive(name);
        archive(stream);
        archive(format);
        archive(fps);
    }

    virtual void initialize() override
    {
        start_timestamp = std::numeric_limits<uint64_t>::max();

        sqlite3 *db;
        if (sqlite3_open(db_path.c_str(), &db) != SQLITE_OK)
        {
            throw std::runtime_error("Failed to open database");
        }

        {
            sqlite3_stmt *stmt;
            if (sqlite3_prepare_v2(db, "SELECT timestamp FROM messages ORDER BY timestamp ASC", -1, &stmt, nullptr) != SQLITE_OK)
            {
                throw std::runtime_error("Failed to prepare statement");
            }

            if (sqlite3_step(stmt) == SQLITE_ROW)
            {
                const auto timestamp = sqlite3_column_int64(stmt, 0);
                start_timestamp = std::min(start_timestamp, static_cast<uint64_t>(timestamp));
            }

            sqlite3_finalize(stmt);
        }

        {
            sqlite3_stmt *stmt;
            if (sqlite3_prepare_v2(db, "SELECT id, name FROM topics WHERE name = ?", -1, &stmt, nullptr) != SQLITE_OK)
            {
                throw std::runtime_error("Failed to prepare statement");
            }

            if (sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_STATIC) != SQLITE_OK)
            {
                throw std::runtime_error("Failed to bind text");
            }

            if (sqlite3_step(stmt) == SQLITE_ROW)
            {
                id = sqlite3_column_int64(stmt, 0);
            }
            else
            {
                throw std::runtime_error("Failed to get topic id");
            }

            sqlite3_finalize(stmt);
        }

        sqlite3_close(db);
    }

    virtual void run() override
    {
        th.reset(new std::thread([this]()
                                 {
            sqlite3 *db;
            if (sqlite3_open(db_path.c_str(), &db) != SQLITE_OK)
            {
                throw std::runtime_error("Failed to open database");
            }

            sqlite3_stmt *stmt;
            if (sqlite3_prepare_v2(db, "SELECT timestamp, topic_id, data FROM messages WHERE topic_id = ? ORDER BY timestamp ASC", -1, &stmt, nullptr) != SQLITE_OK)
            {
                throw std::runtime_error("Failed to prepare statement");
            }
            
            const auto id_str = std::to_string(id);
            if (sqlite3_bind_text(stmt, 1, id_str.c_str(), -1, SQLITE_STATIC) != SQLITE_OK)
            {
                throw std::runtime_error("Failed to bind text");
            }

            playing = true;
            uint64_t position = 0;
            const auto start_time = std::chrono::system_clock::now();
            while (playing && sqlite3_step(stmt) == SQLITE_ROW)
            {
                const auto timestamp = sqlite3_column_int64(stmt, 0);
                const auto elapsed_time = timestamp - start_timestamp;

                const auto data_size = static_cast<size_t>(sqlite3_column_bytes(stmt, 2));
                const auto data_ptr = reinterpret_cast<const uint8_t *>(sqlite3_column_blob(stmt, 2));
                const std::vector<uint8_t> data(data_ptr, data_ptr + data_size);

                std::this_thread::sleep_until(start_time + std::chrono::duration<double>(static_cast<double>(elapsed_time) / 1000.0));

                auto msg = std::make_shared<blob_frame_message>();
                msg->set_data(std::move(data));
                msg->set_frame_number(position);
                msg->set_timestamp(timestamp);
                msg->set_profile(std::make_shared<stream_profile>(
                    stream,
                    0,
                    format,
                    fps,
                    0));

                output->send(msg);

                position++;
            }

            sqlite3_finalize(stmt);
            stmt = nullptr; }));
    }

    virtual void stop() override
    {
        playing.store(false);
        if (th && th->joinable())
        {
            th->join();
        }
    }
};

CEREAL_REGISTER_TYPE(load_blob_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, load_blob_node)

class load_marker_node : public graph_node
{
    std::string db_path;
    std::string name;
    stream_type stream;
    stream_format format;
    int fps;

    std::shared_ptr<std::thread> th;
    std::atomic_bool playing;
    graph_edge_ptr output;

    uint64_t start_timestamp;
    uint64_t id;

public:
    load_marker_node()
        : graph_node(), stream(stream_type::ANY), format(stream_format::ANY), fps(90), output(std::make_shared<graph_edge>(this)), start_timestamp(0)
    {
        set_output(output);
    }

    void set_db_path(std::string value)
    {
        db_path = value;
    }

    void set_name(std::string value)
    {
        name = value;
    }

    void set_stream(stream_type value)
    {
        stream = value;
    }

    void set_format(stream_format value)
    {
        format = value;
    }

    void set_fps(int value)
    {
        fps = value;
    }

    virtual std::string get_proc_name() const override
    {
        return "load_marker_node";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(db_path);
        archive(name);
        archive(stream);
        archive(format);
        archive(fps);
    }

    virtual void initialize() override
    {
        start_timestamp = std::numeric_limits<uint64_t>::max();

        sqlite3 *db;
        if (sqlite3_open(db_path.c_str(), &db) != SQLITE_OK)
        {
            throw std::runtime_error("Failed to open database");
        }

        {
            sqlite3_stmt *stmt;
            if (sqlite3_prepare_v2(db, "SELECT timestamp FROM messages ORDER BY timestamp ASC", -1, &stmt, nullptr) != SQLITE_OK)
            {
                throw std::runtime_error("Failed to prepare statement");
            }

            if (sqlite3_step(stmt) == SQLITE_ROW)
            {
                const auto timestamp = sqlite3_column_int64(stmt, 0);
                start_timestamp = std::min(start_timestamp, static_cast<uint64_t>(timestamp));
            }

            sqlite3_finalize(stmt);
        }

        {
            sqlite3_stmt *stmt;
            if (sqlite3_prepare_v2(db, "SELECT id, name FROM topics WHERE name = ?", -1, &stmt, nullptr) != SQLITE_OK)
            {
                throw std::runtime_error("Failed to prepare statement");
            }

            if (sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_STATIC) != SQLITE_OK)
            {
                throw std::runtime_error("Failed to bind text");
            }

            if (sqlite3_step(stmt) == SQLITE_ROW)
            {
                id = sqlite3_column_int64(stmt, 0);
            }
            else
            {
                throw std::runtime_error("Failed to get topic id");
            }

            sqlite3_finalize(stmt);
        }

        sqlite3_close(db);
    }

    void read_frame(const std::string &text, std::vector<keypoint> &frame_data)
    {
        nlohmann::json j_frame = nlohmann::json::parse(text);

        const auto j_kpts = j_frame["points"];
        const auto timestamp = j_frame["timestamp"].get<double>();

        for (std::size_t j = 0; j < j_kpts.size(); j++)
        {
            frame_data.push_back(keypoint{j_kpts[j]["x"].get<float>(), j_kpts[j]["y"].get<float>()});
        }
    }

    virtual void run() override
    {
        th.reset(new std::thread([this]()
                                 {
            sqlite3 *db;
            if (sqlite3_open(db_path.c_str(), &db) != SQLITE_OK)
            {
                throw std::runtime_error("Failed to open database");
            }

            sqlite3_stmt *stmt;
            if (sqlite3_prepare_v2(db, "SELECT timestamp, topic_id, data FROM messages WHERE topic_id = ? ORDER BY timestamp ASC", -1, &stmt, nullptr) != SQLITE_OK)
            {
                throw std::runtime_error("Failed to prepare statement");
            }
            
            const auto id_str = std::to_string(id);
            if (sqlite3_bind_text(stmt, 1, id_str.c_str(), -1, SQLITE_STATIC) != SQLITE_OK)
            {
                throw std::runtime_error("Failed to bind text");
            }

            playing = true;
            uint64_t position = 0;
            const auto start_time = std::chrono::system_clock::now();
            while (playing && sqlite3_step(stmt) == SQLITE_ROW)
            {
                const auto timestamp = sqlite3_column_int64(stmt, 0);
                const auto elapsed_time = timestamp - start_timestamp;

                const std::string data(reinterpret_cast<const char *>(sqlite3_column_text(stmt, 2)));

                std::this_thread::sleep_until(start_time + std::chrono::duration<double>(static_cast<double>(elapsed_time) / 1000.0));

                std::vector<keypoint> points;
                read_frame(data, points);

                auto msg = std::make_shared<keypoint_frame_message>();
                msg->set_data(std::move(points));
                msg->set_frame_number(position);
                msg->set_timestamp(timestamp);
                msg->set_profile(std::make_shared<stream_profile>(
                    stream,
                    0,
                    format,
                    fps,
                    0));

                output->send(msg);

                position++;
            }

            sqlite3_finalize(stmt);
            stmt = nullptr;
        }));
    }

    virtual void stop() override
    {
        playing.store(false);
        if (th && th->joinable())
        {
            th->join();
        }
    }
};

CEREAL_REGISTER_TYPE(load_marker_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, load_marker_node)

class multiview_capture_pipeline::impl
{
    local_server server;
    asio::io_service io_service;
    graph_proc_client client;
    std::unique_ptr<std::thread> io_thread;

    mutable std::mutex image_frame_mtx;
    std::map<std::string, cv::Mat> frames;
    std::vector<std::function<void(const std::map<std::string, cv::Mat>&)>> image_received;

    std::map<std::string, std::shared_ptr<mask_node>> mask_nodes;
    std::map<std::string, cv::Mat> masks;

    mutable std::mutex marker_frame_mtx;
    std::deque<std::map<std::string, marker_frame_data>> marker_frames;

    mutable std::mutex marker_collecting_clusters_mtx;
    std::unordered_set<std::string> marker_collecting_clusters;

    std::vector<std::function<void(const std::map<std::string, marker_frame_data>&)>> marker_received;

public:
    void add_marker_received(std::function<void(const std::map<std::string, marker_frame_data> &)> f)
    {
        std::lock_guard lock(marker_frame_mtx);
        marker_received.push_back(f);
    }

    void clear_marker_received()
    {
        std::lock_guard lock(marker_frame_mtx);
        marker_received.clear();
    }

    void add_image_received(std::function<void(const std::map<std::string, cv::Mat> &)> f)
    {
        std::lock_guard lock(image_frame_mtx);
        image_received.push_back(f);
    }

    void clear_image_received()
    {
        std::lock_guard lock(image_frame_mtx);
        image_received.clear();
    }

    impl(const std::map<std::string, cv::Mat> &masks) : server(0), masks(masks)
    {
    }

    void run(const std::vector<device_info> &infos)
    {
        int sync_fps = 90;
        std::vector<device_info> device_infos = infos;

        std::vector<std::shared_ptr<remote_cluster>> clusters;
        for (std::size_t i = 0; i < device_infos.size(); i++)
        {
            if (device_infos[i].type == device_type::raspi)
            {
                constexpr int fps = 90;
                sync_fps = std::min(sync_fps, fps);
                std::shared_ptr<image> mask_img;
                if (masks.find(device_infos[i].name) != masks.end())
                {
                    const auto& mask = masks.at(device_infos[i].name);
                    mask_img.reset(new image(mask.cols, mask.rows, CV_8UC1, mask.step, (const uint8_t *)mask.data));
                }
                auto cluster = std::make_shared<remote_cluster_raspi>(fps, mask_img.get());
                mask_nodes.insert(std::make_pair(device_infos[i].name, cluster->mask_node_));
                clusters.emplace_back(cluster);
            }
            else if(device_infos[i].type == device_type::raspi_color)
            {
                constexpr int fps = 30;
                sync_fps = std::min(sync_fps, fps);
                clusters.emplace_back(std::make_shared<remote_cluster_raspi_color>(fps));
            }
            else if (device_infos[i].type == device_type::depthai_color)
            {
                constexpr int fps = 30;
                sync_fps = std::min(sync_fps, fps);
                clusters.emplace_back(std::make_unique<remote_cluster_depthai_color>(fps));
            }
            else if (device_infos[i].type == device_type::rs_d435)
            {
                constexpr int fps = 90;
                sync_fps = std::min(sync_fps, fps);
                constexpr int exposure = 5715;
                constexpr int gain = 248;
                constexpr int laser_power = 150;
                constexpr bool with_image = true;
                constexpr bool emitter_enabled = false;
                clusters.emplace_back(std::make_unique<remote_cluster_rs_d435>(fps, exposure, gain, laser_power, with_image, emitter_enabled));
            }
            else if (device_infos[i].type == device_type::rs_d435_color)
            {
                constexpr int fps = 30;
                sync_fps = std::min(sync_fps, fps);
                clusters.emplace_back(std::make_unique<remote_cluster_rs_d435_color>(fps));
            }
            else if (device_infos[i].type == device_type::raspi_playback)
            {
                constexpr int fps = 90;
                sync_fps = std::min(sync_fps, fps);
                clusters.push_back(nullptr);
            }
        }

        std::shared_ptr<subgraph> g(new subgraph());

        std::vector<std::shared_ptr<graph_node>> rcv_nodes;
        std::vector<std::shared_ptr<graph_node>> rcv_marker_nodes;
        for (std::size_t i = 0; i < device_infos.size(); i++)
        {
            auto &cluster = clusters[i];
            if (cluster && cluster->infra1_output)
            {
                std::shared_ptr<p2p_listener_node> n1(new p2p_listener_node());
                n1->set_input(cluster->infra1_output);
                n1->set_endpoint(device_infos[i].endpoint, 0);
                g->add_node(n1);

                std::shared_ptr<decode_image_node> n7(new decode_image_node());
                n7->set_input(n1->get_output());
                g->add_node(n7);

                rcv_nodes.push_back(n7);

                std::shared_ptr<callback_node> n8(new callback_node());
                n8->set_input(n7->get_output());
                g->add_node(n8);

                n8->set_name("image#" + device_infos[i].name);

#if 0
                const auto data_dir = "../data/output";
                std::shared_ptr<fifo_node> n12(new fifo_node());
                n12->set_input(n1->get_output());
                g->add_node(n12);

                std::shared_ptr<dump_blob_node> n5(new dump_blob_node());
                n5->set_input(n12->get_output(), "image_" + std::to_string(i + 1));
                n5->set_data_dir(data_dir);
                n5->set_ext(".jpg");
                g->add_node(n5);
#endif
            }
            else if (device_infos[i].type == device_type::raspi_playback)
            {
                std::shared_ptr<load_blob_node> n1(new load_blob_node());
                n1->set_name("image_" + std::to_string(i + 1));
                n1->set_db_path(device_infos[i].db_path);
                g->add_node(n1);

                std::shared_ptr<decode_image_node> n7(new decode_image_node());
                n7->set_input(n1->get_output());
                g->add_node(n7);

                rcv_nodes.push_back(n7);

                std::shared_ptr<callback_node> n8(new callback_node());
                n8->set_input(n7->get_output());
                g->add_node(n8);

                n8->set_name("image#" + device_infos[i].name);
            }
            else
            {
                rcv_nodes.push_back(nullptr);
            }

            if (cluster && cluster->infra1_marker_output)
            {
                std::shared_ptr<p2p_listener_node> n2(new p2p_listener_node());
                n2->set_input(cluster->infra1_marker_output);
                n2->set_endpoint(device_infos[i].endpoint, 0);
                g->add_node(n2);

                rcv_marker_nodes.push_back(n2);
            }
            else if (device_infos[i].type == device_type::raspi_playback)
            {
                std::shared_ptr<load_marker_node> n1(new load_marker_node());
                n1->set_name("marker_" + std::to_string(i + 1));
                n1->set_db_path(device_infos[i].db_path);
                g->add_node(n1);

                rcv_marker_nodes.push_back(n1);
            }
            else
            {
                rcv_marker_nodes.push_back(nullptr);
            }
        }

        std::shared_ptr<approximate_time_sync_node> n3(new approximate_time_sync_node());
        for (std::size_t i = 0; i < rcv_nodes.size(); i++)
        {
            if (rcv_nodes[i])
            {
                n3->set_input(rcv_nodes[i]->get_output(), device_infos[i].name);
            }
        }
        // Allow fps variation
        n3->get_config().set_interval(1000.0 / sync_fps + 0.5);
        g->add_node(n3);

        std::shared_ptr<callback_node> n8(new callback_node());
        n8->set_input(n3->get_output());
        g->add_node(n8);

        n8->set_name("images");

        
        std::shared_ptr<approximate_time_sync_node> n6(new approximate_time_sync_node());
        for (std::size_t i = 0; i < rcv_marker_nodes.size(); i++)
        {
            if (rcv_marker_nodes[i])
            {
                n6->set_input(rcv_marker_nodes[i]->get_output(), "camera" + std::to_string(i + 1));
            }
        }
        // Allow fps variation
        n6->get_config().set_interval(1000.0 / sync_fps + 0.5);
        g->add_node(n6);

        std::shared_ptr<callback_node> n9(new callback_node());
        n9->set_input(n6->get_output());
        g->add_node(n9);

        n9->set_name("markers");

        const auto callbacks = std::make_shared<callback_list>();

        callbacks->add([this](const callback_node *node, std::string input_name, graph_message_ptr message)
                       {
            if (node->get_name() == "images")
            {
                if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message))
                {
                    std::map<std::string, cv::Mat> frames;
                    for (const auto &[name, field] : obj_msg->get_fields())
                    {
                        if (auto image_msg = std::dynamic_pointer_cast<frame_message<image>>(field))
                        {
                            const auto &image = image_msg->get_data();

                            int type = -1;
                            if (image_msg->get_profile())
                            {
                                auto format = image_msg->get_profile()->get_format();
                                type = stream_format_to_cv_type(format);
                            }

                            if (type < 0)
                            {
                                throw std::logic_error("Unknown image format");
                            }

                            frames.insert(std::make_pair(name, cv::Mat(image.get_height(), image.get_width(), type, (uchar *)image.get_data(), image.get_stride()).clone()));
                        }
                    }

                    std::lock_guard lock(image_frame_mtx);
                    this->frames = frames;

                    for (const auto &f : image_received)
                    {
                        f(frames);
                    }
                }
            }
            else if (node->get_name() == "markers")
            {
                {
                    std::lock_guard lock(marker_collecting_clusters_mtx);
                    if (marker_collecting_clusters.empty())
                    {
                        return;
                    }
                }
                if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message))
                {
                    std::map<std::string, marker_frame_data> frames;
                    for (const auto &[name, field] : obj_msg->get_fields())
                    {
                        {
                            std::lock_guard lock(marker_collecting_clusters_mtx);
                            if (marker_collecting_clusters.find(name) == marker_collecting_clusters.end())
                            {
                                continue;
                            }
                        }
                        if (const auto keypoints_msg = std::dynamic_pointer_cast<keypoint_frame_message>(field))
                        {
                            const auto &keypoints = keypoints_msg->get_data();

                            marker_frame_data frame;
                            for (const auto &keypoint : keypoints)
                            {
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

                    std::lock_guard lock(marker_frame_mtx);
                    this->marker_frames.push_back(frames);

                    for (const auto &f : marker_received)
                    {
                        f(frames);
                    }
                }
            }
        });

        server.add_resource(callbacks);
        server.run();

        for (std::size_t i = 0; i < clusters.size(); i++)
        {
            if (clusters[i])
            {
                client.deploy(io_service, device_infos[i].address, 31400, clusters[i]->g);
            }
        }
        client.deploy(io_service, "127.0.0.1", server.get_port(), g);

        io_thread.reset(new std::thread([this]
                                        { io_service.run(); }));

        client.run();
    }

    void stop()
    {
        client.stop();
        server.stop();
        io_service.stop();
        if (io_thread && io_thread->joinable())
        {
            io_thread->join();
        }
        io_thread.reset();
    }

    std::map<std::string, cv::Mat> get_frames() const
    {
        std::map<std::string, cv::Mat> result;

        {
            std::lock_guard lock(image_frame_mtx);
            result = this->frames;
        }

        return result;
    }
    void gen_mask()
    {
        const auto frames = get_frames();
        for (const auto &[name, mask_node] : mask_nodes)
        {
            if (frames.find(name) == frames.end())
            {
                continue;
            }
            const auto frame_img = frames.at(name);
            if (frame_img.empty())
            {
                continue;
            }

            cv::Mat mask_img;
            {
                cv::threshold(frame_img, mask_img, 128, 255, cv::THRESH_BINARY);
                cv::morphologyEx(mask_img, mask_img, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2)));
                cv::dilate(mask_img, mask_img, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10)));
                cv::bitwise_not(mask_img, mask_img);
            }

            image mask(mask_img.cols, mask_img.rows, CV_8UC1, mask_img.step, (const uint8_t *)mask_img.data);
            const auto mask_msg = std::make_shared<image_message>();
            mask_msg->set_image(mask);
            client.process(mask_node.get(), "mask", mask_msg);

            masks[name] = mask_img;
        }
    }
    void clear_mask()
    {
        for (const auto &[name, mask_node] : mask_nodes)
        {
            const int width = 820;
            const int height = 616;
            cv::Mat mask_img(height, width, CV_8UC1, cv::Scalar(255));
            image mask(mask_img.cols, mask_img.rows, CV_8UC1, mask_img.step, (const uint8_t *)mask_img.data);
            const auto image_msg = std::make_shared<image_message>();
            image_msg->set_image(mask);
            client.process(mask_node.get(), "mask", image_msg);

            masks[name] = mask_img;
        }
    }
    std::map<std::string, cv::Mat> get_masks() const
    {
        return masks;
    }

    std::vector<std::map<std::string, marker_frame_data>> pop_marker_frames()
    {
        std::vector<std::map<std::string, marker_frame_data>> result;
        std::lock_guard lock(marker_frame_mtx);
        while (!marker_frames.empty())
        {
            result.push_back(marker_frames.front());
            marker_frames.pop_front();
        }
        return result;
    }

    void enable_marker_collecting(std::string name)
    {
        std::lock_guard lock(marker_collecting_clusters_mtx);
        marker_collecting_clusters.insert(name);
    }
    void disable_marker_collecting(std::string name)
    {
        std::lock_guard lock(marker_collecting_clusters_mtx);
        const auto found = marker_collecting_clusters.find(name);
        if (found != marker_collecting_clusters.end())
        {
            marker_collecting_clusters.erase(found);
        }
    }
};

multiview_capture_pipeline::multiview_capture_pipeline()
    : pimpl(new impl(std::map<std::string, cv::Mat>()))
{
}
multiview_capture_pipeline::multiview_capture_pipeline(const std::map<std::string, cv::Mat> &masks)
    : pimpl(new impl(masks))
{
}
multiview_capture_pipeline::~multiview_capture_pipeline() = default;

void multiview_capture_pipeline::run(const std::vector<device_info> &infos)
{
    pimpl->run(infos);
}

void multiview_capture_pipeline::stop()
{
    pimpl->stop();
}
std::map<std::string, cv::Mat> multiview_capture_pipeline::get_frames() const
{
    return pimpl->get_frames();
}
void multiview_capture_pipeline::gen_mask()
{
    pimpl->gen_mask();
}
void multiview_capture_pipeline::clear_mask()
{
    pimpl->clear_mask();
}
std::map<std::string, cv::Mat> multiview_capture_pipeline::get_masks() const
{
    return pimpl->get_masks();
}
std::vector<std::map<std::string, marker_frame_data>> multiview_capture_pipeline::pop_marker_frames()
{
    return pimpl->pop_marker_frames();
}

void multiview_capture_pipeline::enable_marker_collecting(std::string name)
{
    pimpl->enable_marker_collecting(name);
}
void multiview_capture_pipeline::disable_marker_collecting(std::string name)
{
    pimpl->disable_marker_collecting(name);
}
void multiview_capture_pipeline::add_marker_received(std::function<void(const std::map<std::string, marker_frame_data> &)> f)
{
    pimpl->add_marker_received(f);
}
void multiview_capture_pipeline::clear_marker_received()
{
    pimpl->clear_marker_received();
}
void multiview_capture_pipeline::add_image_received(std::function<void(const std::map<std::string, cv::Mat> &)> f)
{
    pimpl->add_image_received(f);
}
void multiview_capture_pipeline::clear_image_received()
{
    pimpl->clear_image_received();
}
