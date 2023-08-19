#include "capture.hpp"

#include "graph_proc.h"
#include "graph_proc_cv.h"
#include "ext/graph_proc_jpeg.h"
#include "ext/graph_proc_cv.h"
#include "ext/graph_proc_libcamera.h"

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
    explicit remote_cluster_raspi(int fps, const image* mask, bool emitter_enabled = true)
    {
        constexpr bool with_marker = true;

        g.reset(new subgraph());

        // Camera Module v2
        const int width = 820;
        const int height = 616;
        // Camera Module 3 Wide
        // const int width = 1536;
        // const int height = 864;

        std::shared_ptr<libcamera_capture_node> n1(new libcamera_capture_node());
        n1->set_stream(stream_type::INFRERED);
        n1->set_option(libcamera_capture_node::option::exposure, 5000);
        n1->set_option(libcamera_capture_node::option::gain, 10);

        n1->set_fps(fps);
        n1->set_width(width);
        n1->set_height(height);
        n1->set_format(image_format::Y8_UINT);
        n1->set_emitter_enabled(emitter_enabled);

        g->add_node(n1);

        std::shared_ptr<fifo_node> n7(new fifo_node());
        n7->set_input(n1->get_output());
        g->add_node(n7);

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

class local_server
{
    asio::io_service io_service;
    std::shared_ptr<resource_list> resources;
    std::shared_ptr<graph_proc_server> server;
    std::shared_ptr<std::thread> th;
    std::atomic_bool running;

public:
    local_server(uint16_t port = 31400)
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

class capture_controller::impl
{
    local_server server;
    asio::io_service io_service;
    graph_proc_client client;
    std::unique_ptr<std::thread> io_thread;

    mutable std::mutex frame_mtx;
    std::shared_ptr<frame_message<image>> frame;

public:
    impl() : server(0)
    {
    }

    void run(const cluster_info &info)
    {
        int fps = 90;
        std::vector<cluster_info> cluster_infos = {info};

        std::vector<std::unique_ptr<remote_cluster>> clusters;
        for (std::size_t i = 0; i < cluster_infos.size(); i++)
        {
            if (cluster_infos[i].type == cluster_type::raspi)
            {
                clusters.emplace_back(std::make_unique<remote_cluster_raspi>(fps, nullptr));
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
                n1->set_endpoint(cluster_infos[i].endpoint, 0);
                g->add_node(n1);

                std::shared_ptr<decode_image_node> n7(new decode_image_node());
                n7->set_input(n1->get_output());
                g->add_node(n7);

                rcv_nodes.push_back(n7);

                std::shared_ptr<callback_node> n8(new callback_node());
                n8->set_input(n7->get_output());
                g->add_node(n8);

                n8->set_name("image#" + cluster_infos[i].name);
            }

            if (cluster->infra1_marker_output)
            {
                std::shared_ptr<p2p_listener_node> n2(new p2p_listener_node());
                n2->set_input(cluster->infra1_marker_output);
                n2->set_endpoint(cluster_infos[i].endpoint, 0);
                g->add_node(n2);

                rcv_marker_nodes.push_back(n2);
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
            } });

        server.add_resource(callbacks);
        server.run();

        for (std::size_t i = 0; i < clusters.size(); i++)
        {
            client.deploy(io_service, cluster_infos[i].address, 31400, clusters[i]->g);
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
};

capture_controller::capture_controller()
    : pimpl(new impl())
{
}
capture_controller::~capture_controller() = default;

void capture_controller::run(const cluster_info &info)
{
    pimpl->run(info);
}

void capture_controller::stop()
{
    pimpl->stop();
}
cv::Mat capture_controller::get_frame() const
{
    return pimpl->get_frame();
}

void capture_controller::set_mask(cv::Mat mask)
{
}

class sync_capture_controller::impl
{
    local_server server;
    asio::io_service io_service;
    graph_proc_client client;
    std::unique_ptr<std::thread> io_thread;

    mutable std::mutex frame_mtx;
    std::map<std::string, std::shared_ptr<frame_message<image>>> frames;

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

    impl(const std::map<std::string, cv::Mat> &masks) : server(0), masks(masks)
    {
    }

    void run(const std::vector<cluster_info> &infos)
    {
        int fps = 90;
        std::vector<cluster_info> cluster_infos = infos;

        std::vector<std::shared_ptr<remote_cluster>> clusters;
        for (std::size_t i = 0; i < cluster_infos.size(); i++)
        {
            if (cluster_infos[i].type == cluster_type::raspi)
            {
                std::shared_ptr<image> mask_img;
                if (masks.find(cluster_infos[i].name) != masks.end())
                {
                    const auto& mask = masks.at(cluster_infos[i].name);
                    mask_img.reset(new image(mask.cols, mask.rows, CV_8UC1, mask.step, (const uint8_t *)mask.data));
                }
                auto cluster = std::make_shared<remote_cluster_raspi>(fps, mask_img.get());
                mask_nodes.insert(std::make_pair(cluster_infos[i].name, cluster->mask_node_));
                clusters.emplace_back(cluster);
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
                n1->set_endpoint(cluster_infos[i].endpoint, 0);
                g->add_node(n1);

                std::shared_ptr<decode_image_node> n7(new decode_image_node());
                n7->set_input(n1->get_output());
                g->add_node(n7);

                rcv_nodes.push_back(n7);

                std::shared_ptr<callback_node> n8(new callback_node());
                n8->set_input(n7->get_output());
                g->add_node(n8);

                n8->set_name("image#" + cluster_infos[i].name);
            }
            else
            {
                rcv_nodes.push_back(nullptr);
            }

            if (cluster->infra1_marker_output)
            {
                std::shared_ptr<p2p_listener_node> n2(new p2p_listener_node());
                n2->set_input(cluster->infra1_marker_output);
                n2->set_endpoint(cluster_infos[i].endpoint, 0);
                g->add_node(n2);

                rcv_marker_nodes.push_back(n2);
            }
            else
            {
                rcv_marker_nodes.push_back(nullptr);
            }
        }

        std::shared_ptr<approximate_time_sync_node> n3(new approximate_time_sync_node());
        for (std::size_t i = 0; i < clusters.size(); i++)
        {
            if (rcv_nodes[i])
            {
                n3->set_input(rcv_nodes[i]->get_output(), cluster_infos[i].name);
            }
        }
        // Allow fps variation
        n3->get_config().set_interval(1000.0 / fps + 0.5);
        g->add_node(n3);

        std::shared_ptr<callback_node> n8(new callback_node());
        n8->set_input(n3->get_output());
        g->add_node(n8);

        n8->set_name("images");

        
        std::shared_ptr<approximate_time_sync_node> n6(new approximate_time_sync_node());
        for (std::size_t i = 0; i < clusters.size(); i++)
        {
            if (rcv_marker_nodes[i])
            {
                n6->set_input(rcv_marker_nodes[i]->get_output(), "camera" + std::to_string(i + 1));
            }
        }
        // Allow fps variation
        n6->get_config().set_interval(1000.0 / fps + 0.5);
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
                    std::map<std::string, std::shared_ptr<frame_message<image>>> frames;
                    for (const auto &[name, field] : obj_msg->get_fields())
                    {
                        if (auto image_msg = std::dynamic_pointer_cast<frame_message<image>>(field))
                        {
                            frames.insert(std::make_pair(name, image_msg));
                        }
                    }

                    std::lock_guard lock(frame_mtx);
                    this->frames = frames;
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
            client.deploy(io_service, cluster_infos[i].address, 31400, clusters[i]->g);
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

        std::map<std::string, std::shared_ptr<frame_message<image>>> frames;
        {
            std::lock_guard lock(frame_mtx);
            frames = this->frames;
        }

        if (frames.empty())
        {
            return result;
        }

        for (const auto &[name, frame] : frames)
        {
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

            result[name] = cv::Mat(image.get_height(), image.get_width(), type, (uchar *)image.get_data(), image.get_stride()).clone();
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

sync_capture_controller::sync_capture_controller()
    : pimpl(new impl(std::map<std::string, cv::Mat>()))
{
}
sync_capture_controller::sync_capture_controller(const std::map<std::string, cv::Mat> &masks)
    : pimpl(new impl(masks))
{
}
sync_capture_controller::~sync_capture_controller() = default;

void sync_capture_controller::run(const std::vector<cluster_info> &infos)
{
    pimpl->run(infos);
}

void sync_capture_controller::stop()
{
    pimpl->stop();
}
std::map<std::string, cv::Mat> sync_capture_controller::get_frames() const
{
    return pimpl->get_frames();
}
void sync_capture_controller::gen_mask()
{
    pimpl->gen_mask();
}
void sync_capture_controller::clear_mask()
{
    pimpl->clear_mask();
}
std::map<std::string, cv::Mat> sync_capture_controller::get_masks() const
{
    return pimpl->get_masks();
}
std::vector<std::map<std::string, marker_frame_data>> sync_capture_controller::pop_marker_frames()
{
    return pimpl->pop_marker_frames();
}

void sync_capture_controller::enable_marker_collecting(std::string name)
{
    pimpl->enable_marker_collecting(name);
}
void sync_capture_controller::disable_marker_collecting(std::string name)
{
    pimpl->disable_marker_collecting(name);
}
void sync_capture_controller::add_marker_received(std::function<void(const std::map<std::string, marker_frame_data> &)> f)
{
    pimpl->add_marker_received(f);
}

void sync_capture_controller::clear_marker_received()
{
    pimpl->clear_marker_received();
}
