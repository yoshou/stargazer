#include "reconstruction.hpp"
#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>

#include "sensor.grpc.pb.h"

#include <spdlog/spdlog.h>

#include "parameters.hpp"
#include "utils.hpp"
#include "correspondance.hpp"
#include "triangulation.hpp"
#include "point_data.hpp"
#include "glm_serialize.hpp"
#include "glm_json.hpp"

#include "graph_proc.h"
#include "graph_proc_cv.h"
#include "graph_proc_img.h"

#include "voxelpose.hpp"
#include "mvpose.hpp"

using namespace coalsack;

static std::string read_text_file(const std::string& filename)
{
    std::ifstream ifs(filename.c_str());
    return std::string(
        std::istreambuf_iterator<char>(ifs),
        std::istreambuf_iterator<char>());
}

class SensorServiceImpl final : public stargazer::Sensor::Service
{
    std::mutex mtx;
    std::unordered_map<std::string, grpc::ServerWriter<stargazer::SphereMessage> *> writers;

public:
    void notify_sphere(const std::vector<glm::vec3> &spheres)
    {
        stargazer::SphereMessage response;
        const auto mutable_values = response.mutable_values();
        for (const auto &sphere : spheres)
        {
            const auto mutable_value = mutable_values->Add();
            mutable_value->mutable_point()->set_x(sphere.x);
            mutable_value->mutable_point()->set_y(sphere.y);
            mutable_value->mutable_point()->set_z(sphere.z);
            mutable_value->set_radius(0.02);
        }
        {
            std::lock_guard<std::mutex> lock(mtx);
            for (const auto &[name, writer] : writers)
            {
                writer->Write(response);
            }
        }
    }

    grpc::Status SubscribeSphere(grpc::ServerContext *context,
                                 const stargazer::SubscribeRequest *request,
                                 grpc::ServerWriter<stargazer::SphereMessage> *writer) override
    {
        {
            std::lock_guard<std::mutex> lock(mtx);
            writers.insert(std::make_pair(request->name(), writer));
        }
        while (!context->IsCancelled())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (const auto iter = writers.find(request->name()); iter != writers.end())
            {
                writers.erase(iter);
            }
        }
        return grpc::Status::OK;
    }
};

static std::vector<glm::vec3> reconstruct(const std::vector<stargazer::camera_t> &camera_list, const std::vector<std::vector<glm::vec2>>& camera_pts, glm::mat4 axis)
{
    if (camera_list.size() < 2)
    {
        return {};
    }
    std::vector<stargazer::reconstruction::node_t> nodes;
    stargazer::reconstruction::adj_list_t adj;

    const auto thresh = 1.0;
    stargazer::reconstruction::find_correspondance(camera_pts, camera_list, nodes, adj, thresh);

#if 0
    stargazer::reconstruction::compute_hard_correspondance(nodes, adj, camera_list);

    std::vector<std::vector<std::size_t>> connected_components;
    stargazer::reconstruction::compute_observations(adj, connected_components);
#else
    stargazer::reconstruction::remove_ambiguous_observations(nodes, adj, camera_list, 0.01);

    std::vector<std::vector<std::size_t>> connected_components;
    stargazer::reconstruction::compute_observations(adj, connected_components);
#endif

    bool all_hard_correspondance = true;
    for (std::size_t i = 0; i < connected_components.size(); i++)
    {
        const auto &connected_graph = connected_components[i];
        const auto has_ambigious = stargazer::reconstruction::has_soft_correspondance(nodes, connected_graph);
        if (has_ambigious)
        {
            all_hard_correspondance = false;
            break;
        }
    }

    if (!all_hard_correspondance)
    {
        std::cout << "Can't find correspondance points on frame" << std::endl;
    }

    std::vector<glm::vec3> markers;
    for (auto &g : connected_components)
    {
        if (g.size() < 2)
        {
            continue;
        }

        std::vector<glm::vec2> pts;
        std::vector<stargazer::camera_t> cams;

        for (std::size_t i = 0; i < g.size(); i++)
        {
            pts.push_back(nodes[g[i]].pt);
            cams.push_back(camera_list[nodes[g[i]].camera_idx]);
        }
        const auto marker = stargazer::reconstruction::triangulate(pts, cams);
        markers.push_back(glm::vec3(axis * glm::vec4(marker, 1.0f)));
    }

    return markers;
}

std::vector<glm::vec3> reconstruct(const std::map<std::string, stargazer::camera_t> &cameras, const std::map<std::string, std::vector<stargazer::point_data>> &frame, glm::mat4 axis)
{
    std::vector<std::vector<glm::vec2>> camera_pts;
    std::vector<stargazer::camera_t> camera_list;
    for (const auto &[camera_name, camera] : cameras)
    {
        std::vector<glm::vec2> pts;
        for (const auto &pt : frame.at(camera_name))
        {
            pts.push_back(pt.point);
        }
        camera_pts.push_back(pts);
        camera_list.push_back(camera);
    }
    return reconstruct(camera_list, camera_pts, axis);
}

struct float2
{
    float x;
    float y;

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(x, y);
    }
};

struct float3
{
    float x;
    float y;
    float z;

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(x, y, z);
    }
};

using float2_list_message = frame_message<std::vector<float2>>;
using float3_list_message = frame_message<std::vector<float3>>;
using mat4_message = frame_message<glm::mat4>;

CEREAL_REGISTER_TYPE(float2_list_message)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::frame_message_base, float2_list_message)

CEREAL_REGISTER_TYPE(float3_list_message)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::frame_message_base, float3_list_message)

CEREAL_REGISTER_TYPE(mat4_message)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::frame_message_base, mat4_message)

CEREAL_REGISTER_TYPE(frame_message<object_message>)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::frame_message_base, frame_message<object_message>)

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

class camera_message : public object_message
{
    stargazer::camera_t camera;

public:
    camera_message() : object_message(), camera()
    {
    }

    camera_message(const stargazer::camera_t &camera) : object_message(), camera(camera)
    {
    }

    stargazer::camera_t get_camera() const
    {
        return camera;
    }

    void set_camera(const stargazer::camera_t &value)
    {
        camera = value;
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(camera);
    }
};

CEREAL_REGISTER_TYPE(camera_message)
CEREAL_REGISTER_POLYMORPHIC_RELATION(object_message, camera_message)

class epipolar_reconstruct_node : public graph_node
{
    mutable std::mutex cameras_mtx;
    std::map<std::string, stargazer::camera_t> cameras;
    glm::mat4 axis;
    graph_edge_ptr output;

public:
    epipolar_reconstruct_node()
        : graph_node(), cameras(), axis(), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    virtual std::string get_proc_name() const override
    {
        return "epipolar_reconstruct_node";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(cameras, axis);
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (input_name == "cameras")
        {
            if (auto camera_msg = std::dynamic_pointer_cast<object_message>(message))
            {
                for (const auto &[name, field] : camera_msg->get_fields())
                {
                    if (auto camera_msg = std::dynamic_pointer_cast<camera_message>(field))
                    {
                        std::lock_guard lock(cameras_mtx);
                        cameras[name] = camera_msg->get_camera();
                    }
                }
            }

            return;
        }
        if (input_name == "axis")
        {
            if (auto mat4_msg = std::dynamic_pointer_cast<mat4_message>(message))
            {
                axis = mat4_msg->get_data();
            }

            return;
        }

        if (auto frame_msg = std::dynamic_pointer_cast<frame_message<object_message>>(message))
        {
            const auto obj_msg = frame_msg->get_data();
            
            std::vector<std::vector<glm::vec2>> camera_pts;
            std::vector<stargazer::camera_t> camera_list;

            std::map<std::string, stargazer::camera_t> cameras;
            {
                std::lock_guard lock(cameras_mtx);
                cameras = this->cameras;
            }

            for (const auto &[name, field] : obj_msg.get_fields())
            {
                if (auto points_msg = std::dynamic_pointer_cast<float2_list_message>(field))
                {
                    if (cameras.find(name) == cameras.end())
                    {
                        continue;
                    }
                    const auto &camera = cameras.at(name);
                    std::vector<glm::vec2> pts;
                    for (const auto &pt : points_msg->get_data())
                    {
                        pts.push_back(glm::vec2(pt.x, pt.y));
                    }
                    camera_pts.push_back(pts);
                    camera_list.push_back(camera);
                }
            }

            const auto markers = reconstruct(camera_list, camera_pts, axis);

            auto marker_msg = std::make_shared<float3_list_message>();
            std::vector<float3> marker_data;
            for (const auto &marker : markers)
            {
                marker_data.push_back({marker.x, marker.y, marker.z});
            }
            marker_msg->set_data(marker_data);
            marker_msg->set_frame_number(frame_msg->get_frame_number());
            output->send(marker_msg);
        }
    }
};

CEREAL_REGISTER_TYPE(epipolar_reconstruct_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, epipolar_reconstruct_node)

class grpc_server_node : public graph_node
{
    std::shared_ptr<std::thread> server_th;
    std::unique_ptr<grpc::Server> server;
    std::unique_ptr<SensorServiceImpl> service;

public:
    grpc_server_node()
        : graph_node(), server_th(), server(), service(std::make_unique<SensorServiceImpl>())
    {
    }

    virtual std::string get_proc_name() const override
    {
        return "grpc_server_node";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
    }

    virtual void run() override
    {
        std::string server_address("0.0.0.0:50051");

        grpc::ServerBuilder builder;
#ifdef USE_SECURE_CREDENTIALS
        std::string ca_crt_content = read_text_file("../data/ca.crt");
        std::string server_crt_content = read_text_file("../data/server.crt");
        std::string server_key_content = read_text_file("../data/server.key");

        grpc::SslServerCredentialsOptions ssl_options;
        grpc::SslServerCredentialsOptions::PemKeyCertPair key_cert = { server_key_content, server_crt_content };
        ssl_options.pem_root_certs = ca_crt_content;
        ssl_options.pem_key_cert_pairs.push_back(key_cert);

        builder.AddListeningPort(server_address, grpc::SslServerCredentials(ssl_options));
#else
        builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
#endif
        builder.RegisterService(service.get());
        server = builder.BuildAndStart();
        spdlog::info("Server listening on " + server_address);

        server_th.reset(new std::thread([this]()
        {
            server->Wait();
        }));
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (input_name == "sphere")
        {
            if (const auto msg = std::dynamic_pointer_cast<float3_list_message>(message))
            {
                std::vector<glm::vec3> spheres;
                for (const auto &data : msg->get_data())
                {
                    spheres.push_back(glm::vec3(data.x, data.y, data.z));
                }

                service->notify_sphere(spheres);
            }
        }
    }

    virtual void stop() override
    {
        if (server)
        {
            server->Shutdown();
            if (server_th && server_th->joinable())
            {
                server_th->join();
            }
        }
    }
};

CEREAL_REGISTER_TYPE(grpc_server_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, grpc_server_node)

class frame_number_numbering_node : public graph_node
{
    uint64_t frame_number;
    graph_edge_ptr output;

public:
    frame_number_numbering_node()
        : graph_node(), frame_number(0), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    virtual std::string get_proc_name() const override
    {
        return "frame_number_numbering_node";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(frame_number);
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto msg = std::dynamic_pointer_cast<frame_message_base>(message))
        {
            msg->set_frame_number(frame_number++);
            output->send(msg);
        }
    }
};

CEREAL_REGISTER_TYPE(frame_number_numbering_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, frame_number_numbering_node)

class parallel_queue_node : public graph_node
{
    std::shared_ptr<task_queue<std::function<void()>>> workers;
    graph_edge_ptr output;

public:
    parallel_queue_node()
        : graph_node(), workers(std::make_shared<task_queue<std::function<void()>>>()), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    virtual std::string get_proc_name() const override
    {
        return "parallel_queue_node";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
    }

    virtual void run() override
    {
    }

    virtual void stop() override
    {
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto msg = std::dynamic_pointer_cast<frame_message_base>(message))
        {
            workers->push_task([this, msg]()
                          {
                output->send(msg);
            });
        }
    }
};

CEREAL_REGISTER_TYPE(parallel_queue_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, parallel_queue_node)

class greater_graph_message_ptr
{
public:
    bool operator()(const graph_message_ptr &lhs, const graph_message_ptr &rhs) const
    {
        return std::dynamic_pointer_cast<frame_message_base>(lhs)->get_frame_number() > std::dynamic_pointer_cast<frame_message_base>(rhs)->get_frame_number();
    }
};

class frame_number_ordering_node : public graph_node
{
    graph_edge_ptr output;
    std::mutex mtx;

    std::priority_queue<
        graph_message_ptr,
        std::deque<graph_message_ptr>,
        greater_graph_message_ptr> messages;

    std::shared_ptr<std::thread> th;
    std::atomic_bool running;
    std::condition_variable cv;
    std::uint32_t max_size;
    std::atomic_ullong frame_number;

public:
    frame_number_ordering_node()
        : graph_node(), output(std::make_shared<graph_edge>(this)), max_size(100), frame_number(0)
    {
        set_output(output);
    }

    virtual std::string get_proc_name() const override
    {
        return "frame_number_ordering_node";
    }

    void set_max_size(std::uint32_t value)
    {
        max_size = value;
    }
    std::uint32_t get_max_size() const
    {
        return max_size;
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(max_size, frame_number);
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (!running)
        {
            return;
        }

        if (input_name == "default")
        {
            std::lock_guard<std::mutex> lock(mtx);

            if (messages.size() >= max_size)
            {
                std::cout << "Fifo overflow" << std::endl;
                spdlog::error("Fifo overflow");
            }
            else
            {
                messages.push(message);
                cv.notify_one();
            }
        }
    }

    virtual void run() override
    {
        running = true;
        th.reset(new std::thread([this]()
                                 {
            while (running.load())
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&]
                        { return (!messages.empty() && std::dynamic_pointer_cast<frame_message_base>(messages.top())->get_frame_number() == frame_number) || !running; });

                if (!running)
                {
                    break;
                }
                if (!messages.empty() && std::dynamic_pointer_cast<frame_message_base>(messages.top())->get_frame_number() == frame_number)
                {
                    const auto message = messages.top();
                    messages.pop();
                    output->send(message);

                    frame_number++;
                }
            } }));
    }

    virtual void stop() override
    {
        if (running.load())
        {
            {
                std::lock_guard<std::mutex> lock(mtx);
                running.store(false);
            }
            cv.notify_one();
            if (th && th->joinable())
            {
                th->join();
            }
        }
    }
};

class epipolar_reconstruction_pipeline
{
    graph_proc graph;

    std::atomic_bool running;

    mutable std::mutex markers_mtx;
    std::vector<glm::vec3> markers;
    std::vector<std::function<void(const std::vector<glm::vec3> &)>> markers_received;

    std::shared_ptr<epipolar_reconstruct_node> reconstruct_node;
    std::shared_ptr<graph_node> input_node;

public:
    void add_markers_received(std::function<void(const std::vector<glm::vec3> &)> f)
    {
        std::lock_guard lock(markers_mtx);
        markers_received.push_back(f);
    }

    void clear_markers_received()
    {
        std::lock_guard lock(markers_mtx);
        markers_received.clear();
    }

    epipolar_reconstruction_pipeline()
        : graph(), running(false), markers(), markers_received(), reconstruct_node(), input_node()
    {
    }

    void set_camera(const std::string &name, const stargazer::camera_t &camera)
    {
        auto camera_msg = std::make_shared<camera_message>(camera);
        camera_msg->set_camera(camera);

        auto obj_msg = std::make_shared<object_message>();
        obj_msg->add_field(name, camera_msg);

        if (reconstruct_node)
        {
            graph.process(reconstruct_node.get(), "cameras", obj_msg);
        }
    }

    void set_axis(const glm::mat4 &axis)
    {
        auto mat4_msg = std::make_shared<mat4_message>();
        mat4_msg->set_data(axis);

        if (reconstruct_node)
        {
            graph.process(reconstruct_node.get(), "axis", mat4_msg);
        }
    }
    
    using frame_type = std::map<std::string, std::vector<stargazer::point_data>>;

    void push_frame(const frame_type &frame)
    {
        if (!running)
        {
            return;
        }

        auto msg = std::make_shared<object_message>();
        for (const auto &[name, field] : frame)
        {
            auto float2_msg = std::make_shared<float2_list_message>();
            std::vector<float2> float2_data;
            for (const auto &pt : field)
            {
                float2_data.push_back({pt.point.x, pt.point.y});
            }
            float2_msg->set_data(float2_data);
            msg->add_field(name, float2_msg);
        }

        auto frame_msg = std::make_shared<frame_message<object_message>>();
        frame_msg->set_data(*msg);

        if (input_node)
        {
            graph.process(input_node.get(), frame_msg);
        }
    }

    void run()
    {
        std::shared_ptr<subgraph> g(new subgraph());

        std::shared_ptr<frame_number_numbering_node> n4(new frame_number_numbering_node());
        g->add_node(n4);

        input_node = n4;

        std::shared_ptr<parallel_queue_node> n6(new parallel_queue_node());
        n6->set_input(n4->get_output());
        g->add_node(n6);

        std::shared_ptr<epipolar_reconstruct_node> n1(new epipolar_reconstruct_node());
        n1->set_input(n6->get_output());
        g->add_node(n1);

        reconstruct_node = n1;

        std::shared_ptr<frame_number_ordering_node> n5(new frame_number_ordering_node());
        n5->set_input(n1->get_output());
        g->add_node(n5);

        std::shared_ptr<callback_node> n2(new callback_node());
        n2->set_input(n5->get_output());
        g->add_node(n2);

        n2->set_name("markers");

        std::shared_ptr<grpc_server_node> n3(new grpc_server_node());
        n3->set_input(n5->get_output(), "sphere");
        g->add_node(n3);

        const auto callbacks = std::make_shared<callback_list>();

        callbacks->add([this](const callback_node *node, std::string input_name, graph_message_ptr message)
                       {
            if (node->get_name() == "markers")
            {
                if (const auto markers_msg = std::dynamic_pointer_cast<float3_list_message>(message))
                {
                    std::vector<glm::vec3> markers;
                    for (const auto &marker : markers_msg->get_data())
                    {
                        markers.push_back(glm::vec3(marker.x, marker.y, marker.z));
                    }

                    {
                        std::lock_guard lock(markers_mtx);
                        this->markers = markers;
                    }

                    for (const auto &f : markers_received)
                    {
                        f(markers);
                    }
                }
            } });

        graph.deploy(g);
        graph.get_resources()->add(callbacks);
        graph.run();

        running = true;
    }

    void stop()
    {
        running.store(false);
        graph.stop();
    }

    std::vector<glm::vec3> get_markers() const
    {
        std::vector<glm::vec3> result;

        {
            std::lock_guard lock(markers_mtx);
            result = this->markers;
        }

        return result;
    }
};

class epipolar_reconstruction::impl
{
public:
    std::shared_ptr<epipolar_reconstruction_pipeline> pipeline;

    impl()
        : pipeline(std::make_shared<epipolar_reconstruction_pipeline>())
    {
    }

    ~impl() = default;

    void run()
    {
        pipeline->run();
    }

    void stop()
    {
        pipeline->stop();
    }
};

epipolar_reconstruction::epipolar_reconstruction()
    : pimpl(new impl()) {}
epipolar_reconstruction::~epipolar_reconstruction() = default;

void epipolar_reconstruction::push_frame(const frame_type &frame)
{
    pimpl->pipeline->push_frame(frame);
}

void epipolar_reconstruction::run()
{
    pimpl->run();
}

void epipolar_reconstruction::stop()
{
    pimpl->stop();
}

std::vector<glm::vec3> epipolar_reconstruction::get_markers() const
{
    return pimpl->pipeline->get_markers();
}

void epipolar_reconstruction::set_camera(const std::string &name, const stargazer::camera_t &camera)
{
    pimpl->pipeline->set_camera(name, camera);
    multiview_point_reconstruction::set_camera(name, camera);
}
void epipolar_reconstruction::set_axis(const glm::mat4 &axis)
{
    pimpl->pipeline->set_axis(axis);
    multiview_point_reconstruction::set_axis(axis);
}

#include <cereal/types/array.hpp>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

#define PANOPTIC

#ifdef PANOPTIC
namespace stargazer::voxelpose
{
    static std::map<std::string, camera_data> load_cameras()
    {
        using namespace stargazer::voxelpose;

        std::map<std::string, camera_data> cameras;

        const auto camera_file = fs::path("/workspace/data/panoptic/calibration_171204_pose1.json");

        std::ifstream f;
        f.open(camera_file, std::ios::in | std::ios::binary);
        std::string str((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());

        nlohmann::json calib = nlohmann::json::parse(str);

        for (const auto &cam : calib["cameras"])
        {
            const auto panel = cam["panel"].get<int32_t>();
            const auto node = cam["node"].get<int32_t>();

            const auto k = cam["K"].get<std::vector<std::vector<double>>>();
            const auto dist_coeffs = cam["distCoef"].get<std::vector<double>>();
            const auto rotation = cam["R"].get<std::vector<std::vector<double>>>();
            const auto translation = cam["t"].get<std::vector<std::vector<double>>>();

            const std::array<std::array<double, 3>, 3> m = {{
                {{1.0, 0.0, 0.0}},
                {{0.0, 0.0, -1.0}},
                {{0.0, 1.0, 0.0}},
            }};

            camera_data cam_data = {};
            cam_data.fx = k[0][0];
            cam_data.fy = k[1][1];
            cam_data.cx = k[0][2];
            cam_data.cy = k[1][2];
            for (size_t i = 0; i < 3; i++)
            {
                for (size_t j = 0; j < 3; j++)
                {
                    for (size_t k = 0; k < 3; k++)
                    {
                        cam_data.rotation[i][j] += rotation[i][k] * m[k][j];
                    }
                }
            }
            for (size_t i = 0; i < 3; i++)
            {
                for (size_t j = 0; j < 3; j++)
                {
                    cam_data.translation[i] += -translation[j][0] * cam_data.rotation[j][i] * 10.0;
                }
            }
            cam_data.k[0] = dist_coeffs[0];
            cam_data.k[1] = dist_coeffs[1];
            cam_data.k[2] = dist_coeffs[4];
            cam_data.p[0] = dist_coeffs[2];
            cam_data.p[1] = dist_coeffs[3];

            cameras[cv::format("%02d_%02d", panel, node)] = cam_data;
        }
        return cameras;
    }
}
namespace stargazer_mvpose
{
    static std::map<std::string, stargazer::camera_t> load_cameras()
    {
        using namespace stargazer::voxelpose;

        std::map<std::string, stargazer::camera_t> cameras;

        const auto camera_file = fs::path("/workspace/data/panoptic/calibration_171204_pose1.json");

        std::ifstream f;
        f.open(camera_file, std::ios::in | std::ios::binary);
        std::string str((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());

        nlohmann::json calib = nlohmann::json::parse(str);

        for (const auto &cam : calib["cameras"])
        {
            const auto panel = cam["panel"].get<int32_t>();
            const auto node = cam["node"].get<int32_t>();

            const auto k = cam["K"].get<std::vector<std::vector<double>>>();
            const auto dist_coeffs = cam["distCoef"].get<std::vector<double>>();
            const auto rotation = cam["R"].get<std::vector<std::vector<double>>>();
            const auto translation = cam["t"].get<std::vector<std::vector<double>>>();

            glm::mat4 camera_pose(1.0f);
            for (size_t i = 0; i < 3; i++)
            {
                for (size_t j = 0; j < 3; j++)
                {
                    camera_pose[j][i] = rotation[i][j];
                }
            }
            for (size_t i = 0; i < 3; i++)
            {
                camera_pose[3][i] = translation[i][0] / 100;
            }

            glm::mat4 cv_to_gl(1.f);
            cv_to_gl[0] = glm::vec4(1.f, 0.f, 0.f, 0.f);
            cv_to_gl[1] = glm::vec4(0.f, -1.f, 0.f, 0.f);
            cv_to_gl[2] = glm::vec4(0.f, 0.f, -1.f, 0.f);

            camera_pose = camera_pose * cv_to_gl;

            stargazer::camera_t cam_data = {};
            cam_data.intrin.fx = k[0][0];
            cam_data.intrin.fy = k[1][1];
            cam_data.intrin.cx = k[0][2];
            cam_data.intrin.cy = k[1][2];
            for (size_t i = 0; i < 3; i++)
            {
                for (size_t j = 0; j < 3; j++)
                {
                    cam_data.extrin.rotation[i][j] = camera_pose[i][j];
                }
            }
            for (size_t i = 0; i < 3; i++)
            {
                cam_data.extrin.translation[i] = camera_pose[3][i];
            }
            for (size_t i = 0; i < 5; i++)
            {
                cam_data.intrin.coeffs[i] = dist_coeffs[i];
            }

            cameras[cv::format("%02d_%02d", panel, node)] = cam_data;
        }
        return cameras;
    }
}
#endif

std::vector<glm::vec3> voxelpose_reconstruction::dnn_reconstruct(const std::map<std::string, stargazer::camera_t> &cameras, const std::map<std::string, cv::Mat> &frame, glm::mat4 axis)
{
    using namespace stargazer::voxelpose;

    std::vector<std::string> names;
    std::vector<cv::Mat> images_list;
    std::vector<camera_data> cameras_list;

    if (frame.size() <= 1)
    {
        return std::vector<glm::vec3>();
    }

#ifdef PANOPTIC
    const auto panoptic_cameras = load_cameras();
#endif

    for (const auto &[camera_name, image] : frame)
    {
        names.push_back(camera_name);
    }

    for (size_t i = 0; i < frame.size(); i++)
    {
        const auto name = names[i];

#ifdef PANOPTIC
        const auto &camera = panoptic_cameras.at(name);
#else
        camera_data camera;

        const auto &src_camera = cameras.at(name);

        camera.fx = src_camera.intrin.fx;
        camera.fy = src_camera.intrin.fy;
        camera.cx = src_camera.intrin.cx;
        camera.cy = src_camera.intrin.cy;
        camera.k[0] = src_camera.intrin.coeffs[0];
        camera.k[1] = src_camera.intrin.coeffs[3];
        camera.k[2] = src_camera.intrin.coeffs[4];
        camera.p[0] = src_camera.intrin.coeffs[1];
        camera.p[1] = src_camera.intrin.coeffs[2];

        glm::mat4 basis(1.f);
        basis[0] = glm::vec4(-1.f, 0.f, 0.f, 0.f);
        basis[1] = glm::vec4(0.f, 0.f, 1.f, 0.f);
        basis[2] = glm::vec4(0.f, 1.f, 0.f, 0.f);

        const auto axis = glm::inverse(basis) * this->axis;
        const auto camera_pose = axis * glm::inverse(src_camera.extrin.transform_matrix());

        for (size_t i = 0; i < 3; i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                camera.rotation[i][j] = camera_pose[i][j];
            }
            camera.translation[i] = camera_pose[3][i] * 1000.0;
        }
#endif

        cameras_list.push_back(camera);
        images_list.push_back(frame.at(name));
    }

#ifdef PANOPTIC
    std::array<float, 3> grid_center = {0.0f, -500.0f, 800.0f};
#else
    std::array<float, 3> grid_center = {0.0f, 0.0f, 0.0f};
#endif

    pose_estimator.set_grid_center(grid_center);

    const auto points = pose_estimator.inference(images_list, cameras_list);

    const auto start5 = std::chrono::system_clock::now();
    coalsack::tensor<float, 4> heatmaps({pose_estimator.get_heatmap_width(), pose_estimator.get_heatmap_height(), pose_estimator.get_num_joints(), (uint32_t)images_list.size()});
    pose_estimator.copy_heatmap_to(images_list.size(), heatmaps.get_data());

    {
        std::lock_guard lock(features_mtx);
        this->names = names;
        this->features = std::move(heatmaps);
    }

    return points;
}

voxelpose_reconstruction::voxelpose_reconstruction()
    : service(new SensorServiceImpl()), reconstruction_workers(std::make_shared<task_queue<std::function<void()>>>(1)), task_id_gen(std::random_device()()) {}
voxelpose_reconstruction::~voxelpose_reconstruction() = default;

void voxelpose_reconstruction::push_frame(const frame_type &frame)
{
    if (!running)
    {
        return;
    }

    if (reconstruction_workers->size() > 10)
    {
        return;
    }

    const auto task_id = task_id_gen();
    {
        std::lock_guard lock(reconstruction_task_wait_queue_mtx);
        reconstruction_task_wait_queue.push_back(task_id);
    }

    reconstruction_task_wait_queue_cv.notify_one();

    reconstruction_workers->push_task([frame, this, task_id]()
                                      {
        const auto start = std::chrono::system_clock::now();
        const auto markers = dnn_reconstruct(cameras, frame, axis);
        const auto end = std::chrono::system_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        // std::cout << "dnn_reconstruct: " << elapsed << std::endl;

        {
            std::unique_lock<std::mutex> lock(reconstruction_task_wait_queue_mtx);
            reconstruction_task_wait_queue_cv.wait(lock, [&]
                                                   { return reconstruction_task_wait_queue.front() == task_id; });

            assert(reconstruction_task_wait_queue.front() == task_id);
            reconstruction_task_wait_queue.pop_front();

            {
                std::lock_guard lock(markers_mtx);
                this->markers = markers;
            }

            service->notify_sphere(markers);
        }

        reconstruction_task_wait_queue_cv.notify_all(); });
}

void voxelpose_reconstruction::run()
{
    running = true;
    server_th.reset(new std::thread([this]()
                                    {
        std::string server_address("0.0.0.0:50052");

        grpc::ServerBuilder builder;
        builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
        builder.RegisterService(service.get());
        server = builder.BuildAndStart();
        spdlog::info("Server listening on " + server_address);
        server->Wait(); }));
}

void voxelpose_reconstruction::stop()
{
    if (running.load())
    {
        {
            std::lock_guard<std::mutex> lock(reconstruction_task_wait_queue_mtx);
            running.store(false);
        }
        server->Shutdown();
        if (server_th && server_th->joinable())
        {
            server_th->join();
        }
    }
}

std::map<std::string, cv::Mat> voxelpose_reconstruction::get_features() const
{
    coalsack::tensor<float, 4> features;
    std::vector<std::string> names;
    {
        std::lock_guard lock(features_mtx);
        features = this->features;
        names = this->names;
    }
    std::map<std::string, cv::Mat> result;
    if (features.get_size() == 0)
    {
        return result;
    }
    for (size_t i = 0; i < names.size(); i++)
    {
        const auto name = names[i];
        const auto heatmap = features.view<3>({features.shape[0], features.shape[1], features.shape[2], 0}, {0, 0, 0, static_cast<uint32_t>(i)}).contiguous().sum<1>({2});

        cv::Mat heatmap_mat;
        cv::Mat(heatmap.shape[1], heatmap.shape[0], CV_32FC1, (float *)heatmap.get_data()).clone().convertTo(heatmap_mat, CV_8U, 255);
        cv::resize(heatmap_mat, heatmap_mat, cv::Size(960, 540));
        cv::cvtColor(heatmap_mat, heatmap_mat, cv::COLOR_GRAY2BGR);

        result[name] = heatmap_mat;
    }
    return result;
}

std::vector<glm::vec3> voxelpose_reconstruction::get_markers() const
{
    std::vector<glm::vec3> result;
    {
        std::lock_guard lock(markers_mtx);
        result = markers;
    }
    return result;
}

output_server::output_server(const std::string &server_address)
    : server_address(server_address), running(false), server_th(), server(), service(std::make_unique<SensorServiceImpl>())
{
}

output_server::~output_server()
{
}

void output_server::run()
{
    running = true;
    grpc::ServerBuilder builder;

#ifdef USE_SECURE_CREDENTIALS
    std::string ca_crt_content = read_text_file("../data/ca.crt");
    std::string server_crt_content = read_text_file("../data/server.crt");
    std::string server_key_content = read_text_file("../data/server.key");

    grpc::SslServerCredentialsOptions ssl_options;
    grpc::SslServerCredentialsOptions::PemKeyCertPair key_cert = { server_key_content, server_crt_content };
    ssl_options.pem_root_certs = ca_crt_content;
    ssl_options.pem_key_cert_pairs.push_back(key_cert);
    builder.AddListeningPort(server_address, grpc::SslServerCredentials(ssl_options));
#else
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
#endif
    builder.RegisterService(service.get());
    server = builder.BuildAndStart();
    spdlog::info("Server listening on " + server_address);
    server_th.reset(new std::thread([this]()
                                    {
        server->Wait(); }));
}

void output_server::stop()
{
    if (running.load())
    {
        running.store(false);
    }
    if (server)
    {
        server->Shutdown();
        if (server_th && server_th->joinable())
        {
            server_th->join();
        }
    }
}

void output_server::notify_sphere(const std::vector<glm::vec3> &spheres)
{
    if (running && service)
    {
        service->notify_sphere(spheres);
    }
}

mvpose_reconstruction::mvpose_reconstruction()
    : output("0.0.0.0:50053"), processor(1)
{
}
mvpose_reconstruction::~mvpose_reconstruction()
{
}

std::tuple<std::vector<std::string>, coalsack::tensor<float, 4>, std::vector<glm::vec3>> mvpose_reconstruction::mvpose_reconstruct(const std::map<std::string, stargazer::camera_t> &cameras, const std::map<std::string, cv::Mat> &frame, glm::mat4 axis)
{
    using namespace stargazer_mvpose;

    std::vector<std::string> names;
    coalsack::tensor<float, 4> heatmaps;
    std::vector<cv::Mat> images_list;
    std::vector<stargazer::camera_t> cameras_list;

    if (frame.size() <= 1)
    {
        std::vector<glm::vec3> points;
        return std::forward_as_tuple(names, heatmaps, points);
    }

#ifdef PANOPTIC
    const auto panoptic_cameras = load_cameras();
#endif

    for (const auto &[camera_name, image] : frame)
    {
        names.push_back(camera_name);
    }

    for (size_t i = 0; i < frame.size(); i++)
    {
        const auto name = names[i];

#ifdef PANOPTIC
        const auto &camera = panoptic_cameras.at(name);
#else
        stargazer::camera_t camera;

        const auto &src_camera = cameras.at(name);

        camera.intrin.fx = src_camera.intrin.fx;
        camera.intrin.fy = src_camera.intrin.fy;
        camera.intrin.cx = src_camera.intrin.cx;
        camera.intrin.cy = src_camera.intrin.cy;
        camera.intrin.coeffs[0] = src_camera.intrin.coeffs[0];
        camera.intrin.coeffs[1] = src_camera.intrin.coeffs[1];
        camera.intrin.coeffs[2] = src_camera.intrin.coeffs[2];
        camera.intrin.coeffs[3] = src_camera.intrin.coeffs[3];
        camera.intrin.coeffs[4] = src_camera.intrin.coeffs[4];

        const auto camera_pose = glm::inverse(axis * glm::inverse(src_camera.extrin.transform_matrix()));

        for (size_t i = 0; i < 3; i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                camera.extrin.rotation[i][j] = camera_pose[i][j];
            }
            camera.extrin.translation[i] = camera_pose[3][i];
        }
#endif

        cameras_list.push_back(camera);
        images_list.push_back(frame.at(name));
    }

    const auto points = pose_estimator.inference(images_list, cameras_list);

    return std::forward_as_tuple(names, heatmaps, points);
}

void mvpose_reconstruction::push_frame(const frame_type &frame)
{
    processor.push([this, frame = frame]() {
        auto [names, features, markers] = mvpose_reconstruct(cameras, frame, axis);

        task_result result;
        result.markers = markers;
        result.camera_names = names;
        result.features = std::move(features);
        return result;
    });
}
void mvpose_reconstruction::run()
{
    output.run();
    processor.run([this](const task_result& result) {
        {
            std::lock_guard lock(markers_mtx);
            this->markers = result.markers;
        }
        {
            std::lock_guard lock(features_mtx);
            this->names = result.camera_names;
            this->features = result.features;
        }
        output.notify_sphere(result.markers);
    });
}
void mvpose_reconstruction::stop()
{
    processor.stop();
    output.stop();
}

std::map<std::string, cv::Mat> mvpose_reconstruction::get_features() const
{
    coalsack::tensor<float, 4> features;
    std::vector<std::string> names;
    {
        std::lock_guard lock(features_mtx);
        features = this->features;
        names = this->names;
    }
    std::map<std::string, cv::Mat> result;
    if (features.get_size() == 0)
    {
        return result;
    }
    for (size_t i = 0; i < names.size(); i++)
    {
        const auto name = names[i];
        const auto heatmap = features.view<3>({features.shape[0], features.shape[1], features.shape[2], 0}, {0, 0, 0, static_cast<uint32_t>(i)}).contiguous().sum<1>({2});

        cv::Mat heatmap_mat;
        cv::Mat(heatmap.shape[1], heatmap.shape[0], CV_32FC1, (float *)heatmap.get_data()).clone().convertTo(heatmap_mat, CV_8U, 255);
        cv::resize(heatmap_mat, heatmap_mat, cv::Size(960, 540));
        cv::cvtColor(heatmap_mat, heatmap_mat, cv::COLOR_GRAY2BGR);

        result[name] = heatmap_mat;
    }
    return result;
}

std::vector<glm::vec3> mvpose_reconstruction::get_markers() const
{
    std::vector<glm::vec3> result;
    {
        std::lock_guard lock(markers_mtx);
        result = markers;
    }
    return result;
}