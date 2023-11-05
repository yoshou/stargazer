#include "reconstruction.hpp"
#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>

#include "sensor.grpc.pb.h"

#include <spdlog/spdlog.h>

#include "camera_info.hpp"
#include "utils.hpp"
#include "correspondance.hpp"
#include "triangulation.hpp"
#include "multiview_point_data.hpp"

#include "voxelpose.hpp"

class SensorServiceImpl final : public stargazer::Sensor::Service
{
    std::mutex mtx;
    std::unordered_map<std::string, grpc::ServerWriter<stargazer::SphereResponse> *> writers;

public:
    void notify_sphere(const std::vector<glm::vec3> &spheres)
    {
        stargazer::SphereResponse response;
        const auto mutable_spheres = response.mutable_spheres();
        for (const auto &sphere : spheres)
        {
            const auto mutable_sphere = mutable_spheres->mutable_values()->Add();
            mutable_sphere->mutable_point()->set_x(sphere.x);
            mutable_sphere->mutable_point()->set_y(sphere.y);
            mutable_sphere->mutable_point()->set_z(sphere.z);
            mutable_sphere->set_radius(0.02);
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
                                 grpc::ServerWriter<stargazer::SphereResponse> *writer) override
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

std::vector<glm::vec3> reconstruct(const std::map<std::string, stargazer::camera_t> &cameras, const std::map<std::string, std::vector<stargazer::point_data>> &frame, glm::mat4 axis)
{
    std::vector<std::vector<glm::vec2>> camera_pts;
    std::vector<std::string> camera_names;
    std::vector<stargazer::camera_t> camera_list;
    for (const auto &[camera_name, camera] : cameras)
    {
        std::vector<glm::vec2> pts;
        for (const auto &pt : frame.at(camera_name))
        {
            pts.push_back(pt.point);
        }
        camera_pts.push_back(pts);
        camera_names.push_back(camera_name);
        camera_list.push_back(camera);
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

marker_stream_server::marker_stream_server()
    : service(new SensorServiceImpl()), reconstruction_workers(std::make_shared<task_queue<std::function<void()>>>(4)), task_id_gen(std::random_device()()) {}
marker_stream_server::~marker_stream_server() = default;

void marker_stream_server::push_frame(const frame_type &frame)
{
    if (!running)
    {
        return;
    }

    const auto cloned_frame = frame;
    const auto task_id = task_id_gen();
    {
        std::lock_guard lock(reconstruction_task_wait_queue_mtx);
        reconstruction_task_wait_queue.push_back(task_id);
    }

    reconstruction_task_wait_queue_cv.notify_one();

    reconstruction_workers->push_task([cloned_frame, this, task_id]()
                                      {
        const auto markers = reconstruct(cameras, cloned_frame, axis);

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

void marker_stream_server::run()
{
    running = true;
    server_th.reset(new std::thread([this]()
                                    {
        std::string server_address("0.0.0.0:50051");

        grpc::ServerBuilder builder;
        builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
        builder.RegisterService(service.get());
        server = builder.BuildAndStart();
        spdlog::info("Server listening on " + server_address);
        server->Wait(); }));
}

void marker_stream_server::stop()
{
    if (running.load())
    {
        running.store(false);
        server->Shutdown();
        if (server_th && server_th->joinable())
        {
            server_th->join();
        }
    }
}

#include <cereal/types/array.hpp>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

// #define PANOPTIC

#ifdef PANOPTIC
static std::map<std::tuple<int32_t, int32_t>, camera_data> load_cameras()
{
    std::map<std::tuple<int32_t, int32_t>, camera_data> cameras;

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

        cameras[std::make_pair(panel, node)] = cam_data;
    }
    return cameras;
}
#endif

std::vector<glm::vec3> dnn_reconstruction::dnn_reconstruct(const std::map<std::string, stargazer::camera_t> &cameras, const std::map<std::string, cv::Mat> &frame, glm::mat4 axis)
{
    std::vector<std::string> names;
    std::vector<cv::Mat> images_list;
    std::vector<camera_data> cameras_list;

    if (frame.size() <= 1)
    {
        return std::vector<glm::vec3>();
    }

#ifdef PANOPTIC
    const auto panoptic_cameras = load_cameras();

    std::vector<std::tuple<int32_t, int32_t>> camera_list = {
        {0, 3},
        {0, 6},
        {0, 12},
        {0, 13},
        {0, 23},
    };
    for (const auto &[camera_panel, camera_node] : camera_list)
    {
        const auto camera_name = cv::format("camera_%02d_%02d", camera_panel, camera_node);

        const auto prefix = cv::format("%02d_%02d", camera_panel, camera_node);
        std::string postfix = "_00000000";
        const auto image_file = (fs::path("/workspace/data/panoptic") / prefix / (prefix + postfix + ".jpg")).string();

        auto data = cv::imread(image_file, cv::IMREAD_UNCHANGED | cv::IMREAD_IGNORE_ORIENTATION);
        // cv::resize(data, data, cv::Size(960, 540));
        images_list.push_back(data);
        
        cameras_list.push_back(panoptic_cameras.at(std::make_tuple(camera_panel, camera_node)));
        names.push_back(camera_name);
    }
#else
    for (const auto &[camera_name, image] : frame)
    {
        names.push_back(camera_name);
    }

    for (size_t i = 0; i < frame.size(); i++)
    {
        const auto name = names[i];

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
        const auto camera_pose = axis * glm::inverse(src_camera.extrin.rotation);

        for (size_t i = 0; i < 3; i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                camera.rotation[i][j] = camera_pose[i][j];
            }
            camera.translation[i] = camera_pose[3][i] * 1000.0;
        }

        cameras_list.push_back(camera);
        images_list.push_back(frame.at(name));
    }
#endif

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

#ifdef PANOPTIC
        std::map<std::string, std::string> name_cvt = {
            {"camera_00_03", "camera101"},
            {"camera_00_06", "camera102"},
            {"camera_00_12", "camera103"},
            {"camera_00_13", "camera104"},
            {"camera_00_23", "camera105"},
        };

        std::vector<std::string> new_names;
        for (const auto& name : names)
        {
            new_names.push_back(name_cvt.at(name));
        }

        this->names = new_names;
#else
        this->names = names;
#endif
        this->features = std::move(heatmaps);
    }

    return points;
}

dnn_reconstruction::dnn_reconstruction()
    : service(new SensorServiceImpl()), reconstruction_workers(std::make_shared<task_queue<std::function<void()>>>(1)), task_id_gen(std::random_device()()) {}
dnn_reconstruction::~dnn_reconstruction() = default;

void dnn_reconstruction::push_frame(const frame_type &frame)
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

void dnn_reconstruction::run()
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

void dnn_reconstruction::stop()
{
    if (running.load())
    {
        running.store(false);
        server->Shutdown();
        if (server_th && server_th->joinable())
        {
            server_th->join();
        }
    }
}

std::map<std::string, cv::Mat> dnn_reconstruction::get_features() const
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