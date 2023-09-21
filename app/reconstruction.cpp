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

#include "graph_proc.h"
#include "graph_proc_img.h"
#include "graph_proc_cv.h"
#include "graph_proc_tensor.h"

#include <fmt/core.h>
#include <cereal/types/array.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <onnxruntime_cxx_api.h>
#include <tensorrt_provider_factory.h>

using namespace coalsack;

namespace fs = std::filesystem;

struct camera_data
{
    double fx;
    double fy;
    double cx;
    double cy;
    std::array<double, 3> k;
    std::array<double, 2> p;
    std::array<std::array<double, 3>, 3> rotation;
    std::array<double, 3> translation;
};

struct roi_data
{
    std::array<double, 2> scale;
    double rotation;
    std::array<double, 2> center;
};

static cv::Mat get_transform(const cv::Point2f &center, const cv::Size2f &scale, const cv::Size2f &output_size)
{
    const auto get_tri_3rd_point = [](const cv::Point2f &a, const cv::Point2f &b)
    {
        const auto direct = a - b;
        return b + cv::Point2f(-direct.y, direct.x);
    };

    const auto get_affine_transform = [&](const cv::Point2f &center, const cv::Size2f &scale, const cv::Size2f &output_size)
    {
        const auto src_w = scale.width * 200.0;
        const auto src_h = scale.height * 200.0;
        const auto dst_w = output_size.width;
        const auto dst_h = output_size.height;

        cv::Point2f src_dir, dst_dir;
        if (src_w >= src_h)
        {
            src_dir = cv::Point2f(0, src_w * -0.5);
            dst_dir = cv::Point2f(0, dst_w * -0.5);
        }
        else
        {
            src_dir = cv::Point2f(src_h * -0.5, 0);
            dst_dir = cv::Point2f(dst_h * -0.5, 0);
        }

        const auto src_tri_a = center;
        const auto src_tri_b = center + src_dir;
        const auto src_tri_c = get_tri_3rd_point(src_tri_a, src_tri_b);
        cv::Point2f src_tri[3] = {src_tri_a, src_tri_b, src_tri_c};

        const auto dst_tri_a = cv::Point2f(dst_w * 0.5, dst_h * 0.5);
        const auto dst_tri_b = dst_tri_a + dst_dir;
        const auto dst_tri_c = get_tri_3rd_point(dst_tri_a, dst_tri_b);

        cv::Point2f dst_tri[3] = {dst_tri_a, dst_tri_b, dst_tri_c};

        return cv::getAffineTransform(src_tri, dst_tri);
    };

    return get_affine_transform(center, scale, output_size);
}

class dnn_reconstruction::dnn_inference
{
    std::vector<uint8_t> model_data;

    Ort::Session session;
    std::vector<std::string> input_node_names;

    std::unordered_map<std::string, std::vector<int64_t>> input_node_dims;

public:
    dnn_inference()
        : session(nullptr)
    {
    }

    std::vector<std::string> output_names;

    void set_model_data(const std::vector<uint8_t> &value)
    {
        model_data = value;
    }

    Ort::Env env{ORT_LOGGING_LEVEL_WARNING};

    virtual void initialize()
    {
        const auto &api = Ort::GetApi();

        // Create session
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#if 0
        OrtCUDAProviderOptions cuda_options{};

        cuda_options.device_id = 0;
        cuda_options.arena_extend_strategy = 0;
        cuda_options.gpu_mem_limit = 2ULL * 1024 * 1024 * 1024;
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        cuda_options.do_copy_in_default_stream = 1;

        session_options.AppendExecutionProvider_CUDA(cuda_options);
#else
        std::string cache_dir = "./cache";
        fs::create_directory(fs::path(cache_dir));

        OrtTensorRTProviderOptions trt_options{};

        trt_options.device_id = 0;
        trt_options.trt_max_workspace_size = 2147483648;
        trt_options.trt_max_partition_iterations = 1000;
        trt_options.trt_min_subgraph_size = 1;
        trt_options.trt_fp16_enable = 1;
        trt_options.trt_int8_enable = 0;
        trt_options.trt_int8_use_native_calibration_table = 0;
        trt_options.trt_engine_cache_enable = 1;
        trt_options.trt_engine_cache_path = cache_dir.c_str();
        trt_options.trt_dump_subgraphs = 1;

        session_options.AppendExecutionProvider_TensorRT(trt_options);
#endif

        session = Ort::Session(env, model_data.data(), model_data.size(), session_options);

        // Iterate over all input nodes
        const size_t num_input_nodes = session.GetInputCount();
        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < num_input_nodes; i++)
        {
            const auto input_name = session.GetInputNameAllocated(i, allocator);
            input_node_names.push_back(input_name.get());

            const auto type_info = session.GetInputTypeInfo(i);
            const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

            const auto type = tensor_info.GetElementType();

            const auto input_shape = tensor_info.GetShape();
            input_node_dims[input_name.get()] = input_shape;
        }
    }

    virtual graph_message_ptr process(std::string input_name, graph_message_ptr message)
    {
        if (auto frame_msg = std::dynamic_pointer_cast<frame_message<tensor<float, 4>>>(message))
        {
            const auto &src = frame_msg->get_data();

            const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            std::vector<const char *> input_node_names;
            std::vector<Ort::Value> input_tensors;
            for (const auto &name : this->input_node_names)
            {
                input_node_names.push_back(name.c_str());

                // const auto dims = input_node_dims.at(name);
                const auto num_dims = input_node_dims.at(name).size();
                std::vector<int64_t> dims;
                std::reverse_copy(src.shape.begin(), src.shape.end(), std::back_inserter(dims));
                while (dims.size() < num_dims)
                {
                    dims.insert(dims.begin(), 1);
                }
                input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, const_cast<float *>(src.get_data()), src.get_size(),
                                                                        dims.data(), dims.size()));
            }

            std::vector<const char *> output_node_names;
            for (const auto &name : output_names)
            {
                output_node_names.push_back(name.c_str());
            }

            const auto output_tensors =
                session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), input_tensors.data(), input_tensors.size(), output_node_names.data(), output_node_names.size());

            assert(output_tensors.size() == output_node_names.size());
            for (std::size_t i = 0; i < output_node_names.size(); i++)
            {
                const auto name = output_node_names.at(i);
                const auto &value = output_tensors.at(i);

                graph_message_ptr output_msg;

                if (value.IsTensor())
                {
                    const auto data = value.GetTensorData<float>();
                    const auto tensor_info = value.GetTensorTypeAndShapeInfo();
                    const auto type = tensor_info.GetElementType();
                    const auto shape = tensor_info.GetShape();

                    if (shape.size() == 4)
                    {
                        constexpr auto num_dims = 4;

                        auto msg = std::make_shared<frame_message<tensor<float, num_dims>>>();
                        tensor<float, num_dims> output_tensor({static_cast<std::uint32_t>(shape.at(3)),
                                                                static_cast<std::uint32_t>(shape.at(2)),
                                                                static_cast<std::uint32_t>(shape.at(1)),
                                                                static_cast<std::uint32_t>(shape.at(0))},
                                                                data);

                        msg->set_data(std::move(output_tensor));
                        msg->set_profile(frame_msg->get_profile());
                        msg->set_timestamp(frame_msg->get_timestamp());
                        msg->set_frame_number(frame_msg->get_frame_number());
                        msg->set_metadata(*frame_msg);

                        output_msg = msg;
                    }
                    else if (shape.size() == 5)
                    {
                        constexpr auto num_dims = 5;

                        auto msg = std::make_shared<frame_message<tensor<float, num_dims>>>();
                        tensor<float, num_dims> output_tensor({static_cast<std::uint32_t>(shape.at(4)),
                                                                static_cast<std::uint32_t>(shape.at(3)),
                                                                static_cast<std::uint32_t>(shape.at(2)),
                                                                static_cast<std::uint32_t>(shape.at(1)),
                                                                static_cast<std::uint32_t>(shape.at(0))},
                                                                data);

                        msg->set_data(std::move(output_tensor));
                        msg->set_profile(frame_msg->get_profile());
                        msg->set_timestamp(frame_msg->get_timestamp());
                        msg->set_frame_number(frame_msg->get_frame_number());
                        msg->set_metadata(*frame_msg);

                        output_msg = msg;
                    }
                }

                return output_msg;
            }
        }
        return nullptr;
    }
};

class dnn_reconstruction::projector
{
    std::array<float, 3> grid_size;
    std::array<int32_t, 3> cube_size;

public:
    projector()
    {
    }

    std::array<float, 3> get_grid_size() const
    {
        return grid_size;
    }
    void set_grid_size(const std::array<float, 3> &value)
    {
        grid_size = value;
    }
    std::array<int32_t, 3> get_cube_size() const
    {
        return cube_size;
    }
    void set_cube_size(const std::array<int32_t, 3> &value)
    {
        cube_size = value;
    }

    std::vector<std::array<float, 3>> compute_grid(const std::array<float, 3> &grid_center) const
    {
        std::vector<std::array<float, 3>> grid;
        for (int32_t x = 0; x < cube_size.at(0); x++)
        {
            for (int32_t y = 0; y < cube_size.at(1); y++)
            {
                for (int32_t z = 0; z < cube_size.at(2); z++)
                {
                    const auto gridx = -grid_size.at(0) / 2 + grid_size.at(0) * x / (cube_size.at(0) - 1) + grid_center.at(0);
                    const auto gridy = -grid_size.at(1) / 2 + grid_size.at(1) * y / (cube_size.at(1) - 1) + grid_center.at(1);
                    const auto gridz = -grid_size.at(2) / 2 + grid_size.at(2) * z / (cube_size.at(2) - 1) + grid_center.at(2);

                    grid.push_back({gridx, gridy, gridz});
                }
            }
        }
        return grid;
    }

    static std::vector<std::array<float, 2>> project_point(const std::vector<std::array<float, 3>> &x, const camera_data &camera)
    {
        std::vector<cv::Point3d> points;

        std::transform(x.begin(), x.end(), std::back_inserter(points), [&](const auto &p)
                       {
            const auto pt_x = p[0] - camera.translation[0];
            const auto pt_y = p[1] - camera.translation[1];
            const auto pt_z = p[2] - camera.translation[2];
            const auto cam_x = pt_x * camera.rotation[0][0] + pt_y * camera.rotation[0][1] + pt_z * camera.rotation[0][2];
            const auto cam_y = pt_x * camera.rotation[1][0] + pt_y * camera.rotation[1][1] + pt_z * camera.rotation[1][2];
            const auto cam_z = pt_x * camera.rotation[2][0] + pt_y * camera.rotation[2][1] + pt_z * camera.rotation[2][2];

            return cv::Point3d(cam_x / (cam_z + 1e-5), cam_y / (cam_z + 1e-5), 1.0); });

        cv::Mat camera_matrix = cv::Mat::eye(3, 3, cv::DataType<double>::type);
        camera_matrix.at<double>(0, 0) = camera.fx;
        camera_matrix.at<double>(1, 1) = camera.fy;
        camera_matrix.at<double>(0, 2) = camera.cx;
        camera_matrix.at<double>(1, 2) = camera.cy;

        cv::Mat dist_coeffs(5, 1, cv::DataType<double>::type);
        dist_coeffs.at<double>(0) = camera.k[0];
        dist_coeffs.at<double>(1) = camera.k[1];
        dist_coeffs.at<double>(2) = camera.p[0];
        dist_coeffs.at<double>(3) = camera.p[1];
        dist_coeffs.at<double>(4) = camera.k[2];

        cv::Mat rvec = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
        cv::Mat tvec = cv::Mat::zeros(3, 1, cv::DataType<double>::type);

        std::vector<cv::Point2d> projected_points;
        cv::projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs, projected_points);

        std::vector<std::array<float, 2>> y;

        std::transform(projected_points.begin(), projected_points.end(), std::back_inserter(y), [](const auto &p)
                       { return std::array<float, 2>{static_cast<float>(p.x), static_cast<float>(p.y)}; });

        return y;
    }

    static tensor<float, 4> grid_sample(const tensor<float, 4> &src, const std::vector<std::array<float, 2>> &grid, bool align_corner = false)
    {
        const auto num_o = src.shape[3];
        const auto num_c = src.shape[2];
        const auto num_h = src.shape[1];
        const auto num_w = src.shape[0];

        tensor<float, 4> dst({static_cast<uint32_t>(grid.size()), 1, num_c, num_o});

        constexpr size_t num_size = SHRT_MAX - 1;

        for (size_t offset = 0; offset < grid.size(); offset += num_size)
        {
            const auto grid_num = std::min(num_size, grid.size() - offset);

            cv::Mat map_x(grid_num, 1, cv::DataType<float>::type);
            cv::Mat map_y(grid_num, 1, cv::DataType<float>::type);

            if (align_corner)
            {
                for (size_t i = 0; i < grid_num; i++)
                {
                    const auto x = ((grid[i + offset][0] + 1) / 2) * (num_w - 1);
                    const auto y = ((grid[i + offset][1] + 1) / 2) * (num_h - 1);
                    map_x.at<float>(i, 0) = x;
                    map_y.at<float>(i, 0) = y;
                }
            }
            else
            {
                for (size_t i = 0; i < grid_num; i++)
                {
                    const auto x = ((grid[i + offset][0] + 1) * num_w - 1) / 2;
                    const auto y = ((grid[i + offset][1] + 1) * num_h - 1) / 2;
                    map_x.at<float>(i, 0) = x;
                    map_y.at<float>(i, 0) = y;
                }
            }

            for (uint32_t o = 0; o < num_o; o++)
            {
                for (uint32_t c = 0; c < num_c; c++)
                {
                    cv::Mat plane(num_h, num_w, cv::DataType<float>::type, const_cast<float *>(src.get_data()) + c * src.stride[2] + o * src.stride[3]);
                    cv::Mat remapped(grid_num, 1, cv::DataType<float>::type, dst.get_data() + offset + c * dst.stride[2] + o * dst.stride[3]);
                    cv::remap(plane, remapped, map_x, map_y, cv::INTER_LINEAR);
                }
            }
        }

        return dst;
    }

    std::tuple<tensor<float, 4>, std::vector<std::array<float, 3>>> get_voxel(const std::vector<tensor<float, 4>> &heatmaps, const std::vector<camera_data> &cameras, const std::vector<roi_data> &rois, const std::array<float, 3> &grid_center) const
    {
        const auto num_bins = static_cast<uint32_t>(std::accumulate(cube_size.begin(), cube_size.end(), 1, std::multiplies<int32_t>()));
        const auto num_joints = static_cast<uint32_t>(heatmaps.at(0).shape[2]);
        const auto num_cameras = static_cast<uint32_t>(heatmaps.size());
        const auto w = heatmaps.at(0).shape[0];
        const auto h = heatmaps.at(0).shape[1];
        const auto grid = compute_grid(grid_center);

        auto cubes = tensor<float, 4>::zeros({num_cameras, num_bins, 1, num_joints});
        auto bounding = tensor<float, 4>::zeros({num_cameras, num_bins, 1, 1});

        for (uint32_t c = 0; c < num_cameras; c++)
        {
            const auto &roi = rois.at(c);
            const auto &&image_size = cv::Size2f(960, 512);
            const auto center = cv::Point2f(roi.center[0], roi.center[1]);
            const auto scale = cv::Size2f(roi.scale[0], roi.scale[1]);
            const auto width = center.x * 2;
            const auto height = center.y * 2;

            const auto trans = get_transform(center, scale, image_size);
            cv::Mat transf;
            trans.convertTo(transf, cv::DataType<float>::type);

            const auto xy = project_point(grid, cameras[c]);

            auto camera_bounding = bounding.view({0, cubes.shape[1], 0, 0}, {c, 0, 0, 0});

            camera_bounding.assign([&xy, width, height](const float value, const size_t w, auto...)
                                   { return (xy[w][0] >= 0 && xy[w][0] < width && xy[w][1] >= 0 && xy[w][1] < height); });

            std::vector<std::array<float, 2>> sample_grid;
            std::transform(xy.begin(), xy.end(), std::back_inserter(sample_grid), [&](const auto &p)
                           {
                const auto x0 = p[0];
                const auto y0 = p[1];

                const auto x1 = std::clamp(x0, -1.0f, std::max(width, height));
                const auto y1 = std::clamp(y0, -1.0f, std::max(width, height));

                const auto x2 = x1 * transf.at<float>(0, 0) + y1 * transf.at<float>(0, 1) + transf.at<float>(0, 2);
                const auto y2 = x1 * transf.at<float>(1, 0) + y1 * transf.at<float>(1, 1) + transf.at<float>(1, 2);

                const auto x3 = x2 * w / image_size.width;
                const auto y3 = y2 * h / image_size.height;

                const auto x4 = x3 / (w - 1) * 2.0f - 1.0f;
                const auto y4 = y3 / (h - 1) * 2.0f - 1.0f;

                const auto x5 = std::clamp(x4, -1.1f, 1.1f);
                const auto y5 = std::clamp(y4, -1.1f, 1.1f);

                return std::array<float, 2>{x5, y5}; });

            const auto cube = grid_sample(heatmaps[c], sample_grid, true);

            auto camera_cubes = cubes.view({0, cubes.shape[1], cubes.shape[2], cubes.shape[3]}, {c, 0, 0, 0});

            camera_cubes.assign(cube.view(), [](const float value1, const float value2, auto...)
                                { return value1 + value2; });
        }

        const auto bounding_count = bounding.sum<1>({0});
        const auto merged_cubes = cubes
                                      .transform(bounding,
                                                 [](const float value1, const float value2, auto...)
                                                 {
                                                     return value1 * value2;
                                                 })
                                      .sum<1>({0})
                                      .transform(bounding_count,
                                                 [](const float value1, const float value2, auto...)
                                                 {
                                                     return std::clamp(value1 / (value2 + 1e-6f), 0.f, 1.f);
                                                 });

        const auto output_cubes = merged_cubes.view<4>({static_cast<uint32_t>(cube_size[2]), static_cast<uint32_t>(cube_size[1]), static_cast<uint32_t>(cube_size[0]), num_joints}).contiguous();
        return std::forward_as_tuple(output_cubes, grid);
    }
};

class dnn_reconstruction::get_proposal
{
    uint32_t max_num;
    float threshold;
    std::array<float, 3> grid_size;
    std::array<float, 3> grid_center;
    std::array<int32_t, 3> cube_size;

public:
    get_proposal()
    {
    }

    void set_max_num(uint32_t value)
    {
        max_num = value;
    }
    void set_threshold(float value)
    {
        threshold = value;
    }
    std::array<float, 3> get_grid_size() const
    {
        return grid_size;
    }
    void set_grid_size(const std::array<float, 3> &value)
    {
        grid_size = value;
    }
    std::array<float, 3> get_grid_center() const
    {
        return grid_center;
    }
    void set_grid_center(const std::array<float, 3> &value)
    {
        grid_center = value;
    }
    std::array<int32_t, 3> get_cube_size() const
    {
        return cube_size;
    }
    void set_cube_size(const std::array<int32_t, 3> &value)
    {
        cube_size = value;
    }

    static tensor<float, 4> max_pool(const tensor<float, 4> &inputs, size_t kernel = 3)
    {
        const auto padding = (kernel - 1) / 2;
        const auto max = inputs.max_pool3d(kernel, 1, padding, 1);
        const auto keep = inputs
                              .transform(max,
                                         [](const float value1, const float value2, auto...)
                                         {
                                             return value1 == value2 ? value1 : 0.f;
                                         });
        return keep;
    }

    static tensor<uint64_t, 2> get_index(const tensor<uint64_t, 1> &indices, const std::array<uint64_t, 3> &shape)
    {
        const auto num_people = indices.shape[3];
        const auto result = indices
                                .transform_expand<1>({3},
                                                     [shape](const uint64_t value, auto...)
                                                     {
                                                         const auto index_x = value / (shape[1] * shape[0]);
                                                         const auto index_y = value % (shape[1] * shape[0]) / shape[0];
                                                         const auto index_z = value % shape[0];
                                                         return std::array<uint64_t, 3>{index_x, index_y, index_z};
                                                     });
        return result;
    }

    tensor<float, 2> get_real_loc(const tensor<uint64_t, 2> &index)
    {
        const auto loc = index.cast<float>()
                             .transform(
                                 [this](const float value, const size_t i, const size_t j)
                                 {
                                     return value / (cube_size[i] - 1) * grid_size[i] + grid_center[i] - grid_size[i] / 2.0f;
                                 });
        return loc;
    }

    tensor<float, 2> get_centers(const tensor<float, 5> &src)
    {
        const auto root_cubes = src.view<4>({src.shape[0], src.shape[1], src.shape[2], src.shape[3]}).contiguous();
        const auto root_cubes_nms = max_pool(root_cubes);
        const auto [topk_values, topk_index] = root_cubes_nms.view<1>({src.shape[0] * src.shape[1] * src.shape[2]})
                                                    .topk(max_num);

        const auto topk_unravel_index = get_index(topk_index, {src.shape[0], src.shape[1], src.shape[2]});
        const auto topk_loc = get_real_loc(topk_unravel_index);

        auto grid_centers = tensor<float, 2>::zeros({5, max_num});
        grid_centers.view({3, grid_centers.shape[1]}, {0, 0})
            .assign(topk_loc.view(), [](auto, const float value, auto...)
                    { return value; });
        grid_centers.view<1>({0, grid_centers.shape[1]}, {4, 0})
            .assign(topk_values.view(), [](auto, const float value, auto...)
                    { return value; });
        grid_centers.view<1>({0, grid_centers.shape[1]}, {3, 0})
            .assign(topk_values.view(), [this](auto, const float value, auto...)
                    { return (value > threshold ? 1.f : 0.f) - 1.f; });

        return grid_centers;
    }
};

std::vector<glm::vec3> dnn_reconstruction::dnn_reconstruct(const std::map<std::string, stargazer::camera_t> &cameras, const std::map<std::string, cv::Mat> &frame, glm::mat4 axis)
{
    std::vector<tensor<float, 4>> input_img_tensors;

    std::vector<std::string> names;
    std::vector<roi_data> rois_list;
    for (const auto &[camera_name, image] : frame)
    {
        names.push_back(camera_name);
        auto data = image;

        const auto get_scale = [](const cv::Size2f &image_size, const cv::Size2f &resized_size)
        {
            float w_pad, h_pad;
            if (image_size.width / resized_size.width < image_size.height / resized_size.height)
            {
                w_pad = image_size.height / resized_size.height * resized_size.width;
                h_pad = image_size.height;
            }
            else
            {
                w_pad = image_size.width;
                h_pad = image_size.width / resized_size.width * resized_size.height;
            }

            return cv::Size2f(w_pad / 200.0, h_pad / 200.0);
        };

        const auto &&image_size = cv::Size2f(960, 512);
        const auto scale = get_scale(data.size(), image_size);
        const auto center = cv::Point2f(data.size().width / 2.0, data.size().height / 2.0);
        const auto rotation = 0.0;

        const auto trans = get_transform(center, scale, image_size);

        cv::Mat input_img = data;
        cv::warpAffine(input_img, input_img, trans, cv::Size(image_size), cv::INTER_LINEAR);
        cv::cvtColor(input_img, input_img, cv::COLOR_BGR2RGB);

        tensor<uint8_t, 4> input_img_tensor({static_cast<std::uint32_t>(input_img.size().width),
                                             static_cast<std::uint32_t>(input_img.size().height),
                                             static_cast<std::uint32_t>(input_img.elemSize()),
                                             1},
                                            (const uint8_t *)input_img.data,
                                            {static_cast<std::uint32_t>(input_img.step[1]),
                                             static_cast<std::uint32_t>(input_img.step[0]),
                                             static_cast<std::uint32_t>(1),
                                             static_cast<std::uint32_t>(input_img.total())});

        const std::array<float, 3> mean = {0.485, 0.456, 0.406};
        const std::array<float, 3> std = {0.229, 0.224, 0.225};

        const auto input_img_tensor_f = input_img_tensor.cast<float>().transform([this, mean, std](const float value, const size_t w, const size_t h, const size_t c, const size_t n)
                                                                                 { return (value / 255.0f - mean[c]) / std[c]; });

        input_img_tensors.emplace_back(std::move(input_img_tensor_f));

        roi_data roi = {{scale.width, scale.height}, rotation, {center.x, center.y}};
        rois_list.push_back(roi);
    }

    auto frame_msg = std::make_shared<frame_message<tensor<float, 4>>>();
    const auto&& input_img_tensor = tensor_f32_4::concat<3>(input_img_tensors);

    frame_msg->set_data(std::move(input_img_tensor));
    frame_msg->set_timestamp(0);
    frame_msg->set_frame_number(0);

    const auto heatmaps_msg = inference_heatmap->process("input", frame_msg);

    const auto &heatmaps = std::dynamic_pointer_cast<frame_message<tensor<float, 4>>>(heatmaps_msg)->get_data();

    std::map<std::string, cv::Mat> features;

    for (size_t i = 0; i < frame.size(); i++)
    {
        const auto name = names[i];

        const auto heatmap = heatmaps.view<3>({heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2], 0}, {0, 0, 0, static_cast<uint32_t>(i)}).contiguous().sum<1>({2});

        cv::Mat heatmap_mat;
        cv::Mat(heatmap.shape[1], heatmap.shape[0], CV_32FC1, (float*)heatmap.data.data()).convertTo(heatmap_mat, CV_8U, 255);
        cv::resize(heatmap_mat, heatmap_mat, cv::Size(960, 540));
        cv::cvtColor(heatmap_mat, heatmap_mat, cv::COLOR_GRAY2BGR);

        features[name] = heatmap_mat;
    }

    std::vector<tensor<float, 4>> heatmaps_list;
    std::vector<camera_data> cameras_list;

    for (size_t i = 0; i < frame.size(); i++)
    {
        const auto name = names[i];

        const auto heatmap = heatmaps.view<4>({heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2], 1}, {0, 0, 0, static_cast<uint32_t>(i)}).contiguous();

        camera_data camera;

        const auto& src_camera = cameras.at(name);

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
        const auto camera_pose = src_camera.extrin.rotation * glm::inverse(axis);

        for (size_t i = 0; i < 3; i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                camera.rotation[i][j] = camera_pose[j][i];
            }
            camera.translation[i] = camera_pose[3][i];
        }

        heatmaps_list.push_back(heatmap);
        cameras_list.push_back(camera);
    }
    glm::vec3 grid_center = {0.0f, 0.0f, 0.0f};

    proj->set_grid_size({8000.0, 8000.0, 2000.0});
    proj->set_cube_size({{80, 80, 20}});

    const auto [cubes, grid] = proj->get_voxel(heatmaps_list, cameras_list, rois_list, {grid_center[0], grid_center[1], grid_center[2]});

    auto cubes_msg = std::make_shared<frame_message<tensor<float, 4>>>();

    cubes_msg->set_data(std::move(cubes));
    cubes_msg->set_timestamp(0);
    cubes_msg->set_frame_number(0);

    const auto proposal_msg = inference_proposal->process("input", cubes_msg);

    const auto &proposal = std::dynamic_pointer_cast<frame_message<tensor<float, 5>>>(proposal_msg)->get_data();

    prop->set_max_num(10);
    prop->set_threshold(0.3f);
    prop->set_grid_size({8000.0, 8000.0, 2000.0});
    prop->set_grid_center({0.0, 0.0, 0.0});
    prop->set_cube_size({{80, 80, 20}});

    const auto centers = prop->get_centers(proposal);

    {
        std::lock_guard lock(features_mtx);
        this->features = features;
    }

    std::vector<glm::vec3> points;
    return points;
}

dnn_reconstruction::dnn_reconstruction()
    : inference_heatmap(new dnn_inference()), inference_proposal(new dnn_inference()), proj(new projector()), prop(new get_proposal()), service(new SensorServiceImpl()), reconstruction_workers(std::make_shared<task_queue<std::function<void()>>>(1)), task_id_gen(std::random_device()()) {}
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
        const auto markers = dnn_reconstruct(cameras, frame, axis);

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

static void load_model(std::string model_path, std::vector<uint8_t> &data)
{
    std::ifstream ifs;
    ifs.open(model_path, std::ios_base::in | std::ios_base::binary);
    if (ifs.fail())
    {
        std::cerr << "File open error: " << model_path << "\n";
        std::quick_exit(0);
    }

    ifs.seekg(0, std::ios::end);
    const auto length = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    data.resize(length);

    ifs.read((char *)data.data(), length);
    if (ifs.fail())
    {
        std::cerr << "File read error: " << model_path << "\n";
        std::quick_exit(0);
    }
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

    std::vector<uint8_t> backbone_model_data;
    {
        const auto model_path = "backbone.onnx";
        std::vector<uint8_t> data;
        load_model(model_path, data);

        backbone_model_data = std::move(data);
    }

    inference_heatmap->output_names.push_back("output");

    inference_heatmap->set_model_data(backbone_model_data);
    inference_heatmap->initialize();

    std::vector<uint8_t> proposal_v2v_net_model_data;
    {
        const auto model_path = "proposal_v2v_net.onnx";
        std::vector<uint8_t> data;
        load_model(model_path, data);

        proposal_v2v_net_model_data = std::move(data);
    }

    inference_proposal->output_names.push_back("output");

    inference_proposal->set_model_data(proposal_v2v_net_model_data);
    inference_proposal->initialize();
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