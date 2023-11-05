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
#include "voxelpose_cuda.hpp"

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

#include <cuda_runtime.h>

#define CUDA_SAFE_CALL(func)                                                                                                  \
    do                                                                                                                        \
    {                                                                                                                         \
        cudaError_t err = (func);                                                                                             \
        if (err != cudaSuccess)                                                                                               \
        {                                                                                                                     \
            fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(err);                                                                                                        \
        }                                                                                                                     \
    } while (0)

namespace fs = std::filesystem;

#define ENABLE_ONNXRUNTIME

#ifdef ENABLE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#include <tensorrt_provider_factory.h>

class dnn_reconstruction::dnn_inference
{
    std::vector<uint8_t> model_data;

    Ort::Env env{ORT_LOGGING_LEVEL_WARNING};
    Ort::Session session;
    Ort::IoBinding io_binding;
    Ort::MemoryInfo info_cuda{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};
    Ort::Allocator cuda_allocator{nullptr};

    float *input_data = nullptr;
    float *output_data = nullptr;

    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;

    std::unordered_map<std::string, std::vector<int64_t>> input_node_dims;
    std::unordered_map<std::string, std::vector<int64_t>> output_node_dims;

public:
    dnn_inference(const std::vector<uint8_t> &model_data, std::string cache_dir)
        : session(nullptr), io_binding(nullptr)
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
        io_binding = Ort::IoBinding(session);
        io_binding = Ort::IoBinding(session);
        cuda_allocator = Ort::Allocator(session, info_cuda);

        Ort::AllocatorWithDefaultOptions allocator;

        // Iterate over all input nodes
        const size_t num_input_nodes = session.GetInputCount();
        for (size_t i = 0; i < num_input_nodes; i++)
        {
            const auto input_name = session.GetInputNameAllocated(i, allocator);
            input_node_names.push_back(input_name.get());

            const auto type_info = session.GetInputTypeInfo(i);
            const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            const auto input_shape = tensor_info.GetShape();
            input_node_dims[input_name.get()] = input_shape;
        }

        // Iterate over all output nodes
        const size_t num_output_nodes = session.GetOutputCount();
        for (size_t i = 0; i < num_output_nodes; i++)
        {
            const auto output_name = session.GetOutputNameAllocated(i, allocator);
            output_node_names.push_back(output_name.get());

            const auto type_info = session.GetOutputTypeInfo(i);
            const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            const auto output_shape = tensor_info.GetShape();
            output_node_dims[output_name.get()] = output_shape;
        }
        assert(input_node_names.size() == 1);
        assert(input_node_names[0] == "input");

        assert(output_node_names.size() == 1);
        assert(output_node_names[0] == "output");

        {
            const auto dims = input_node_dims.at(input_node_names[0]);
            const auto input_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

            input_data = reinterpret_cast<float *>(cuda_allocator.GetAllocation(input_size * sizeof(float)).get());
        }

        {
            const auto dims = output_node_dims.at(output_node_names[0]);
            const auto output_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

            output_data = reinterpret_cast<float *>(cuda_allocator.GetAllocation(output_size * sizeof(float)).get());
        }
    }

    void inference(const float* input)
    {
        assert(input_node_names.size() == 1);
        assert(input_node_names[0] == "input");

        assert(output_node_names.size() == 1);
        assert(output_node_names[0] == "output");

        std::vector<const char *> input_node_names;
        {
            input_node_names.push_back(this->input_node_names[0].c_str());
        }

        std::vector<const char *> output_node_names;
        {
            output_node_names.push_back(this->output_node_names[0].c_str());
        }

        io_binding.ClearBoundInputs();
        io_binding.ClearBoundOutputs();

        std::vector<Ort::Value> input_tensors;
        {
            const auto dims = input_node_dims.at(input_node_names[0]);
            const auto input_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

            CUDA_SAFE_CALL(cudaMemcpy(input_data, input, input_size * sizeof(float), cudaMemcpyDeviceToDevice));

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(info_cuda, input_data, input_size, dims.data(), dims.size());

            io_binding.BindInput(input_node_names[0], input_tensor);

            input_tensors.emplace_back(std::move(input_tensor));
        }

        std::vector<Ort::Value> output_tensors;
        {
            const auto dims = output_node_dims.at(output_node_names[0]);
            const auto output_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

            Ort::Value output_tensor = Ort::Value::CreateTensor(info_cuda, output_data, output_size, dims.data(), dims.size());

            io_binding.BindOutput(output_node_names[0], output_tensor);

            output_tensors.emplace_back(std::move(output_tensor));
        }

        io_binding.SynchronizeInputs();

        session.Run(Ort::RunOptions{nullptr}, io_binding);

        io_binding.SynchronizeOutputs();
    }

    const float *get_output_data() const
    {
        return output_data;
    }
};

class dnn_reconstruction::dnn_inference_heatmap
{
    std::vector<uint8_t> model_data;

    Ort::Env env{ORT_LOGGING_LEVEL_WARNING};
    Ort::Session session;
    Ort::IoBinding io_binding;
    Ort::MemoryInfo info_cuda{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};
    Ort::Allocator cuda_allocator{nullptr};

    float* input_data = nullptr;
    float *output_data = nullptr;

    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;

    std::unordered_map<std::string, std::vector<int64_t>> input_node_dims;
    std::unordered_map<std::string, std::vector<int64_t>> output_node_dims;

    int input_image_width = 960;
    int input_image_height = 540;

    // int input_image_width = 1920;
    // int input_image_height = 1080;

    uint8_t* input_image_data = nullptr;

public:

    dnn_inference_heatmap(const std::vector<uint8_t> &model_data, size_t max_views)
        : session(nullptr), io_binding(nullptr)
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
        io_binding = Ort::IoBinding(session);
        cuda_allocator = Ort::Allocator(session, info_cuda);

        Ort::AllocatorWithDefaultOptions allocator;

        // Iterate over all input nodes
        const size_t num_input_nodes = session.GetInputCount();
        for (size_t i = 0; i < num_input_nodes; i++)
        {
            const auto input_name = session.GetInputNameAllocated(i, allocator);
            input_node_names.push_back(input_name.get());

            const auto type_info = session.GetInputTypeInfo(i);
            const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            const auto input_shape = tensor_info.GetShape();
            input_node_dims[input_name.get()] = input_shape;
        }

        // Iterate over all output nodes
        const size_t num_output_nodes = session.GetOutputCount();
        for (size_t i = 0; i < num_output_nodes; i++)
        {
            const auto output_name = session.GetOutputNameAllocated(i, allocator);
            output_node_names.push_back(output_name.get());

            const auto type_info = session.GetOutputTypeInfo(i);
            const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            const auto output_shape = tensor_info.GetShape();
            output_node_dims[output_name.get()] = output_shape;
        }
        assert(input_node_names.size() == 1);
        assert(input_node_names[0] == "input");

        assert(output_node_names.size() == 1);
        assert(output_node_names[0] == "output");

        const auto input_size = 960 * 512 * 3 * max_views;

        input_data = reinterpret_cast<float*>(cuda_allocator.GetAllocation(input_size * sizeof(float)).get());

        const auto output_size = 240 * 128 * 15 * max_views;

        output_data = reinterpret_cast<float*>(cuda_allocator.GetAllocation(output_size * sizeof(float)).get());

        cudaMalloc(&input_image_data, input_image_width * input_image_height * 3 * max_views);
    }

    ~dnn_inference_heatmap()
    {
        cudaFree(input_image_data);
    }

    void process(const std::vector<cv::Mat>& images, std::vector<roi_data>& rois)
    {
        const auto &&image_size = cv::Size(960, 512);

        for (size_t i = 0; i < images.size(); i++)
        {
            const auto &data = images.at(i);

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

            assert(data.size().width == input_image_width);
            assert(data.size().height == input_image_height);

            const auto scale = get_scale(data.size(), image_size);
            const auto center = cv::Point2f(data.size().width / 2.0, data.size().height / 2.0);
            const auto rotation = 0.0;

            roi_data roi = {{scale.width, scale.height}, rotation, {center.x, center.y}};
            rois.push_back(roi);

            const std::array<float, 3> mean = {0.485, 0.456, 0.406};
            const std::array<float, 3> std = {0.229, 0.224, 0.225};

            CUDA_SAFE_CALL(cudaMemcpy2D(input_image_data + i * input_image_width * 3 * input_image_height, input_image_width * 3, data.data, data.step, data.cols * 3, data.rows, cudaMemcpyHostToDevice));

            preprocess_cuda(input_image_data + i * input_image_width * 3 * input_image_height, input_image_width, input_image_height, input_image_width * 3, input_data + i * 960 * 512 * 3, 960, 512, 960, mean, std);
        }

        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        inference(images.size());
    }

    void inference(size_t num_views)
    {
        assert(input_node_names.size() == 1);
        assert(input_node_names[0] == "input");

        assert(output_node_names.size() == 1);
        assert(output_node_names[0] == "output");

        std::vector<const char *> input_node_names;
        {
            input_node_names.push_back(this->input_node_names[0].c_str());
        }

        std::vector<const char *> output_node_names;
        {
            output_node_names.push_back(this->output_node_names[0].c_str());
        }

        io_binding.ClearBoundInputs();
        io_binding.ClearBoundOutputs();

        std::vector<Ort::Value> input_tensors;
        {
            auto dims = input_node_dims.at(input_node_names[0]);
            dims[0] = num_views;

            const auto input_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(info_cuda, input_data, input_size, dims.data(), dims.size());

            io_binding.BindInput(input_node_names[0], input_tensor);

            input_tensors.emplace_back(std::move(input_tensor));
        }

        std::vector<Ort::Value> output_tensors;
        {
            auto dims = output_node_dims.at(output_node_names[0]);
            dims[0] = num_views;
            const auto output_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

            Ort::Value output_tensor = Ort::Value::CreateTensor(info_cuda, output_data, output_size, dims.data(), dims.size());

            io_binding.BindOutput(output_node_names[0], output_tensor);

            output_tensors.emplace_back(std::move(output_tensor));
        }

        io_binding.SynchronizeInputs();

        session.Run(Ort::RunOptions{nullptr}, io_binding);

        io_binding.SynchronizeOutputs();
    }

    const float* get_heatmaps() const
    {
        return output_data;
    }

    int get_heatmap_width() const
    {
        return 240;
    }

    int get_heatmap_height() const
    {
        return 128;
    }
};
#else
#include <opencv2/dnn/dnn.hpp>

class dnn_reconstruction::dnn_inference
{
    cv::dnn::Net net;

    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;

    std::unordered_map<std::string, std::vector<int64_t>> input_node_dims;
    std::unordered_map<std::string, std::vector<int64_t>> output_node_dims;

    std::vector<float> input_data;
    float* output_data = nullptr;

public:
    dnn_inference(const std::vector<uint8_t>& model_data, std::string cache_dir)
    {
        const auto backend = cv::dnn::getAvailableBackends();
        net = cv::dnn::readNetFromONNX(model_data);

        std::vector<cv::dnn::MatShape> input_layer_shapes;
        std::vector<cv::dnn::MatShape> output_layer_shapes;
        net.getLayerShapes(cv::dnn::MatShape(), 0, input_layer_shapes, output_layer_shapes);

        assert(input_layer_shapes.size() == 1);
        assert(output_layer_shapes.size() == 1);

        const auto input_size = std::accumulate(input_layer_shapes[0].begin(), input_layer_shapes[0].end(), 1, std::multiplies<int64_t>());
        input_data.resize(input_size);

        const auto output_size = std::accumulate(output_layer_shapes[0].begin(), output_layer_shapes[0].end(), 1, std::multiplies<int64_t>());
        cudaMalloc(&output_data, output_size * sizeof(float));
    }

    void inference(const float* input)
    {
        std::vector<cv::dnn::MatShape> input_layer_shapes;
        std::vector<cv::dnn::MatShape> output_layer_shapes;
        net.getLayerShapes(cv::dnn::MatShape(), 0, input_layer_shapes, output_layer_shapes);

        assert(input_layer_shapes.size() == 1);
        assert(output_layer_shapes.size() == 1);

        cudaMemcpy(input_data.data(), input, input_data.size(), cudaMemcpyDeviceToHost);

        cv::Mat input_mat(input_layer_shapes[0], CV_32FC1, (void*)input_data.data());
        net.setInput(input_mat);
        const auto output_mat = net.forward();

        cudaMemcpy(output_data, output_mat.data, output_mat.total() * sizeof(float), cudaMemcpyHostToDevice);
    }

    const float* get_output_data() const
    {
        return output_data;
    }
};

class dnn_reconstruction::dnn_inference_heatmap
{
    cv::dnn::Net net;

    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;

    std::unordered_map<std::string, std::vector<int64_t>> input_node_dims;
    std::unordered_map<std::string, std::vector<int64_t>> output_node_dims;

    float* input_data = nullptr;
    float* output_data = nullptr;
    std::vector<float> input_data_cpu;

    int input_image_width = 960;
    int input_image_height = 540;

    // int input_image_width = 1920;
    // int input_image_height = 1080;

    uint8_t* input_image_data = nullptr;

public:

    dnn_inference_heatmap(const std::vector<uint8_t>& model_data, size_t max_views)
    {
        const auto backends = cv::dnn::getAvailableBackends();
        net = cv::dnn::readNetFromONNX(model_data);

        const auto input_size = 960 * 512 * 3 * max_views;
        cudaMalloc(&input_data, input_size * sizeof(float));
        input_data_cpu.resize(input_size);

        const auto output_size = 240 * 128 * 15 * max_views;
        cudaMalloc(&output_data, output_size * sizeof(float));

        cudaMalloc(&input_image_data, input_image_width * input_image_height * 3 * max_views);
    }

    ~dnn_inference_heatmap()
    {
    }

    void process(const std::vector<cv::Mat>& images, std::vector<roi_data>& rois)
    {
        const auto&& image_size = cv::Size(960, 512);

        for (size_t i = 0; i < images.size(); i++)
        {
            const auto& data = images.at(i);

            const auto get_scale = [](const cv::Size2f& image_size, const cv::Size2f& resized_size)
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

            assert(data.size().width == input_image_width);
            assert(data.size().height == input_image_height);

            const auto scale = get_scale(data.size(), image_size);
            const auto center = cv::Point2f(data.size().width / 2.0, data.size().height / 2.0);
            const auto rotation = 0.0;

            roi_data roi = { {scale.width, scale.height}, rotation, {center.x, center.y} };
            rois.push_back(roi);

            const std::array<float, 3> mean = { 0.485, 0.456, 0.406 };
            const std::array<float, 3> std = { 0.229, 0.224, 0.225 };

            CUDA_SAFE_CALL(cudaMemcpy2D(input_image_data + i * input_image_width * 3 * input_image_height, input_image_width * 3, data.data, data.step, data.cols * 3, data.rows, cudaMemcpyHostToDevice));

            preprocess_cuda(input_image_data + i * input_image_width * 3 * input_image_height, input_image_width, input_image_height, input_image_width * 3, input_data + i * 960 * 512 * 3, 960, 512, 960, mean, std);
        }

        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        inference(images.size());
    }

    void inference(size_t num_views)
    {
        cudaMemcpy(input_data_cpu.data(), input_data, num_views * 960 * 512 * 3, cudaMemcpyDeviceToHost);

        const cv::dnn::MatShape input_shape = { static_cast<int>(num_views), 3, 512, 960 };

        cv::Mat input_mat(input_shape, CV_32FC1, (void*)input_data_cpu.data());
        net.setInput(input_mat);
        const auto output_mat = net.forward();

        cudaMemcpy(output_data, output_mat.data, output_mat.total() * sizeof(float), cudaMemcpyHostToDevice);
    }

    const float* get_heatmaps() const
    {
        return output_data;
    }

    int get_heatmap_width() const
    {
        return 240;
    }

    int get_heatmap_height() const
    {
        return 128;
    }
};
#endif

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

    static coalsack::tensor<float, 4> max_pool(const coalsack::tensor<float, 4> &inputs, size_t kernel = 3)
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

    static coalsack::tensor<uint64_t, 2> get_index(const coalsack::tensor<uint64_t, 1> &indices, const std::array<uint64_t, 3> &shape)
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

    coalsack::tensor<float, 2> get_real_loc(const coalsack::tensor<uint64_t, 2> &index)
    {
        const auto loc = index.cast<float>()
                             .transform(
                                 [this](const float value, const size_t i, const size_t j)
                                 {
                                     return value / (cube_size[i] - 1) * grid_size[i] + grid_center[i] - grid_size[i] / 2.0f;
                                 });
        return loc;
    }

    coalsack::tensor<float, 2> get_centers(const coalsack::tensor<float, 5> &src)
    {
        const auto root_cubes = src.view<4>({src.shape[0], src.shape[1], src.shape[2], src.shape[3]}).contiguous();
        const auto root_cubes_nms = max_pool(root_cubes);
        const auto [topk_values, topk_index] = root_cubes_nms.view<1>({src.shape[0] * src.shape[1] * src.shape[2]})
                                                    .topk(max_num);

        const auto topk_unravel_index = get_index(topk_index, {src.shape[0], src.shape[1], src.shape[2]});
        const auto topk_loc = get_real_loc(topk_unravel_index);

        auto grid_centers = coalsack::tensor<float, 2>::zeros({5, max_num});
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
    const auto start = std::chrono::system_clock::now();

    std::vector<roi_data> rois_list;
    inference_heatmap->process(images_list, rois_list);

#ifdef PANOPTIC
    std::array<float, 3> grid_center = {0.0f, -500.0f, 800.0f};
#else
    std::array<float, 3> grid_center = {0.0f, 0.0f, 0.0f};
#endif
    std::array<int32_t, 3> cube_size = {80, 80, 20};
    std::array<float, 3> grid_size = {8000.0, 8000.0, 2000.0};

    global_proj->set_grid_size(grid_size);
    global_proj->set_cube_size(cube_size);

    global_proj->get_voxel(inference_heatmap->get_heatmaps(), images_list.size(), inference_heatmap->get_heatmap_width(), inference_heatmap->get_heatmap_height(), cameras_list, rois_list, grid_center);

#if 0
    {
        static int counter = 0;
        counter++;

        const auto num_bins = static_cast<uint32_t>(std::accumulate(cube_size.begin(), cube_size.end(), 1, std::multiplies<int32_t>()));
        const auto num_joints = 15;
        coalsack::tensor<float, 4> temp_cubes({num_bins, 1, num_joints, 1});
        CUDA_SAFE_CALL(cudaMemcpy(temp_cubes.get_data(), global_proj->get_cubes(), temp_cubes.get_size() * sizeof(float), cudaMemcpyDeviceToHost));

        const auto cubes = temp_cubes.reshape_move<4>({static_cast<uint32_t>(cube_size[2]), static_cast<uint32_t>(cube_size[1]), static_cast<uint32_t>(cube_size[0]), num_joints});
        
        {
            std::ofstream ofs;
            ofs.open("./cubes" + std::to_string(counter) + ".bin", std::ios::out | std::ios::binary);

            ofs.write((const char*)cubes.get_data(), cubes.get_size() * sizeof(float));
        }
    }
#endif

    inference_proposal->inference(global_proj->get_cubes());

    coalsack::tensor<float, 5> proposal({20, 80, 80, 1, 1});
    CUDA_SAFE_CALL(cudaMemcpy(proposal.get_data(), inference_proposal->get_output_data(), proposal.get_size() * sizeof(float), cudaMemcpyDeviceToHost));

    prop->set_max_num(10);
    prop->set_threshold(0.3f);
    prop->set_grid_size(grid_size);
    prop->set_grid_center(grid_center);
    prop->set_cube_size(cube_size);

    const auto centers = prop->get_centers(proposal);

    std::vector<glm::vec3> points;

    for (uint32_t i = 0; i < centers.shape[1]; i++)
    {
        const auto score = centers.get({4, i});
        if (score > 0.3f)
        {
            const std::array<float, 3> center = {centers.get({0, i}), centers.get({1, i}), centers.get({2, i})};

            std::array<int32_t, 3> cube_size = {64, 64, 64};
            std::array<float, 3> grid_size = {2000.0, 2000.0, 2000.0};

            local_proj->set_grid_size(grid_size);
            local_proj->set_cube_size(cube_size);

            local_proj->get_voxel(inference_heatmap->get_heatmaps(), images_list.size(), inference_heatmap->get_heatmap_width(), inference_heatmap->get_heatmap_height(), cameras_list, rois_list, center);

            inference_pose->inference(local_proj->get_cubes());

            joint_extract->soft_argmax(inference_pose->get_output_data(), 100, grid_size, cube_size, center);

            std::vector<glm::vec3> joints(15);

            CUDA_SAFE_CALL(cudaMemcpy(&joints[0][0], joint_extract->get_joints(), 3 * 15 * sizeof(float), cudaMemcpyDeviceToHost));

            glm::mat4 basis(1.f);
            basis[0] = glm::vec4(-1.f, 0.f, 0.f, 0.f);
            basis[1] = glm::vec4(0.f, 0.f, 1.f, 0.f);
            basis[2] = glm::vec4(0.f, 1.f, 0.f, 0.f);

            for (const auto &joint : joints)
            {
                points.push_back(basis * glm::vec4(joint / 1000.0f, 1.0f));
            }

#if 0
            {
                static int counter = 0;
                counter++;

                const auto num_bins = static_cast<uint32_t>(std::accumulate(cube_size.begin(), cube_size.end(), 1, std::multiplies<int32_t>()));
                const auto num_joints = 15;
                coalsack::tensor<float, 4> temp_cubes({num_bins, 1, num_joints, 1});
                CUDA_SAFE_CALL(cudaMemcpy(temp_cubes.get_data(), inference_pose->get_output_data(), temp_cubes.get_size() * sizeof(float), cudaMemcpyDeviceToHost));

                const auto cubes = temp_cubes.reshape_move<4>({static_cast<uint32_t>(cube_size[2]), static_cast<uint32_t>(cube_size[1]), static_cast<uint32_t>(cube_size[0]), num_joints});
                
                {
                    std::ofstream ofs;
                    ofs.open("./ind_cubes" + std::to_string(counter) + ".bin", std::ios::out | std::ios::binary);

                    ofs.write((const char*)cubes.get_data(), cubes.get_size() * sizeof(float));
                }
            }
#endif

#if 0
            {
                static int counter = 0;
                counter++;

                const auto num_points = 15;

                std::ofstream ofs;
                ofs.open("./result" + std::to_string(counter) + ".pcd", std::ios::out);

                ofs << "VERSION 0.7" << std::endl;
                ofs << "FIELDS x y z rgba" << std::endl;
                ofs << "SIZE 4 4 4 4" << std::endl;
                ofs << "TYPE F F F U" << std::endl;
                ofs << "COUNT 1 1 1 1" << std::endl;
                ofs << "WIDTH " << num_points << std::endl;
                ofs << "HEIGHT 1" << std::endl;
                ofs << "VIEWPOINT 0 0 0 1 0 0 0" << std::endl;
                ofs << "POINTS " << num_points << std::endl;
                ofs << "DATA ascii" << std::endl;

                for (size_t j = 0; j < num_points; j++)
                {
                    const auto joint = basis * glm::vec4(joints[j] / 1000.0f, 1.0f);
                    ofs << joint.x << " " << joint.y << " " << joint.z << " " << 16711680 << std::endl;
                }
            }
#endif
        }
    }
    const auto end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // std::cout << "voxelpose: " << elapsed << std::endl;

    const auto start5 = std::chrono::system_clock::now();
    coalsack::tensor<float, 4> heatmaps({240, 128, 15, (uint32_t)images_list.size()});
    CUDA_SAFE_CALL(cudaMemcpy(heatmaps.get_data(), inference_heatmap->get_heatmaps(), heatmaps.get_size() * sizeof(float), cudaMemcpyDeviceToHost));

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
    : inference_heatmap(), inference_proposal(), inference_pose(), global_proj(new voxel_projector()), local_proj(new voxel_projector()), prop(new get_proposal()), joint_extract(new joint_extractor()), service(new SensorServiceImpl()), reconstruction_workers(std::make_shared<task_queue<std::function<void()>>>(1)), task_id_gen(std::random_device()()) {}
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

    inference_heatmap.reset(new dnn_inference_heatmap(backbone_model_data, 5));

    std::vector<uint8_t> proposal_v2v_net_model_data;
    {
        const auto model_path = "proposal_v2v_net.onnx";
        std::vector<uint8_t> data;
        load_model(model_path, data);

        proposal_v2v_net_model_data = std::move(data);
    }

    inference_proposal.reset(new dnn_inference(proposal_v2v_net_model_data, "./proposal_model_cache"));

    std::vector<uint8_t> pose_v2v_net_model_data;
    {
        const auto model_path = "pose_v2v_net.onnx";
        std::vector<uint8_t> data;
        load_model(model_path, data);

        pose_v2v_net_model_data = std::move(data);
    }

    inference_pose.reset(new dnn_inference(pose_v2v_net_model_data, "./pose_model_cache"));
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