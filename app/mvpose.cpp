#include "mvpose.hpp"

#include <filesystem>
#include <numeric>
#include <fstream>
#include <iostream>

#include <spdlog/spdlog.h>
#include <glm/ext.hpp>

#include <cuda_runtime.h>

#include "mvpose_cuda.hpp"

#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/calib3d/calib3d.hpp>

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

#define ENABLE_ONNXRUNTIME

#ifdef ENABLE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#include <tensorrt_provider_factory.h>
#include <cpu_provider_factory.h>

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>

namespace
{
    using namespace cv;
    using namespace std;

    template <typename T>
    void homogeneousToEuclidean(const Mat &X_, Mat &x_)
    {
        int d = X_.rows - 1;

        const Mat_<T> &X_rows = X_.rowRange(0, d);
        const Mat_<T> h = X_.row(d);

        const T *h_ptr = h[0], *h_ptr_end = h_ptr + h.cols;
        const T *X_ptr = X_rows[0];
        T *x_ptr = x_.ptr<T>(0);
        for (; h_ptr != h_ptr_end; ++h_ptr, ++X_ptr, ++x_ptr)
        {
            const T *X_col_ptr = X_ptr;
            T *x_col_ptr = x_ptr, *x_col_ptr_end = x_col_ptr + d * x_.step1();
            for (; x_col_ptr != x_col_ptr_end; X_col_ptr += X_rows.step1(), x_col_ptr += x_.step1())
                *x_col_ptr = (*X_col_ptr) / (*h_ptr);
        }
    }

    void homogeneousToEuclidean(InputArray X_, OutputArray x_)
    {
        // src
        const Mat X = X_.getMat();

        // dst
        x_.create(X.rows - 1, X.cols, X.type());
        Mat x = x_.getMat();

        // type
        if (X.depth() == CV_32F)
        {
            homogeneousToEuclidean<float>(X, x);
        }
        else
        {
            homogeneousToEuclidean<double>(X, x);
        }
    }

    /** @brief Triangulates the a 3d position between two 2d correspondences, using the DLT.
      @param xl Input vector with first 2d point.
      @param xr Input vector with second 2d point.
      @param Pl Input 3x4 first projection matrix.
      @param Pr Input 3x4 second projection matrix.
      @param objectPoint Output vector with computed 3d point.

      Reference: @cite HartleyZ00 12.2 pag.312
     */
    static void
    triangulateDLT(const Vec2d &xl, const Vec2d &xr,
                   const Matx34d &Pl, const Matx34d &Pr,
                   Vec3d &point3d)
    {
        Matx44d design;
        for (int i = 0; i < 4; ++i)
        {
            design(0, i) = xl(0) * Pl(2, i) - Pl(0, i);
            design(1, i) = xl(1) * Pl(2, i) - Pl(1, i);
            design(2, i) = xr(0) * Pr(2, i) - Pr(0, i);
            design(3, i) = xr(1) * Pr(2, i) - Pr(1, i);
        }

        Vec4d XHomogeneous;
        cv::SVD::solveZ(design, XHomogeneous);

        homogeneousToEuclidean(XHomogeneous, point3d);
    }

    /** @brief Triangulates the 3d position of 2d correspondences between n images, using the DLT
     * @param x Input vectors of 2d points (the inner vector is per image). Has to be 2xN
     * @param Ps Input vector with 3x4 projections matrices of each image.
     * @param X Output vector with computed 3d point.

     * Reference: it is the standard DLT; for derivation see appendix of Keir's thesis
     */
    static void
    triangulateNViews(const Mat_<double> &x, const std::vector<Matx34d> &Ps, Vec3d &X)
    {
        CV_Assert(x.rows == 2);
        unsigned nviews = x.cols;
        CV_Assert(nviews == Ps.size());

        cv::Mat_<double> design = cv::Mat_<double>::zeros(3 * nviews, 4 + nviews);
        for (unsigned i = 0; i < nviews; ++i)
        {
            for (char jj = 0; jj < 3; ++jj)
                for (char ii = 0; ii < 4; ++ii)
                    design(3 * i + jj, ii) = -Ps[i](jj, ii);
            design(3 * i + 0, 4 + i) = x(0, i);
            design(3 * i + 1, 4 + i) = x(1, i);
            design(3 * i + 2, 4 + i) = 1.0;
        }

        Mat X_and_alphas;
        cv::SVD::solveZ(design, X_and_alphas);
        homogeneousToEuclidean(X_and_alphas.rowRange(0, 4), X);
    }

    void
    triangulatePoints(InputArrayOfArrays _points2d, InputArrayOfArrays _projection_matrices,
                      OutputArray _points3d)
    {
        // check
        size_t nviews = (unsigned)_points2d.total();
        CV_Assert(nviews >= 2 && nviews == _projection_matrices.total());

        // inputs
        size_t n_points;
        std::vector<Mat_<double>> points2d(nviews);
        std::vector<Matx34d> projection_matrices(nviews);
        {
            std::vector<Mat> points2d_tmp;
            _points2d.getMatVector(points2d_tmp);
            n_points = points2d_tmp[0].cols;

            std::vector<Mat> projection_matrices_tmp;
            _projection_matrices.getMatVector(projection_matrices_tmp);

            // Make sure the dimensions are right
            for (size_t i = 0; i < nviews; ++i)
            {
                CV_Assert(points2d_tmp[i].rows == 2 && points2d_tmp[i].cols == n_points);
                if (points2d_tmp[i].type() == CV_64F)
                    points2d[i] = points2d_tmp[i];
                else
                    points2d_tmp[i].convertTo(points2d[i], CV_64F);

                CV_Assert(projection_matrices_tmp[i].rows == 3 && projection_matrices_tmp[i].cols == 4);
                if (projection_matrices_tmp[i].type() == CV_64F)
                    projection_matrices[i] = projection_matrices_tmp[i];
                else
                    projection_matrices_tmp[i].convertTo(projection_matrices[i], CV_64F);
            }
        }

        // output
        _points3d.create(3, n_points, CV_64F);
        cv::Mat points3d = _points3d.getMat();

        // Two view
        if (nviews == 2)
        {
            const Mat_<double> &xl = points2d[0], &xr = points2d[1];

            const Matx34d &Pl = projection_matrices[0]; // left matrix projection
            const Matx34d &Pr = projection_matrices[1]; // right matrix projection

            // triangulate
            for (unsigned i = 0; i < n_points; ++i)
            {
                Vec3d point3d;
                triangulateDLT(Vec2d(xl(0, i), xl(1, i)), Vec2d(xr(0, i), xr(1, i)), Pl, Pr, point3d);
                for (char j = 0; j < 3; ++j)
                    points3d.at<double>(j, i) = point3d[j];
            }
        }
        else if (nviews > 2)
        {
            // triangulate
            for (unsigned i = 0; i < n_points; ++i)
            {
                // build x matrix (one point per view)
                Mat_<double> x(2, nviews);
                for (unsigned k = 0; k < nviews; ++k)
                {
                    points2d.at(k).col(i).copyTo(x.col(k));
                }

                Vec3d point3d;
                triangulateNViews(x, projection_matrices, point3d);
                for (char j = 0; j < 3; ++j)
                    points3d.at<double>(j, i) = point3d[j];
            }
        }
    }
}

namespace stargazer_mvpose
{
    class dnn_inference_pose
    {
        std::vector<uint8_t> model_data;

        Ort::Env env{ORT_LOGGING_LEVEL_WARNING};
        Ort::Session session;
        Ort::IoBinding io_binding;
        Ort::MemoryInfo info_cuda{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};
        Ort::Allocator cuda_allocator{nullptr};

        float *input_data = nullptr;
        float *simcc_x_data = nullptr;
        float *simcc_y_data = nullptr;

        std::vector<std::string> input_node_names;
        std::vector<std::string> output_node_names;

        std::unordered_map<std::string, std::vector<int64_t>> input_node_dims;
        std::unordered_map<std::string, std::vector<int64_t>> output_node_dims;

        uint8_t *input_image_data = nullptr;

    public:
        static const int max_input_image_width = 1920;
        static const int max_input_image_height = 1080;

        static const int image_width = 288;
        static const int image_height = 384;

        static const auto num_joints = 133;

        dnn_inference_pose(const std::vector<uint8_t> &model_data, size_t max_batch_size, std::string cache_dir)
            : session(nullptr), io_binding(nullptr)
        {
            namespace fs = std::filesystem;

            const auto &api = Ort::GetApi();

            // Create session
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(4);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#if 0
            try
            {
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
            }
            catch (const Ort::Exception &e)
            {
                spdlog::info(e.what());
            }
#endif

            try
            {
                OrtCUDAProviderOptions cuda_options{};

                cuda_options.device_id = 0;
                cuda_options.arena_extend_strategy = 1;
                cuda_options.gpu_mem_limit = 2ULL * 1024 * 1024 * 1024;
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
                cuda_options.do_copy_in_default_stream = 1;

                session_options.AppendExecutionProvider_CUDA(cuda_options);
            }
            catch (const Ort::Exception &e)
            {
                spdlog::info(e.what());
            }

            OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 0);

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

            assert(output_node_names.size() == 2);
            assert(output_node_names[0] == "simcc_x");
            assert(output_node_names[1] == "simcc_y");

            const auto &&image_size = cv::Size(image_width, image_height);

            const auto input_size = image_size.width * image_size.height * 3 * max_batch_size;

            input_data = reinterpret_cast<float *>(cuda_allocator.GetAllocation(input_size * sizeof(float)).get());

            const auto simcc_x_size = image_size.width * 2 * num_joints * max_batch_size;

            // simcc_y_data = reinterpret_cast<float *>(cuda_allocator.GetAllocation(simcc_x_size * sizeof(float)).get());
            simcc_x_data = reinterpret_cast<float *>(std::malloc(simcc_x_size * sizeof(float)));

            const auto simcc_y_size = image_size.height * 2 * num_joints * max_batch_size;

            // simcc_y_data = reinterpret_cast<float *>(cuda_allocator.GetAllocation(simcc_y_size * sizeof(float)).get());
            simcc_y_data = reinterpret_cast<float *>(std::malloc(simcc_y_size * sizeof(float)));

            CUDA_SAFE_CALL(cudaMalloc(&input_image_data, max_input_image_width * max_input_image_height * 3 * max_batch_size));
        }

        ~dnn_inference_pose()
        {
            cudaFree(input_image_data);
            // cudaFree(simcc_x_data);
            // cudaFree(simcc_y_data);
            std::free(simcc_x_data);
            std::free(simcc_y_data);
        }

        void process(const cv::Mat &image, const cv::Rect2f& rect, roi_data &roi)
        {
            const auto &&image_size = cv::Size(image_width, image_height);

            {
                const auto &data = image;

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

                    return cv::Size2f(w_pad * 1.2 / 200.0, h_pad * 1.2 / 200.0);
                };

                const auto input_image_width = data.size().width;
                const auto input_image_height = data.size().height;

                assert(input_image_width <= max_input_image_width);
                assert(input_image_height <= max_input_image_height);

                const auto scale = get_scale(rect.size(), image_size);
                const auto center = cv::Point2f(rect.x + rect.size().width / 2.0, rect.y + rect.size().height / 2.0);
                const auto rotation = 0.0;

                roi = {{scale.width, scale.height}, rotation, {center.x, center.y}};

                const auto trans = get_transform(center, scale, image_size);

                const std::array<float, 3> mean = {0.485, 0.456, 0.406};
                const std::array<float, 3> std = {0.229, 0.224, 0.225};

                CUDA_SAFE_CALL(cudaMemcpy2D(input_image_data, input_image_width * 3, data.data, data.step, data.cols * 3, data.rows, cudaMemcpyHostToDevice));

                preprocess_cuda(input_image_data, input_image_width, input_image_height, input_image_width * 3, input_data, image_size.width, image_size.height, image_size.width, trans, mean, std);
            }

            CUDA_SAFE_CALL(cudaDeviceSynchronize());

            inference();
        }

        void inference()
        {
            assert(input_node_names.size() == 1);
            assert(input_node_names[0] == "input");

            assert(output_node_names.size() == 2);
            assert(output_node_names[0] == "simcc_x");
            assert(output_node_names[1] == "simcc_y");

            std::vector<const char *> input_node_names;
            {
                input_node_names.push_back(this->input_node_names[0].c_str());
            }

            std::vector<const char *> output_node_names;
            {
                output_node_names.push_back(this->output_node_names[0].c_str());
                output_node_names.push_back(this->output_node_names[1].c_str());
            }

            io_binding.ClearBoundInputs();
            io_binding.ClearBoundOutputs();

            std::vector<Ort::Value> input_tensors;
            {
                auto dims = input_node_dims.at(input_node_names[0]);
                dims[0] = 1;

                const auto input_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(info_cuda, input_data, input_size, dims.data(), dims.size());

                io_binding.BindInput(input_node_names[0], input_tensor);

                input_tensors.emplace_back(std::move(input_tensor));
            }

            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
            std::vector<Ort::Value> output_tensors;
            {
                auto dims = output_node_dims.at(output_node_names[0]);
                dims[0] = 1;
                dims[1] = num_joints;
                const auto simcc_x_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

                Ort::Value simcc_x_tensor = Ort::Value::CreateTensor<float>(memory_info, simcc_x_data, simcc_x_size, dims.data(), dims.size());

                io_binding.BindOutput(output_node_names[0], simcc_x_tensor);

                output_tensors.emplace_back(std::move(simcc_x_tensor));
            }

            {
                auto dims = output_node_dims.at(output_node_names[1]);
                dims[0] = 1;
                dims[1] = num_joints;
                const auto simcc_y_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

                Ort::Value simcc_y_tensor = Ort::Value::CreateTensor<float>(memory_info, simcc_y_data, simcc_y_size, dims.data(), dims.size());

                io_binding.BindOutput(output_node_names[1], simcc_y_tensor);

                output_tensors.emplace_back(std::move(simcc_y_tensor));
            }

            io_binding.SynchronizeInputs();

            session.Run(Ort::RunOptions{nullptr}, io_binding);

            io_binding.SynchronizeOutputs();
        }

        void copy_simcc_x_to_cpu(float *simcc_x) const
        {
            // CUDA_SAFE_CALL(cudaMemcpy(simcc_x, simcc_x_data, image_width * 2 * num_joints * sizeof(float), cudaMemcpyDeviceToHost));
            std::copy_n(simcc_x_data, image_width * 2 * num_joints, simcc_x);
        }

        void copy_simcc_y_to_cpu(float *simcc_y) const
        {
            // CUDA_SAFE_CALL(cudaMemcpy(simcc_y, simcc_y_data, image_height * 2 * num_joints * sizeof(float), cudaMemcpyDeviceToHost));
            std::copy_n(simcc_y_data, image_height * 2 * num_joints, simcc_y);
        }

        const float *get_simcc_x() const
        {
            return simcc_x_data;
        }

        const float *get_simcc_y() const
        {
            return simcc_y_data;
        }
    };

    class dnn_inference_det
    {
        std::vector<uint8_t> model_data;

        Ort::Env env{ORT_LOGGING_LEVEL_WARNING};
        Ort::Session session;
        Ort::IoBinding io_binding;
        Ort::MemoryInfo info_cuda{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};
        Ort::Allocator cuda_allocator{nullptr};

        float *input_data = nullptr;
        float *dets_data = nullptr;
        int64_t *labels_data = nullptr;

        std::vector<std::string> input_node_names;
        std::vector<std::string> output_node_names;

        std::unordered_map<std::string, std::vector<int64_t>> input_node_dims;
        std::unordered_map<std::string, std::vector<int64_t>> output_node_dims;

        uint8_t *input_image_data = nullptr;

    public:
        static const int max_input_image_width = 1920;
        static const int max_input_image_height = 1080;

        static const int image_width = 640;
        static const int image_height = 640;

        static constexpr auto num_people = 14;

        dnn_inference_det(const std::vector<uint8_t> &model_data, std::string cache_dir)
            : session(nullptr), io_binding(nullptr)
        {
            namespace fs = std::filesystem;

            const auto &api = Ort::GetApi();

            // Create session
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(4);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#if 0
            try
            {
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
            }
            catch (const Ort::Exception &e)
            {
                spdlog::info(e.what());
            }
#endif

            try
            {
                OrtCUDAProviderOptions cuda_options{};

                cuda_options.device_id = 0;
                cuda_options.arena_extend_strategy = 1;
                cuda_options.gpu_mem_limit = 2ULL * 1024 * 1024 * 1024;
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
                cuda_options.do_copy_in_default_stream = 1;

                session_options.AppendExecutionProvider_CUDA(cuda_options);
            }
            catch (const Ort::Exception &e)
            {
                spdlog::info(e.what());
            }

            OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 0);

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

            assert(output_node_names.size() == 2);
            assert(output_node_names[0] == "dets");
            assert(output_node_names[1] == "labels");

            const auto &&image_size = cv::Size(image_width, image_height);

            const auto input_size = image_size.width * image_size.height * 3;

            input_data = reinterpret_cast<float *>(cuda_allocator.GetAllocation(input_size * sizeof(float)).get());

            const auto dets_size = num_people * 5;

            // dets_data = reinterpret_cast<float *>(cuda_allocator.GetAllocation(dets_size * sizeof(float)).get());
            dets_data = reinterpret_cast<float *>(std::malloc(dets_size * sizeof(float)));

            const auto labels_size = num_people;

            // labels_data = reinterpret_cast<int64_t *>(cuda_allocator.GetAllocation(labels_size * sizeof(int64_t)).get());
            labels_data = reinterpret_cast<int64_t *>(std::malloc(labels_size * sizeof(int64_t)));

            CUDA_SAFE_CALL(cudaMalloc(&input_image_data, max_input_image_width * max_input_image_height * 3));
        }

        ~dnn_inference_det()
        {
            cudaFree(input_image_data);
            std::free(dets_data);
            std::free(labels_data);
        }

        void process(const cv::Mat &image, roi_data &roi)
        {
            const auto &&image_size = cv::Size(image_width, image_height);

            {
                const auto &data = image;

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

                const auto input_image_width = data.size().width;
                const auto input_image_height = data.size().height;

                assert(input_image_width <= max_input_image_width);
                assert(input_image_height <= max_input_image_height);

                const auto scale = get_scale(data.size(), image_size);
                const auto center = cv::Point2f(data.size().width / 2.0, data.size().height / 2.0);
                const auto rotation = 0.0;

                roi = {{scale.width, scale.height}, rotation, {center.x, center.y}};

                const std::array<float, 3> mean = {0.485, 0.456, 0.406};
                const std::array<float, 3> std = {0.229, 0.224, 0.225};

                CUDA_SAFE_CALL(cudaMemcpy2D(input_image_data, input_image_width * 3, data.data, data.step, data.cols * 3, data.rows, cudaMemcpyHostToDevice));

                preprocess_cuda(input_image_data, input_image_width, input_image_height, input_image_width * 3, input_data, image_size.width, image_size.height, image_size.width, mean, std);
            }

            CUDA_SAFE_CALL(cudaDeviceSynchronize());

            inference();
        }

        void inference()
        {
            assert(input_node_names.size() == 1);
            assert(input_node_names[0] == "input");

            assert(output_node_names.size() == 2);
            assert(output_node_names[0] == "dets");
            assert(output_node_names[1] == "labels");

            std::vector<const char *> input_node_names;
            {
                input_node_names.push_back(this->input_node_names[0].c_str());
            }

            std::vector<const char *> output_node_names;
            {
                output_node_names.push_back(this->output_node_names[0].c_str());
                output_node_names.push_back(this->output_node_names[1].c_str());
            }

#if 0
            io_binding.ClearBoundInputs();
            io_binding.ClearBoundOutputs();
#endif

            std::vector<Ort::Value> input_tensors;
            {
                auto dims = input_node_dims.at(input_node_names[0]);

                const auto input_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(info_cuda, input_data, input_size, dims.data(), dims.size());

                io_binding.BindInput(input_node_names[0], input_tensor);

                input_tensors.emplace_back(std::move(input_tensor));
            }

#if 0
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
            std::vector<Ort::Value> output_tensors;
            {
                auto dims = output_node_dims.at(output_node_names[0]);
                dims[1] = num_people;
                const auto dets_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

                Ort::Value dets_tensor = Ort::Value::CreateTensor<float>(memory_info, dets_data, dets_size, dims.data(), dims.size());

                io_binding.BindOutput(output_node_names[0], dets_tensor);

                output_tensors.emplace_back(std::move(dets_tensor));
            }

            {
                auto dims = output_node_dims.at(output_node_names[1]);
                dims[1] = num_people;
                const auto labels_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());

                Ort::Value labels_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, labels_data, labels_size, dims.data(), dims.size());

                io_binding.BindOutput(output_node_names[1], labels_tensor);

                output_tensors.emplace_back(std::move(labels_tensor));
            }

            io_binding.SynchronizeInputs();

            session.Run(Ort::RunOptions{nullptr}, io_binding);

            io_binding.SynchronizeOutputs();
#else
            const auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), input_tensors.data(), input_node_names.size(), output_node_names.data(), output_node_names.size());

            std::copy_n(output_tensors[0].GetTensorData<float>(), 5 * num_people, dets_data);
            std::copy_n(output_tensors[1].GetTensorData<int64_t>(), num_people, labels_data);
#endif
        }

        void copy_labels_to_cpu(int64_t* labels) const
        {
            // CUDA_SAFE_CALL(cudaMemcpy(labels, labels_data, num_people * sizeof(int32_t), cudaMemcpyDeviceToHost));
            std::copy_n(labels_data, num_people, labels);
        }

        void copy_dets_to_cpu(float *dets) const
        {
            // CUDA_SAFE_CALL(cudaMemcpy(dets, dets_data, 5 * num_people * sizeof(float), cudaMemcpyDeviceToHost));
            std::copy_n(dets_data, 5 * num_people, dets);
        }

        const int64_t *get_labels() const
        {
            return labels_data;
        }

        const float *get_dets() const
        {
            return dets_data;
        }
    };
}
#else
#endif

#include <Eigen/Core>
#include <Eigen/Dense>

namespace stargazer_mvpose
{
    static cv::Point2f operator*(cv::Mat M, const cv::Point2f &p)
    {
        cv::Mat_<double> src(3, 1);

        src(0, 0) = p.x;
        src(1, 0) = p.y;
        src(2, 0) = 1.0;

        cv::Mat_<double> dst = M * src;
        return cv::Point2f(dst(0, 0), dst(1, 0));
    }
    using pose_joints_t = std::vector<std::tuple<cv::Point2f, float>>;

    static glm::mat3 get_matrix(const camera_data &camera)
    {
        return glm::mat3(
            camera.fx, 0, 0,
            0, camera.fy, 0,
            camera.cx, camera.cy, 1);
    }

    static glm::mat4 get_pose(const camera_data &camera)
    {
        glm::mat4 m(1.0f);
        for (size_t i = 0; i < 3; i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                m[i][j] = camera.rotation[i][j];
            }
            m[3][i] = camera.translation[i];
        }
        return m;
    }

    class mvpose_matcher
    {
    public:
        float factor;
        mvpose_matcher(float factor = 5.0f)
            : factor(factor)
        {
        }

        static glm::mat3 calculate_fundametal_matrix(const glm::mat3 &camera_mat1, const glm::mat3 &camera_mat2,
                                                     const glm::mat4 &camera_pose1, const glm::mat4 &camera_pose2)
        {
            const auto pose = camera_pose2 * glm::inverse(camera_pose1);

            const auto R = glm::mat3(pose);
            const auto t = glm::vec3(pose[3]);

            const auto Tx = glm::mat3(
                0, -t[2], t[1],
                t[2], 0, -t[0],
                -t[1], t[0], 0);

            const auto E = Tx * R;

            const auto F = glm::inverse(glm::transpose(camera_mat2)) * E * glm::inverse(camera_mat1);

            return F;
        }

        static glm::mat3 calculate_fundametal_matrix(const camera_data &camera1, const camera_data &camera2)
        {
            const auto camera_mat1 = get_matrix(camera1);
            const auto camera_mat2 = get_matrix(camera2);

            return calculate_fundametal_matrix(camera_mat1, camera_mat2, get_pose(camera1), get_pose(camera2));
        }

        static glm::vec3 normalize_line(const glm::vec3 &v)
        {
            const auto c = std::sqrt(v.x * v.x + v.y * v.y);
            return v / c;
        }

        static glm::vec3 compute_correspond_epiline(const glm::mat3 &F, const glm::vec2 &p)
        {
            const auto l = F * glm::vec3(p, 1.f);
            return normalize_line(l);
            // return l;
        }

        static glm::vec3 compute_correspond_epiline(const camera_data &camera1, const camera_data &camera2, const glm::vec2 &p)
        {
            const auto F = calculate_fundametal_matrix(camera1, camera2);
            return compute_correspond_epiline(F, p);
        }

        static float distance_sq_line_point(const glm::vec3 &line, const glm::vec2 &point)
        {
            const auto a = line.x;
            const auto b = line.y;
            const auto c = line.z;

            const auto num = a * point.x + b * point.y + c;
            const auto distsq = num * num / (a * a + b * b);

            return distsq;
        }

        static cv::Mat glm2cv_mat3(const glm::mat4 &m)
        {
            cv::Mat ret(3, 3, CV_32F);
            for (std::size_t i = 0; i < 3; i++)
            {
                for (std::size_t j = 0; j < 3; j++)
                {
                    ret.at<float>(i, j) = m[j][i];
                }
            }
            return ret;
        }

        static glm::vec2 undistort(const glm::vec2 &pt, const camera_data &camera)
        {
            auto pts = std::vector<cv::Point2f>{cv::Point2f(pt.x, pt.y)};
            cv::Mat m = glm2cv_mat3(get_matrix(camera));
            cv::Mat coeffs(5, 1, CV_32F);
            coeffs.at<float>(0) = camera.k[0];
            coeffs.at<float>(1) = camera.k[1];
            coeffs.at<float>(4) = camera.k[2];
            coeffs.at<float>(2) = camera.p[0];
            coeffs.at<float>(3) = camera.p[1];

            std::vector<cv::Point2f> norm_pts;
            cv::undistortPoints(pts, norm_pts, m, coeffs);

            return glm::vec2(norm_pts[0].x, norm_pts[0].y);
        }

        static glm::vec2 project_undistorted(const glm::vec2 &pt, const camera_data &camera)
        {
            const auto p = get_matrix(camera) * glm::vec3(pt.x, pt.y, 1.0f);
            return glm::vec2(p.x / p.z, p.y / p.z);
        }

        template <typename T, int m, int n>
        static inline glm::mat<m, n, float, glm::precision::highp> eigen2glm(const Eigen::Matrix<T, m, n> &src)
        {
            glm::mat<m, n, float, glm::precision::highp> dst;
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    dst[j][i] = src(i, j);
                }
            }
            return dst;
        }

        static float projected_distance(const pose_joints_t &points1, const pose_joints_t &points2, const Eigen::Matrix<float, 3, 3> &F)
        {
            const auto num_points = 17;
            auto result = 0.0f;
            {
                for (size_t i = 0; i < num_points; i++)
                {
                    const auto pt1 = glm::vec2(std::get<0>(points1[i]).x, std::get<0>(points1[i]).y);
                    const auto pt2 = glm::vec2(std::get<0>(points2[i]).x, std::get<0>(points2[i]).y);
                    const auto line = compute_correspond_epiline(eigen2glm(F), pt1);
                    const auto dist = std::abs(glm::dot(line, glm::vec3(pt2, 1.f)));
                    result += dist;
                }
            }
            return result / num_points;
        }

        static float projected_distance(const pose_joints_t &points1, const pose_joints_t &points2, const camera_data &camera1, const camera_data &camera2)
        {
            const auto num_points = 17;
            auto result = 0.0f;
            {
                const auto F = calculate_fundametal_matrix(camera1, camera2);
                for (size_t i = 0; i < num_points; i++)
                {
                    const auto pt1 = project_undistorted(undistort(glm::vec2(std::get<0>(points1[i]).x, std::get<0>(points1[i]).y), camera1), camera1);
                    const auto pt2 = project_undistorted(undistort(glm::vec2(std::get<0>(points2[i]).x, std::get<0>(points2[i]).y), camera2), camera2);
                    const auto line = compute_correspond_epiline(F, pt1);
                    const auto dist = distance_sq_line_point(line, pt2);
                    result += dist;
                }
            }
            {
                const auto F = calculate_fundametal_matrix(camera2, camera1);
                for (size_t i = 0; i < num_points; i++)
                {
                    const auto pt1 = project_undistorted(undistort(glm::vec2(std::get<0>(points1[i]).x, std::get<0>(points1[i]).y), camera1), camera1);
                    const auto pt2 = project_undistorted(undistort(glm::vec2(std::get<0>(points2[i]).x, std::get<0>(points2[i]).y), camera2), camera2);
                    const auto line = compute_correspond_epiline(F, pt2);
                    const auto dist = distance_sq_line_point(line, pt1);
                    result += dist;
                }
            }
            return result / (num_points * 2);
        }

        static Eigen::MatrixXf compute_geometry_affinity(const std::vector<pose_joints_t> &points_set, const std::vector<size_t> &dim_group, const std::vector<std::vector<Eigen::Matrix<float, 3, 3>>> &cameras_list, float factor)
        {
            const auto M = points_set.size();
            Eigen::MatrixXf dist = Eigen::MatrixXf::Ones(M, M) * (factor * factor);
            dist.diagonal() = Eigen::VectorXf::Zero(M);

            for (size_t i = 0; i < dim_group.size() - 1; i++)
            {
                if (dim_group[i] == dim_group[i + 1])
                {
                    continue;
                }
                for (size_t j = i + 1; j < dim_group.size() - 1; j++)
                {
                    if (dim_group[j] == dim_group[j + 1])
                    {
                        continue;
                    }

                    for (size_t m = dim_group[i]; m < dim_group[i + 1]; m++)
                    {
                        for (size_t n = dim_group[j]; n < dim_group[j + 1]; n++)
                        {
                            const auto d1 = projected_distance(points_set[m], points_set[n], cameras_list[i][j]);
                            const auto d2 = projected_distance(points_set[n], points_set[m], cameras_list[j][i]);
                            dist(n, m) = dist(m, n) = (d1 + d2) / 2;
                        }
                    }
                }
            }

            const auto calc_std_dev = [](const auto &value)
            {
                return std::sqrt((value.array() - value.array().mean()).square().sum() / (value.array().size() - 1));
            };

            {
                const auto std_dev = calc_std_dev(dist);
                if (std_dev < factor)
                {
                    for (size_t i = 0; i < M; i++)
                    {
                        dist(i, i) = dist.array().mean();
                    }
                }
            }

            Eigen::MatrixXf affinity_matrix = -(dist.array() - dist.array().mean()) / (calc_std_dev(dist) + 1e-12);
            affinity_matrix = 1 / (1 + (-factor * affinity_matrix.array()).exp());

            return affinity_matrix;
        }

        static Eigen::MatrixXf compute_geometry_affinity(const std::vector<pose_joints_t> &points_set, const std::vector<size_t> &dim_group, const std::vector<camera_data> &cameras_list, float factor)
        {
            const auto M = points_set.size();
            Eigen::MatrixXf dist = Eigen::MatrixXf::Ones(M, M) * (factor * factor);
            dist.diagonal() = Eigen::VectorXf::Zero(M);

            for (size_t i = 0; i < dim_group.size() - 1; i++)
            {
                if (dim_group[i] == dim_group[i + 1])
                {
                    continue;
                }
                for (size_t j = i + 1; j < dim_group.size() - 1; j++)
                {
                    if (dim_group[j] == dim_group[j + 1])
                    {
                        continue;
                    }

                    for (size_t m = dim_group[i]; m < dim_group[i + 1]; m++)
                    {
                        for (size_t n = dim_group[j]; n < dim_group[j + 1]; n++)
                        {
                            const auto d = projected_distance(points_set[m], points_set[n], cameras_list[i], cameras_list[j]);
                            dist(m, n) = d;
                            dist(n, m) = d;
                        }
                    }
                }
            }

            const auto calc_std_dev = [](const auto &value)
            {
                return std::sqrt((value.array() - value.array().mean()).square().sum() / (value.array().size() - 1));
            };

            {
                const auto std_dev = calc_std_dev(dist);
                if (std_dev < factor)
                {
                    for (size_t i = 0; i < M; i++)
                    {
                        dist(i, i) = dist.array().mean();
                    }
                }
            }

            Eigen::MatrixXf affinity_matrix = -(dist.array() - dist.array().mean()) / (calc_std_dev(dist) + 1e-12);
            affinity_matrix = 1 / (1 + (-factor * affinity_matrix.array()).exp());

            return affinity_matrix;
        }

        static Eigen::MatrixXi transform_closure(const Eigen::MatrixXi& X_binary)
        {
            // Convert binary relation matrix to permutation matrix.
            Eigen::MatrixXi temp = Eigen::MatrixXi::Zero(X_binary.rows(), X_binary.cols());
            const int N = X_binary.rows();
            for (size_t k = 0; k < N; k++)
            {
                for (size_t i = 0; i < N; i++)
                {
                    for (size_t j = 0; j < N; j++)
                    {
                        temp(i, j) = X_binary(i, j) | (X_binary(i, k) & X_binary(k, j));
                    }
                }
            }
            Eigen::VectorXi vis = Eigen::VectorXi::Zero(N);
            Eigen::MatrixXi match = Eigen::MatrixXi::Zero(X_binary.rows(), X_binary.cols());
            for (size_t i = 0; i < N; i++)
            {
                if (vis(i))
                {
                    continue;
                }
                for (size_t j = 0; j < N; j++)
                {
                    if (temp(i, j))
                    {
                        vis(j) = 1;
                        match(j, i) = 1;
                    }
                }
            }
            return match;
        }

        /*
            Wang, Qianqian, Xiaowei Zhou, and Kostas Daniilidis. "Multi-Image Semantic
            Matching by Mining Consistent Features." arXiv preprint arXiv:1711.07641(2017).
        */

        static Eigen::VectorXf proj2pav(Eigen::VectorXf y)
        {
            y = y.array().max(0);
            Eigen::VectorXf x = Eigen::VectorXf::Zero(y.size());
            if (y.array().sum() < 1)
            {
                x.array() += y.array();
            }
            else
            {
                Eigen::VectorXf u = y;
                std::sort(u.begin(), u.end(), std::greater{});

                Eigen::VectorXf sv = Eigen::VectorXf::Zero(y.size());
                std::partial_sum(u.begin(), u.end(), sv.begin());

                int rho = -1;
                for (size_t i = 0; i < y.size(); i++)
                {
                    if (u(i) > (sv(i) - 1) / (i + 1))
                    {
                        rho = i;
                    }
                }
                assert(rho >= 0);
                const auto theta = std::max(0.0f, (sv(rho) - 1) / (rho + 1));
                x.array() += (y.array() - theta).max(0);
            }
            return x;
        }

        static Eigen::MatrixXf project_row(Eigen::MatrixXf X)
        {
            for (size_t i = 0; i < X.rows(); i++)
            {
                X.row(i) = proj2pav(X.row(i));
            }
            return X;
        }
        static Eigen::MatrixXf project_col(Eigen::MatrixXf X)
        {
            for (size_t j = 0; j < X.cols(); j++)
            {
                X.col(j) = proj2pav(X.col(j));
            }
            return X;
        }

        static Eigen::MatrixXf proj2dpam(const Eigen::MatrixXf& Y, const float tol = 1e-4f)
        {
            Eigen::MatrixXf X0 = Y;
            Eigen::MatrixXf X = Y;
            Eigen::MatrixXf I2 = Eigen::MatrixXf::Zero(Y.rows(), Y.cols());

            for (size_t iter = 0; iter < 10; iter++)
            {
                Eigen::MatrixXf X1 = project_row(X0.array() + I2.array());
                Eigen::MatrixXf I1 = X1.array() - (X0.array() + I2.array());
                Eigen::MatrixXf X2 = project_col(X0.array() + I1.array());
                I2 = X2.array() - (X0.array() + I1.array());

                const auto change = (X2.array() - X.array()).abs().mean();
                X = X2;
                if (change < tol)
                {
                    return X;
                }
            }
            return X;
        }

        static Eigen::MatrixXi solve_svt(Eigen::MatrixXf affinity_matrix, const std::vector<size_t> &dim_group)
        {
            const bool dual_stochastic = true;
            const int N = affinity_matrix.rows();
            const int max_iter = 500;
            const float alpha = 0.1f;
            const float lambda = 50.0f;
            const float tol = 5e-4f;
            const int p_select = 1;
            float mu = 64.0f;

            affinity_matrix.diagonal() = Eigen::VectorXf::Zero(N);
            affinity_matrix = (affinity_matrix.array() + affinity_matrix.transpose().array()) / 2;

            Eigen::MatrixXf X = affinity_matrix;
            Eigen::MatrixXf Y = Eigen::MatrixXf::Zero(affinity_matrix.rows(), affinity_matrix.cols());
            Eigen::MatrixXf W = alpha - affinity_matrix.array();

            for (int iter = 0; iter < max_iter; iter++)
            {
                Eigen::MatrixXf X0 = X;

                // Update Q with SVT
                Eigen::JacobiSVD<Eigen::MatrixXf> svd(1.0f / mu * Y.array() + X.array(), Eigen::ComputeFullU | Eigen::ComputeFullV);

                Eigen::VectorXf diagS = svd.singularValues().array() - static_cast<float>(lambda) / mu;
                diagS = diagS.array().max(0.0f);
                const Eigen::MatrixXf Q = svd.matrixU() * diagS.asDiagonal() * svd.matrixV().transpose();

                // Update X
                X = Q.array() - (W.array() + Y.array()) / mu;

                // Project X
                for (size_t i = 0; i < dim_group.size() - 1; i++)
                {
                    X.block(dim_group[i], dim_group[i], dim_group[i + 1] - dim_group[i], dim_group[i + 1] - dim_group[i]).array() = 0;
                }
                if (p_select == 1)
                {
                    X.diagonal() = Eigen::VectorXf::Ones(N);
                }
                X = X.array().max(0.0f).min(1.0f);

                if (dual_stochastic)
                {
                    // Projection for double stochastic constraint
                    for (size_t i = 0; i < dim_group.size() - 1; i++)
                    {
                        const auto row_begin = dim_group[i];
                        const auto row_end = dim_group[i + 1];
                        for (size_t j = 0; j < dim_group.size() - 1; j++)
                        {
                            const auto col_begin = dim_group[j];
                            const auto col_end = dim_group[j + 1];

                            if (row_end > row_begin && col_end > col_begin)
                            {
                                X.block(row_begin, col_begin, row_end - row_begin, col_end - col_begin).array() = proj2dpam(
                                    X.block(row_begin, col_begin, row_end - row_begin, col_end - col_begin), 1e-2);
                            }
                        }
                    }
                }

                X = (X.array() + X.transpose().array()) / 2;

                // Update Y
                Y = Y.array() + mu * (X.array() - Q.array());

                // Test if convergence
                const auto p_res = (X.array() - Q.array()).matrix().norm() / N;
                const auto d_res = mu * (X.array() - X0.array()).matrix().norm() / N;

                if (p_res < tol && d_res < tol)
                {
                    break;
                }

                if (p_res > 10 * d_res)
                {
                    mu = 2 * mu;
                }
                else if (d_res > 10 * p_res)
                {
                    mu = mu / 2;
                }
            }

            X = (X.array() + X.transpose().array()) / 2;

            const auto match = transform_closure(X.array().unaryExpr([](float p) { return p > 0.5f; }).template cast<int>());
            return match;
        }

        using pose_id_t = std::pair<size_t, size_t>;

        std::vector<std::vector<pose_id_t>> compute_matches(const std::vector<std::vector<pose_joints_t>> &pose_joints_list, const std::vector<camera_data> &cameras_list)
        {
            std::vector<pose_joints_t> points_set;
            std::vector<size_t> dim_group;
            for (const auto& persons : pose_joints_list)
            {
                dim_group.push_back(points_set.size());
                for (const auto &pose_joints : persons)
                {
                    points_set.push_back(pose_joints);
                }
            }
            dim_group.push_back(points_set.size());

            const auto affinity_matrix = compute_geometry_affinity(points_set, dim_group, cameras_list, factor);
            const auto match_matrix = solve_svt(affinity_matrix, dim_group);

            const size_t num_camera_min = 2;

            std::vector<std::vector<pose_id_t>> matched_list;
            for (size_t j = 0; j < match_matrix.cols(); j++)
            {
                if (match_matrix.col(j).array().sum() >= num_camera_min)
                {
                    std::vector<pose_id_t> matched;
                    for (size_t k = 0; k < dim_group.size() - 1; k++)
                    {
                        for (size_t i = dim_group[k]; i < dim_group[k + 1]; i++)
                        {
                            if (match_matrix(i, j) > 0)
                            {
                                matched.push_back(std::make_pair(k, i - dim_group[k]));
                            }
                        }
                    }
                    matched_list.push_back(matched);
                }
            }

            return matched_list;
        }
    };

    void get_cv_intrinsic(const camera_data &camera, cv::Mat &camera_matrix, cv::Mat &dist_coeffs)
    {
        camera_matrix = cv::Mat::eye(3, 3, CV_64F);
        camera_matrix.at<double>(0, 0) = camera.fx;
        camera_matrix.at<double>(1, 1) = camera.fy;
        camera_matrix.at<double>(0, 2) = camera.cx;
        camera_matrix.at<double>(1, 2) = camera.cy;

        dist_coeffs = cv::Mat::zeros(5, 1, CV_64FC1);
        dist_coeffs.at<double>(0) = camera.k[0];
        dist_coeffs.at<double>(1) = camera.k[1];
        dist_coeffs.at<double>(2) = camera.p[0];
        dist_coeffs.at<double>(3) = camera.p[1];
        dist_coeffs.at<double>(4) = camera.k[2];
    }

    static cv::Mat glm_to_cv_mat3x4(const glm::mat4 &m)
    {
        cv::Mat ret(3, 4, CV_64F);
        for (size_t i = 0; i < 3; i++)
        {
            for (size_t j = 0; j < 4; j++)
            {
                ret.at<double>(i, j) = m[j][i];
            }
        }
        return ret;
    }

    static glm::vec3 triangulate(const std::vector<glm::vec2> &points, const std::vector<camera_data> &cameras)
    {
        assert(points.size() == cameras.size());
        std::vector<cv::Mat> pts(points.size());
        std::vector<cv::Mat> projs(points.size());
        for (std::size_t i = 0; i < points.size(); i++)
        {
            cv::Mat camera_mat;
            cv::Mat dist_coeffs;
            get_cv_intrinsic(cameras[i], camera_mat, dist_coeffs);

            std::vector<cv::Point2d> pt = {cv::Point2d(points[i].x, points[i].y)};
            std::vector<cv::Point2d> undistort_pt;
            cv::undistortPoints(pt, undistort_pt, camera_mat, dist_coeffs);

            cv::Mat pt_mat(2, undistort_pt.size(), CV_64F);
            for (std::size_t j = 0; j < undistort_pt.size(); j++)
            {
                pt_mat.at<double>(0, j) = undistort_pt[j].x;
                pt_mat.at<double>(1, j) = undistort_pt[j].y;
            }
            pts[i] = pt_mat;
            projs[i] = glm_to_cv_mat3x4(get_pose(cameras[i]));
        }

        cv::Mat output;
        triangulatePoints(pts, projs, output);

        return glm::vec3(
            output.at<double>(0, 0),
            output.at<double>(1, 0),
            output.at<double>(2, 0));
    }

    std::vector<glm::vec3> mvpose::inference(const std::vector<cv::Mat> &images_list, const std::vector<camera_data> &cameras_list)
    {
        std::vector<std::vector<cv::Rect2f>> rects(images_list.size());
        for (size_t i = 0; i < images_list.size(); i++)
        {  
            roi_data roi;
            inference_det->process(images_list.at(i), roi);

            const auto num_people = dnn_inference_det::num_people;

            std::vector<float> dets(5 * num_people);
            std::vector<int64_t> labels(num_people);

            inference_det->copy_dets_to_cpu(dets.data());
            inference_det->copy_labels_to_cpu(labels.data());

            const auto &&resized_size = cv::Size(640, 640);
            const auto trans = get_transform(cv::Point2f(roi.center[0], roi.center[1]), cv::Size2f(roi.scale[0], roi.scale[1]), resized_size);

            cv::Mat inv_trans;
            cv::invertAffineTransform(trans, inv_trans);

            for (size_t j = 0; j < num_people; j++)
            {
                if (labels[j] == 0)
                {
                    const auto score = dets[j * 5 + 4];
                    if (score > 0.3f)
                    {
                        const auto bbox_left = dets[j * 5];
                        const auto bbox_top = dets[j * 5 + 1];
                        const auto bbox_right = dets[j * 5 + 2];
                        const auto bbox_bottom = dets[j * 5 + 3];

                        const auto bbox_left_top = inv_trans * cv::Point2f(bbox_left, bbox_top);
                        const auto bbox_right_bottom = inv_trans * cv::Point2f(bbox_right, bbox_bottom);

#if 0
                        cv::Mat trans = cv::Mat::zeros(2, 3, CV_64FC1);
                        trans.at<double>(0, 0) = 1;
                        trans.at<double>(1, 1) = 1;
                        trans.at<double>(0, 2) = -bbox_left_top.x;
                        trans.at<double>(1, 2) = -bbox_left_top.y;

                        cv::Mat image_part;
                        cv::warpAffine(images_list.at(i), image_part, trans, cv::Rect(bbox_left_top, bbox_right_bottom).size(), cv::INTER_LINEAR);

                        static int count = 0;
                        cv::imwrite("image_" + std::to_string(count++) + ".jpg", image_part);
#endif

                        rects[i].push_back(cv::Rect2f(bbox_left_top, bbox_right_bottom));
                    }
                }
            }
        }

        std::vector<std::vector<pose_joints_t>> pose_joints_list(images_list.size());
        for (size_t i = 0; i < images_list.size(); i++)
        {
            for (size_t j = 0; j < rects[i].size(); j++)
            {
                roi_data roi;
                inference_pose->process(images_list.at(i), rects[i][j], roi);

                const auto &&resized_size = cv::Size(288, 384);
                const auto num_joints = dnn_inference_pose::num_joints;
                const auto extend_width = resized_size.width * 2;
                const auto extend_height = resized_size.height * 2;

                std::vector<float> simcc_x(extend_width * num_joints);
                std::vector<float> simcc_y(extend_height * num_joints);

                inference_pose->copy_simcc_x_to_cpu(simcc_x.data());
                inference_pose->copy_simcc_y_to_cpu(simcc_y.data());

                const auto trans = get_transform(cv::Point2f(roi.center[0], roi.center[1]), cv::Size2f(roi.scale[0], roi.scale[1]), resized_size);

                cv::Mat inv_trans;
                cv::invertAffineTransform(trans, inv_trans);

                pose_joints_t pose_joints;
                for (int k = 0; k < num_joints; ++k)
                {
                    const auto x_biggest_iter = std::max_element(simcc_x.begin() + k * extend_width, simcc_x.begin() + k * extend_width + extend_width);
                    const auto max_x_pos = std::distance(simcc_x.begin() + k * extend_width, x_biggest_iter);
                    const auto pose_x = max_x_pos / 2.0f;
                    const auto score_x = *x_biggest_iter;

                    const auto y_biggest_iter = std::max_element(simcc_y.begin() + k * extend_height, simcc_y.begin() + k * extend_height + extend_height);
                    const auto max_y_pos = std::distance(simcc_y.begin() + k * extend_height, y_biggest_iter);
                    const auto pose_y = max_y_pos / 2.0f;
                    const auto score_y = *y_biggest_iter;

                    const auto score = (score_x + score_y) / 2;
                    // const auto score = std::max(score_x, score_y);

                    const auto pose = inv_trans * cv::Point2f(pose_x, pose_y);
                    const auto joint = std::make_tuple(pose, score);
                    pose_joints.emplace_back(joint);
                }

                pose_joints_list[i].push_back(pose_joints);
            }
        }

        const auto matched_list = matcher->compute_matches(pose_joints_list, cameras_list);

        glm::mat4 axis(1.0f);
        std::vector<glm::vec3> markers;
        for (const auto &matched : matched_list)
        {
#if 0
            static int count = 0;
            for (std::size_t i = 0; i < matched.size(); i++)
            {
                const auto rect = rects[matched[i].first][matched[i].second];

                cv::Mat affine = cv::Mat::zeros(2, 3, CV_64FC1);
                affine.at<double>(0, 0) = 1;
                affine.at<double>(1, 1) = 1;
                affine.at<double>(0, 2) = -rect.x;
                affine.at<double>(1, 2) = -rect.y;

                cv::Mat image_part;
                cv::warpAffine(images_list.at(matched[i].first), image_part, affine, cv::Rect(rect).size(), cv::INTER_LINEAR);

                for (size_t j = 5; j < 17; j++)
                {
                    const auto pt = std::get<0>(pose_joints_list[matched[i].first][matched[i].second][j]);
                    const auto score = std::get<1>(pose_joints_list[matched[i].first][matched[i].second][j]);
                    if (score > 0.7f)
                    {
                        cv::circle(image_part, cv::Point(pt.x - rect.x, pt.y - rect.y), 3, cv::Scalar(255, 0, 0), cv::FILLED);
                        cv::putText(image_part, std::to_string(score), cv::Point(pt.x - rect.x + 5, pt.y - rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
                    }
                }

                cv::imwrite("image_" + std::to_string(count) + "_" + std::to_string(matched[i].first) + ".jpg", image_part);
            }
            count++;
#endif

            for (size_t j = 5; j < 17; j++)
            {
                std::vector<glm::vec2> pts;
                std::vector<camera_data> cams;
                for (std::size_t i = 0; i < matched.size(); i++)
                {
                    const auto pt = std::get<0>(pose_joints_list[matched[i].first][matched[i].second][j]);
                    const auto score = std::get<1>(pose_joints_list[matched[i].first][matched[i].second][j]);
                    if (score > 0.7f)
                    {
                        pts.push_back(glm::vec2(pt.x, pt.y));
                        cams.push_back(cameras_list[matched[i].first]);
                    }
                }
                if (pts.size() >= 2)
                {
                    const auto marker = triangulate(pts, cams);
                    markers.push_back(marker);
                }
            }
        }
        return markers;
    }

    static void load_model(std::string model_path, std::vector<uint8_t> &data)
    {
        std::ifstream ifs;
        ifs.open(model_path, std::ios_base::in | std::ios_base::binary);
        if (ifs.fail())
        {
            spdlog::error("File open error: %s", model_path);
            std::quick_exit(0);
        }

        ifs.seekg(0, std::ios::end);
        const auto length = ifs.tellg();
        ifs.seekg(0, std::ios::beg);

        data.resize(length);

        ifs.read((char *)data.data(), length);
        if (ifs.fail())
        {
            spdlog::error("File read error: %s", model_path);
            std::quick_exit(0);
        }
    }

    mvpose::mvpose()
    {
        std::vector<uint8_t> det_model_data;
        {
            const auto model_path = "../data/mvpose/rtmdet_m_640-8xb32_coco-person_infer.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            det_model_data = std::move(data);
        }

        inference_det.reset(new dnn_inference_det(det_model_data, "./rtmdet_model_cache"));

        std::vector<uint8_t> pose_model_data;
        {
            const auto model_path = "../data/mvpose/rtmpose-l_8xb32-270e_coco-wholebody-384x288.onnx";
            std::vector<uint8_t> data;
            load_model(model_path, data);

            pose_model_data = std::move(data);
        }

        inference_pose.reset(new dnn_inference_pose(pose_model_data, 1, "./rtmpose_model_cache"));

        matcher.reset(new mvpose_matcher());
    }

    mvpose::~mvpose() = default;
}