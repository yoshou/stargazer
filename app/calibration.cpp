#include "calibration.hpp"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <spdlog/spdlog.h>

#include <unordered_map>
#include <unordered_set>
#include <random>

#include "camera_info.hpp"
#include "utils.hpp"
#include "graph_proc.h"
#include "graph_proc_cv.h"
#include "graph_proc_img.h"
#include "glm_serialize.hpp"
#include "triangulation.hpp"

class three_point_bar_calibration_target : public calibration_target
{
    template <typename T>
    static void sort(T &a, T &b, T &c)
    {
        if (a > b)
            std::swap(a, b);
        if (b > c)
            std::swap(b, c);
        if (a > b)
            std::swap(a, b);
    }

public:
    virtual std::vector<glm::vec2> detect_points(const std::vector<stargazer::point_data> &markers) override
    {
        std::vector<glm::vec2> points;
        combination(markers, 3, [&](const std::vector<stargazer::point_data> &target_markers)
                    {  
            auto x1 = target_markers[0].point.x;
            auto y1 = target_markers[0].point.y;
            auto x2 = target_markers[1].point.x;
            auto y2 = target_markers[1].point.y;
            auto x3 = target_markers[2].point.x;
            auto y3 = target_markers[2].point.y;

            sort(x1, x2, x3);
            sort(y1, y2, y3);

            if (std::abs(x2 - x1) > std::abs(x2 - x3))
            {
                std::swap(x1, x3);
            }
            if (std::abs(y2 - y1) > std::abs(y2 - y3))
            {
                std::swap(y1, y3);
            }

            const auto la = (y3 - y1) / (x3 - x1);
            const auto lb = -1;
            const auto lc = y1 - la * x1;

            const auto d = std::abs(la * x2 + lb * y2 + lc) / std::sqrt(la * la + lb * lb);

            if (d < 1.0)
            {
                points.push_back(glm::vec2(x2, y2));
                return true;
            }

            return false; });

        return points;
    }
};

class pattern_board_calibration_target : public calibration_target
{
    std::vector<cv::Point3f> object_points;
    stargazer::camera_t camera;

public:
    pattern_board_calibration_target(const std::vector<cv::Point3f>& object_points, const stargazer::camera_t& camera)
        : object_points(object_points), camera(camera)
    {
    }

    virtual std::vector<glm::vec2> detect_points(const std::vector<stargazer::point_data> &markers) override
    {
        if (markers.size() == object_points.size())
        {
            std::vector<cv::Point2f> image_points;

            std::transform(markers.begin(), markers.end(), std::back_inserter(image_points), [](const auto &pt)
                           { return cv::Point2f(pt.point.x, pt.point.y); });

            cv::Mat rvec, tvec;

            cv::Mat camera_matrix;
            cv::Mat dist_coeffs;
            stargazer::get_cv_intrinsic(camera.intrin, camera_matrix, dist_coeffs);

            cv::solvePnP(object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec);

            std::vector<cv::Point2f> proj_points;
            cv::projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs, proj_points);

            double error = 0.0;
            for (size_t i = 0; i < object_points.size(); i++)
            {
                error += cv::norm(image_points[i] - proj_points[i]);
            }

            if (error / object_points.size() > 4.0)
            {
                return {};
            }

            return {glm::vec2(proj_points[0].x, proj_points[0].y)};
        }
        return {};
    }
};

using namespace coalsack;

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

using float2_list_message = frame_message<std::vector<float2>>;

CEREAL_REGISTER_TYPE(float2_list_message)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::frame_message_base, float2_list_message)

CEREAL_REGISTER_TYPE(frame_message<object_message>)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::frame_message_base, frame_message<object_message>)

class camera_message : public graph_message
{
    stargazer::camera_t camera;

public:
    camera_message() : graph_message(), camera()
    {
    }

    camera_message(const stargazer::camera_t &camera) : graph_message(), camera(camera)
    {
    }

    static std::string get_type()
    {
        return "camera";
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
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_message, camera_message)

static void zip_points(const std::vector<observed_points_t> &points1, const std::vector<observed_points_t> &points2,
                        std::vector<std::pair<glm::vec2, glm::vec2>> &corresponding_points)
{
    const auto size = std::min(points1.size(), points2.size());
    for (size_t i = 0; i < size; i++)
    {
        if (points1[i].points.size() != points2[i].points.size())
        {
            continue;
        }
        for (size_t j = 0; j < points1[i].points.size(); j++)
        {
            corresponding_points.emplace_back(points1[i].points[j], points2[i].points[j]);
        }
    }
}

static void zip_points(const std::vector<observed_points_t> &points1, const std::vector<observed_points_t> &points2, const std::vector<observed_points_t> &points3,
                       std::vector<std::tuple<glm::vec2, glm::vec2, glm::vec2>> &corresponding_points)
{
    const auto size = std::min(points1.size(), std::min(points2.size(), points3.size()));
    for (size_t i = 0; i < size; i++)
    {
        if (points1[i].points.size() == 0 || points2[i].points.size() == 0 || points3[i].points.size() == 0)
        {
            continue;
        }
        assert(points1[i].points.size() == points2[i].points.size());
        assert(points1[i].points.size() == points3[i].points.size());
        for (size_t j = 0; j < points1[i].points.size(); j++)
        {
            corresponding_points.emplace_back(points1[i].points[j], points2[i].points[j], points3[i].points[j]);
        }
    }
}

static float compute_diff_camera_angle(const glm::mat3 &r1, const glm::mat3 &r2)
{
    const auto r = glm::transpose(r1) * r2;
    const auto r_quat = glm::quat_cast(r);
    return glm::angle(r_quat);
}

static glm::mat4 estimate_relative_pose(const std::vector<std::pair<glm::vec2, glm::vec2>> &corresponding_points,
                                        const stargazer::camera_t &base_camera, const stargazer::camera_t &target_camera, bool use_lmeds = false)
{
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    for (const auto &[point1, point2] : corresponding_points)
    {
        points1.emplace_back(point1.x, point1.y);
        points2.emplace_back(point2.x, point2.y);
    }

    cv::Mat camera_matrix1, camera_matrix2;
    cv::Mat coeffs1, coeffs2;
    stargazer::get_cv_intrinsic(base_camera.intrin, camera_matrix1, coeffs1);
    stargazer::get_cv_intrinsic(target_camera.intrin, camera_matrix2, coeffs2);

    cv::Mat R, t;
    if (use_lmeds)
    {
        cv::Mat E;
        cv::recoverPose(points1, points2, camera_matrix1, coeffs1, camera_matrix2, coeffs2, E, R, t, cv::LMEDS);
    }
    else
    {
        std::vector<cv::Point2f> norm_points1;
        std::vector<cv::Point2f> norm_points2;
        cv::undistortPoints(points1, norm_points1, camera_matrix1, coeffs1);
        cv::undistortPoints(points2, norm_points2, camera_matrix2, coeffs2);

        cv::Mat mask;
        const auto E = cv::findEssentialMat(norm_points1, norm_points2, 1.0, cv::Point2d(0.0, 0.0), cv::RANSAC, 0.99, 0.003, mask);

        cv::recoverPose(E, norm_points1, norm_points2, R, t, 1.0, cv::Point2d(0.0, 0.0), mask);
    }

    const auto r_mat = stargazer::cv_to_glm_mat3x3(R);
    const auto t_vec = stargazer::cv_to_glm_vec3(t);

    return glm::mat4(
        glm::vec4(r_mat[0], 0.f),
        glm::vec4(r_mat[1], 0.f),
        glm::vec4(r_mat[2], 0.f),
        glm::vec4(t_vec, 1.f));
}

static glm::mat4 estimate_pose(const std::vector<std::tuple<glm::vec2, glm::vec2, glm::vec2>> &corresponding_points,
                               const stargazer::camera_t &base_camera1, const stargazer::camera_t &base_camera2, const stargazer::camera_t &target_camera)
{
    std::vector<cv::Point2d> points1;
    std::vector<cv::Point2d> points2;
    std::vector<cv::Point2d> points3;

    cv::Mat camera_matrix1, camera_matrix2, camera_matrix3;
    cv::Mat coeffs1, coeffs2, coeffs3;
    stargazer::get_cv_intrinsic(base_camera1.intrin, camera_matrix1, coeffs1);
    stargazer::get_cv_intrinsic(base_camera2.intrin, camera_matrix2, coeffs2);
    stargazer::get_cv_intrinsic(target_camera.intrin, camera_matrix3, coeffs3);

    std::vector<cv::Point2d> norm_points1;
    std::vector<cv::Point2d> norm_points2;
    std::vector<cv::Point2d> norm_points3;
    for (const auto &[point1, point2, point3] : corresponding_points)
    {
        points1.emplace_back(point1.x, point1.y);
        points2.emplace_back(point2.x, point2.y);
        points3.emplace_back(point3.x, point3.y);
    }
    cv::undistortPoints(points1, norm_points1, camera_matrix1, coeffs1);
    cv::undistortPoints(points2, norm_points2, camera_matrix2, coeffs2);
    cv::undistortPoints(points3, norm_points3, camera_matrix3, coeffs3);

    cv::Mat point4d;
    cv::triangulatePoints(stargazer::glm_to_cv_mat3x4(base_camera1.extrin.rotation),
                          stargazer::glm_to_cv_mat3x4(base_camera2.extrin.rotation), norm_points1, norm_points2, point4d);

    std::vector<cv::Point3d> point3d;
    for (size_t i = 0; i < static_cast<size_t>(point4d.cols); i++)
    {
        point3d.emplace_back(
            point4d.at<double>(0, i) / point4d.at<double>(3, i),
            point4d.at<double>(1, i) / point4d.at<double>(3, i),
            point4d.at<double>(2, i) / point4d.at<double>(3, i));
    }

    // Check reprojection error
    {
        const auto Rt1 = stargazer::glm_to_cv_mat3x4(base_camera1.extrin.rotation);
        const auto Rt2 = stargazer::glm_to_cv_mat3x4(base_camera2.extrin.rotation);
        
        cv::Mat R1 = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat tvec1 = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat rvec1;
        for (size_t i = 0; i < 3; i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                R1.at<double>(i, j) = Rt1.at<double>(i, j);
            }
        }
        cv::Rodrigues(R1, rvec1);
        tvec1.at<double>(0, 0) = Rt1.at<double>(0, 3);
        tvec1.at<double>(1, 0) = Rt1.at<double>(1, 3);
        tvec1.at<double>(2, 0) = Rt1.at<double>(2, 3);
        
        cv::Mat R2 = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat tvec2 = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat rvec2;
        for (size_t i = 0; i < 3; i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                R2.at<double>(i, j) = Rt2.at<double>(i, j);
            }
        }
        cv::Rodrigues(R2, rvec2);
        tvec2.at<double>(0, 0) = Rt2.at<double>(0, 3);
        tvec2.at<double>(1, 0) = Rt2.at<double>(1, 3);
        tvec2.at<double>(2, 0) = Rt2.at<double>(2, 3);

        std::vector<cv::Point2d> proj_points1;
        std::vector<cv::Point2d> proj_points2;
        cv::projectPoints(point3d, rvec1, tvec1, camera_matrix1, coeffs1, proj_points1);
        cv::projectPoints(point3d, rvec2, tvec2, camera_matrix2, coeffs2, proj_points2);

        double error1 = 0.0;
        double error2 = 0.0;
        for (size_t i = 0; i < point3d.size(); i++)
        {
            error1 += cv::norm(points1[i] - proj_points1[i]);
            error2 += cv::norm(points2[i] - proj_points2[i]);
        }

        error1 /= point3d.size();
        error2 /= point3d.size();

        spdlog::info("reprojection error1: {}", error1);
        spdlog::info("reprojection error2: {}", error2);
    }

    cv::Mat r, t;
    cv::solvePnP(point3d, norm_points3, cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(1, 5, CV_64F), r, t);

    cv::Mat R;
    cv::Rodrigues(r, R);

    // Check reprojection error
    {
        std::vector<cv::Point2d> proj_points3;
        cv::projectPoints(point3d, r, t, camera_matrix3, coeffs3, proj_points3);

        double error3 = 0.0;
        for (size_t i = 0; i < point3d.size(); i++)
        {
            error3 += cv::norm(points3[i] - proj_points3[i]);
        }

        error3 /= point3d.size();

        spdlog::info("reprojection error3: {}", error3);
    }

    const auto r_mat = stargazer::cv_to_glm_mat3x3(R);
    const auto t_vec = stargazer::cv_to_glm_vec3(t);

    return glm::mat4(
        glm::vec4(r_mat[0], 0.f),
        glm::vec4(r_mat[1], 0.f),
        glm::vec4(r_mat[2], 0.f),
        glm::vec4(t_vec, 1.f));
}

static constexpr auto num_parameters = 18;

template <bool is_radial_distortion = false>
struct reprojection_error_functor
{
    reprojection_error_functor(double observed_x, double observed_y)
        : observed_x(observed_x), observed_y(observed_y) {}

    template <typename T>
    bool operator()(const T *const camera,
                    const T *const point,
                    T *residuals) const
    {
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);

        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        T predicted_x;
        T predicted_y;

        const T &fx = camera[6];
        const T &fy = camera[7];
        const T &cx = camera[8];
        const T &cy = camera[9];
        const T r2 = xp * xp + yp * yp;

        const T &k1 = camera[10];
        const T &k2 = camera[11];
        const T &k3 = camera[12];
        const T &p1 = camera[13];
        const T &p2 = camera[14];

        if constexpr (is_radial_distortion)
        {
            const T &k4 = camera[15];
            const T &k5 = camera[16];
            const T &k6 = camera[17];

            const T distortion = (1.0 + (k1 + (k2 + k3 * r2) * r2) * r2) / (1.0 + (k4 + (k5 + k6 * r2) * r2) * r2);

            predicted_x = fx * (distortion * xp + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp)) + cx;
            predicted_y = fy * (distortion * yp + 2.0 * p2 * xp * yp + p1 * (r2 + 2.0 * yp * yp)) + cy;
        }
        else
        {
            const T distortion = (1.0 + (k1 + (k2 + k3 * r2) * r2) * r2);

            predicted_x = fx * (distortion * xp + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp)) + cx;
            predicted_y = fy * (distortion * yp + 2.0 * p2 * xp * yp + p1 * (r2 + 2.0 * yp * yp)) + cy;
        }

        const T err_x = predicted_x - observed_x;
        const T err_y = predicted_y - observed_y;
        residuals[0] = err_x;
        residuals[1] = err_y;

        return true;
    }

    static ceres::CostFunction *create(const double observed_x,
                                       const double observed_y)
    {
        return (new ceres::AutoDiffCostFunction<reprojection_error_functor, 2, num_parameters, 3>(
            new reprojection_error_functor(observed_x, observed_y)));
    }

    double observed_x;
    double observed_y;
};

class pattern_board_calibration_target_detector_node : public graph_node
{
    stargazer::camera_t camera;
    std::unique_ptr<pattern_board_calibration_target> detector;
    graph_edge_ptr output;

    std::vector<cv::Point3f> get_object_points()
    {
        std::vector<cv::Point3f> object_points;
        calc_board_corner_positions(cv::Size(2, 9), cv::Size2f(1.0f, 1.0f), object_points, calibration_pattern::ASYMMETRIC_CIRCLES_GRID);
        return object_points;
    }

public:
    pattern_board_calibration_target_detector_node()
        : graph_node(), detector(), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    virtual std::string get_proc_name() const override
    {
        return "pattern_board_calibration_target_detector_node";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(camera);
    }

    virtual void run() override
    {
        detector = std::make_unique<pattern_board_calibration_target>(get_object_points(), camera);
    }

    void set_camera(const stargazer::camera_t &camera)
    {
        this->camera = camera;
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (const auto frame_msg = std::dynamic_pointer_cast<float2_list_message>(message))
        {
            if (detector)
            {
                std::vector<stargazer::point_data> markers;
                for (const auto &pt : frame_msg->get_data())
                {
                    markers.push_back({{pt.x, pt.y}, 0.0, 0.0});
                }

                const auto points = detector->detect_points(markers);

                std::vector<float2> float2_data;
                for (const auto &pt : points)
                {
                    float2_data.push_back({pt.x, pt.y});
                }

                auto msg = std::make_shared<float2_list_message>();
                msg->set_data(float2_data);
                msg->set_frame_number(std::dynamic_pointer_cast<frame_message_base>(message)->get_frame_number());
                msg->set_timestamp(std::dynamic_pointer_cast<frame_message_base>(message)->get_timestamp());

                output->send(msg);
            }
        }
    }
};

CEREAL_REGISTER_TYPE(pattern_board_calibration_target_detector_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, pattern_board_calibration_target_detector_node)

class three_point_bar_calibration_target_detector_node : public graph_node
{
    std::unique_ptr<three_point_bar_calibration_target> detector;
    graph_edge_ptr output;

public:
    three_point_bar_calibration_target_detector_node()
        : graph_node(), detector(), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    virtual std::string get_proc_name() const override
    {
        return "three_point_bar_calibration_target_detector_node";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
    }

    virtual void run() override
    {
        detector = std::make_unique<three_point_bar_calibration_target>();
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (const auto frame_msg = std::dynamic_pointer_cast<float2_list_message>(message))
        {
            if (detector)
            {
                std::vector<stargazer::point_data> markers;
                for (const auto &pt : frame_msg->get_data())
                {
                    markers.push_back({{pt.x, pt.y}, 0.0, 0.0});
                }

                const auto points = detector->detect_points(markers);

                std::vector<float2> float2_data;
                for (const auto &pt : points)
                {
                    float2_data.push_back({pt.x, pt.y});
                }

                auto msg = std::make_shared<float2_list_message>();
                msg->set_data(float2_data);
                msg->set_frame_number(std::dynamic_pointer_cast<frame_message_base>(message)->get_frame_number());
                msg->set_timestamp(std::dynamic_pointer_cast<frame_message_base>(message)->get_timestamp());

                output->send(msg);
            }
        }
    }
};

CEREAL_REGISTER_TYPE(three_point_bar_calibration_target_detector_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, three_point_bar_calibration_target_detector_node)

class calibration_node : public graph_node
{
    mutable std::mutex frames_mtx;
    std::unordered_map<uint32_t, uint32_t> timestamp_to_index;
    std::map<std::string, size_t> camera_name_to_index;
    std::map<std::string, std::vector<observed_points_t>> observed_frames;
    std::map<std::string, size_t> num_frames;

    bool only_extrinsic;
    bool robust;

    mutable std::mutex cameras_mtx;
    std::vector<std::string> camera_names;
    std::unordered_map<std::string, stargazer::camera_t> cameras;
    std::unordered_map<std::string, stargazer::camera_t> calibrated_cameras;

    graph_edge_ptr output;

public:
    calibration_node()
        : graph_node(), only_extrinsic(true), robust(false), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    virtual std::string get_proc_name() const override
    {
        return "calibration_node";
    }

    void set_cameras(const std::unordered_map<std::string, stargazer::camera_t> &cameras)
    {
        this->cameras = cameras;
    }

    void set_only_extrinsic(bool only_extrinsic)
    {
        this->only_extrinsic = only_extrinsic;
    }

    void set_robust(bool robust)
    {
        this->robust = robust;
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(cameras, only_extrinsic, robust);
    }

    size_t get_num_frames(std::string name) const
    {
        if (num_frames.find(name) == num_frames.end())
        {
            return 0;
        }
        return num_frames.at(name);
    }

    const std::vector<observed_points_t> get_observed_points(std::string name) const
    {
        static std::vector<observed_points_t> empty;
        if (observed_frames.find(name) == observed_frames.end())
        {
            return empty;
        }
        std::vector<observed_points_t> observed_points;
        {
            std::lock_guard<std::mutex> lock(frames_mtx);
            observed_points = observed_frames.at(name);
        }
        return observed_points;
    }

    void calibrate()
    {
        {
            std::string base_camera_name1;
            std::string base_camera_name2;
            glm::mat4 base_camera_pose1;
            glm::mat4 base_camera_pose2;

            bool found_base_pair = false;
            const float min_base_angle = 15.0f;

            for (const auto &camera_name1 : camera_names)
            {
                if (found_base_pair)
                {
                    break;
                }
                for (const auto &camera_name2 : camera_names)
                {
                    if (camera_name1 == camera_name2)
                    {
                        continue;
                    }
                    if (found_base_pair)
                    {
                        break;
                    }

                    std::vector<std::pair<glm::vec2, glm::vec2>> corresponding_points;
                    zip_points(observed_frames.at(camera_name1), observed_frames.at(camera_name2), corresponding_points);

                    const auto pose1 = glm::mat4(1.0);
                    const auto pose2 = estimate_relative_pose(corresponding_points, cameras.at(camera_name1), cameras.at(camera_name2));

                    const auto angle = compute_diff_camera_angle(glm::mat3(pose1), glm::mat3(pose2));

                    if (glm::degrees(angle) > min_base_angle)
                    {
                        base_camera_name1 = camera_name1;
                        base_camera_name2 = camera_name2;
                        base_camera_pose1 = pose1;
                        base_camera_pose2 = pose2;
                        found_base_pair = true;
                        break;
                    }
                }
            }

            cameras[base_camera_name1].extrin.rotation = base_camera_pose1;
            cameras[base_camera_name1].extrin.translation = glm::vec3(base_camera_pose1[3]);
            cameras[base_camera_name2].extrin.rotation = base_camera_pose2;
            cameras[base_camera_name2].extrin.translation = glm::vec3(base_camera_pose2[3]);

            std::vector<std::string> processed_cameras = {base_camera_name1, base_camera_name2};
            for (const auto &camera_name : camera_names)
            {
                if (camera_name == base_camera_name1)
                {
                    continue;
                }
                if (camera_name == base_camera_name2)
                {
                    continue;
                }

                std::vector<std::tuple<glm::vec2, glm::vec2, glm::vec2>> corresponding_points;
                zip_points(observed_frames.at(base_camera_name1), observed_frames.at(base_camera_name2), observed_frames.at(camera_name), corresponding_points);

                if (corresponding_points.size() < 7)
                {
                    continue;
                }

                const auto pose = estimate_pose(corresponding_points, cameras.at(base_camera_name1), cameras.at(base_camera_name2), cameras.at(camera_name));
                cameras[camera_name].extrin.rotation = pose;
                cameras[camera_name].extrin.translation = glm::vec3(pose[3]);

                processed_cameras.push_back(camera_name);
            }

            assert(processed_cameras.size() == camera_names.size());
        }

        stargazer::calibration::bundle_adjust_data ba_data;

        for (const auto &camera_name : camera_names)
        {
            const auto &camera = cameras[camera_name];
            cv::Mat rot_vec;
            cv::Rodrigues(stargazer::glm_to_cv_mat3(camera.extrin.rotation), rot_vec);
            const auto trans_vec = camera.extrin.translation;

            std::vector<double> camera_params;
            for (size_t j = 0; j < 3; j++)
            {
                camera_params.push_back(rot_vec.at<float>(j));
            }
            for (size_t j = 0; j < 3; j++)
            {
                camera_params.push_back(trans_vec[j]);
            }
            camera_params.push_back(camera.intrin.fx);
            camera_params.push_back(camera.intrin.fy);
            camera_params.push_back(camera.intrin.cx);
            camera_params.push_back(camera.intrin.cy);
            camera_params.push_back(camera.intrin.coeffs[0]);
            camera_params.push_back(camera.intrin.coeffs[1]);
            camera_params.push_back(camera.intrin.coeffs[4]);
            camera_params.push_back(camera.intrin.coeffs[2]);
            camera_params.push_back(camera.intrin.coeffs[3]);
            for (size_t j = 0; j < 3; j++)
            {
                camera_params.push_back(0.0);
            }

            ba_data.add_camera(camera_params.data());
        }

        {
            std::vector<stargazer::camera_t> camera_list;
            std::map<std::string, size_t> camera_name_to_index;
            for (size_t i = 0; i < camera_names.size(); i++)
            {
                camera_list.push_back(cameras.at(camera_names[i]));
                camera_name_to_index.insert(std::make_pair(camera_names[i], i));
            }
            size_t point_idx = 0;
            for (size_t f = 0; f < observed_frames.begin()->second.size(); f++)
            {
                std::vector<std::vector<glm::vec2>> pts;
                std::vector<size_t> camera_idxs;
                for (const auto &camera_name : camera_names)
                {
                    const auto &point = observed_frames.at(camera_name).at(f);
                    if (point.points.size() == 0)
                    {
                        continue;
                    }
                    std::vector<glm::vec2> pt;
                    std::copy(point.points.begin(), point.points.end(), std::back_inserter(pt));

                    pts.push_back(pt);
                    camera_idxs.push_back(camera_name_to_index.at(camera_name));
                }

                if (pts.size() < 2)
                {
                    continue;
                }

                std::vector<stargazer::camera_t> view_cameras;
                for (const auto i : camera_idxs)
                {
                    view_cameras.push_back(camera_list[i]);
                }

                const auto num_points = std::min_element(pts.begin(), pts.end(), [](const auto &a, const auto &b)
                                                         { return a.size() < b.size(); })
                                            ->size();

                std::vector<glm::vec3> point3ds(num_points);

                for (size_t i = 0; i < num_points; i++)
                {
                    std::vector<glm::vec2> point2ds;
                    for (size_t j = 0; j < pts.size(); j++)
                    {
                        point2ds.push_back(pts[j][i]);
                    }

                    const auto point3d = stargazer::reconstruction::triangulate(point2ds, view_cameras);
                    point3ds[i] = point3d;
                }

                for (const auto &point3d : point3ds)
                {
                    std::vector<double> point_params;
                    for (size_t i = 0; i < 3; i++)
                    {
                        point_params.push_back(point3d[i]);
                    }
                    ba_data.add_point(point_params.data());
                }

                for (size_t i = 0; i < pts.size(); i++)
                {
                    for (size_t j = 0; j < pts[i].size(); j++)
                    {
                        const auto &pt = pts[i][j];
                        std::array<double, 2> observation;
                        for (size_t k = 0; k < 2; k++)
                        {
                            observation[k] = pt[k];
                        }
                        ba_data.add_observation(observation.data(), camera_idxs[i], point_idx + j);
                    }
                }

                point_idx += point3ds.size();
            }
        }

        {
            const double *observations = ba_data.observations();
            ceres::Problem problem;

            ceres::SubsetManifold *constant_params_manifold = nullptr;
            if (only_extrinsic)
            {
                std::vector<int> constant_params;

                for (int i = 6; i < num_parameters; i++)
                {
                    constant_params.push_back(i);
                }

                constant_params_manifold =
                    new ceres::SubsetManifold(num_parameters, constant_params);
            }

            for (int i = 0; i < ba_data.num_observations(); ++i)
            {
                ceres::LossFunction *loss_function = robust ? new ceres::HuberLoss(1.0) : nullptr;
                
                ceres::CostFunction *cost_function =
                    reprojection_error_functor<false>::create(observations[2 * i + 0],
                                                                                        observations[2 * i + 1]);
                problem.AddResidualBlock(cost_function,
                                         loss_function,
                                         ba_data.mutable_camera_for_observation(i),
                                         ba_data.mutable_point_for_observation(i));

                if (only_extrinsic)
                {
                    problem.SetManifold(ba_data.mutable_camera_for_observation(i), constant_params_manifold);
                }
            }

            std::cout << "Num cameras: " << ba_data.num_cameras() << std::endl;
            std::cout << "Num points: " << ba_data.num_points() << std::endl;
            std::cout << "Num Observations: " << ba_data.num_observations() << std::endl;

            ceres::Solver::Options options;
            options.use_nonmonotonic_steps = true;
            options.preconditioner_type = ceres::SCHUR_JACOBI;
            options.linear_solver_type = ceres::DENSE_SCHUR;
            options.use_inner_iterations = true;
            options.max_num_iterations = 100;
            options.minimizer_progress_to_stdout = true;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            std::cout << summary.FullReport() << "\n";

            calibrated_cameras = cameras;

            {
                for (size_t i = 0; i < camera_names.size(); i++)
                {
                    calibrated_cameras[camera_names[i]].extrin.rotation = ba_data.get_camera_extrinsic(i);
                    calibrated_cameras[camera_names[i]].extrin.translation = ba_data.get_camera_extrinsic(i)[3];
                }
            }

            if (!only_extrinsic)
            {
                for (size_t i = 0; i < camera_names.size(); i++)
                {
                    const auto intrin = &ba_data.mutable_camera(i)[6];
                    calibrated_cameras[camera_names[i]].intrin.fx = intrin[0];
                    calibrated_cameras[camera_names[i]].intrin.fy = intrin[1];
                    calibrated_cameras[camera_names[i]].intrin.cx = intrin[2];
                    calibrated_cameras[camera_names[i]].intrin.cy = intrin[3];
                    calibrated_cameras[camera_names[i]].intrin.coeffs[0] = intrin[4];
                    calibrated_cameras[camera_names[i]].intrin.coeffs[1] = intrin[5];
                    calibrated_cameras[camera_names[i]].intrin.coeffs[4] = intrin[6];
                    calibrated_cameras[camera_names[i]].intrin.coeffs[2] = intrin[7];
                    calibrated_cameras[camera_names[i]].intrin.coeffs[3] = intrin[8];
                }
            }
        }
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (input_name == "calibrate")
        {
            camera_names.clear();

            if (auto camera_msg = std::dynamic_pointer_cast<object_message>(message))
            {
                for (const auto &[name, field] : camera_msg->get_fields())
                {
                    if (auto camera_msg = std::dynamic_pointer_cast<camera_message>(field))
                    {
                        std::lock_guard lock(cameras_mtx);
                        cameras[name] = camera_msg->get_camera();
                        camera_names.push_back(name);
                    }
                }
            }

            calibrate();

            std::shared_ptr<object_message> msg(new object_message());
            for (const auto &[name, camera] : calibrated_cameras)
            {
                std::shared_ptr<camera_message> camera_msg(new camera_message(camera));
                msg->add_field(name, camera_msg);
            }
            output->send(msg);

            return;
        }

        if (auto points_msg = std::dynamic_pointer_cast<float2_list_message>(message))
        {
            if (timestamp_to_index.find(points_msg->get_frame_number()) == timestamp_to_index.end())
            {
                timestamp_to_index.insert(std::make_pair(points_msg->get_frame_number(), timestamp_to_index.size()));
            }

            const auto index = timestamp_to_index.at(points_msg->get_frame_number());
            const auto &name = input_name;

            if (num_frames.find(name) == num_frames.end())
            {
                num_frames.insert(std::make_pair(name, 0));
            }

            if (camera_name_to_index.find(name) == camera_name_to_index.end())
            {
                camera_name_to_index.insert(std::make_pair(name, camera_name_to_index.size()));
            }

            if (observed_frames.find(name) == observed_frames.end())
            {
                if (observed_frames.empty())
                {
                    std::lock_guard lock(frames_mtx);
                    observed_frames.insert(std::make_pair(name, std::vector<observed_points_t>()));
                }
                else
                {
                    observed_points_t obs = {};
                    obs.camera_idx = camera_name_to_index.at(name);
                    std::lock_guard lock(frames_mtx);
                    observed_frames.insert(std::make_pair(name, std::vector<observed_points_t>(observed_frames.begin()->second.size(), obs)));
                }
            }

            for (auto &[name, observed_points] : observed_frames)
            {
                observed_points.resize(timestamp_to_index.size());
            }

            observed_points_t obs = {};
            obs.camera_idx = camera_name_to_index.at(name);
            for (const auto &pt : points_msg->get_data())
            {
                obs.points.emplace_back(pt.x, pt.y);
            }

            {
                std::lock_guard lock(frames_mtx);
                observed_frames[name][index] = obs;
            }

            if (obs.points.size() > 0)
            {
                num_frames.at(name) += 1;
            }
        }
    }
};

CEREAL_REGISTER_TYPE(calibration_node);
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, calibration_node);

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
        if (auto msg = std::dynamic_pointer_cast<object_message>(message))
        {
            for (const auto &[name, field] : msg->get_fields())
            {
                if (auto frame_msg = std::dynamic_pointer_cast<frame_message_base>(field))
                {
                    frame_msg->set_frame_number(frame_number);
                }
            }
            frame_number++;
            output->send(msg);
        }
    }
};

CEREAL_REGISTER_TYPE(frame_number_numbering_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, frame_number_numbering_node)

class object_map_node : public graph_node
{
public:
    object_map_node()
        : graph_node()
    {
    }

    virtual std::string get_proc_name() const override
    {
        return "object_map_node";
    }

    template <typename Archive>
    void save(Archive &archive) const
    {
        std::vector<std::string> output_names;
        auto outputs = get_outputs();
        for (auto output : outputs)
        {
            output_names.push_back(output.first);
        }
        archive(output_names);
    }

    template <typename Archive>
    void load(Archive &archive)
    {
        std::vector<std::string> output_names;
        archive(output_names);
        for (auto output_name : output_names)
        {
            set_output(std::make_shared<graph_edge>(this), output_name);
        }
    }

    graph_edge_ptr add_output(const std::string &name)
    {
        auto outputs = get_outputs();
        auto it = outputs.find(name);
        if (it == outputs.end())
        {
            auto output = std::make_shared<graph_edge>(this);
            set_output(output, name);
            return output;
        }
        return it->second;
    }

    virtual void process(std::string input_name, graph_message_ptr message) override
    {
        if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message))
        {
            for (const auto& [name, field] : obj_msg->get_fields())
            {
                if (auto frame_msg = std::dynamic_pointer_cast<frame_message_base>(field))
                {
                    try
                    {
                        const auto output = get_output(name);
                        output->send(field);
                    }
                    catch (const std::exception &e)
                    {
                        spdlog::error(e.what());
                    }
                }
            }
        }
    }
};

CEREAL_REGISTER_TYPE(object_map_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_node, object_map_node)

class calibration::impl
{
public:
    graph_proc graph;

    std::atomic_bool running;

    std::shared_ptr<calibration_node> calib_node;
    std::shared_ptr<graph_node> input_node;

    std::unordered_map<std::string, stargazer::camera_t> cameras;
    std::unordered_map<std::string, stargazer::camera_t> calibrated_cameras;

    std::vector<std::function<void(const std::unordered_map<std::string, stargazer::camera_t> &)>> calibrated;

    void add_calibrated(std::function<void(const std::unordered_map<std::string, stargazer::camera_t> &)> callback)
    {
        calibrated.push_back(callback);
    }

    void clear_calibrated()
    {
        calibrated.clear();
    }

    void push_frame(const std::map<std::string, std::vector<stargazer::point_data>> &frame)
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

        if (input_node)
        {
            graph.process(input_node.get(), msg);
        }
    }

    void calibrate(const std::unordered_map<std::string, stargazer::camera_t>& cameras)
    {
        std::shared_ptr<object_message> msg(new object_message());
        for (const auto &[name, camera] : cameras)
        {
            std::shared_ptr<camera_message> camera_msg(new camera_message(camera));
            msg->add_field(name, camera_msg);
        }
        graph.process(calib_node.get(), "calibrate", msg);
    }

    void run(const std::vector<node_info> &infos)
    {
        std::shared_ptr<subgraph> g(new subgraph());

        std::shared_ptr<frame_number_numbering_node> n4(new frame_number_numbering_node());
        g->add_node(n4);

        input_node = n4;

        std::shared_ptr<object_map_node> n5(new object_map_node());
        n5->set_input(n4->get_output());
        g->add_node(n5);

        std::unordered_map<std::string, graph_node_ptr> detector_nodes;

        for (const auto &info : infos)
        {
            if (info.type == node_type::pattern_board_calibration_target_detector)
            {
                for (const auto &[name, camera] : cameras)
                {
                    std::shared_ptr<pattern_board_calibration_target_detector_node> n1(new pattern_board_calibration_target_detector_node());
                    n1->set_input(n5->add_output(name));
                    n1->set_camera(camera);
                    g->add_node(n1);

                    detector_nodes[name] = n1;
                }
            }
            if (info.type == node_type::three_point_bar_calibration_target_detector)
            {
                for (const auto &[name, camera] : cameras)
                {
                    std::shared_ptr<three_point_bar_calibration_target_detector_node> n1(new three_point_bar_calibration_target_detector_node());
                    n1->set_input(n5->add_output(name));
                    g->add_node(n1);

                    detector_nodes[name] = n1;
                }
            }
        }

        for (const auto& info : infos)
        {
            if (info.type == node_type::calibration)
            {
                std::shared_ptr<calibration_node> n1(new calibration_node());
                for (const auto &[name, node] : detector_nodes)
                {
                    n1->set_input(node->get_output(), name);
                }
                n1->set_cameras(cameras);
                n1->set_only_extrinsic(std::get<bool>(info.params.at("only_extrinsic")));
                n1->set_robust(std::get<bool>(info.params.at("robust")));
                g->add_node(n1);

                calib_node = n1;
            }
        }

        if (calib_node == nullptr)
        {
            spdlog::error("Calibration node not found");
            return;
        }

        std::shared_ptr<callback_node> n2(new callback_node());
        n2->set_input(calib_node->get_output());
        g->add_node(n2);

        n2->set_name("cameras");

        const auto callbacks = std::make_shared<callback_list>();

        callbacks->add([this](const callback_node *node, std::string input_name, graph_message_ptr message)
                       {
            if (node->get_name() == "cameras")
            {
                if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message))
                {
                    for (const auto &[name, field] : obj_msg->get_fields())
                    {
                        if (auto camera_msg = std::dynamic_pointer_cast<camera_message>(field))
                        {
                            calibrated_cameras[name] = camera_msg->get_camera();
                        }
                    }
                    for (const auto &f : calibrated)
                    {
                        f(calibrated_cameras);
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

    size_t get_num_frames(std::string name) const
    {
        if (!calib_node)
        {
            return 0;
        }
        return calib_node->get_num_frames(name);
    }

    const std::vector<observed_points_t> get_observed_points(std::string name) const
    {
        if (!calib_node)
        {
            static std::vector<observed_points_t> empty;
            return empty;
        }
        return calib_node->get_observed_points(name);
    }
};

calibration::calibration() : pimpl(std::make_unique<impl>())
{
}

calibration::~calibration() = default;

void calibration::set_camera(const std::string &name, const stargazer::camera_t &camera)
{
    pimpl->cameras[name] = camera;
}

size_t calibration::get_camera_size() const
{
    return pimpl->cameras.size();
}

const std::unordered_map<std::string, stargazer::camera_t> &calibration::get_cameras() const
{
    return pimpl->cameras;
}

std::unordered_map<std::string, stargazer::camera_t> &calibration::get_cameras()
{
    return pimpl->cameras;
}

void calibration::run(const std::vector<node_info> &infos)
{
    pimpl->run(infos);
}

void calibration::stop()
{
    pimpl->stop();
}

void calibration::add_calibrated(std::function<void(const std::unordered_map<std::string, stargazer::camera_t> &)> callback)
{
    pimpl->add_calibrated(callback);
}

void calibration::clear_calibrated()
{
    pimpl->clear_calibrated();
}

size_t calibration::get_num_frames(std::string name) const
{
    return pimpl->get_num_frames(name);
}

const std::vector<observed_points_t> calibration::get_observed_points(std::string name) const
{
    return pimpl->get_observed_points(name);
}

const std::unordered_map<std::string, stargazer::camera_t> &calibration::get_calibrated_cameras() const
{
    return pimpl->calibrated_cameras;
}

void calibration::push_frame(const std::map<std::string, std::vector<stargazer::point_data>> &frame)
{
    pimpl->push_frame(frame);
}

void calibration::calibrate()
{
    pimpl->calibrate(pimpl->cameras);
}

void calc_board_corner_positions(cv::Size board_size, cv::Size2f square_size, std::vector<cv::Point3f> &corners, const calibration_pattern pattern_type)
{
    corners.clear();
    switch (pattern_type)
    {
    case calibration_pattern::CHESSBOARD:
    case calibration_pattern::CIRCLES_GRID:
        for (int i = 0; i < board_size.height; ++i)
        {
            for (int j = 0; j < board_size.width; ++j)
            {
                corners.push_back(cv::Point3f(j * square_size.width, i * square_size.height, 0));
            }
        }
        break;
    case calibration_pattern::ASYMMETRIC_CIRCLES_GRID:
        for (int i = 0; i < board_size.height; i++)
        {
            for (int j = 0; j < board_size.width; j++)
            {
                corners.push_back(cv::Point3f((2 * j + i % 2) * square_size.width, i * square_size.height, 0));
            }
        }
        break;
    default:
        break;
    }
}

bool detect_calibration_board(cv::Mat frame, std::vector<cv::Point2f> &points, const calibration_pattern pattern_type)
{
    if (frame.empty())
    {
        return false;
    }
    constexpr auto use_fisheye = false;

    cv::Size board_size;
    switch (pattern_type)
    {
    case calibration_pattern::CHESSBOARD:
        board_size = cv::Size(10, 7);
        break;
    case calibration_pattern::CIRCLES_GRID:
        board_size = cv::Size(10, 7);
        break;
    case calibration_pattern::ASYMMETRIC_CIRCLES_GRID:
        board_size = cv::Size(4, 11);
        break;
    }
    const auto win_size = 5;

    int chessboard_flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
    if (!use_fisheye)
    {
        chessboard_flags |= cv::CALIB_CB_FAST_CHECK;
    }

    bool found = false;
    switch (pattern_type)
    {
    case calibration_pattern::CHESSBOARD:
        found = cv::findChessboardCorners(frame, board_size, points, chessboard_flags);
        break;
    case calibration_pattern::CIRCLES_GRID:
        found = cv::findCirclesGrid(frame, board_size, points);
        break;
    case calibration_pattern::ASYMMETRIC_CIRCLES_GRID:
    {
        auto params = cv::SimpleBlobDetector::Params();
        params.minDistBetweenBlobs = 3;
        auto detector = cv::SimpleBlobDetector::create(params);
        found = cv::findCirclesGrid(frame, board_size, points, cv::CALIB_CB_ASYMMETRIC_GRID, detector);
    }
    break;
    default:
        found = false;
        break;
    }

    if (found)
    {
        if (pattern_type == calibration_pattern::CHESSBOARD)
        {
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            cv::cornerSubPix(gray, points, cv::Size(win_size, win_size),
                             cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.0001));
        }
    }

    return found;
}

#include <opencv2/aruco/charuco.hpp>

static inline cv::aruco::CharucoBoard create_charuco_board()
{
    const auto dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
    const auto board = cv::aruco::CharucoBoard(cv::Size(3, 5), 0.0575, 0.0575 * 0.75f, dictionary);
    return board;
}

static bool detect_charuco_board(cv::Mat image, std::vector<cv::Point2f> &points, std::vector<int> &ids)
{
    cv::aruco::DetectorParameters detector_params = cv::aruco::DetectorParameters();
    cv::aruco::CharucoParameters charuco_params = cv::aruco::CharucoParameters();
    const auto board = create_charuco_board();
    const auto detector = cv::aruco::CharucoDetector(board, charuco_params, detector_params);
    std::vector<int> marker_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners;
    std::vector<int> charuco_ids;
    std::vector<cv::Point2f> charuco_corners;
    detector.detectBoard(image, charuco_corners, charuco_ids, marker_corners, marker_ids);

    if (charuco_ids.size() == 0)
    {
        return false;
    }

    points = charuco_corners;
    ids = charuco_ids;

    return true;
}

static void detect_aruco_marker(cv::Mat image, std::vector<std::vector<cv::Point2f>> &points, std::vector<int> &ids)
{
    cv::aruco::DetectorParameters detector_params = cv::aruco::DetectorParameters();
    cv::aruco::RefineParameters refine_params = cv::aruco::RefineParameters();
    const auto dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
    const auto detector = cv::aruco::ArucoDetector(dictionary, detector_params, refine_params);

    points.clear();
    ids.clear();
    detector.detectMarkers(image, points, ids);
}

static bool detect_aruco_marker(cv::Mat image, std::vector<cv::Point2f> &points, std::vector<int> &ids)
{
    std::vector<int> marker_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners;
    detect_aruco_marker(image, marker_corners, marker_ids);

    for (size_t i = 0; i < marker_ids.size(); i++)
    {
        const auto marker_id = marker_ids[i];
        const auto &marker_corner = marker_corners[i];
        if (marker_id == 0)
        {
            for (size_t i = 0; i < 4; i++)
            {
                points.push_back(marker_corner[i]);
                ids.push_back(marker_id * 4 + i);
            }
            return true;
        }
    }

    return false;
}

static std::vector<size_t> create_random_indices(size_t size)
{
    std::vector<size_t> data(size);
    for (size_t i = 0; i < size; i++)
    {
        data[i] = i;
    }

    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());

    std::shuffle(data.begin(), data.end(), engine);

    return data;
}

void intrinsic_calibration::add_frame(const std::vector<stargazer::point_data> &frame)
{
    frames.push_back(frame);
}

void intrinsic_calibration::calibrate()
{
    const auto square_size = cv::Size2f(2.41, 2.4); // TODO: Define as config
    const auto board_size = cv::Size(10, 7);        // TODO: Define as config
    const auto image_size = cv::Size(image_width, image_height);

    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points;

    std::vector<cv::Point3f> object_point;
    calc_board_corner_positions(board_size, square_size, object_point);

    const auto max_num_frames = 100; // TODO: Define as config

    const auto frame_indices = create_random_indices(std::min(frames.size(), static_cast<size_t>(max_num_frames)));

    for (const auto& frame_index : frame_indices)
    {
        const auto& frame = frames.at(frame_index);

        object_points.push_back(object_point);

        std::vector<cv::Point2f> image_point;
        for (const auto& point : frame)
        {
            image_point.push_back(cv::Point2f(point.point.x, point.point.y));
        }

        image_points.push_back(image_point);
    }

    std::vector<cv::Mat> rvecs, tvecs;
    cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);

    rms = cv::calibrateCamera(object_points, image_points, image_size, camera_matrix, dist_coeffs, rvecs, tvecs);

    calibrated_camera.intrin.fx = camera_matrix.at<double>(0, 0);
    calibrated_camera.intrin.fy = camera_matrix.at<double>(1, 1);
    calibrated_camera.intrin.cx = camera_matrix.at<double>(0, 2);
    calibrated_camera.intrin.cy = camera_matrix.at<double>(1, 2);
    calibrated_camera.intrin.coeffs[0] = dist_coeffs.at<double>(0);
    calibrated_camera.intrin.coeffs[1] = dist_coeffs.at<double>(1);
    calibrated_camera.intrin.coeffs[2] = dist_coeffs.at<double>(2);
    calibrated_camera.intrin.coeffs[3] = dist_coeffs.at<double>(3);
    calibrated_camera.intrin.coeffs[4] = dist_coeffs.at<double>(4);
    calibrated_camera.width = image_width;
    calibrated_camera.height = image_height;
}

static std::vector<glm::vec2> convert_cv_to_glm_point2f(const std::vector<cv::Point2f>& points)
{
    std::vector<glm::vec2> glm_points;
    for (const auto& point : points)
    {
        glm_points.emplace_back(point.x, point.y);
    }
    return glm_points;
}


static std::vector<glm::vec3> get_target_object_points(int board_size_x, int board_size_y, float square_size_x, float square_size_y, calibration_pattern target)
{
    std::vector<cv::Point3f> corners;
    calc_board_corner_positions(cv::Size(board_size_x, board_size_y), cv::Size2f(square_size_x, square_size_y), corners, target);

    std::vector<glm::vec3> points;
    for (const auto& corner : corners)
    {
        points.emplace_back(corner.x, corner.y, corner.z);
    }
    return points;
}

static glm::mat4 compute_axis(const observed_points_t& points, const std::vector<glm::vec3>& target_points, cv::Mat camera_matrix, cv::Mat dist_coeffs)
{
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;

    std::transform(points.points.begin(), points.points.end(), std::back_inserter(image_points), [](const auto &pt)
                   { return cv::Point2f(pt.x, pt.y); });

    std::transform(target_points.begin(), target_points.end(), std::back_inserter(object_points), [](const auto &pt)
                   { return cv::Point3f(pt.x, pt.y, pt.z); });

    cv::Mat rvec, tvec;

    const auto rms = cv::solvePnP(object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec);

    glm::mat4 axis(1.0f);
    cv::Mat rmat;
    cv::Rodrigues(rvec, rmat);
    for (size_t i = 0; i < 3; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            axis[i][j] = rmat.at<double>(j, i);
        }
    }
    glm::vec3 origin(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
    axis[3] = glm::vec4(origin, 1.0f);

    return glm::inverse(axis);
}