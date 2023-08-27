#include "calibration.hpp"
#include <experimental/filesystem>

#include <glm/gtx/string_cast.hpp>
#include <opencv2/sfm/triangulation.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <unordered_map>
#include <unordered_set>
#include "utils.hpp"

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

calibration_model::calibration_model() : detector(std::make_shared<three_point_bar_calibration_target>())
{
    namespace fs = std::experimental::filesystem;
    const std::string prefix = "calibrate";
    const std::string data_dir = "../data";

    std::ifstream ifs;
    ifs.open((fs::path(data_dir) / "config.json").string(), std::ios::in);
    nlohmann::json j_config = nlohmann::json::parse(ifs);

    camera_names = j_config["cameras"].get<std::vector<std::string>>();
    const auto camera_ids = j_config["camera_ids"].get<std::vector<std::string>>();
    const auto num_cameras = camera_names.size();

    {
        const auto camera_params = stargazer::load_camera_params("../ir_camera_calib/camera_params.yml");

        for (std::size_t i = 0; i < camera_names.size(); i++)
        {
            cameras.insert(std::make_pair(camera_names[i], camera_params.at(camera_ids[i]).infra1));
        }
    }
    assert(cameras.size() == num_cameras);

    for (auto& [camera_name, camera] : cameras)
    {
        camera.extrin.rotation = glm::mat4(1.0);
        camera.extrin.translation = glm::vec3(1.0);
    }
}

void calibration_model::add_frame(const std::map<std::string, std::vector<stargazer::point_data>> &frame)
{
    for (const auto &p : frame)
    {
        const auto &name = p.first;
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
                observed_frames.insert(std::make_pair(name, std::vector<observed_points_t>()));
            }
            else
            {
                observed_points_t obs = {};
                obs.camera_idx = camera_name_to_index.at(name);
                observed_frames.insert(std::make_pair(name, std::vector<observed_points_t>(observed_frames.begin()->second.size(), obs)));
            }
        }
    }
    for (auto &[name, observed_points] : observed_frames)
    {
        observed_points_t obs = {};
        obs.camera_idx = camera_name_to_index.at(name);
        if (frame.find(name) != frame.end())
        {
            const auto points = detector->detect_points(frame.at(name));
            if (points.size() > 0)
            {
                obs.points = std::move(points);
            }
        }
        observed_points.push_back(obs);

        if (obs.points.size() > 0)
        {
            num_frames.at(name) += 1;
        }
    }
};

static void zip_points(const std::vector<observed_points_t> &points1, const std::vector<observed_points_t> &points2,
                        std::vector<std::pair<glm::vec2, glm::vec2>> &corresponding_points)
{
    assert(points1.size() == points2.size());
    for (size_t i = 0; i < points1.size(); i++)
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
    assert(points1.size() == points2.size());
    assert(points1.size() == points3.size());
    for (size_t i = 0; i < points1.size(); i++)
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
                                        const stargazer::camera_t &base_camera, const stargazer::camera_t &target_camera)
{
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    cv::Mat camera_matrix1, camera_matrix2;
    cv::Mat coeffs1, coeffs2;
    stargazer::get_cv_intrinsic(base_camera.intrin, camera_matrix1, coeffs1);
    stargazer::get_cv_intrinsic(target_camera.intrin, camera_matrix2, coeffs2);

    std::vector<cv::Point2f> norm_points1;
    std::vector<cv::Point2f> norm_points2;
    for (const auto &[point1, point2] : corresponding_points)
    {
        points1.emplace_back(point1.x, point1.y);
        points2.emplace_back(point2.x, point2.y);
    }
    cv::undistortPoints(points1, norm_points1, camera_matrix1, coeffs1);
    cv::undistortPoints(points2, norm_points2, camera_matrix2, coeffs2);

    cv::Mat mask;
    const auto E = cv::findEssentialMat(norm_points1, norm_points2, 1.0, cv::Point2d(0.0, 0.0), cv::RANSAC, 0.99, 0.003, mask);

    cv::Mat R, t;
    cv::recoverPose(E, norm_points1, norm_points2, R, t, 1.0, cv::Point2d(0.0, 0.0), mask);

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
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    std::vector<cv::Point2f> points3;

    cv::Mat camera_matrix1, camera_matrix2, camera_matrix3;
    cv::Mat coeffs1, coeffs2, coeffs3;
    stargazer::get_cv_intrinsic(base_camera1.intrin, camera_matrix1, coeffs1);
    stargazer::get_cv_intrinsic(base_camera2.intrin, camera_matrix2, coeffs2);
    stargazer::get_cv_intrinsic(target_camera.intrin, camera_matrix3, coeffs3);

    std::vector<cv::Point2f> norm_points1;
    std::vector<cv::Point2f> norm_points2;
    std::vector<cv::Point2f> norm_points3;
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

    std::vector<cv::Point3f> point3d;
    for (size_t i = 0; i < static_cast<size_t>(point4d.cols); i++)
    {
        point3d.emplace_back(
            point4d.at<float>(0, i) / point4d.at<float>(3, i),
            point4d.at<float>(1, i) / point4d.at<float>(3, i),
            point4d.at<float>(2, i) / point4d.at<float>(3, i));
    }

    cv::Mat r, t;
    cv::solvePnP(point3d, norm_points3, cv::Mat::eye(3, 3, CV_32F), cv::Mat::zeros(1, 5, CV_32F), r, t);

    cv::Mat R;
    cv::Rodrigues(r, R);

    const auto r_mat = stargazer::cv_to_glm_mat3x3(R);
    const auto t_vec = stargazer::cv_to_glm_vec3(t);

    return glm::mat4(
        glm::vec4(r_mat[0], 0.f),
        glm::vec4(r_mat[1], 0.f),
        glm::vec4(r_mat[2], 0.f),
        glm::vec4(t_vec, 1.f));
}

namespace stargazer::reconstruction
{
    std::vector<glm::vec3> triangulate(const std::vector<std::vector<glm::vec2>> &points, const std::vector<size_t> &camera_idxs, const std::vector<camera_t> &cameras)
    {
        assert(points.size() == camera_idxs.size());
        std::vector<cv::Mat> pts;
        std::vector<cv::Mat> projs;
        for (std::size_t i = 0; i < points.size(); i++)
        {
            cv::Mat camera_mat;
            cv::Mat dist_coeffs;
            get_cv_intrinsic(cameras[camera_idxs[i]].intrin, camera_mat, dist_coeffs);

            std::vector<cv::Point2d> pt;
            for (std::size_t j = 0; j < points[i].size(); j++)
            {
                pt.emplace_back(points[i][j].x, points[i][j].y);
            }
            std::vector<cv::Point2d> undistort_pt;
            cv::undistortPoints(pt, undistort_pt, camera_mat, dist_coeffs);

            cv::Mat pt_mat(2, undistort_pt.size(), CV_64F);
            for (std::size_t j = 0; j < undistort_pt.size(); j++)
            {
                pt_mat.at<double>(0, j) = undistort_pt[j].x;
                pt_mat.at<double>(1, j) = undistort_pt[j].y;
            }
            pts.push_back(pt_mat);
            projs.push_back(glm_to_cv_mat3x4(cameras[camera_idxs[i]].extrin.rotation));
        }

        cv::Mat output;
        cv::sfm::triangulatePoints(pts, projs, output);

        std::vector<glm::vec3> result;
        for (size_t i = 0; i < static_cast<size_t>(output.cols); i++)
        {
            result.emplace_back(
                output.at<double>(0, i),
                output.at<double>(1, i),
                output.at<double>(2, i));
        }
        return result;
    }
}

static constexpr auto num_parameters = 18;

struct snavely_reprojection_error_extrinsics
{
    snavely_reprojection_error_extrinsics(double observed_x, double observed_y, const double *intrinsics)
        : observed_x(observed_x), observed_y(observed_y), fx(intrinsics[0]), fy(intrinsics[1]), cx(intrinsics[2]), cy(intrinsics[3]), k1(intrinsics[4]), k2(intrinsics[5]), k3(intrinsics[6]), p1(intrinsics[7]), p2(intrinsics[8]) {}

    template <typename T>
    bool operator()(const T *const camera,
                    const T *const point,
                    T *residuals) const
    {
        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        // Compute the center of distortion. The sign change comes from
        // the camera model that Noah Snavely's Bundler assumes, whereby
        // the camera coordinate system has a negative z axis.
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];
        // Compute final projected point position.
        T r2 = xp * xp + yp * yp;
        T distortion = 1.0 + (k1 + (k2 + k3 * r2) * r2) * r2;
        T predicted_x = fx * (distortion * xp + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp)) + cx;
        T predicted_y = fy * (distortion * yp + 2.0 * p2 * xp * yp + p1 * (r2 + 2.0 * yp * yp)) + cy;
        // The error is the difference between the predicted and observed position.
        T err_x = predicted_x - observed_x;
        T err_y = predicted_y - observed_y;
        residuals[0] = err_x;
        residuals[1] = err_y;

        if (observed_x == -1)
        {
            return false;
        }
        return true;
    }

    static ceres::CostFunction *create(const double observed_x,
                                       const double observed_y, const double *intrinsics)
    {
        return (new ceres::AutoDiffCostFunction<snavely_reprojection_error_extrinsics, 2, 6, 3>(
            new snavely_reprojection_error_extrinsics(observed_x, observed_y, intrinsics)));
    }
    double observed_x;
    double observed_y;
    double fx;
    double fy;
    double cx;
    double cy;
    double k1;
    double k2;
    double k3;
    double p1;
    double p2;
};

struct snavely_reprojection_error_extrinsic_intrinsic
{
    snavely_reprojection_error_extrinsic_intrinsic(double observed_x, double observed_y, double point_x, double point_y, double point_z)
        : observed_x(observed_x), observed_y(observed_y), point_x(point_x), point_y(point_y), point_z(point_z) {}

    template <typename T>
    bool operator()(const T *const camera,
                    T *residuals) const
    {
        // camera[0,1,2] are the angle-axis rotation.
        T point[3] = {T(point_x), T(point_y), T(point_z)};
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        // Compute the center of distortion. The sign change comes from
        // the camera model that Noah Snavely's Bundler assumes, whereby
        // the camera coordinate system has a negative z axis.
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];
        // Compute final projected point position.
        T predicted_x = xp;
        T predicted_y = yp;
        // The error is the difference between the predicted and observed position.
        T err_x = predicted_x - observed_x;
        T err_y = predicted_y - observed_y;
        residuals[0] = err_x;
        residuals[1] = err_y;

        if (observed_x == -1)
        {
            return false;
        }
        return true;
    }

    static ceres::CostFunction *create(const double observed_x,
                                       const double observed_y, double point_x, double point_y, double point_z)
    {
        return (new ceres::AutoDiffCostFunction<snavely_reprojection_error_extrinsic_intrinsic, 2, num_parameters>(
            new snavely_reprojection_error_extrinsic_intrinsic(observed_x, observed_y, point_x, point_y, point_z)));
    }
    double observed_x;
    double observed_y;
    double point_x;
    double point_y;
    double point_z;
};

template <bool is_full_intrinsics>
struct snavely_reprojection_error_extrinsic_intrinsic_point
{
    snavely_reprojection_error_extrinsic_intrinsic_point(double observed_x, double observed_y)
        : observed_x(observed_x), observed_y(observed_y) {}

    template <typename T>
    bool operator()(const T *const camera,
                    const T *const point,
                    T *residuals) const
    {
        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        // Compute the center of distortion. The sign change comes from
        // the camera model that Noah Snavely's Bundler assumes, whereby
        // the camera coordinate system has a negative z axis.
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        T predicted_x;
        T predicted_y;

        if constexpr (is_full_intrinsics)
        {
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
            const T &k4 = camera[15];
            const T &k5 = camera[16];
            const T &k6 = camera[17];
            const T distortion = (1.0 + (k1 + (k2 + k3 * r2) * r2) * r2);
            // T distortion = (1.0 + (k1 + (k2 + k3 * r2) * r2) * r2) / (1.0 + (k4 + (k5 + k6 * r2) * r2) * r2);

            // Compute final projected point position.
            predicted_x = fx * (distortion * xp + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp)) + cx;
            predicted_y = fy * (distortion * yp + 2.0 * p2 * xp * yp + p1 * (r2 + 2.0 * yp * yp)) + cy;
        }
        else
        {
            // Apply second and fourth order radial distortion.
            const T &l1 = camera[8];
            const T &l2 = camera[9];
            const T r2 = xp * xp + yp * yp;
            const T distortion = 1.0 + r2 * (l1 + l2 * r2);
            // Compute final projected point position.
            const T &focal_x = camera[6];
            const T &focal_y = camera[7];

            predicted_x = focal_x * distortion * xp;
            predicted_y = focal_y * distortion * yp;
        }

        // The error is the difference between the predicted and observed position.
        const T err_x = predicted_x - observed_x;
        const T err_y = predicted_y - observed_y;
        residuals[0] = err_x;
        residuals[1] = err_y;

        if (observed_x == -1)
        {
            return false;
        }
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction *create(const double observed_x,
                                       const double observed_y)
    {
        return (new ceres::AutoDiffCostFunction<snavely_reprojection_error_extrinsic_intrinsic_point, 2, num_parameters, 3>(
            new snavely_reprojection_error_extrinsic_intrinsic_point(observed_x, observed_y)));
    }
    double observed_x;
    double observed_y;
};

void calibration_model::calibrate()
{
    {
        std::string base_camera_name1;
        std::string base_camera_name2;
        glm::mat4 base_camera_pose1;
        glm::mat4 base_camera_pose2;

        bool found_base_pair = false;
        const float min_base_angle = 15.0f;

        for (const auto& camera_name1 : camera_names)
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
        for (const auto& camera_name : camera_names)
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

    for (const auto& camera_name : camera_names)
    {
        const auto& camera = cameras[camera_name];
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
            for (const auto& camera_name : camera_names)
            {
                const auto& point = observed_frames.at(camera_name).at(f);
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

            const auto point3ds = stargazer::reconstruction::triangulate(pts, camera_idxs, camera_list);

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
        const auto only_extrinsic = false;

        const double *observations = ba_data.observations();
        ceres::Problem problem;
        for (int i = 0; i < ba_data.num_observations(); ++i)
        {
            ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
            if (only_extrinsic)
            {
                ceres::CostFunction *cost_function =
                    snavely_reprojection_error_extrinsics::create(observations[2 * i + 0],
                                                                  observations[2 * i + 1], ba_data.mutable_camera_for_observation(i) + 6);
                problem.AddResidualBlock(cost_function,
                                         loss_function /* squared loss */,
                                         ba_data.mutable_camera_for_observation(i),
                                         ba_data.mutable_point_for_observation(i));
            }
            else
            {
                ceres::CostFunction *cost_function =
                    snavely_reprojection_error_extrinsic_intrinsic_point<true>::create(observations[2 * i + 0],
                                                                                       observations[2 * i + 1]);
                problem.AddResidualBlock(cost_function,
                                         NULL /* squared loss */,
                                         ba_data.mutable_camera_for_observation(i),
                                         ba_data.mutable_point_for_observation(i));
            }
        }

        std::cout << "num_cameras" << ba_data.num_cameras() << std::endl;
        std::cout << "num_points" << ba_data.num_points() << std::endl;
        std::cout << "num_observations" << ba_data.num_observations() << std::endl;

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 100;
        // options.gradient_tolerance = 1e-16;
        // options.function_tolerance = 1e-16;
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
                double *intrin = &ba_data.mutable_camera(i)[6];
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
