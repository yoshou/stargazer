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
        triangulatePoints(pts, projs, output);

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

class calibration_node : public graph_node
{
    mutable std::mutex cameras_mtx;

    std::map<std::string, std::vector<observed_points_t>> observed_frames;
    std::map<std::string, size_t> num_frames;

    std::shared_ptr<calibration_target> detector;

    std::map<std::string, size_t> camera_name_to_index;

    std::vector<std::string> camera_names;
    std::vector<std::string> camera_ids;

    std::unordered_map<std::string, stargazer::camera_t> cameras;
    std::unordered_map<std::string, stargazer::camera_t> calibrated_cameras;

    graph_edge_ptr output;

public:
    calibration_node()
        : graph_node(), detector(std::make_shared<three_point_bar_calibration_target>()), output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    virtual std::string get_proc_name() const override
    {
        return "calibration_node";
    }

    template <typename Archive>
    void serialize(Archive &archive)
    {
        archive(cameras);
    }

    size_t get_num_frames(std::string name) const
    {
        if (num_frames.find(name) == num_frames.end())
        {
            return 0;
        }
        return num_frames.at(name);
    }

    const std::vector<observed_points_t> &get_observed_points(std::string name) const
    {
        static std::vector<observed_points_t> empty;
        if (observed_frames.find(name) == observed_frames.end())
        {
            return empty;
        }
        return observed_frames.at(name);
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

            std::cout << "Num cameras: " << ba_data.num_cameras() << std::endl;
            std::cout << "Num points: " << ba_data.num_points() << std::endl;
            std::cout << "Num Observations: " << ba_data.num_observations() << std::endl;

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

    void push_frame(const std::map<std::string, std::vector<stargazer::point_data>> &frame)
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

        if (auto frame_msg = std::dynamic_pointer_cast<frame_message<object_message>>(message))
        {
            const auto obj_msg = frame_msg->get_data();

            std::map<std::string, std::vector<stargazer::point_data>> frame;

            for (const auto &[name, field] : obj_msg.get_fields())
            {
                if (auto points_msg = std::dynamic_pointer_cast<float2_list_message>(field))
                {
                    std::vector<stargazer::point_data> points;
                    for (const auto &pt : points_msg->get_data())
                    {
                        points.push_back(stargazer::point_data{glm::vec2(pt.x, pt.y), 0, 0});
                    }
                    frame.insert(std::make_pair(name, points));
                }
            }

            push_frame(frame);
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

class calibration::impl
{
public:
    graph_proc graph;

    std::shared_ptr<calibration_node> calib_node;

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
        calib_node->push_frame(frame);
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

    void run()
    {
        std::shared_ptr<subgraph> g(new subgraph());

        std::shared_ptr<calibration_node> n1(new calibration_node());
        g->add_node(n1);

        calib_node = n1;

        std::shared_ptr<callback_node> n2(new callback_node());
        n2->set_input(n1->get_output());
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
    }

    void stop()
    {
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

    const std::vector<observed_points_t> &get_observed_points(std::string name) const
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

void calibration::run()
{
    pimpl->run();
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

const std::vector<observed_points_t> &calibration::get_observed_points(std::string name) const
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
    pimpl->calibrate(cameras);
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

extrinsic_calibration::extrinsic_calibration(calibration_pattern pattern)
    : pattern(pattern), workers(std::make_shared<task_queue<std::function<void()>>>(4)), task_id_gen(std::random_device()()), axis(1.0f)
{
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

#define USE_CHARUCO 1

std::vector<std::unordered_map<std::string, observed_points_t>> extrinsic_calibration::detect_pattern(const frame_type &frame)
{
#if USE_CHARUCO

    std::unordered_map<int, std::vector<std::pair<std::string, cv::Point2f>>> detect_points;
    for (const auto &[camera_name, camera_image] : frame)
    {
        if (camera_name_to_index.find(camera_name) == camera_name_to_index.end())
        {
            camera_name_to_index.insert(std::make_pair(camera_name, camera_name_to_index.size()));
        }

        std::vector<cv::Point2f> points;
        std::vector<int> ids;
#if 0
        if (detect_charuco_board(camera_image, points, ids))
#else
        if (detect_aruco_marker(camera_image, points, ids))
#endif
        {
            for (size_t i = 0; i < ids.size(); i++)
            {
                auto &camera_points = detect_points[ids[i]];
                camera_points.push_back(std::make_pair(camera_name, points[i]));
            }
        }
    }

    std::vector<std::unordered_map<std::string, observed_points_t>> results;
    if (detect_points.size() == 0)
    {
        return results;
    }

    for (const auto &[id, points] : detect_points)
    {
        if (points.size() >= 3)
        {
            std::unordered_map<std::string, observed_points_t> frame_points;
            for (const auto &[camera_name, point] : points)
            {
                auto &obs = frame_points[camera_name];
                obs.camera_idx = camera_name_to_index[camera_name];
                obs.points = {glm::vec2(point.x, point.y)};
            }
            results.emplace_back(std::move(frame_points));
        }
    }

    return results;
#else
    size_t detected_view_count = 0;

    std::unordered_map<std::string, observed_points_t> frame_points;
    for (const auto &[camera_name, camera_image] : frame)
    {
        auto &observed_point = frame_points[camera_name];

        if (camera_name_to_index.find(camera_name) == camera_name_to_index.end())
        {
            camera_name_to_index.insert(std::make_pair(camera_name, camera_name_to_index.size()));
        }

        std::vector<cv::Point2f> points;
        if (detect_calibration_board(camera_image, points, calibration_pattern::ASYMMETRIC_CIRCLES_GRID))
        {
            observed_point.camera_idx = camera_name_to_index.at(camera_name);
            observed_point.points = convert_cv_to_glm_point2f(points);
            detected_view_count++;
        }
    }

    std::vector<std::unordered_map<std::string, observed_points_t>> results;
    if (detected_view_count >= 3)
    {
        results.emplace_back(std::move(frame_points));
    }

    return results;
#endif
}

void extrinsic_calibration::add_frame(const frame_type &frame)
{
    if (task_wait_queue.size() > 50)
    {
        return;
    }

    const auto task_id = task_id_gen();
    {
        std::lock_guard lock(task_wait_queue_mtx);
        task_wait_queue.push_back(task_id);
    }

    task_wait_queue_cv.notify_one();

    workers->push_task([frame, this, task_id]()
                       {
        const auto frame_patterns_list = detect_pattern(frame);

        {
            std::unique_lock<std::mutex> lock(task_wait_queue_mtx);
            task_wait_queue_cv.wait(lock, [&]
                                    { return task_wait_queue.front() == task_id; });

            assert(task_wait_queue.front() == task_id);
            task_wait_queue.pop_front();

            if (frame_patterns_list.size() > 0)
            {
                std::lock_guard lock(observed_frames_mtx);

                for (const auto& frame_patterns : frame_patterns_list)
                {
                    for (const auto &[camera_name, points] : frame_patterns)
                    {
                        if (points.points.size() > 0)
                        {
                            observed_frames[camera_name];
                        }
                    }

                    size_t max_frame_count = 0;
                    for (const auto &[camera_name, observed_frame] : observed_frames)
                    {
                        max_frame_count = std::max(observed_frame.size(), max_frame_count);
                    }
                    for (auto &[camera_name, observed_frame] : observed_frames)
                    {
                        observed_frame.resize(max_frame_count + 1);
                    }
                    for (const auto& [camera_name, points] : frame_patterns)
                    {
                        if (points.points.size() > 0)
                        {
                            observed_frames[camera_name].back() = points;
                            num_frames[camera_name]++;
                        }
                    }
                }
            }
        }

        task_wait_queue_cv.notify_all(); });
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

void extrinsic_calibration::calibrate()
{
    std::vector<std::string> camera_names;
    for (const auto &[camera_name, frame] : observed_frames)
    {
        camera_names.push_back(camera_name);
    }

    std::sort(camera_names.begin(), camera_names.end());
    {
        std::vector<std::string> processed_cameras;

        std::string base_camera_name1;
        std::string base_camera_name2;
        glm::mat4 base_camera_pose1;
        glm::mat4 base_camera_pose2;

        bool found_base_pair = false;
        const float min_base_angle = 0.0f;

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

        if (!found_base_pair)
        {
            spdlog::error("Two base cameras could not be found. At least two cameras with more minimum angles are required.");
            return;
        }

#if 0
        {
            const auto mm_to_m = 0.001f;
            const auto board_size = cv::Size(4, 11);                                                                // TODO: Define as config
            const auto square_size = cv::Size2f(117.0f / (board_size.width - 1) / 2 * mm_to_m, 196.0f / (board_size.height - 1) * mm_to_m); // TODO: Define as config

            const auto target = get_target_object_points(board_size.width, board_size.height, square_size.width, square_size.height, calibration_pattern::ASYMMETRIC_CIRCLES_GRID);

            const auto get_points = [&observed_frames = this->observed_frames](const std::string &camera_name)
            {
                for (const auto &frame : observed_frames.at(camera_name))
                {
                    if (!frame.points.empty())
                    {
                        return frame;
                    }
                }
            };

            const auto &points1 = get_points(base_camera_name1);
            const auto &points2 = get_points(base_camera_name2);

            assert(points1.points.size() > 0);
            assert(points2.points.size() > 0);

            cv::Mat camera_matrix1, dist_coeffs1;
            stargazer::get_cv_intrinsic(cameras.at(base_camera_name1).intrin, camera_matrix1, dist_coeffs1);
            cv::Mat camera_matrix2, dist_coeffs2;
            stargazer::get_cv_intrinsic(cameras.at(base_camera_name2).intrin, camera_matrix2, dist_coeffs2);
            const auto camera1_axis = compute_axis(points1, target, camera_matrix1, dist_coeffs1);
            const auto camera2_axis = compute_axis(points2, target, camera_matrix2, dist_coeffs2);

            const auto scale = glm::distance(glm::vec3(camera1_axis[3]), glm::vec3(camera2_axis[3]));

            axis = camera1_axis * glm::scale(glm::mat4(1.f), glm::vec3(scale, scale, scale));
        }
#endif

        std::cout << "Base camera1: " << base_camera_name1 << std::endl;
        std::cout << "Base camera2: " << base_camera_name2 << std::endl;

        cameras[base_camera_name1].extrin.rotation = base_camera_pose1;
        cameras[base_camera_name1].extrin.translation = glm::vec3(base_camera_pose1[3]);
        cameras[base_camera_name2].extrin.rotation = base_camera_pose2;
        cameras[base_camera_name2].extrin.translation = glm::vec3(base_camera_pose2[3]);

        processed_cameras.push_back(base_camera_name1);
        processed_cameras.push_back(base_camera_name2);

        while (processed_cameras.size() != camera_names.size())
        {
            size_t num_process = 0;
            for (const auto &camera_name : camera_names)
            {
                if (std::find(processed_cameras.begin(), processed_cameras.end(), camera_name) != processed_cameras.end())
                {
                    continue;
                }

                bool processed = false;

                for (const auto &camera_name1 : processed_cameras)
                {
                    if (processed)
                    {
                        break;
                    }
                    for (const auto &camera_name2 : processed_cameras)
                    {
                        if (processed)
                        {
                            break;
                        }

                        if (camera_name1 == camera_name2)
                        {
                            continue;
                        }

                        std::vector<std::tuple<glm::vec2, glm::vec2, glm::vec2>> corresponding_points;
                        zip_points(observed_frames.at(camera_name1), observed_frames.at(camera_name2), observed_frames.at(camera_name), corresponding_points);

                        if (corresponding_points.size() < 7)
                        {
                            continue;
                        }

                        const auto pose = estimate_pose(corresponding_points, cameras.at(camera_name1), cameras.at(camera_name2), cameras.at(camera_name));
                        cameras[camera_name].extrin.rotation = pose;
                        cameras[camera_name].extrin.translation = glm::vec3(pose[3]);

                        processed = true;
                        num_process++;
                    }
                }

                if (processed)
                {
                    processed_cameras.push_back(camera_name);
                }
            }

            if (num_process == 0)
            {
                break;
            }
        }

        if (processed_cameras.size() != camera_names.size())
        {
            std::vector<std::string> missing_cameras;
            for (const auto &camera_name : camera_names)
            {
                if (std::find(processed_cameras.begin(), processed_cameras.end(), camera_name) != processed_cameras.end())
                {
                    continue;
                }
                missing_cameras.push_back(camera_name);
            }
            std::stringstream ss;
            ss << "The location of ";
            for (size_t i = 0; i < missing_cameras.size(); i++)
            {
                if (i > 0)
                {
                    ss << ", ";
                }
                ss << missing_cameras[i];
            }
            ss << " is unknown. Markers must be visible from at least three cameras.";
            spdlog::error(ss.str());
            return;
        }
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
        const auto only_extrinsic = true;
        const auto robust = false;

        const double *observations = ba_data.observations();
        ceres::Problem problem;
        for (int i = 0; i < ba_data.num_observations(); ++i)
        {
            ceres::LossFunction *loss_function = robust ? new ceres::HuberLoss(1.0) : nullptr;
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

        std::cout << "Num cameras: " << ba_data.num_cameras() << std::endl;
        std::cout << "Num points: " << ba_data.num_points() << std::endl;
        std::cout << "Num Observations: " << ba_data.num_observations() << std::endl;

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
