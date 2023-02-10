#include "triangulation.hpp"

#include <glm/glm.hpp>
#include <opencv2/opencv.hpp>
#include "utils.hpp"
#include <opencv2/sfm/triangulation.hpp>

static inline void swap_rows(cv::Mat &m, int i, int j)
{
    for (int k = 0; k < m.cols; k++)
    {
        auto temp = m.at<float>(i, k);
        m.at<float>(i, k) = m.at<float>(j, k);
        m.at<float>(j, k) = temp;
    }
}

static inline void compute_gauss_elimination(const cv::Mat &m, cv::Mat &result)
{
    const int num = m.rows;
    auto a = m.clone();

    for (int k = 0; k < num - 1; k++)
    {
        int pivot = k;
        float pivot_val = a.at<float>(k, k);
        for (int i = k + 1; i < num; i++)
        {
            if (abs(a.at<float>(i, k)) > abs(pivot_val))
            {
                pivot_val = a.at<float>(i, k);
                pivot = i;
            }
        }
        if (pivot != k)
        {
            swap_rows(a, k, pivot);
        }
        for (int i = k + 1; i < num; i++)
        {
            auto d = a.at<float>(i, k) / a.at<float>(k, k);
            for (int j = k; j <= num; j++)
            {
                a.at<float>(i, j) = a.at<float>(i, j) - a.at<float>(k, j) * d;
            }
        }
    }

    for (int i = num - 1; i >= 0; i--)
    {
        auto d = a.at<float>(i, num);
        for (int j = i + 1; j < num; j++)
        {
            d -= a.at<float>(i, j) * a.at<float>(j, num);
        }
        a.at<float>(i, num) = d / a.at<float>(i, i);
    }

    result.create(1, num, CV_32F);
    for (int k = 0; k < num; k++)
    {
        result.at<float>(0, k) = a.at<float>(k, num);
    }
}

static glm::vec3 triangulate(const glm::mat4 &m1, const glm::mat4 &m2, const glm::vec2 &p1, const glm::vec2 &p2, glm::vec2 &s)
{
    auto u1 = p1.x;
    auto v1 = p1.y;
    auto u2 = p2.x;
    auto v2 = p2.y;

    cv::Mat coeff(5, 6, CV_32F);

    coeff.at<float>(0, 0) = m1[0][0];
    coeff.at<float>(0, 1) = m1[1][0];
    coeff.at<float>(0, 2) = m1[2][0];
    coeff.at<float>(0, 3) = -u1;
    coeff.at<float>(0, 4) = 0;
    coeff.at<float>(0, 5) = -m1[3][0];

    coeff.at<float>(1, 0) = m1[0][1];
    coeff.at<float>(1, 1) = m1[1][1];
    coeff.at<float>(1, 2) = m1[2][1];
    coeff.at<float>(1, 3) = -v1;
    coeff.at<float>(1, 4) = 0;
    coeff.at<float>(1, 5) = -m1[3][1];

    coeff.at<float>(2, 0) = m1[0][2];
    coeff.at<float>(2, 1) = m1[1][2];
    coeff.at<float>(2, 2) = m1[2][2];
    coeff.at<float>(2, 3) = -1;
    coeff.at<float>(2, 4) = 0;
    coeff.at<float>(2, 5) = -m1[3][2];

    coeff.at<float>(3, 0) = m2[0][0];
    coeff.at<float>(3, 1) = m2[1][0];
    coeff.at<float>(3, 2) = m2[2][0];
    coeff.at<float>(3, 3) = 0;
    coeff.at<float>(3, 4) = -u2;
    coeff.at<float>(3, 5) = -m2[3][0];

    coeff.at<float>(4, 0) = m2[0][2];
    coeff.at<float>(4, 1) = m2[1][2];
    coeff.at<float>(4, 2) = m2[2][2];
    coeff.at<float>(4, 3) = 0;
    coeff.at<float>(4, 4) = -1;
    coeff.at<float>(4, 5) = -m2[3][2];

    cv::Mat result_pt;
    compute_gauss_elimination(coeff, result_pt);
    glm::vec3 pt(result_pt.at<float>(0, 0), result_pt.at<float>(0, 1), result_pt.at<float>(0, 2));

    s = glm::vec2(result_pt.at<float>(0, 3), result_pt.at<float>(0, 4));

    return pt;
}

static glm::vec3 deproject_pixel_to_point2(float fx, float fy, float ppx, float ppy, const glm::vec2 &pixel, float depth)
{
    float x = (pixel[0] - ppx) / fx;
    float y = (pixel[1] - ppy) / fy;

    glm::vec3 point;
    point[0] = depth * x;
    point[1] = depth * y;
    point[2] = depth;

    return point;
}

template <typename Scalar>
static Scalar compute_depth(Scalar fx, Scalar rx, Scalar lx, Scalar base_line, Scalar depth_unit = Scalar(0.001))
{
    return fx * base_line / (depth_unit * std::abs(rx - lx));
}

namespace stargazer::reconstruction
{
    glm::vec3 triangulate(const glm::vec2 pt1, const glm::vec2 pt2, const camera_t &cm1, const camera_t &cm2)
    {
        cv::Mat camera_mat1, camera_mat2;
        cv::Mat dist_coeffs1, dist_coeffs2;
        get_cv_intrinsic(cm1.intrin, camera_mat1, dist_coeffs1);
        get_cv_intrinsic(cm2.intrin, camera_mat2, dist_coeffs2);

        std::vector<cv::Point2d> pts1, pts2;
        pts1.push_back(cv::Point2d(pt1.x, pt1.y));
        pts2.push_back(cv::Point2d(pt2.x, pt2.y));

        std::vector<cv::Point2d> undistort_pts1, undistort_pts2;
        cv::undistortPoints(pts1, undistort_pts1, camera_mat1, dist_coeffs1);
        cv::undistortPoints(pts2, undistort_pts2, camera_mat2, dist_coeffs2);

        cv::Mat proj1 = glm_to_cv_mat3x4(cm1.extrin.rotation);
        cv::Mat proj2 = glm_to_cv_mat3x4(cm2.extrin.rotation);

        cv::Mat output;
        cv::triangulatePoints(proj1, proj2, undistort_pts1, undistort_pts2, output);

        return glm::vec3(
            output.at<double>(0, 0) / output.at<double>(3, 0),
            output.at<double>(1, 0) / output.at<double>(3, 0),
            output.at<double>(2, 0) / output.at<double>(3, 0));
    }

    glm::vec3 triangulate_undistorted(const glm::vec2 pt1, const glm::vec2 pt2, const camera_t &cm1, const camera_t &cm2)
    {
        cv::Mat camera_mat1, camera_mat2;
        cv::Mat dist_coeffs1, dist_coeffs2;
        get_cv_intrinsic(cm1.intrin, camera_mat1, dist_coeffs1);
        get_cv_intrinsic(cm2.intrin, camera_mat2, dist_coeffs2);

        std::vector<cv::Point2d> pts1, pts2;
        pts1.push_back(cv::Point2d(pt1.x, pt1.y));
        pts2.push_back(cv::Point2d(pt2.x, pt2.y));

        cv::Mat proj1 = glm_to_cv_mat3x4(cm1.extrin.rotation);
        cv::Mat proj2 = glm_to_cv_mat3x4(cm2.extrin.rotation);

        cv::Mat output;
        cv::triangulatePoints(proj1, proj2, pts1, pts2, output);

        return glm::vec3(
            output.at<double>(0, 0) / output.at<double>(3, 0),
            output.at<double>(1, 0) / output.at<double>(3, 0),
            output.at<double>(2, 0) / output.at<double>(3, 0));
    }

    glm::vec3 triangulate(const std::vector<glm::vec2> &points, const std::vector<camera_t> &cameras)
    {
        assert(points.size() == cameras.size());
        std::vector<cv::Mat> pts(points.size());
        std::vector<cv::Mat> projs(points.size());
        for (std::size_t i = 0; i < points.size(); i++)
        {
            cv::Mat camera_mat;
            cv::Mat dist_coeffs;
            get_cv_intrinsic(cameras[i].intrin, camera_mat, dist_coeffs);

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
            projs[i] = glm_to_cv_mat3x4(cameras[i].extrin.rotation);
        }

        cv::Mat output;
        cv::sfm::triangulatePoints(pts, projs, output);

        return glm::vec3(
            output.at<double>(0, 0),
            output.at<double>(1, 0),
            output.at<double>(2, 0));
    }
}
