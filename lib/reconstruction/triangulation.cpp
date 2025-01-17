#include "triangulation.hpp"

#include <glm/glm.hpp>
#include <opencv2/opencv.hpp>
#include "utils.hpp"

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>

namespace
{
    using namespace cv;
    using namespace std;

    template <typename T>
    void
    homogeneousToEuclidean(const Mat &X_, Mat &x_)
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

    void
    homogeneousToEuclidean(InputArray X_, OutputArray x_)
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
        triangulatePoints(pts, projs, output);

        return glm::vec3(
            output.at<double>(0, 0),
            output.at<double>(1, 0),
            output.at<double>(2, 0));
    }
}
