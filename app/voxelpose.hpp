#pragma once

#include <array>
#include <memory>

#include <opencv2/imgproc/imgproc.hpp>

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

static bool operator==(const camera_data &camera1, const camera_data &camera2)
{
    if (camera1.fx != camera2.fx || camera1.fy != camera2.fy || camera1.cx != camera2.cx || camera1.cy != camera2.cy)
    {
        return false;
    }
    if (!std::equal(camera1.k.begin(), camera1.k.end(), camera2.k.begin()))
    {
        return false;
    }
    if (!std::equal(camera1.p.begin(), camera1.p.end(), camera2.p.begin()))
    {
        return false;
    }
    for (size_t i = 0; i < camera1.rotation.size(); i++)
    {
        if (!std::equal(camera1.rotation[i].begin(), camera1.rotation[i].end(), camera2.rotation[i].begin()))
        {
            return false;
        }
    }
    if (!std::equal(camera1.translation.begin(), camera1.translation.end(), camera2.translation.begin()))
    {
        return false;
    }
    return true;
}

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
