#pragma once

#include <array>
#include <memory>

#include "voxelpose.hpp"

class voxel_projector
{
    std::array<float, 3> grid_size;
    std::array<int32_t, 3> cube_size;

    struct cuda_data;

    std::unique_ptr<cuda_data> cuda_data_;

public:
    voxel_projector();
    ~voxel_projector();

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

    void get_voxel(const float *heatmaps, int num_cameras, int heatmap_width, int heatmap_height, const std::vector<camera_data> &cameras, const std::vector<roi_data> &rois, const std::array<float, 3> &grid_center);

    const float* get_cubes() const;
};

void preprocess_cuda(const uint8_t *src_data, int src_width, int src_height, int src_step, float *dst_data, int dst_width, int dst_height, int dst_step, const std::array<float, 3> &mean, const std::array<float, 3> &std);

class joint_extractor
{
    struct cuda_data;

    std::unique_ptr<cuda_data> cuda_data_;

    int num_joints;

public:
    joint_extractor(int num_joints = 15);
    ~joint_extractor();
    void soft_argmax(const float *src_data, float beta, const std::array<float, 3> &grid_size, const std::array<int32_t, 3> &cube_size, const std::array<float, 3> &grid_center);
    const float * get_joints() const;
};

class proposal_extractor
{
    uint32_t max_num;
    float threshold;
    std::array<float, 3> grid_size;
    std::array<float, 3> grid_center;
    std::array<int32_t, 3> cube_size;

public:
    proposal_extractor()
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
};
