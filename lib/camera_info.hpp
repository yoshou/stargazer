#pragma once

#include <glm/glm.hpp>
#include <array>
#include <string>
#include <fstream>
#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

namespace stargazer
{
    struct camera_intrin_t
    {
        float fx, fy;
        float cx, cy;
        std::array<float, 5> coeffs = {};

        glm::mat3 get_matrix() const
        {
            return glm::mat3(
                fx, 0, 0,
                0, fy, 0,
                cx, cy, 1);
        }
    };

    struct camera_extrin_t
    {
        glm::vec3 translation;
        glm::mat4 rotation;
    };

    struct camera_t
    {
        camera_intrin_t intrin;
        camera_extrin_t extrin;
        uint32_t width, height;
    };

    struct rs_d435_camera_module_t
    {
        camera_t infra1;
        camera_t infra2;
        camera_t color;
    };

    std::map<std::string, rs_d435_camera_module_t> load_camera_params(std::string path);

    void get_cv_intrinsic(const camera_intrin_t &intrin, cv::Mat &camera_matrix, cv::Mat &dist_coeffs);
}
