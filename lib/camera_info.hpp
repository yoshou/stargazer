#pragma once

#include <glm/glm.hpp>
#include <array>
#include <string>
#include <fstream>
#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>

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

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(fx, fy, cx, cy, coeffs);
        }
    };

    struct camera_extrin_t
    {
        glm::vec3 translation;
        glm::mat4 rotation;

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(translation, rotation);
        }
    };

    struct camera_t
    {
        camera_intrin_t intrin;
        camera_extrin_t extrin;
        uint32_t width, height;

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(intrin, extrin, width, height);
        }
    };

    struct camera_module_t
    {
        std::map<std::string, camera_t> cameras;
    };

    std::map<std::string, camera_module_t> load_camera_params(std::string path);

    void save_camera_params(std::string path, const std::map<std::string, camera_module_t>& params);

    void get_cv_intrinsic(const camera_intrin_t &intrin, cv::Mat &camera_matrix, cv::Mat &dist_coeffs);
}
