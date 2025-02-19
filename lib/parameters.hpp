#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <array>
#include <string>
#include <variant>
#include <unordered_map>
#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <cereal/types/array.hpp>

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
        glm::mat3 rotation;

        glm::mat4 transform_matrix() const
        {
            return glm::translate(glm::mat4(1.0f), translation) * glm::mat4(rotation);
        }

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

    struct scene_t
    {
        glm::mat4 axis;

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(axis);
        }
    };

    class parameters_t
    {
        std::unordered_map<std::string, std::variant<camera_t, scene_t>> parameters;
        std::string path;

    public:

        parameters_t(const std::string &path) : path(path) {}

        void load();

        void save() const;

        std::variant<camera_t, scene_t> operator[](const std::string &key) const
        {
            return parameters.at(key);
        }
        std::variant<camera_t, scene_t>& operator[](const std::string &key)
        {
            return parameters[key];
        }

        std::variant<camera_t, scene_t> at(const std::string &key) const
        {
            return parameters.at(key);
        }
        std::variant<camera_t, scene_t>& at(const std::string &key)
        {
            return parameters.at(key);
        }

        bool contains(const std::string &key) const
        {
            return parameters.find(key) != parameters.end();
        }
    };

    void get_cv_intrinsic(const camera_intrin_t &intrin, cv::Mat &camera_matrix, cv::Mat &dist_coeffs);
}
