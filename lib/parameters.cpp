#include "parameters.hpp"

#include <glm/glm.hpp>
#include <array>
#include <string>
#include <fstream>
#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>

#include "glm_json.hpp"

namespace stargazer
{
    void get_cv_intrinsic(const camera_intrin_t &intrin, cv::Mat &camera_matrix, cv::Mat &dist_coeffs)
    {
        camera_matrix = cv::Mat::eye(3, 3, CV_64F);
        camera_matrix.at<double>(0, 0) = intrin.fx;
        camera_matrix.at<double>(1, 1) = intrin.fy;
        camera_matrix.at<double>(0, 2) = intrin.cx;
        camera_matrix.at<double>(1, 2) = intrin.cy;

        dist_coeffs = cv::Mat::zeros(5, 1, CV_64FC1);
        dist_coeffs.at<double>(0) = intrin.coeffs[0];
        dist_coeffs.at<double>(1) = intrin.coeffs[1];
        dist_coeffs.at<double>(2) = intrin.coeffs[2];
        dist_coeffs.at<double>(3) = intrin.coeffs[3];
        dist_coeffs.at<double>(4) = intrin.coeffs[4];
    }

    static void to_json(nlohmann::json &j, const camera_intrin_t &intrin)
    {
        j = {
            {"fx", intrin.fx},
            {"fy", intrin.fy},
            {"cx", intrin.cx},
            {"cy", intrin.cy},
            {"coeffs", intrin.coeffs},
        };
    }

    static void from_json(const nlohmann::json &j, camera_intrin_t &intrin)
    {
        intrin.fx = j["fx"].get<float>();
        intrin.fy = j["fy"].get<float>();
        intrin.cx = j["cx"].get<float>();
        intrin.cy = j["cy"].get<float>();
        intrin.coeffs = j["coeffs"].get<std::array<float, 5>>();
    }

    static void to_json(nlohmann::json &j, const camera_extrin_t &extrin)
    {
        j = {
            {"rotation", extrin.rotation},
            {"translation", extrin.translation},
        };
    }

    static void from_json(const nlohmann::json &j, camera_extrin_t &extrin)
    {
        extrin.rotation = j["rotation"].get<glm::mat3>();
        extrin.translation = j["translation"].get<glm::vec3>();
    }

    static void to_json(nlohmann::json &j, const camera_t &camera)
    {
        j = {
            {"intrin", camera.intrin},
            {"extrin", camera.extrin},
            {"width", camera.width},
            {"height", camera.height},
        };
    }

    static void from_json(const nlohmann::json &j, camera_t &camera)
    {
        camera.intrin = j["intrin"].get<camera_intrin_t>();
        camera.extrin = j["extrin"].get<camera_extrin_t>();
        camera.width = j["width"].get<uint32_t>();
        camera.height = j["height"].get<uint32_t>();
    }

    static void to_json(nlohmann::json &j, const scene_t &scene)
    {
        j = {
            {"axis", scene.axis},
        };
    }

    static void from_json(const nlohmann::json &j, scene_t &scene)
    {
        scene.axis = j["axis"].get<glm::mat4>();
    }

    void parameters_t::load()
    {
        std::ifstream ifs;
        ifs.open(path, std::ios::binary | std::ios::in);

        const auto j = nlohmann::json::parse(ifs);

        for (const auto &[name, item] : j.items())
        {
            if (item["type"] == "camera")
            {
                auto camera = item.get<camera_t>();
                parameters[name] = camera;
            }
            else if (item["type"] == "scene")
            {
                auto scene = item.get<scene_t>();
                parameters[name] = scene;
            }
        }
    }

    void parameters_t::save() const
    {
        std::ofstream ofs;
        ofs.open(path, std::ios::out);

        nlohmann::json j;

        for (const auto &[name, param] : parameters)
        {
            nlohmann::json j_param;

            if (std::holds_alternative<camera_t>(param))
            {
                j_param = std::get<camera_t>(param);
                j_param["type"] = "camera";
            }
            else if (std::holds_alternative<scene_t>(param))
            {
                j_param = std::get<scene_t>(param);
                j_param["type"] = "scene";
            }

            j[name] = j_param;
        }

        ofs << j.dump(2);
    }
}
