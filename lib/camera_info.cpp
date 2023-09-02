#include "camera_info.hpp"
#include <glm/glm.hpp>
#include <array>
#include <string>
#include <fstream>
#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>

void stargazer::get_cv_intrinsic(const camera_intrin_t &intrin, cv::Mat &camera_matrix, cv::Mat &dist_coeffs)
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

namespace glm
{
    static void to_json(nlohmann::json &j, const glm::vec2 &v)
    {
        j = {v.x, v.y};
    }
    static void from_json(const nlohmann::json &j, glm::vec2 &v)
    {
        v.x = j[0].get<float>();
        v.y = j[1].get<float>();
    }
    static void to_json(nlohmann::json &j, const glm::vec3 &v)
    {
        j = {v.x, v.y, v.z};
    }
    static void from_json(const nlohmann::json &j, glm::vec3 &v)
    {
        v.x = j[0].get<float>();
        v.y = j[1].get<float>();
        v.z = j[2].get<float>();
    }
    static void to_json(nlohmann::json &j, const glm::vec4 &v)
    {
        j = {v.x, v.y, v.z, v.w};
    }
    static void from_json(const nlohmann::json &j, glm::vec4 &v)
    {
        v.x = j[0].get<float>();
        v.y = j[1].get<float>();
        v.z = j[2].get<float>();
        v.w = j[3].get<float>();
    }
    static void to_json(nlohmann::json &j, const glm::mat4 &m)
    {
        j = {m[0], m[1], m[2], m[3]};
    }
    static void from_json(const nlohmann::json &j, glm::mat4 &m)
    {
        m[0] = j[0].get<glm::vec4>();
        m[1] = j[1].get<glm::vec4>();
        m[2] = j[2].get<glm::vec4>();
        m[3] = j[3].get<glm::vec4>();
    }
}

namespace stargazer
{
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
        extrin.rotation = j["rotation"].get<glm::mat4>();
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

    void to_json(nlohmann::json &j, const camera_module_t &p)
    {
        j = nlohmann::json{{"cameras", p.cameras}};
    }

    void from_json(const nlohmann::json &j, camera_module_t &p)
    {
        j.at("cameras").get_to(p.cameras);
    }
}

std::map<std::string, stargazer::camera_module_t> stargazer::load_camera_params(std::string path)
{
    std::ifstream ifs;
    ifs.open(path, std::ios::binary | std::ios::in);

    const auto j = nlohmann::json::parse(ifs);
    return j["devices"].get<std::map<std::string, stargazer::camera_module_t>>();
}

void stargazer::save_camera_params(std::string path, const std::map<std::string, camera_module_t> &params)
{
    std::ofstream ofs;
    ofs.open(path, std::ios::out);

    const auto j = nlohmann::json{{"devices", params}};
    ofs << j.dump(2);
}
