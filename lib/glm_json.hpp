#pragma once

#include <nlohmann/json.hpp>
#include <glm/glm.hpp>

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