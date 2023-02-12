#pragma once

#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace stargazer
{
    struct point_data
    {
        glm::vec2 point;
        double size;
        double timestamp;
    };

    void list_frame_numbers(const std::string &directory, std::vector<std::uint64_t> &frame_numbers, const std::string &prefix = "marker_");

    void read_frame(std::string filename, const std::vector<std::string> &names, std::vector<std::vector<point_data>> &frame_data);

    void read_points(const std::string &directory, const std::vector<std::string> &names, std::vector<std::vector<std::vector<point_data>>> &frame_data, const std::string &prefix = "marker_");
}
