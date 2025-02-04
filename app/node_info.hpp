#pragma once

#include <string>
#include <unordered_map>
#include <variant>

enum class node_type
{
    raspi,
    raspi_color,
    depthai_color,
    rs_d435,
    rs_d435_color,
    raspi_playback,
    record,
    calibration,
};

struct node_info
{
    std::string name;
    node_type type;
    std::string id;
    std::string address;
    std::string endpoint;
    std::string db_path;
    std::unordered_map<std::string, std::string> inputs;
    std::unordered_map<std::string, std::variant<float, bool>> params;

    bool is_camera() const
    {
        return type == node_type::raspi ||
                type == node_type::raspi_color ||
                type == node_type::depthai_color ||
                type == node_type::rs_d435 ||
                type == node_type::rs_d435_color ||
                type == node_type::raspi_playback;
    }
};
