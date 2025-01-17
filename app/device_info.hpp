#pragma once

#include <string>
#include <unordered_map>

enum class device_type
{
    raspi,
    raspi_color,
    depthai_color,
    rs_d435,
    rs_d435_color,
    raspi_playback,
    record,
};

struct device_info
{
    std::string name;
    device_type type;
    std::string id;
    std::string address;
    std::string endpoint;
    std::string db_path;
    std::unordered_map<std::string, std::string> inputs;
    std::unordered_map<std::string, float> params;

    bool is_camera() const
    {
        return type == device_type::raspi ||
                type == device_type::raspi_color ||
                type == device_type::depthai_color ||
                type == device_type::rs_d435 ||
                type == device_type::rs_d435_color ||
                type == device_type::raspi_playback;
    }
};
