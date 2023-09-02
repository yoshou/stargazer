#pragma once

#include <string>

enum class device_type
{
    raspi,
    raspi_color,
    depthai_color,
    rs_d435,
    rs_d435_color,
};

struct device_info
{
    std::string name;
    device_type type;
    std::string id;
    std::string address;
    std::string endpoint;
};
