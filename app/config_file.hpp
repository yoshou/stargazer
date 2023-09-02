#pragma once

#include <string>
#include <vector>
#include <fstream>

#include <nlohmann/json.hpp>

#include "device_info.hpp"

namespace stargazer
{
    class configuration_file
    {
        std::string path;
        std::vector<device_info> device_infos;

    public:
        configuration_file(const std::string& path) : path(path)
        {
            std::ifstream ifs;
            ifs.open(path, std::ios::in);

            if (ifs)
            {
                const auto j = nlohmann::json::parse(ifs);
                for (const auto &j_camera : j["cameras"])
                {
                    device_info cluster;
                    const auto type = j_camera["node"]["type"].get<std::string>();
                    if (type == "raspi")
                    {
                        cluster.type = device_type::raspi;
                    }
                    else if (type == "rs_d435")
                    {
                        cluster.type = device_type::rs_d435;
                    }
                    else if (type == "rs_d435_color")
                    {
                        cluster.type = device_type::rs_d435_color;
                    }
                    else if (type == "depthai_color")
                    {
                        cluster.type = device_type::depthai_color;
                    }
                    else
                    {
                        throw std::runtime_error("Invalid node type");
                    }
                    cluster.id = j_camera["node"]["id"].get<std::string>();
                    cluster.address = j_camera["node"]["address"].get<std::string>();
                    cluster.endpoint = j_camera["node"]["gateway"].get<std::string>();
                    cluster.name = j_camera["name"].get<std::string>();
                    device_infos.push_back(cluster);
                }
            }
        }

        const std::vector<device_info>& get_device_infos() const
        {
            return device_infos;
        }
        std::vector<device_info> &get_device_infos()
        {
            return device_infos;
        }

        void update()
        {
            std::vector<nlohmann::json> j_devices;
            for (const auto &device : device_infos)
            {
                nlohmann::json j_device;

                std::string device_type_name;
                switch (device.type)
                {
                case device_type::raspi:
                    device_type_name = "raspi";
                    break;
                case device_type::rs_d435:
                    device_type_name = "rs_d435";
                    break;
                case device_type::rs_d435_color:
                    device_type_name = "rs_d435_color";
                    break;
                case device_type::depthai_color:
                    device_type_name = "depthai_color";
                    break;
                default:
                    throw std::runtime_error("Invalid node type");
                }

                j_device["node"]["type"] = device_type_name;
                j_device["node"]["id"] = device.id;
                j_device["node"]["address"] = device.address;
                j_device["node"]["gateway"] = device.endpoint;
                j_device["name"] = device.name;
                j_devices.push_back(j_device);
            }

            
            nlohmann::json j;
            j["cameras"] = j_devices;

            std::ofstream ofs;
            ofs.open(path, std::ios::out);
            ofs << j.dump(2);
        }
    };
}
