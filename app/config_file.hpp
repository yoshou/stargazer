#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include "device_info.hpp"

namespace stargazer
{
    class configuration_file
    {
        std::string path;
        std::vector<device_info> device_infos;
        std::string pipeline_name;
        std::unordered_set<std::string> node_names;

    public:
        configuration_file(const std::string& path) : path(path)
        {
            std::ifstream ifs;
            ifs.open(path, std::ios::in);

            if (ifs)
            {
                const auto j = nlohmann::json::parse(ifs);

                pipeline_name = j["pipeline"].get<std::string>();

                for (const auto &j_node : j["pipelines"][pipeline_name]["nodes"])
                {
                    device_info device;
                    const auto type = j_node["type"].get<std::string>();
                    if (type == "raspi")
                    {
                        device.type = device_type::raspi;
                    }
                    else if (type == "raspi_color")
                    {
                        device.type = device_type::raspi_color;
                    }
                    else if (type == "rs_d435")
                    {
                        device.type = device_type::rs_d435;
                    }
                    else if (type == "rs_d435_color")
                    {
                        device.type = device_type::rs_d435_color;
                    }
                    else if (type == "depthai_color")
                    {
                        device.type = device_type::depthai_color;
                    }
                    else if (type == "raspi_playback")
                    {
                        device.type = device_type::raspi_playback;
                    }
                    else if (type == "record")
                    {
                        device.type = device_type::record;
                    }
                    else
                    {
                        throw std::runtime_error("Invalid node type");
                    }
                    
                    device.name = j_node["name"].get<std::string>();
                    if (type == "raspi_playback")
                    {
                        device.id = j_node["id"].get<std::string>();
                        device.db_path = j_node["db_path"].get<std::string>();
                        device.name = j_node["name"].get<std::string>();
                    }
                    else if (type == "record")
                    {
                        device.db_path = j_node["db_path"].get<std::string>();
                        device.name = j_node["name"].get<std::string>();
                    }
                    else
                    {
                        device.id = j_node["id"].get<std::string>();
                        device.address = j_node["address"].get<std::string>();
                        device.endpoint = j_node["gateway"].get<std::string>();
                    }
                    if (j_node.contains("params"))
                    {
                        device.params = j_node["params"].get<std::unordered_map<std::string, float>>();
                    }
                    if (j_node.contains("inputs"))
                    {
                        device.inputs = j_node["inputs"].get<std::unordered_map<std::string, std::string>>();
                    }
                    device_infos.push_back(device);

                    if (node_names.find(device.name) != node_names.end())
                    {
                        throw std::runtime_error("Duplicate node name");
                    }
                    node_names.insert(device.name);
                }
            }
        }

        const std::string& get_pipeline_name() const
        {
            return pipeline_name;
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
            std::vector<nlohmann::json> j_nodes;
            for (const auto &device : device_infos)
            {
                nlohmann::json j_node;

                std::string device_type_name;
                switch (device.type)
                {
                case device_type::raspi:
                    device_type_name = "raspi";
                    break;
                case device_type::raspi_color:
                    device_type_name = "raspi_color";
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
                case device_type::raspi_playback:
                    device_type_name = "raspi_playback";
                    break;
                case device_type::record:
                    device_type_name = "record";
                    break;
                default:
                    throw std::runtime_error("Invalid node type");
                }

                j_node["type"] = device_type_name;
                j_node["name"] = device.name;
                j_node["params"] = device.params;
                if (device.type == device_type::raspi_playback)
                {
                    j_node["id"] = device.id;
                    j_node["db_path"] = device.db_path;
                }
                else if (device.type == device_type::record)
                {
                    j_node["db_path"] = device.db_path;
                }
                else
                {
                    j_node["id"] = device.id;
                    j_node["address"] = device.address;
                    j_node["gateway"] = device.endpoint;
                }
                j_nodes.push_back(j_node);
            }

            
            nlohmann::json j;
            j["nodes"] = j_nodes;

            std::ofstream ofs;
            ofs.open(path, std::ios::out);
            ofs << j.dump(2);
        }
    };
}
