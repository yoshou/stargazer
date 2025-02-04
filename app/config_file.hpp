#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "node_info.hpp"

namespace stargazer
{
    class configuration_file
    {
        std::string path;
        std::unordered_map<std::string, std::vector<node_info>> pipeline_nodes;
        std::unordered_map<std::string, std::string> pipeline_names;

    public:
        configuration_file(const std::string &path);

        void update();

        const std::vector<node_info>& get_node_infos() const
        {
            return pipeline_nodes.at(pipeline_names.at("pipeline"));
        }
        std::vector<node_info> &get_node_infos()
        {
            return pipeline_nodes.at(pipeline_names.at("pipeline"));
        }

        const std::vector<node_info> &get_node_infos(const std::string& pipeline) const
        {
            return pipeline_nodes.at(pipeline_names.at(pipeline));
        }
        std::vector<node_info> &get_node_infos(const std::string &pipeline)
        {
            return pipeline_nodes.at(pipeline_names.at(pipeline));
        }
    };
}
