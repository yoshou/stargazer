#include "multiview_point_data.hpp"

#include <regex>
#include <fstream>
#include <nlohmann/json.hpp>

#include <filesystem>
namespace fs = std::filesystem;

namespace stargazer
{
    void list_frame_numbers(const std::string &directory, std::vector<std::uint64_t> &frame_numbers, const std::string& prefix)
    {
        std::regex path_pattern(prefix + "([0-9]+).json");

        for (const auto &entry : fs::directory_iterator(directory))
        {
            const std::string s = entry.path().filename().string();
            std::smatch m;
            if (std::regex_search(s, m, path_pattern))
            {
                if (m[1].matched)
                {
                    const auto frame_no = std::stoull(m[1].str());
                    frame_numbers.push_back(frame_no);
                }
            }
        }

        std::sort(frame_numbers.begin(), frame_numbers.end());
    }

    void read_frame(std::string filename, const std::vector<std::string> &names, std::vector<std::vector<point_data>> &frame_data)
    {
        std::ifstream f;
        f.open(filename, std::ios::in | std::ios::binary);
        std::string str((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());

        nlohmann::json j_frame = nlohmann::json::parse(str);

        for (std::size_t i = 0; i < names.size(); i++)
        {
            if (j_frame[names[i]].is_null())
            {
                continue;
            }
            const auto j_kpts = j_frame[names[i]]["points"];
            const auto timestamp = j_frame[names[i]]["timestamp"].get<double>();

            for (std::size_t j = 0; j < j_kpts.size(); j++)
            {
                frame_data[i].push_back(point_data{glm::vec2(j_kpts[j]["x"].get<float>(), j_kpts[j]["y"].get<float>()), j_kpts[j]["r"].get<double>(), timestamp});
            }
        }
    }

    void read_points(const std::string &directory, const std::vector<std::string> &names, std::vector<std::vector<std::vector<point_data>>> &frame_data, const std::string &prefix)
    {
        std::vector<std::uint64_t> frame_numbers;
        list_frame_numbers(directory, frame_numbers, prefix);

        for (std::size_t camera = 0; camera < frame_data.size(); camera++)
        {
            frame_data[camera].resize(frame_numbers.size());
        }
        for (std::uint64_t frame_idx = 0; frame_idx < frame_numbers.size(); frame_idx++)
        {
            std::string filename = fs::path(directory).append(prefix + std::to_string(frame_numbers[frame_idx]) + ".json").string();

            if (!fs::exists(filename))
            {
                continue;
            }

            std::vector<std::vector<point_data>> frame(names.size());
            read_frame(filename, names, frame);

            for (std::size_t i = 0; i < names.size(); i++)
            {
                for (std::size_t j = 0; j < frame[i].size(); j++)
                {
                    frame_data[i][frame_idx].push_back(frame[i][j]);
                }
            }
        }
    }
}
