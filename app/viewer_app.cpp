#include <functional>
#include <iostream>
#include "viewer.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>

#include <glad/glad.h>
#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>

#include <sstream>
#include <map>
#include <cmath>
#include <memory>

#include "camera_info.hpp"
#include "correspondance.hpp"
#include "capture.hpp"
#include "views.hpp"
#include "calibration.hpp"
#include "reconstruction.hpp"
#include "config_file.hpp"
#include "glm_json.hpp"
#include "triangulation.hpp"

#include <filesystem>

class playback_stream
{
    std::string directory;
    std::size_t initial_frame_no;
    std::vector<std::uint64_t> frame_numbers;
    std::shared_ptr<std::thread> th;
    std::atomic_bool playing;

public:
    playback_stream(std::string directory, std::size_t initial_frame_no = 0)
        : directory(directory), initial_frame_no(initial_frame_no), playing(false)
    {
        stargazer::list_frame_numbers(directory, frame_numbers);

        std::size_t max_frames = frame_numbers.size();
    }

    void start(const std::vector<std::string> &names, std::function<void(const std::map<std::string, std::vector<stargazer::point_data>> &)> callback)
    {
        namespace fs = std::filesystem;
        th.reset(new std::thread([this, callback, names]()
                                               {
            playing = true;
            auto frame_no = initial_frame_no;
            auto next_time = std::chrono::system_clock::now() + std::chrono::duration<double>(1.0 / 90);
            while (playing && frame_no < frame_numbers.size()) {
                std::string filename = (fs::path(directory) / ("marker_" + std::to_string(frame_numbers[frame_no]) + ".json")).string();
                if (!fs::exists(filename))
                {
                    continue;
                }

                std::vector<std::vector<stargazer::point_data>> frame_data(names.size());
                read_frame(filename, names, frame_data);

                std::map<std::string, std::vector<stargazer::point_data>> markers;
                for (size_t i = 0; i < names.size(); i++)
                {
                    markers.insert(std::make_pair(names[i], frame_data[i]));
                }

                callback(markers);

                frame_no++;

                std::this_thread::sleep_until(next_time);
                next_time = next_time + std::chrono::duration<double>(1.0 / 90);
            } }));
    }

    void stop()
    {
        playing.store(false);
        if (th && th->joinable())
        {
            th->join();
        }
    }
};

const int SCREEN_WIDTH = 1680;
const int SCREEN_HEIGHT = 1050;
struct reconstruction_viewer : public window_base
{
    std::mutex mtx;

    static std::string get_calibration_config_path()
    {
        namespace fs = std::filesystem;
        const std::string data_dir = "../data";

        return (fs::path(data_dir) / "config.json").string();
    }

    reconstruction_viewer()
        : window_base("Reconstruction Viewer", SCREEN_WIDTH, SCREEN_HEIGHT), calib(get_calibration_config_path()), multiview_image_reconstruction_(std::make_unique<voxelpose_reconstruction>())
    {
    }

    virtual void initialize() override
    {
        window_base::initialize();
    }

    ImFont *large_font;
    ImFont *default_font;

    std::unique_ptr<view_context> context;
    std::unique_ptr<top_bar_view> top_bar_view_;
    std::unique_ptr<capture_panel_view> capture_panel_view_;
    std::unique_ptr<calibration_panel_view> calibration_panel_view_;
    std::unique_ptr<reconstruction_panel_view> reconstruction_panel_view_;
    std::unique_ptr<frame_tile_view> frame_tile_view_;
    std::unique_ptr<frame_tile_view> contrail_tile_view_;
    std::unique_ptr<pose_view> pose_view_;
    std::shared_ptr<azimuth_elevation> view_controller;

    std::map<std::string, std::shared_ptr<capture_pipeline>> captures;
    std::shared_ptr<multiview_capture_pipeline> multiview_capture;

    epipolar_reconstruction marker_server;
    std::unique_ptr<multiview_image_reconstruction> multiview_image_reconstruction_;
    std::unique_ptr<playback_stream> playback;
    std::unique_ptr<stargazer::configuration_file> config;

    std::string generate_new_id() const
    {
        uint64_t max_id = 0;
        for (const auto &device_info : config->get_device_infos())
        {
            size_t idx = 0;
            const auto id = std::stoull(device_info.id, &idx);

            if (device_info.id.size() == idx)
            {
                max_id = std::max(max_id, static_cast<uint64_t>(id + 1));
            }
        }
        return fmt::format("{:>012d}", max_id);
    }

    void init_capture_panel()
    {
        capture_panel_view_ = std::make_unique<capture_panel_view>();
        for (const auto &device_info : config->get_device_infos())
        {
            capture_panel_view_->devices.push_back(capture_panel_view::device_info{device_info.name, device_info.address, device_info.params});
        }

        capture_panel_view_->is_streaming_changed.push_back([this](const capture_panel_view::device_info &device) {
            if (device.is_streaming == true)
            {
                const auto& device_infos = config->get_device_infos();
                auto found = std::find_if(device_infos.begin(), device_infos.end(), [device](const auto& x) { return x.name == device.name; });
                if (found == device_infos.end()) {
                    return false;
                }

                const auto capture = std::make_shared<capture_pipeline>();

                try
                {
                    capture->run(*found);
                }
                catch (std::exception& e)
                {
                    std::cout << "Failed to start capture: " << e.what() << std::endl;
                    return false;
                }
                captures.insert(std::make_pair(device.name, capture));
                const int width = static_cast<int>(std::round(device.params.at("width")));
                const int height = static_cast<int>(std::round(device.params.at("height")));

                const auto stream = std::make_shared<frame_tile_view::stream_info>(device.name, float2{(float)width, (float)height});
                frame_tile_view_->streams.push_back(stream);
            }
            else
            {
                auto it = captures.find(device.name);
                it->second->stop();
                if (it != captures.end())
                {
                    captures.erase(captures.find(device.name));
                }

                const auto stream_it = std::find_if(frame_tile_view_->streams.begin(), frame_tile_view_->streams.end(), [&](const auto &x)
                                                    { return x->name == device.name; });

                if (stream_it != frame_tile_view_->streams.end())
                {
                    frame_tile_view_->streams.erase(stream_it);
                }
            }

            return true;
        });

        capture_panel_view_->on_add_device.push_back([this](const std::string& device_name, device_type device_type, const std::string& ip_address, const std::string& gateway_address) {
            device_info new_device {};
            new_device.id = generate_new_id();
            new_device.name = device_name;
            new_device.address = ip_address;
            new_device.endpoint = gateway_address;
            new_device.type = device_type;

            switch (device_type)
            {
            case device_type::depthai_color:
                new_device.params["width"] = 960;
                new_device.params["height"] = 540;
                break;
            case device_type::raspi:
                new_device.params["width"] = 820;
                new_device.params["height"] = 616;
                break;
            case device_type::raspi_color:
                new_device.params["width"] = 820;
                new_device.params["height"] = 616;
                break;
            case device_type::rs_d435:
                new_device.params["width"] = 640;
                new_device.params["height"] = 480;
                break;
            case device_type::rs_d435_color:
                new_device.params["width"] = 960;
                new_device.params["height"] = 540;
                break;
            }

            auto &device_infos = config->get_device_infos();
            if (const auto found = std::find_if(device_infos.begin(), device_infos.end(), [&](const auto &x) {
                return x.name == device_name; });
                found == device_infos.end())
            {
                device_infos.push_back(new_device);

                for (const auto &device_info : device_infos)
                {
                    capture_panel_view_->devices.push_back(capture_panel_view::device_info{device_info.name, device_info.address, device_info.params});
                }
                config->update();
            }
        });

        capture_panel_view_->on_remove_device.push_back([this](const std::string& device_name) {
            auto& device_infos = config->get_device_infos();
            if (const auto found = std::find_if(device_infos.begin(), device_infos.end(), [&](const auto &x) {
                return x.name == device_name; });
                found != device_infos.end())
            {
                device_infos.erase(found);
                capture_panel_view_->devices.clear();

                for (const auto &device_info : device_infos)
                {
                    capture_panel_view_->devices.push_back(capture_panel_view::device_info{device_info.name, device_info.address, device_info.params});
                }
                config->update();
            }
        });
    }

    std::map<std::string, cv::Mat> masks;

    void init_calibration_panel()
    {
        calibration_panel_view_ = std::make_unique<calibration_panel_view>();
        for (const auto &device_info : config->get_device_infos())
        {
            calibration_panel_view_->devices.push_back(calibration_panel_view::device_info{device_info.id, device_info.name, device_info.address, device_info.params});
        }

        calibration_panel_view_->is_streaming_changed.push_back([this](const std::vector<calibration_panel_view::device_info> &devices, bool is_streaming)
                                                               {
            if (is_streaming)
            {
                if (multiview_capture)
                {
                    return false;
                }

                if (calibration_panel_view_->calibration_target_index == 0)
                {
                    std::vector<device_info> infos;

                    for (const auto& device : devices)
                    {
                        const auto &device_infos = config->get_device_infos();
                        auto found = std::find_if(device_infos.begin(), device_infos.end(), [device](const auto &x)
                                                { return x.name == device.name; });
                        if (found == device_infos.end())
                        {
                            return false;
                        }
                        
                        infos.push_back(*found);
                    }

                    if (calibration_panel_view_->is_masking)
                    {
                        multiview_capture.reset(new multiview_capture_pipeline(masks));
                    }
                    else
                    {
                        multiview_capture.reset(new multiview_capture_pipeline());
                    }

                    multiview_capture->add_marker_received([this](const std::map<std::string, marker_frame_data> &marker_frame)
                                                           {
                        std::map<std::string, std::vector<stargazer::point_data>> frame;
                        for (const auto &[name, markers] : marker_frame)
                        {
                            std::vector<stargazer::point_data> points;
                            for (const auto &marker : markers.markers)
                            {
                                points.push_back(stargazer::point_data{glm::vec2(marker.x, marker.y), marker.r, markers.timestamp});
                            }
                            frame.insert(std::make_pair(name, points));
                        }
                        marker_server.push_frame(frame); });

                    multiview_capture->add_image_received([this](const std::map<std::string, cv::Mat> &image_frame)
                                                          {
                        if (!calibration_panel_view_->is_marker_collecting)
                        {
                            return;
                        }
                        std::unordered_map<std::string, cv::Mat> color_camera_images;
                        for (const auto &[camera_name, camera_image] : image_frame)
                        {
                            const auto &device_infos = config->get_device_infos();
                            if (const auto device_info = std::find_if(device_infos.begin(), device_infos.end(), [&camera_name = camera_name](const auto &x)
                                                                    { return x.name == camera_name; });
                                device_info != device_infos.end())
                            {
                                if (device_info->type == device_type::depthai_color || device_info->type == device_type::raspi_color || device_info->type == device_type::rs_d435_color)
                                {
                                    color_camera_images[camera_name] = camera_image;
                                }
                            }
                        }

                        if (color_camera_images.size() > 0)
                        {
                            extrinsic_calib.add_frame(color_camera_images);
                        }
                    });

                    multiview_capture->run(infos);

                    for (const auto& device : devices)
                    {
                        const auto stream = std::make_shared<frame_tile_view::stream_info>(device.name, float2{(float)width, (float)height});
                        frame_tile_view_->streams.push_back(stream);
                    }
                }
                else if (calibration_panel_view_->calibration_target_index == 1)
                {
                    const auto& device = devices[calibration_panel_view_->intrinsic_calibration_device_index];

                    const auto &device_infos = config->get_device_infos();
                    auto found = std::find_if(device_infos.begin(), device_infos.end(), [device](const auto &x)
                                              { return x.name == device.name; });
                    if (found == device_infos.end())
                    {
                        return false;
                    }

                    const auto capture = std::make_shared<capture_pipeline>();

                    try
                    {
                        capture->run(*found);
                    }
                    catch (std::exception &e)
                    {
                        std::cout << "Failed to start capture: " << e.what() << std::endl;
                        return false;
                    }
                    captures.insert(std::make_pair(device.name, capture));
                    const int width = static_cast<int>(std::round(device.params.at("width")));
                    const int height = static_cast<int>(std::round(device.params.at("height")));

                    const auto stream = std::make_shared<frame_tile_view::stream_info>(device.name, float2{(float)width, (float)height});
                    frame_tile_view_->streams.push_back(stream);
                }
            }
            else
            {
                if (calibration_panel_view_->calibration_target_index == 0)
                {
                    multiview_capture->stop();
                    multiview_capture.reset();

                    for (const auto &device : devices)
                    {
                        const auto stream_it = std::find_if(frame_tile_view_->streams.begin(), frame_tile_view_->streams.end(), [&](const auto &x)
                                                            { return x->name == device.name; });

                        if (stream_it != frame_tile_view_->streams.end())
                        {
                            frame_tile_view_->streams.erase(stream_it);
                        }
                    }
                }
                else if (calibration_panel_view_->calibration_target_index == 1)
                {
                    const auto &device = devices[calibration_panel_view_->intrinsic_calibration_device_index];

                    auto it = captures.find(device.name);
                    it->second->stop();
                    if (it != captures.end())
                    {
                        captures.erase(captures.find(device.name));
                    }

                    const auto stream_it = std::find_if(frame_tile_view_->streams.begin(), frame_tile_view_->streams.end(), [&](const auto &x)
                                                        { return x->name == device.name; });

                    if (stream_it != frame_tile_view_->streams.end())
                    {
                        frame_tile_view_->streams.erase(stream_it);
                    }
                }

                calibration_panel_view_->is_marker_collecting = false;
            }
            return true; });

        calibration_panel_view_->is_masking_changed.push_back([this](const std::vector<calibration_panel_view::device_info> &devices, bool is_masking)
                                                             {
            if (!multiview_capture)
            {
                return false;
            }
            if (is_masking)
            {
                multiview_capture->gen_mask();
                masks = multiview_capture->get_masks();
            }
            else
            {
                multiview_capture->clear_mask();
                masks.clear();
            }
            return true; });

        calibration_panel_view_->is_marker_collecting_changed.push_back([this](const std::vector<calibration_panel_view::device_info> &devices, bool is_marker_collecting)
                                                                       {
            if (!multiview_capture)
            {
                return false;
            }
            if (is_marker_collecting)
            {
                for (const auto &device : devices)
                {
                    multiview_capture->enable_marker_collecting(device.name);
                }
            }
            else
            {
                for (const auto &device : devices)
                {
                    multiview_capture->disable_marker_collecting(device.name);
                }
            }
            return true; });

        calibration_panel_view_->on_intrinsic_calibration_device_changed.push_back([this](const calibration_panel_view::device_info &device) {
            const auto &params = camera_params[device.id].cameras["infra1"];
            calibration_panel_view_->fx = params.intrin.fx;
            calibration_panel_view_->fy = params.intrin.fy;
            calibration_panel_view_->cx = params.intrin.cx;
            calibration_panel_view_->cy = params.intrin.cy;
            calibration_panel_view_->k0 = params.intrin.coeffs[0];
            calibration_panel_view_->k1 = params.intrin.coeffs[1];
            calibration_panel_view_->k2 = params.intrin.coeffs[4];
            calibration_panel_view_->p0 = params.intrin.coeffs[2];
            calibration_panel_view_->p1 = params.intrin.coeffs[3];
            calibration_panel_view_->rms = 0;
        });

        calibration_panel_view_->on_calibrate.push_back([this](const std::vector<calibration_panel_view::device_info> &devices, bool on_calibrate)
                                                       {
                                            
            if (calibration_panel_view_->calibration_target_index == 0)
            {
                for (const auto& device : devices)
                {
                    if (calib.get_num_frames(device.name) > 0)
                    {
                        calib.calibrate();

                        for (const auto &[name, camera] : calib.calibrated_cameras)
                        {
                            marker_server.cameras[name] = camera;
                        }
                        break;
                    }
                }

                for (const auto &device : devices)
                {
                    if (extrinsic_calib.get_num_frames(device.name) > 0)
                    {
                        std::unordered_map<std::string, std::string> device_name_to_id;
                        for (const auto& device : devices)
                        {
                            device_name_to_id[device.name] = device.id;
                        }
                        for (const auto& device : devices)
                        {
                            extrinsic_calib.cameras.insert(std::make_pair(device.name, camera_params.at(device.id).cameras.at("infra1")));
                        }

                        for (auto &[camera_name, camera] : extrinsic_calib.cameras)
                        {
                            camera.extrin.rotation = glm::mat4(1.0);
                            camera.extrin.translation = glm::vec3(1.0);
                        }

                        extrinsic_calib.calibrate();

                        for (const auto &[camera_name, camera] : extrinsic_calib.calibrated_cameras)
                        {
                            const auto& camera_id = device_name_to_id.at(camera_name);

                            marker_server.cameras[name].extrin = camera.extrin;
                            camera_params[camera_id].cameras["infra1"].extrin = camera.extrin;
                            // multiview_image_reconstruction_->cameras[camera_name].intrin = camera.intrin;
                            multiview_image_reconstruction_->cameras[camera_name].extrin = camera.extrin;
                        }

                        stargazer::save_camera_params("../data/config/camera_params.json", camera_params);
                        break;
                    }
                }
                return true;
            }
            else if (calibration_panel_view_->calibration_target_index == 1)
            {
                const auto &device = devices.at(calibration_panel_view_->intrinsic_calibration_device_index);

                intrinsic_calib.image_width = static_cast<int>(std::round(device.params.at("width")));
                intrinsic_calib.image_height = static_cast<int>(std::round(device.params.at("height")));

                intrinsic_calib.calibrate();

                calibration_panel_view_->fx = intrinsic_calib.calibrated_camera.intrin.fx;
                calibration_panel_view_->fy = intrinsic_calib.calibrated_camera.intrin.fy;
                calibration_panel_view_->cx = intrinsic_calib.calibrated_camera.intrin.cx;
                calibration_panel_view_->cy = intrinsic_calib.calibrated_camera.intrin.cy;
                calibration_panel_view_->k0 = intrinsic_calib.calibrated_camera.intrin.coeffs[0];
                calibration_panel_view_->k1 = intrinsic_calib.calibrated_camera.intrin.coeffs[1];
                calibration_panel_view_->k2 = intrinsic_calib.calibrated_camera.intrin.coeffs[4];
                calibration_panel_view_->p0 = intrinsic_calib.calibrated_camera.intrin.coeffs[2];
                calibration_panel_view_->p1 = intrinsic_calib.calibrated_camera.intrin.coeffs[3];
                calibration_panel_view_->rms = intrinsic_calib.rms;

                auto &params = camera_params[device.id].cameras["infra1"];
                params.intrin.fx = intrinsic_calib.calibrated_camera.intrin.fx;
                params.intrin.fy = intrinsic_calib.calibrated_camera.intrin.fy;
                params.intrin.cx = intrinsic_calib.calibrated_camera.intrin.cx;
                params.intrin.cy = intrinsic_calib.calibrated_camera.intrin.cy;
                params.intrin.coeffs[0] = intrinsic_calib.calibrated_camera.intrin.coeffs[0];
                params.intrin.coeffs[1] = intrinsic_calib.calibrated_camera.intrin.coeffs[1];
                params.intrin.coeffs[2] = intrinsic_calib.calibrated_camera.intrin.coeffs[2];
                params.intrin.coeffs[3] = intrinsic_calib.calibrated_camera.intrin.coeffs[3];
                params.intrin.coeffs[4] = intrinsic_calib.calibrated_camera.intrin.coeffs[4];
                params.width = intrinsic_calib.calibrated_camera.width;
                params.height = intrinsic_calib.calibrated_camera.height;
                stargazer::save_camera_params("../data/config/camera_params.json", camera_params);
                return true;
            }
        });
    }

    static bool compute_axis(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::mat4 &axis)
    {
        if (!(std::abs(glm::dot(p1 - p0, p2 - p0)) < 0.01))
        {
            return false;
        }
        const auto origin = p0;
        const auto e1 = p1 - p0;
        const auto e2 = p2 - p0;

        glm::vec3 x_axis = e1;
        glm::vec3 y_axis = e2;

        auto z_axis = glm::cross(glm::normalize(x_axis), glm::normalize(y_axis));
        z_axis = glm::normalize(z_axis);

        const auto y_axis_length = 0.196f;
        const auto scale = y_axis_length / glm::length(y_axis);
        x_axis = glm::normalize(x_axis);
        y_axis = glm::normalize(y_axis);

        axis = glm::mat4(1.0f);

        axis[0] = glm::vec4(x_axis / scale, 0.0f);
        axis[1] = glm::vec4(y_axis / scale, 0.0f);
        axis[2] = glm::vec4(z_axis / scale, 0.0f);
        axis[3] = glm::vec4(origin, 1.0f);

        axis = glm::inverse(axis);

        return true;
    }

    static bool detect_axis(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::mat4& axis)
    {
        glm::vec3 origin;
        glm::vec3 e1, e2;

        // Find origin
        if (std::abs(glm::dot(p1 - p0, p2 - p0)) < 0.01)
        {
            origin = p0;
            e1 = p1 - p0;
            e2 = p2 - p0;
        }
        else if (std::abs(glm::dot(p0 - p1, p2 - p1)) < 0.01)
        {
            origin = p1;
            e1 = p0 - p1;
            e2 = p2 - p1;
        }
        else if (std::abs(glm::dot(p0 - p2, p1 - p2)) < 0.01)
        {
            origin = p2;
            e1 = p0 - p2;
            e2 = p1 - p2;
        }
        else
        {
            return false;
        }

        glm::vec3 x_axis, y_axis;
        if (glm::length(e1) < glm::length(e2))
        {
            x_axis = e1;
            y_axis = e2;
        }
        else
        {
            x_axis = e2;
            y_axis = e1;
        }

        auto z_axis = glm::cross(x_axis, y_axis);
        z_axis = glm::normalize(z_axis);

        const auto y_axis_length = 0.17f;
        const auto scale = y_axis_length / glm::length(y_axis);
        x_axis = glm::normalize(x_axis);
        y_axis = glm::normalize(y_axis);

        axis[0] = glm::vec4(x_axis * scale, 0.0f);
        axis[1] = glm::vec4(y_axis * scale, 0.0f);
        axis[2] = glm::vec4(z_axis * scale, 0.0f);
        axis[3] = glm::vec4(glm::mat3(axis) * -origin, 1.0f);

        return true;
    }

    std::vector<glm::vec3> points;

    void init_reconstruction_panel()
    {
        reconstruction_panel_view_ = std::make_unique<reconstruction_panel_view>();
        for (const auto &device_info : config->get_device_infos())
        {
            reconstruction_panel_view_->devices.push_back(reconstruction_panel_view::device_info{device_info.name, device_info.address});
        }

        reconstruction_panel_view_->is_streaming_changed.push_back([this](const std::vector<reconstruction_panel_view::device_info> &devices, bool is_streaming)
                                                                  {
            namespace fs = std::filesystem;
            if (is_streaming)
            {
                if (reconstruction_panel_view_->source == 0)
                {
                    if (playback)
                    {
                        return false;
                    }

                    std::vector<std::string> names;
                    for (const auto& device : devices)
                    {
                        names.push_back(device.name);
                    }

                    const auto data_dir = "../data";

                    std::ifstream ifs;
                    ifs.open((fs::path(data_dir) / "config.json").string(), std::ios::in);
                    nlohmann::json j_config = nlohmann::json::parse(ifs);
                    const std::string prefix = "capture";

                    const auto markers_directory = fs::path(data_dir) / j_config["directory"].get<std::string>() / (prefix + "_5715_248_150_90");

                    playback.reset(new playback_stream(markers_directory.string(), 0));
                    playback->start(names, [this](const std::map<std::string, std::vector<stargazer::point_data>> &frame)
                                    {
                        marker_server.push_frame(frame); });
                }
                else if (reconstruction_panel_view_->source == 1 || reconstruction_panel_view_->source == 2)
                {
                    if (multiview_capture)
                    {
                        return false;
                    }

                    std::vector<device_info> infos;

                    for (const auto &device : devices)
                    {
                        const auto &device_infos = config->get_device_infos();
                        auto found = std::find_if(device_infos.begin(), device_infos.end(), [device](const auto &x)
                                                  { return x.name == device.name; });
                        if (found == device_infos.end())
                        {
                            return false;
                        }

                        infos.push_back(*found);
                    }

                    if (calibration_panel_view_->is_masking)
                    {
                        multiview_capture.reset(new multiview_capture_pipeline(masks));
                    }
                    else
                    {
                        multiview_capture.reset(new multiview_capture_pipeline());
                    }

                    multiview_capture->add_marker_received([this](const std::map<std::string, marker_frame_data> &marker_frame)
                                                           {
                        std::map<std::string, std::vector<stargazer::point_data>> frame;
                        for (const auto &[name, markers] : marker_frame)
                        {
                            std::vector<stargazer::point_data> points;
                            for (const auto &marker : markers.markers)
                            {
                                points.push_back(stargazer::point_data{glm::vec2(marker.x, marker.y), marker.r, markers.timestamp});
                            }
                            frame.insert(std::make_pair(name, points));
                        }
                        marker_server.push_frame(frame); });

                    multiview_capture->add_image_received([this](const std::map<std::string, cv::Mat> &image_frame)
                                                          {
                        std::map<std::string, cv::Mat> color_image_frame;
                        for (const auto& [name, image] : image_frame)
                        {
                            if (image.channels() == 3 && image.depth() == cv::DataType<uchar>::depth)
                            {
                                color_image_frame[name] = image;
                            }
                        }
                        multiview_image_reconstruction_->push_frame(color_image_frame); });

                    multiview_capture->run(infos);

                    for (const auto &device : devices)
                    {
                        const auto stream = std::make_shared<frame_tile_view::stream_info>(device.name, float2{(float)width, (float)height});
                        frame_tile_view_->streams.push_back(stream);
                    }
                }
            }
            else
            {
                if (playback)
                {
                    playback->stop();
                    playback.reset();
                }
                if (multiview_capture)
                {
                    multiview_capture->stop();
                    multiview_capture.reset();

                    for (const auto &device : devices)
                    {
                        const auto stream_it = std::find_if(frame_tile_view_->streams.begin(), frame_tile_view_->streams.end(), [&](const auto &x)
                                                            { return x->name == device.name; });

                        if (stream_it != frame_tile_view_->streams.end())
                        {
                            frame_tile_view_->streams.erase(stream_it);
                        }
                    }
                }
            }
            return true; });

        reconstruction_panel_view_->set_axis_pressed.push_back([this](const std::vector<reconstruction_panel_view::device_info> &devices)
                                                                  {
            namespace fs = std::filesystem;
            {
                glm::mat4 axis(1.0);

                if (calib.calibrated_cameras.size() > 0)
                {
                    std::vector<std::string> names;
                    for (const auto &device : devices)
                    {
                        names.push_back(device.name);
                    }

                    const auto data_dir = "../data";

                    std::ifstream ifs;
                    ifs.open((fs::path(data_dir) / "config.json").string(), std::ios::in);
                    nlohmann::json j_config = nlohmann::json::parse(ifs);
                    const std::string prefix = "axis";

                    const auto markers_directory = fs::path(data_dir) / j_config["directory"].get<std::string>() / (prefix + "_5715_248_150_90");

                    const auto directory = markers_directory;
                    std::vector<std::uint64_t> frame_numbers;
                    stargazer::list_frame_numbers(directory.string(), frame_numbers);

                    std::size_t max_frames = frame_numbers.size();

                    for (size_t frame_no = 0; frame_no < frame_numbers.size(); frame_no++)
                    {
                        std::string filename = (fs::path(directory) / ("marker_" + std::to_string(frame_numbers[frame_no]) + ".json")).string();
                        if (!fs::exists(filename))
                        {
                            continue;
                        }

                        std::vector<std::vector<stargazer::point_data>> frame_data(names.size());
                        read_frame(filename, names, frame_data);

                        std::map<std::string, std::vector<stargazer::point_data>> points;
                        for (size_t i = 0; i < names.size(); i++)
                        {
                            points.insert(std::make_pair(names[i], frame_data[i]));
                        }

                        const std::map<std::string, stargazer::camera_t> cameras(calib.calibrated_cameras.begin(), calib.calibrated_cameras.end());

                        const auto markers = reconstruct(cameras, points);

                        if (markers.size() == 3)
                        {
                            if (detect_axis(markers[0], markers[1], markers[2], axis))
                            {
                                break;
                            }
                        }
                    }

                    glm::mat4 basis(1.f);
                    basis[0] = glm::vec4(-1.f, 0.f, 0.f, 0.f);
                    basis[1] = glm::vec4(0.f, 0.f, 1.f, 0.f);
                    basis[2] = glm::vec4(0.f, 1.f, 0.f, 0.f);

                    // z up -> opengl
                    axis = basis * axis;
                }
                if (reconstruction_panel_view_->source == 2)
                {
                    const auto images = multiview_capture->get_frames();

                    std::map<std::string, std::vector<stargazer::point_data>> points;
                    std::map<std::string, stargazer::camera_t> cameras;

                    for (const auto& [name, image]: images)
                    {
                        std::vector<int> marker_ids;
                        std::vector<std::vector<cv::Point2f>> marker_corners;
                        detect_aruco_marker(image, marker_corners, marker_ids);

                        std::cout << name << ", " << marker_ids.size() << std::endl;

                        for (size_t i = 0; i < marker_ids.size(); i++)
                        {
                            if (marker_ids[i] == 0)
                            {
                                auto& corner_points = points[name];
                                for (size_t j = 0; j < 3; j++)
                                {
                                    stargazer::point_data point {};
                                    point.point.x = marker_corners[i][j].x;
                                    point.point.y = marker_corners[i][j].y;
                                    corner_points.push_back(point);
                                }

                                for (const auto &device : config->get_device_infos())
                                {
                                    if (name == device.name)
                                    {
                                        cameras[name] = camera_params.at(device.id).cameras.at("infra1");
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    std::vector<glm::vec3> markers;
                    for (size_t j = 0; j < 3; j++)
                    {
                        std::vector<glm::vec2> pts;
                        std::vector<stargazer::camera_t> cams;

                        for (const auto& [name, camera]: cameras)
                        {
                            pts.push_back(points[name][j].point);
                            cams.push_back(camera);
                        }
                        const auto marker = stargazer::reconstruction::triangulate(pts, cams);
                        markers.push_back(marker);
                    }

                    std::cout << markers.size() << std::endl;

                    if (markers.size() == 3)
                    {
                        if (!compute_axis(markers[1], markers[0], markers[2], axis))
                        {
                            std::cout << "Failed to compute axis" << std::endl;
                            return true;
                        }
                    }

                    glm::mat4 basis(1.f);
                    basis[0] = glm::vec4(-1.f, 0.f, 0.f, 0.f);
                    basis[1] = glm::vec4(0.f, 0.f, 1.f, 0.f);
                    basis[2] = glm::vec4(0.f, 1.f, 0.f, 0.f);

                    glm::mat4 cv_to_gl(1.f);
                    cv_to_gl[0] = glm::vec4(1.f, 0.f, 0.f, 0.f);
                    cv_to_gl[1] = glm::vec4(0.f, -1.f, 0.f, 0.f);
                    cv_to_gl[2] = glm::vec4(0.f, 0.f, -1.f, 0.f);

                    // z down -> z up -> opengl
                    axis = basis * axis;
                    // axis = basis * cv_to_gl * extrinsic_calib.axis;

                    this->points.clear();
                    for (const auto &marker : markers)
                    {
                        this->points.push_back(glm::vec3(axis * glm::vec4(marker, 1.0f)));
                    }
                }

                this->axis = axis;

                save_scene();

                this->marker_server.axis = axis;
                this->multiview_image_reconstruction_->axis = axis;
                this->pose_view_->axis = axis;
            }
            return true; });
    }

    void init_gui()
    {
        const char *glsl_version = "#version 130";

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO &io = ImGui::GetIO();
        (void)io;
        // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
        // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();
        // ImGui::StyleColorsLight();

        // Setup Platform/Renderer backends
        ImGui_ImplGlfw_InitForOpenGL((GLFWwindow *)get_handle(), true);
        ImGui_ImplOpenGL3_Init(glsl_version);

        static const ImWchar icons_ranges[] = {0xf000, 0xf999, 0};

        {
            const int OVERSAMPLE = true;

            ImFontConfig config_words;
            config_words.OversampleV = OVERSAMPLE;
            config_words.OversampleH = OVERSAMPLE;
            default_font = io.Fonts->AddFontFromFileTTF("../fonts/mplus/fonts/ttf/Mplus2-Regular.ttf", 16.0f, &config_words, io.Fonts->GetGlyphRangesJapanese());

            ImFontConfig config_glyphs;
            config_glyphs.MergeMode = true;
            config_glyphs.OversampleV = OVERSAMPLE;
            config_glyphs.OversampleH = OVERSAMPLE;
            default_font = io.Fonts->AddFontFromMemoryCompressedTTF(font_awesome_compressed_data,
                                                                    font_awesome_compressed_size, 14.f, &config_glyphs, icons_ranges);
        }
        IM_ASSERT(default_font != NULL);

        {
            const int OVERSAMPLE = true;

            ImFontConfig config_words;
            config_words.OversampleV = OVERSAMPLE;
            config_words.OversampleH = OVERSAMPLE;
            large_font = io.Fonts->AddFontFromFileTTF("../fonts/mplus/fonts/ttf/Mplus2-Regular.ttf", 20.0f, &config_words, io.Fonts->GetGlyphRangesJapanese());

            ImFontConfig config_glyphs;
            config_glyphs.MergeMode = true;
            config_glyphs.OversampleV = OVERSAMPLE;
            config_glyphs.OversampleH = OVERSAMPLE;
            large_font = io.Fonts->AddFontFromMemoryCompressedTTF(font_awesome_compressed_data,
                                                                  font_awesome_compressed_size, 20.f, &config_glyphs, icons_ranges);
        }
        IM_ASSERT(large_font != NULL);

        {
            ImGuiStyle &style = ImGui::GetStyle();

            style.WindowRounding = 0.0f;
            style.ScrollbarRounding = 0.0f;

            style.Colors[ImGuiCol_WindowBg] = dark_window_background;
            style.Colors[ImGuiCol_Border] = black;
            style.Colors[ImGuiCol_BorderShadow] = transparent;
            style.Colors[ImGuiCol_FrameBg] = dark_window_background;
            style.Colors[ImGuiCol_ScrollbarBg] = scrollbar_bg;
            style.Colors[ImGuiCol_ScrollbarGrab] = scrollbar_grab;
            style.Colors[ImGuiCol_ScrollbarGrabHovered] = scrollbar_grab + 0.1f;
            style.Colors[ImGuiCol_ScrollbarGrabActive] = scrollbar_grab + (-0.1f);
            // style.Colors[ImGuiCol_ComboBg] = dark_window_background;
            style.Colors[ImGuiCol_CheckMark] = regular_blue;
            style.Colors[ImGuiCol_SliderGrab] = regular_blue;
            style.Colors[ImGuiCol_SliderGrabActive] = regular_blue;
            style.Colors[ImGuiCol_Button] = button_color;
            style.Colors[ImGuiCol_ButtonHovered] = button_color + 0.1f;
            style.Colors[ImGuiCol_ButtonActive] = button_color + (-0.1f);
            style.Colors[ImGuiCol_Header] = header_color;
            style.Colors[ImGuiCol_HeaderActive] = header_color + (-0.1f);
            style.Colors[ImGuiCol_HeaderHovered] = header_color + 0.1f;
            style.Colors[ImGuiCol_TitleBg] = title_color;
            style.Colors[ImGuiCol_TitleBgCollapsed] = title_color;
            style.Colors[ImGuiCol_TitleBgActive] = header_color;
        }
    }

    std::map<std::string, stargazer::camera_module_t> camera_params;

    void load_camera_params()
    {
        const auto &camera_ids = calib.get_camera_ids();
        const auto &camera_names = calib.get_camera_names();
        const auto num_cameras = camera_names.size();

        camera_params = stargazer::load_camera_params("../data/config/camera_params.json");

        for (std::size_t i = 0; i < camera_names.size(); i++)
        {
            calib.cameras.insert(std::make_pair(camera_names[i], camera_params.at(camera_ids[i]).cameras.at("infra1")));
        }
        assert(calib.cameras.size() == num_cameras);

        for (auto &[camera_name, camera] : calib.cameras)
        {
            camera.extrin.rotation = glm::mat4(1.0);
            camera.extrin.translation = glm::vec3(1.0);
        }
    }

    glm::mat4 axis;

    void load_scene()
    {
        const auto path = "../data/config/reconstruction.json";

        std::ifstream ifs;
        ifs.open(path, std::ios::binary | std::ios::in);

        if (!ifs)
        {
            axis = glm::mat4(1.0f);
            return;
        }

        const auto j = nlohmann::json::parse(ifs);
        axis = j["scene"]["axis"].get<glm::mat4>();
    }

    void save_scene()
    {
        const auto path = "../data/config/reconstruction.json";

        std::ofstream ofs;
        ofs.open(path, std::ios::out);

        auto j = nlohmann::json{};
        j["scene"]["axis"] = axis;
        ofs << j.dump(2);
    }

    virtual void show() override
    {
        gladLoadGL();

        config.reset(new stargazer::configuration_file("../data/config/capture.json"));

        load_camera_params();

        for (const auto& device : config->get_device_infos())
        {
            multiview_image_reconstruction_->cameras.insert(std::make_pair(device.name, camera_params.at(device.id).cameras.at("infra1")));
        }

        load_scene();

        init_gui();

        context = std::make_unique<view_context>();
        context->window = (GLFWwindow *)get_handle();
        context->default_font = default_font;
        context->large_font = large_font;

        top_bar_view_ = std::make_unique<top_bar_view>();
        frame_tile_view_ = std::make_unique<frame_tile_view>();
        contrail_tile_view_ = std::make_unique<frame_tile_view>();

        init_capture_panel();
        init_calibration_panel();
        init_reconstruction_panel();
#if 0
        {
            struct camera_data
            {
                double fx;
                double fy;
                double cx;
                double cy;
                std::array<double, 3> k;
                std::array<double, 2> p;
                std::array<std::array<double, 3>, 3> rotation;
                std::array<double, 3> translation;
            };

            const auto& devices = this->config->get_device_infos();
            for (size_t i = 0; i < devices.size(); i++)
            {
                const auto name = devices[i].name;

                camera_data camera;

                const auto &src_camera = camera_params.at(devices[i].id).cameras.at("infra1");

                camera.fx = src_camera.intrin.fx;
                camera.fy = src_camera.intrin.fy;
                camera.cx = src_camera.intrin.cx;
                camera.cy = src_camera.intrin.cy;
                camera.k[0] = src_camera.intrin.coeffs[0];
                camera.k[1] = src_camera.intrin.coeffs[3];
                camera.k[2] = src_camera.intrin.coeffs[4];
                camera.p[0] = src_camera.intrin.coeffs[1];
                camera.p[1] = src_camera.intrin.coeffs[2];

                std::cout << name << std::endl;
                std::cout << "camera.fx = " << camera.fx << ";" << std::endl;
                std::cout << "camera.fy = " << camera.fy << ";" << std::endl;
                std::cout << "camera.cx = " << camera.cx << ";" << std::endl;
                std::cout << "camera.cy = " << camera.cy << ";" << std::endl;
                std::cout << "camera.k[0] = " << camera.k[0] << ";" << std::endl;
                std::cout << "camera.k[1] = " << camera.k[1] << ";" << std::endl;
                std::cout << "camera.k[2] = " << camera.k[2] << ";" << std::endl;
                std::cout << "camera.p[0] = " << camera.p[0] << ";" << std::endl;
                std::cout << "camera.p[1] = " << camera.p[1] << ";" << std::endl;

                glm::mat4 basis(1.f);
                basis[0] = glm::vec4(-1.f, 0.f, 0.f, 0.f);
                basis[1] = glm::vec4(0.f, 0.f, 1.f, 0.f);
                basis[2] = glm::vec4(0.f, 1.f, 0.f, 0.f);

                const auto axis = glm::inverse(basis) * this->axis;
                const auto camera_pose = axis * glm::inverse(src_camera.extrin.rotation);

                for (size_t i = 0; i < 3; i++)
                {
                    for (size_t j = 0; j < 3; j++)
                    {
                        camera.rotation[i][j] = camera_pose[i][j];
                    }
                    camera.translation[i] = camera_pose[3][i];
                }

                std::cout << "camera.rotation = {{";
                for (size_t i = 0; i < 3; i++)
                {
                    std::cout << "{" << std::endl;
                    for (size_t j = 0; j < 3; j++)
                    {
                        std::cout << camera.rotation[i][j] << ", ";
                    }
                    std::cout << "}," << std::endl;
                }
                std::cout << "}};" << std::endl;
                std::cout << "camera.translation = {";
                for (size_t i = 0; i < 3; i++)
                {
                    std::cout << camera.translation[i] << ", ";
                    std::cout << std::endl;
                }
                std::cout << "};" << std::endl;
            }
        }
#endif

        view_controller = std::make_shared<azimuth_elevation>(glm::u32vec2(0, 0), glm::u32vec2(width, height));
        pose_view_ = std::make_unique<pose_view>();

        {
            this->marker_server.axis = this->axis;
            this->multiview_image_reconstruction_->axis = this->axis;
            this->pose_view_->axis = this->axis;
        }

        {
            namespace fs = std::filesystem;
            const std::string prefix = "calibrate";
            const std::string data_dir = "../data";

            std::ifstream ifs;
            ifs.open((fs::path(data_dir) / "config.json").string(), std::ios::in);
            nlohmann::json j_config = nlohmann::json::parse(ifs);

            const auto camera_names = j_config["cameras"].get<std::vector<std::string>>();
            const auto camera_ids = j_config["camera_ids"].get<std::vector<std::string>>();
            const auto num_cameras = camera_names.size();

            const auto markers_directory = fs::path(data_dir) / j_config["directory"].get<std::string>() / (prefix + "_5715_248_150_90");

            std::vector<std::vector<std::vector<stargazer::point_data>>> frames_data(num_cameras);
            read_points(markers_directory.string(), camera_names, frames_data);

            std::size_t max_frames = 0;
            for (const auto &frames : frames_data)
            {
                max_frames = std::max(max_frames, frames.size());
            }
            for (auto &frames : frames_data)
            {
                frames.resize(max_frames);
            }

            for (size_t f = 0; f < max_frames; f++)
            {
                std::map<std::string, std::vector<stargazer::point_data>> frame;
                for (size_t c = 0; c < num_cameras; c++)
                {
                    frame.insert(std::make_pair(camera_names[c], frames_data[c][f]));
                }
                calib.add_frame(frame);
            }
        }

        marker_server.run();
        multiview_image_reconstruction_->run();

        window_base::show();
    }

    virtual void on_close() override
    {
        marker_server.stop();
        multiview_image_reconstruction_->stop();

        std::lock_guard<std::mutex> lock(mtx);
        window_manager::get_instance()->exit();
        window_base::on_close();
    }

    virtual void on_scroll(double x, double y) override
    {
        if (view_controller)
        {
            view_controller->scroll(x, y);
        }
    }

    calibration calib;
    intrinsic_calibration intrinsic_calib;
    extrinsic_calibration extrinsic_calib;

    virtual void update() override
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (handle == nullptr)
        {
            return;
        }
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

#if 1
        std::unordered_map<std::string, cv::Mat> images;
#endif

        std::vector<std::map<std::string, marker_frame_data>> marker_frames;
        if (calibration_panel_view_->is_marker_collecting)
        {
            if (multiview_capture)
            {
                const auto marker_frames = multiview_capture->pop_marker_frames();

                for (const auto &marker_frame : marker_frames)
                {
                    std::map<std::string, std::vector<stargazer::point_data>> frame;
                    for (const auto &[name, markers] : marker_frame)
                    {
                        std::vector<stargazer::point_data> points;
                        for (const auto &marker : markers.markers)
                        {
                            points.push_back(stargazer::point_data{glm::vec2(marker.x, marker.y), marker.r, markers.timestamp});
                        }
                        frame.insert(std::make_pair(name, points));
                    }
                    calib.add_frame(frame);
                }
            }
            else
            {
                for (const auto &device : capture_panel_view_->devices)
                {
                    const auto capture_it = captures.find(device.name);
                    if (capture_it != captures.end())
                    {
                        const auto capture = capture_it->second;
                        auto frame = capture->get_frame();

                        if (!frame.empty())
                        {
                            std::vector<cv::Point2f> board;
                            if (detect_calibration_board(frame, board))
                            {
                                std::vector<stargazer::point_data> points;
                                for (const auto &point : board)
                                {
                                    points.push_back(stargazer::point_data{glm::vec2(point.x, point.y), 0, 0});
                                }
                                intrinsic_calib.add_frame(points);
                            }
#if 0
                            {
                                std::vector<cv::Point2f> board;
                                if (detect_calibration_board(frame, board, calibration_pattern::ASYMMETRIC_CIRCLES_GRID))
                                {
                                    const auto board_size = cv::Size(4, 11);
                                    cv::drawChessboardCorners(frame, board_size, cv::Mat(board), true);

                                    {
                                        const auto mm_to_m = 0.001f;                                                                                                    // TODO: Define as config
                                        const auto square_size = cv::Size2f(117.0f / (board_size.width - 1) / 2 * mm_to_m, 196.0f / (board_size.height - 1) * mm_to_m); // TODO: Define as config
                                        const auto image_size = cv::Size(960, 540);

                                        std::vector<cv::Point3f> object_point;
                                        std::vector<cv::Point2f> image_point = board;
                                        calc_board_corner_positions(board_size, square_size, object_point, calibration_pattern::ASYMMETRIC_CIRCLES_GRID);

                                        cv::Mat camera_matrix;
                                        cv::Mat dist_coeffs;
                                        const auto &device_infos = config->get_device_infos();
                                        if (const auto device_info = std::find_if(device_infos.begin(), device_infos.end(), [&](const auto &x)
                                                                                  { return x.name == device.name; });
                                            device_info != device_infos.end())
                                        {
                                            stargazer::get_cv_intrinsic(camera_params.at(device_info->id).cameras.at("infra1").intrin, camera_matrix, dist_coeffs);
                                        }

                                        cv::Mat rvec, tvec;
                                        cv::solvePnP(object_point, image_point, camera_matrix, dist_coeffs, rvec, tvec);

                                        {
                                            constexpr auto length = 0.2f;
                                            std::vector<cv::Point3f> object_points = {
                                                cv::Point3f(0, 0, 0),
                                                cv::Point3f(length, 0, 0),
                                                cv::Point3f(length, length, 0),
                                                cv::Point3f(0, length, 0),
                                                cv::Point3f(0, 0, -length),
                                                cv::Point3f(length, 0, -length),
                                                cv::Point3f(length, length, -length),
                                                cv::Point3f(0, length, -length),
                                            };
                                            std::vector<cv::Point2f> image_points;
                                            cv::projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs, image_points);

                                            cv::line(frame, image_points.at(1), image_points.at(0), cv::Scalar(0, 0, 255));
                                            cv::line(frame, image_points.at(2), image_points.at(1), cv::Scalar(0, 0, 255));
                                            cv::line(frame, image_points.at(3), image_points.at(2), cv::Scalar(0, 0, 255));
                                            cv::line(frame, image_points.at(0), image_points.at(3), cv::Scalar(0, 0, 255));
                                            cv::line(frame, image_points.at(5), image_points.at(4), cv::Scalar(255, 0, 0));
                                            cv::line(frame, image_points.at(6), image_points.at(5), cv::Scalar(255, 0, 0));
                                            cv::line(frame, image_points.at(7), image_points.at(6), cv::Scalar(255, 0, 0));
                                            cv::line(frame, image_points.at(4), image_points.at(7), cv::Scalar(255, 0, 0));
                                            cv::line(frame, image_points.at(4), image_points.at(0), cv::Scalar(0, 255, 0));
                                            cv::line(frame, image_points.at(5), image_points.at(1), cv::Scalar(0, 255, 0));
                                            cv::line(frame, image_points.at(6), image_points.at(2), cv::Scalar(0, 255, 0));
                                            cv::line(frame, image_points.at(7), image_points.at(3), cv::Scalar(0, 255, 0));
                                            // cv::line(frame, image_points.at(0), image_points.at(4), cv::Scalar(255, 0, 0));
                                        }
                                    }

                                    images[device.name] = frame;
                                }
                            }
#endif
#if 0
                            {
                                cv::aruco::DetectorParameters detector_params = cv::aruco::DetectorParameters();
                                detector_params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_CONTOUR;
                                // detector_params.minMarkerPerimeterRate = 0.001;
                                // detector_params.adaptiveThreshWinSizeStep = 10;
                                // detector_params.adaptiveThreshWinSizeMax = 23;
                                // detector_params.adaptiveThreshWinSizeMax = 73;
                                // detector_params.polygonalApproxAccuracyRate = 0.1;
                                // detector_params.adaptiveThreshWinSizeMax = 43;
                                // detector_params.adaptiveThreshConstant = 50;
                                cv::aruco::CharucoParameters charucoParams = cv::aruco::CharucoParameters();
                                const auto dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
                                const auto board = cv::aruco::CharucoBoard(cv::Size(3, 5), 0.0575, 0.0575 * 0.75f, dictionary);
                                const auto detector = cv::aruco::CharucoDetector(board, charucoParams, detector_params);
                                std::vector<int> markerIds;
                                std::vector<std::vector<cv::Point2f>> markerCorners;
                                std::vector<int> charucoIds;
                                std::vector<cv::Point2f> charucoCorners;

                                try
                                {
                                    detector.detectBoard(frame, charucoCorners, charucoIds, markerCorners, markerIds);
                                }
                                catch(const cv::Exception& e)
                                {
                                    std::cerr << e.what() << '\n';
                                }
                                
                                cv::aruco::drawDetectedCornersCharuco(frame, charucoCorners, charucoIds);
                                images[device.name] = frame;
                            }
#endif
#if 1
                            {
                                cv::aruco::DetectorParameters detector_params = cv::aruco::DetectorParameters();
                                detector_params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_CONTOUR;
                                // detector_params.minMarkerPerimeterRate = 0.001;
                                // detector_params.adaptiveThreshWinSizeStep = 10;
                                // detector_params.adaptiveThreshWinSizeMax = 23;
                                // detector_params.adaptiveThreshWinSizeMax = 73;
                                // detector_params.polygonalApproxAccuracyRate = 0.1;
                                // detector_params.adaptiveThreshWinSizeMax = 43;
                                // detector_params.adaptiveThreshConstant = 50;
                                const auto dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
                                const auto detector = cv::aruco::ArucoDetector(dictionary, detector_params);
                                std::vector<int> markerIds;
                                std::vector<std::vector<cv::Point2f>> markerCorners;

                                try
                                {
                                    detector.detectMarkers(frame, markerCorners, markerIds);
                                }
                                catch (const cv::Exception &e)
                                {
                                    std::cerr << e.what() << '\n';
                                }

                                cv::aruco::drawDetectedMarkers(frame, markerCorners, markerIds);
                                images[device.name] = frame;
                            }
#endif
#if 0
                            {
                                const auto markers = capture->get_markers();

                                std::vector<int> charucoIds;
                                std::vector<cv::Point2f> charucoCorners;

                                for (const auto& marker : markers)
                                {
                                    charucoIds.push_back(marker.first);
                                    charucoCorners.push_back(marker.second);
                                }
                                cv::aruco::drawDetectedCornersCharuco(frame, charucoCorners, charucoIds);
                            }
#endif
                        }
                    }
                }
            }
        }

        top_bar_view_->render(context.get());

        if (top_bar_view_->view_type == top_bar_view::ViewType::Image)
        {
            if (multiview_capture)
            {
                const auto frames = multiview_capture->get_frames();
                for (const auto &[name, frame] : frames)
                {
                    const auto device_name = name;
                    if (!frame.empty())
                    {
                        const auto stream_it = std::find_if(frame_tile_view_->streams.begin(), frame_tile_view_->streams.end(), [device_name](const auto &x)
                                                            { return x->name == device_name; });

                        if (stream_it != frame_tile_view_->streams.end())
                        {
                            cv::Mat image = frame;
                            cv::Mat color_image;
                            if (image.channels() == 1)
                            {
                                cv::cvtColor(image, color_image, cv::COLOR_GRAY2RGB);
                            }
                            else if (image.channels() == 3)
                            {
                                cv::cvtColor(image, color_image, cv::COLOR_BGR2RGB);
                            }
                            (*stream_it)->texture.upload_image(color_image.cols, color_image.rows, color_image.data, GL_RGB);
                        }
                    }
                }
            }

            for (const auto &device : capture_panel_view_->devices)
            {
                const auto capture_it = captures.find(device.name);
                if (capture_it != captures.end())
                {
                    const auto capture = capture_it->second;
                    auto frame = capture->get_frame();

                    if (!frame.empty())
                    {
                        const auto stream_it = std::find_if(frame_tile_view_->streams.begin(), frame_tile_view_->streams.end(), [&](const auto &x)
                                                            { return x->name == device.name; });

                        if (stream_it != frame_tile_view_->streams.end())
                        {
                            cv::Mat image = frame;
#if 1
                            if (images.find(device.name) != images.end())
                            {
                                image = images.at(device.name);
                            }
#endif
                            cv::Mat color_image;
                            if (image.channels() == 1)
                            {
                                cv::cvtColor(image, color_image, cv::COLOR_GRAY2RGB);
                            }
                            else if (image.channels() == 3)
                            {
                                cv::cvtColor(image, color_image, cv::COLOR_BGR2RGB);
                            }
                            (*stream_it)->texture.upload_image(color_image.cols, color_image.rows, color_image.data, GL_RGB);
                        }
                    }
                }
            }

            frame_tile_view_->render(context.get());
        }

        if (top_bar_view_->view_mode == top_bar_view::Mode::Capture)
        {
            capture_panel_view_->render(context.get());
        }
        else if (top_bar_view_->view_mode == top_bar_view::Mode::Calibration)
        {
            if (calibration_panel_view_->calibration_target_index == 0)
            {
                for (auto &device : calibration_panel_view_->devices)
                {
                    const auto &device_infos = config->get_device_infos();
                    if (const auto device_info = std::find_if(device_infos.begin(), device_infos.end(), [&](const auto &x)
                                                              { return x.name == device.name; });
                        device_info != device_infos.end())
                    {
                        if (device_info->type == device_type::depthai_color || device_info->type == device_type::raspi_color || device_info->type == device_type::rs_d435_color)
                        {
                            device.num_points = extrinsic_calib.get_num_frames(device.name);
                        }
                        else
                        {
                            device.num_points = calib.get_num_frames(device.name);
                        }
                    }
                }
            }
            else if (calibration_panel_view_->calibration_target_index == 1)
            {
                calibration_panel_view_->devices[calibration_panel_view_->intrinsic_calibration_device_index].num_points = intrinsic_calib.get_num_frames();
            }

            calibration_panel_view_->render(context.get());

            if (top_bar_view_->view_type == top_bar_view::ViewType::Pose)
            {
                view_controller->update(mouse_state::get_mouse_state(handle));
                float radius = view_controller->get_radius();
                glm::vec3 forward(0.f, 0.f, 1.f);
                glm::vec3 up(0.0f, 1.0f, 0.0f);
                glm::vec3 view_pos = glm::rotate(glm::inverse(view_controller->get_rotation_quaternion()), forward * radius);
                glm::mat4 view = glm::lookAt(view_pos, glm::vec3(0, 0, 0), up);

                context->view = view;

                for (const auto &device : config->get_device_infos())
                {
                    const auto &camera = camera_params.at(device.id).cameras.at("infra1");
                    pose_view_->cameras[device.name] = pose_view::camera_t{
                        (int)camera.width,
                        (int)camera.height,
                        camera.intrin.cx,
                        camera.intrin.cy,
                        camera.intrin.fx,
                        camera.intrin.fy,
                        camera.intrin.coeffs,
                        glm::inverse(camera.extrin.rotation),
                    };
                }

                pose_view_->render(context.get());
            }
            else if (top_bar_view_->view_type == top_bar_view::ViewType::Contrail)
            {
                for (const auto &device : calibration_panel_view_->devices)
                {
                    std::shared_ptr<frame_tile_view::stream_info> stream;
                    const int width = static_cast<int>(std::round(device.params.at("width")));
                    const int height = static_cast<int>(std::round(device.params.at("height")));

                    const auto found = std::find_if(contrail_tile_view_->streams.begin(), contrail_tile_view_->streams.end(), [&](const auto &x)
                                                    { return x->name == device.name; });
                    if (found == contrail_tile_view_->streams.end())
                    {
                        stream = std::make_shared<frame_tile_view::stream_info>(device.name, float2{(float)width, (float)height});
                        contrail_tile_view_->streams.push_back(stream);
                    }
                    else
                    {
                        stream = *found;
                    }

                    const auto observed_points = calib.get_observed_points(device.name);
                    cv::Mat cloud_image(height, width, CV_8UC3, cv::Scalar::all(0));

                    for (const auto &observed_point : observed_points)
                    {
                        for (const auto &point : observed_point.points)
                        {
                            cv::circle(cloud_image, cv::Point(point.x, point.y), 5, cv::Scalar(255, 0, 0));
                        }
                    }
                    stream->texture.upload_image(cloud_image.cols, cloud_image.rows, cloud_image.data, GL_RGB);
                }

                contrail_tile_view_->render(context.get());
            }
        }
        else if (top_bar_view_->view_mode == top_bar_view::Mode::Reconstruction)
        {
            reconstruction_panel_view_->render(context.get());

            if (top_bar_view_->view_type == top_bar_view::ViewType::Pose)
            {
                view_controller->update(mouse_state::get_mouse_state(handle));
                float radius = view_controller->get_radius();
                glm::vec3 forward(0.f, 0.f, 1.f);
                glm::vec3 up(0.0f, 1.0f, 0.0f);
                glm::vec3 view_pos = glm::rotate(glm::inverse(view_controller->get_rotation_quaternion()), forward * radius);
                glm::mat4 view = glm::lookAt(view_pos, glm::vec3(0, 0, 0), up);

                context->view = view;

                for (const auto &device : config->get_device_infos())
                {
                    const auto &camera = camera_params.at(device.id).cameras.at("infra1");
                    pose_view_->cameras[device.name] = pose_view::camera_t{
                        (int)camera.width,
                        (int)camera.height,
                        camera.intrin.cx,
                        camera.intrin.cy,
                        camera.intrin.fx,
                        camera.intrin.fy,
                        camera.intrin.coeffs,
                        glm::inverse(camera.extrin.rotation),
                    };
                }
                pose_view_->points.clear();
                for (const auto& point : marker_server.get_markers())
                {
                    pose_view_->points.push_back(point);
                }
                for (const auto &point : multiview_image_reconstruction_->get_markers())
                {
                    pose_view_->points.push_back(point);
                }
                for (const auto &point : this->points)
                {
                    pose_view_->points.push_back(point);
                }

                pose_view_->render(context.get());
            }
            else if (top_bar_view_->view_type == top_bar_view::ViewType::Point)
            {
                if (multiview_capture)
                {
                    const auto frames = multiview_image_reconstruction_->get_features();
                    for (const auto &[name, frame] : frames)
                    {
                        const auto device_name = name;
                        if (!frame.empty())
                        {
                            const auto stream_it = std::find_if(frame_tile_view_->streams.begin(), frame_tile_view_->streams.end(), [device_name](const auto &x)
                                                                { return x->name == device_name; });

                            if (stream_it != frame_tile_view_->streams.end())
                            {
                                cv::Mat image = frame;
                                cv::Mat color_image;
                                if (image.channels() == 1)
                                {
                                    cv::cvtColor(image, color_image, cv::COLOR_GRAY2RGB);
                                }
                                else if (image.channels() == 3)
                                {
                                    cv::cvtColor(image, color_image, cv::COLOR_BGR2RGB);
                                }
                                (*stream_it)->texture.upload_image(color_image.cols, color_image.rows, color_image.data, GL_RGB);
                            }
                        }
                    }
                }

                for (const auto &device : capture_panel_view_->devices)
                {
                    const auto capture_it = captures.find(device.name);
                    if (capture_it != captures.end())
                    {
                        const auto capture = capture_it->second;
                        auto frame = capture->get_frame();

                        if (!frame.empty())
                        {
                            const auto stream_it = std::find_if(frame_tile_view_->streams.begin(), frame_tile_view_->streams.end(), [&](const auto &x)
                                                                { return x->name == device.name; });

                            if (stream_it != frame_tile_view_->streams.end())
                            {
                                cv::Mat image = frame;
                                cv::Mat color_image;
                                if (image.channels() == 1)
                                {
                                    cv::cvtColor(image, color_image, cv::COLOR_GRAY2RGB);
                                }
                                else if (image.channels() == 3)
                                {
                                    cv::cvtColor(image, color_image, cv::COLOR_BGR2RGB);
                                }
                                (*stream_it)->texture.upload_image(color_image.cols, color_image.rows, color_image.data, GL_RGB);
                            }
                        }
                    }
                }

                frame_tile_view_->render(context.get());
            }
        }

        ImGui::Render();

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

    virtual void on_char(unsigned int codepoint) override
    {
    }
};

static std::vector<std::function<void()>> on_shutdown_handlers;
static std::atomic_bool exit_flag(false);

static void shutdown()
{
    std::for_each(std::rbegin(on_shutdown_handlers), std::rend(on_shutdown_handlers), [](auto handler)
                  { handler(); });
    exit_flag.store(true);
}

int reconstruction_viewer_main()
{
    const auto win_mgr = window_manager::get_instance();
    win_mgr->initialize();

    on_shutdown_handlers.push_back([win_mgr]()
                                   { win_mgr->terminate(); });

    const auto viewer = std::make_shared<reconstruction_viewer>();

    const auto rendering_th = std::make_shared<rendering_thread>();
    rendering_th->start(viewer.get());

    on_shutdown_handlers.push_back([rendering_th, viewer]()
                                   {
        rendering_th->stop();
        viewer->destroy(); });

    while (!win_mgr->should_close())
    {
        win_mgr->handle_event();
    }

    shutdown();

    return 0;
}

int main()
{
    return reconstruction_viewer_main();
}
