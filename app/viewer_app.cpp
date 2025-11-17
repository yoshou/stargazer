#include <glad/glad.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "viewer.hpp"
#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>
#include <signal.h>

#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <memory>

#include "calibration_pipeline.hpp"
#include "capture_pipeline.hpp"
#include "config.hpp"
#include "parameters.hpp"
#include "reconstruction_pipeline.hpp"
#include "views.hpp"

using namespace stargazer;

const int SCREEN_WIDTH = 1680;
const int SCREEN_HEIGHT = 1050;

class viewer_app : public window_base {
  ImFont* large_font;
  ImFont* default_font;

  std::unique_ptr<view_context> context;
  std::unique_ptr<top_bar_view> top_bar_view_;
  std::unique_ptr<capture_panel_view> capture_panel_view_;
  std::unique_ptr<calibration_panel_view> calibration_panel_view_;
  std::unique_ptr<reconstruction_panel_view> reconstruction_panel_view_;
  std::unique_ptr<image_tile_view> image_tile_view_;
  std::unique_ptr<image_tile_view> contrail_tile_view_;
  std::unique_ptr<pose_view> pose_view_;
  std::shared_ptr<azimuth_elevation> view_controller;

  std::shared_ptr<parameters_t> parameters;
  std::map<std::string, cv::Mat> masks;

  std::map<std::string, std::shared_ptr<capture_pipeline>> captures;
  std::shared_ptr<multiview_capture_pipeline> multiview_capture;

  std::unique_ptr<calibration_pipeline> calib;
  std::unique_ptr<intrinsic_calibration_pipeline> intrinsic_calib;
  std::unique_ptr<axis_calibration_pipeline> axis_calib;

  std::unique_ptr<multiview_point_reconstruction_pipeline> multiview_point_reconstruction_pipeline_;
  std::unique_ptr<multiview_image_reconstruction_pipeline> multiview_image_reconstruction_pipeline_;

  std::unique_ptr<configuration> capture_config;
  std::unique_ptr<configuration> reconstruction_config;
  std::unique_ptr<configuration> calibration_config;

  std::string generate_new_id() const {
    uint64_t max_id = 0;
    for (const auto& node_info : capture_config->get_node_infos()) {
      size_t idx = 0;
      const auto id_str = node_info.get_param<std::string>("id");
      const auto id = std::stoull(id_str, &idx);

      if (id_str.size() == idx) {
        max_id = std::max(max_id, static_cast<uint64_t>(id + 1));
      }
    }
    return fmt::format("{:>012d}", max_id);
  }

  void init_capture_panel() {
    capture_panel_view_ = std::make_unique<capture_panel_view>();
    for (const auto& node_info : capture_config->get_node_infos()) {
      std::string path;
      if (node_info.contains_param("address")) {
        path = node_info.get_param<std::string>("address");
      }
      if (node_info.contains_param("db_path")) {
        path = node_info.get_param<std::string>("db_path");
      }
      capture_panel_view_->devices.push_back(
          capture_panel_view::node_info{node_info.name, path, node_info.params});
    }

    capture_panel_view_->is_streaming_changed.push_back(
        [this](const capture_panel_view::node_info& device) {
          if (device.is_streaming == true) {
            const auto& node_infos = capture_config->get_node_infos();
            auto found = std::find_if(node_infos.begin(), node_infos.end(),
                                      [device](const auto& x) { return x.name == device.name; });
            if (found == node_infos.end()) {
              spdlog::error("Device {} not found in capture config", device.name);
              return false;
            }
            const auto& node_info = *found;

            const auto capture = std::make_shared<capture_pipeline>();

            try {
              capture->run(node_info);
            } catch (std::exception& e) {
              spdlog::error("Failed to start capture for {}: {}", device.name, e.what());
              return false;
            }
            captures.insert(std::make_pair(device.name, capture));
            const auto width = static_cast<int>(std::round(node_info.get_param<float>("width")));
            const auto height = static_cast<int>(std::round(node_info.get_param<float>("height")));

            const auto stream = std::make_shared<image_tile_view::stream_info>(
                device.name, float2{(float)width, (float)height});
            image_tile_view_->streams.push_back(stream);
          } else {
            auto it = captures.find(device.name);
            it->second->stop();
            if (it != captures.end()) {
              captures.erase(captures.find(device.name));
            }

            const auto stream_it =
                std::find_if(image_tile_view_->streams.begin(), image_tile_view_->streams.end(),
                             [&](const auto& x) { return x->name == device.name; });

            if (stream_it != image_tile_view_->streams.end()) {
              image_tile_view_->streams.erase(stream_it);
            }
          }

          return true;
        });

    capture_panel_view_->is_all_streaming_changed.push_back(
        [this](const std::vector<capture_panel_view::node_info>& devices, bool is_streaming) {
          if (is_streaming) {
            if (multiview_capture) {
              return false;
            }

            const auto& node_infos = capture_config->get_node_infos();

            if (calibration_panel_view_->is_masking) {
              multiview_capture.reset(new multiview_capture_pipeline(masks));
            } else {
              multiview_capture.reset(new multiview_capture_pipeline());
            }

            for (const auto& node_info : node_infos) {
              if (node_info.is_camera()) {
                multiview_capture->enable_marker_collecting(node_info.name);
              }
            }

            multiview_capture->run(node_infos);

            for (const auto& node_info : node_infos) {
              if (node_info.is_camera()) {
                const auto width =
                    static_cast<int>(std::round(node_info.get_param<float>("width")));
                const auto height =
                    static_cast<int>(std::round(node_info.get_param<float>("height")));
                const auto stream = std::make_shared<image_tile_view::stream_info>(
                    node_info.name, float2{(float)width, (float)height});
                image_tile_view_->streams.push_back(stream);
              }
            }
          } else {
            if (multiview_capture) {
              multiview_capture->stop();
              multiview_capture.reset();

              const auto& node_infos = capture_config->get_node_infos();

              for (const auto& node_info : node_infos) {
                const auto stream_it =
                    std::find_if(image_tile_view_->streams.begin(), image_tile_view_->streams.end(),
                                 [&](const auto& x) { return x->name == node_info.name; });

                if (stream_it != image_tile_view_->streams.end()) {
                  image_tile_view_->streams.erase(stream_it);
                }
              }
            }
          }
          return true;
        });

    capture_panel_view_->on_add_device.push_back(
        [this](const std::string& device_name, node_type node_type, const std::string& ip_address,
               const std::string& gateway_address) {
          node_info new_device{};
          new_device.name = device_name;
          new_device.set_type(node_type);
          new_device.params["id"] = generate_new_id();
          new_device.params["address"] = ip_address;
          new_device.params["gateway"] = gateway_address;

          switch (node_type) {
            case node_type::depthai_color:
              new_device.params["width"] = 960.0f;
              new_device.params["height"] = 540.0f;
              break;
            case node_type::raspi:
              new_device.params["width"] = 820.0f;
              new_device.params["height"] = 616.0f;
              break;
            case node_type::raspi_color:
              new_device.params["width"] = 820.0f;
              new_device.params["height"] = 616.0f;
              break;
            case node_type::rs_d435:
              new_device.params["width"] = 640.0f;
              new_device.params["height"] = 480.0f;
              break;
            case node_type::rs_d435_color:
              new_device.params["width"] = 960.0f;
              new_device.params["height"] = 540.0f;
              break;
            default:
              break;
          }

          auto& node_infos = capture_config->get_node_infos();
          if (const auto found = std::find_if(node_infos.begin(), node_infos.end(),
                                              [&](const auto& x) { return x.name == device_name; });
              found == node_infos.end()) {
            node_infos.push_back(new_device);
            capture_panel_view_->devices.clear();

            for (const auto& node_info : node_infos) {
              std::string path;
              if (node_info.contains_param("address")) {
                path = node_info.get_param<std::string>("address");
              }
              if (node_info.contains_param("db_path")) {
                path = node_info.get_param<std::string>("db_path");
              }
              capture_panel_view_->devices.push_back(
                  capture_panel_view::node_info{node_info.name, path, node_info.params});
            }
            capture_config->update();
          }
        });

    capture_panel_view_->on_remove_device.push_back([this](const std::string& device_name) {
      auto& node_infos = capture_config->get_node_infos();
      if (const auto found = std::find_if(node_infos.begin(), node_infos.end(),
                                          [&](const auto& x) { return x.name == device_name; });
          found != node_infos.end()) {
        node_infos.erase(found);
        capture_panel_view_->devices.clear();

        for (const auto& node_info : node_infos) {
          std::string path;
          if (node_info.contains_param("address")) {
            path = node_info.get_param<std::string>("address");
          }
          if (node_info.contains_param("db_path")) {
            path = node_info.get_param<std::string>("db_path");
          }
          capture_panel_view_->devices.push_back(
              capture_panel_view::node_info{node_info.name, path, node_info.params});
        }
        capture_config->update();
      }
    });
  }

  void init_calibration_panel() {
    calibration_panel_view_ = std::make_unique<calibration_panel_view>();
    for (const auto& node_info : calibration_config->get_node_infos()) {
      std::string path;
      if (node_info.contains_param("address")) {
        path = node_info.get_param<std::string>("address");
      }
      if (node_info.contains_param("db_path")) {
        path = node_info.get_param<std::string>("db_path");
      }
      calibration_panel_view_->devices.push_back(
          calibration_panel_view::node_info{node_info.name, path, node_info.params});
    }

    calibration_panel_view_->is_streaming_changed.push_back(
        [this](const std::vector<calibration_panel_view::node_info>& devices, bool is_streaming) {
          if (is_streaming) {
            if (multiview_capture) {
              return false;
            }

            if (calibration_panel_view_->calibration_target_index == 0) {
              // Extrinsic calibration
              const auto& node_infos = calibration_config->get_node_infos();

              if (calibration_panel_view_->is_masking) {
                multiview_capture.reset(new multiview_capture_pipeline(masks));
              } else {
                multiview_capture.reset(new multiview_capture_pipeline());
              }

              multiview_capture->add_marker_received(
                  [this](const std::map<std::string, marker_frame_data>& marker_frame) {
                    std::map<std::string, std::vector<point_data>> frame;
                    for (const auto& [name, markers] : marker_frame) {
                      std::vector<point_data> points;
                      for (const auto& marker : markers.markers) {
                        points.push_back(
                            point_data{glm::vec2(marker.x, marker.y), marker.r, markers.timestamp});
                      }
                      frame.insert(std::make_pair(name, points));
                    }
                    calib->push_frame(frame);
                  });

              multiview_capture->run(node_infos);

              for (const auto& node_info : node_infos) {
                if (node_info.is_camera()) {
                  const auto width =
                      static_cast<int>(std::round(node_info.get_param<float>("width")));
                  const auto height =
                      static_cast<int>(std::round(node_info.get_param<float>("height")));
                  const auto stream = std::make_shared<image_tile_view::stream_info>(
                      node_info.name, float2{(float)width, (float)height});
                  image_tile_view_->streams.push_back(stream);
                }
              }
            } else if (calibration_panel_view_->calibration_target_index == 1) {
              // Intrinsic calibration
              const auto& device =
                  devices[calibration_panel_view_->intrinsic_calibration_device_index];

              const auto& node_infos = calibration_config->get_node_infos();
              auto found = std::find_if(node_infos.begin(), node_infos.end(),
                                        [device](const auto& x) { return x.name == device.name; });
              if (found == node_infos.end()) {
                return false;
              }
              const auto& node_info = *found;

              const auto capture = std::make_shared<capture_pipeline>();

              capture->add_image_received([this](const cv::Mat& frame) {
                if (!frame.empty() && calibration_panel_view_->is_marker_collecting) {
                  intrinsic_calib->push_frame(frame);
                }
              });

              try {
                capture->run(node_info);
              } catch (std::exception& e) {
                std::cout << "Failed to start capture: " << e.what() << std::endl;
                return false;
              }
              captures.insert(std::make_pair(device.name, capture));
              const auto width = static_cast<int>(std::round(node_info.get_param<float>("width")));
              const auto height =
                  static_cast<int>(std::round(node_info.get_param<float>("height")));

              const auto stream = std::make_shared<image_tile_view::stream_info>(
                  device.name, float2{(float)width, (float)height});
              image_tile_view_->streams.push_back(stream);
            } else if (calibration_panel_view_->calibration_target_index == 2) {
              // Axis calibration
              const auto& node_infos = calibration_config->get_node_infos();

              if (calibration_panel_view_->is_masking) {
                multiview_capture.reset(new multiview_capture_pipeline(masks));
              } else {
                multiview_capture.reset(new multiview_capture_pipeline());
              }

              multiview_capture->add_marker_received(
                  [this](const std::map<std::string, marker_frame_data>& marker_frame) {
                    std::map<std::string, std::vector<point_data>> frame;
                    for (const auto& [name, markers] : marker_frame) {
                      std::vector<point_data> points;
                      for (const auto& marker : markers.markers) {
                        points.push_back(
                            point_data{glm::vec2(marker.x, marker.y), marker.r, markers.timestamp});
                      }
                      frame.insert(std::make_pair(name, points));
                    }
                    axis_calib->push_frame(frame);
                  });

              multiview_capture->run(node_infos);

              for (const auto& node_info : node_infos) {
                if (node_info.is_camera()) {
                  const auto width =
                      static_cast<int>(std::round(node_info.get_param<float>("width")));
                  const auto height =
                      static_cast<int>(std::round(node_info.get_param<float>("height")));
                  const auto stream = std::make_shared<image_tile_view::stream_info>(
                      node_info.name, float2{(float)width, (float)height});
                  image_tile_view_->streams.push_back(stream);
                }
              }
            }
          } else {
            if (calibration_panel_view_->calibration_target_index == 0) {
              multiview_capture->stop();
              multiview_capture.reset();

              const auto& node_infos = calibration_config->get_node_infos();

              for (const auto& node_info : node_infos) {
                const auto stream_it =
                    std::find_if(image_tile_view_->streams.begin(), image_tile_view_->streams.end(),
                                 [&](const auto& x) { return x->name == node_info.name; });

                if (stream_it != image_tile_view_->streams.end()) {
                  image_tile_view_->streams.erase(stream_it);
                }
              }
            } else if (calibration_panel_view_->calibration_target_index == 1) {
              const auto& device =
                  devices[calibration_panel_view_->intrinsic_calibration_device_index];

              auto it = captures.find(device.name);
              it->second->stop();
              if (it != captures.end()) {
                captures.erase(captures.find(device.name));
              }

              const auto stream_it =
                  std::find_if(image_tile_view_->streams.begin(), image_tile_view_->streams.end(),
                               [&](const auto& x) { return x->name == device.name; });

              if (stream_it != image_tile_view_->streams.end()) {
                image_tile_view_->streams.erase(stream_it);
              }
            } else if (calibration_panel_view_->calibration_target_index == 2) {
              multiview_capture->stop();
              multiview_capture.reset();

              const auto& node_infos = calibration_config->get_node_infos();

              for (const auto& node_info : node_infos) {
                const auto stream_it =
                    std::find_if(image_tile_view_->streams.begin(), image_tile_view_->streams.end(),
                                 [&](const auto& x) { return x->name == node_info.name; });

                if (stream_it != image_tile_view_->streams.end()) {
                  image_tile_view_->streams.erase(stream_it);
                }
              }
            }

            calibration_panel_view_->is_marker_collecting = false;
          }
          return true;
        });

    calibration_panel_view_->is_masking_changed.push_back(
        [this](const std::vector<calibration_panel_view::node_info>& devices, bool is_masking) {
          if (!multiview_capture) {
            return false;
          }
          if (is_masking) {
            multiview_capture->gen_mask();
            masks = multiview_capture->get_masks();
          } else {
            multiview_capture->clear_mask();
            masks.clear();
          }
          return true;
        });

    calibration_panel_view_->is_marker_collecting_changed.push_back(
        [this](const std::vector<calibration_panel_view::node_info>& devices,
               bool is_marker_collecting) {
          if (!multiview_capture) {
            return false;
          }
          if (is_marker_collecting) {
            for (const auto& device : devices) {
              multiview_capture->enable_marker_collecting(device.name);
            }
          } else {
            for (const auto& device : devices) {
              multiview_capture->disable_marker_collecting(device.name);
            }
          }
          return true;
        });

    calibration_panel_view_->on_intrinsic_calibration_device_changed.push_back(
        [this](const calibration_panel_view::node_info& device) {
          const auto& node_infos = calibration_config->get_node_infos();
          auto found = std::find_if(node_infos.begin(), node_infos.end(),
                                    [&](const auto& x) { return x.name == device.name; });
          if (found == node_infos.end()) {
            return;
          }
          const auto& node_info = *found;
          const auto& params =
              std::get<camera_t>(parameters->at(node_info.get_param<std::string>("id")));
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

    calibration_panel_view_->on_calibrate.push_back(
        [this](const std::vector<calibration_panel_view::node_info>& devices, bool on_calibrate) {
          if (calibration_panel_view_->calibration_target_index == 0) {
            for (const auto& device : devices) {
              if (calib->get_num_frames(device.name) > 0) {
                spdlog::info("Start calibration");

                calib->calibrate();

                const auto& node_infos = calibration_config->get_node_infos();

                std::unordered_map<std::string, std::string> device_name_to_id;
                for (const auto& device : devices) {
                  auto found = std::find_if(node_infos.begin(), node_infos.end(),
                                            [&](const auto& x) { return x.name == device.name; });
                  if (found == node_infos.end()) {
                    spdlog::error("Device {} not found in calibration config", device.name);
                    return false;
                  }
                  const auto& node_info = *found;
                  if (node_info.is_camera()) {
                    device_name_to_id[device.name] = node_info.get_param<std::string>("id");
                  }
                }

                for (const auto& [camera_name, camera] : calib->get_calibrated_cameras()) {
                  const auto& camera_id = device_name_to_id.at(camera_name);

                  auto& params = std::get<camera_t>(parameters->at(camera_id));
                  params.extrin = camera.extrin;
                  params.intrin = camera.intrin;
                }

                parameters->save();

                spdlog::info("End calibration");

                break;
              }
            }
            return true;
          } else if (calibration_panel_view_->calibration_target_index == 1) {
            const auto& device =
                devices.at(calibration_panel_view_->intrinsic_calibration_device_index);

            const auto& node_infos = calibration_config->get_node_infos();
            auto found = std::find_if(node_infos.begin(), node_infos.end(),
                                      [&](const auto& x) { return x.name == device.name; });
            if (found == node_infos.end()) {
              return false;
            }
            const auto& node_info = *found;

            const auto image_width =
                static_cast<int>(std::round(node_info.get_param<float>("width")));
            const auto image_height =
                static_cast<int>(std::round(node_info.get_param<float>("height")));

            intrinsic_calib->set_image_size(image_width, image_height);

            intrinsic_calib->calibrate();

            const auto& calibrated_camera = intrinsic_calib->get_calibrated_camera();

            calibration_panel_view_->fx = calibrated_camera.intrin.fx;
            calibration_panel_view_->fy = calibrated_camera.intrin.fy;
            calibration_panel_view_->cx = calibrated_camera.intrin.cx;
            calibration_panel_view_->cy = calibrated_camera.intrin.cy;
            calibration_panel_view_->k0 = calibrated_camera.intrin.coeffs[0];
            calibration_panel_view_->k1 = calibrated_camera.intrin.coeffs[1];
            calibration_panel_view_->k2 = calibrated_camera.intrin.coeffs[4];
            calibration_panel_view_->p0 = calibrated_camera.intrin.coeffs[2];
            calibration_panel_view_->p1 = calibrated_camera.intrin.coeffs[3];
            calibration_panel_view_->rms = intrinsic_calib->get_rms();
            auto& params =
                std::get<camera_t>(parameters->at(node_info.get_param<std::string>("id")));
            params.intrin.fx = calibrated_camera.intrin.fx;
            params.intrin.fy = calibrated_camera.intrin.fy;
            params.intrin.cx = calibrated_camera.intrin.cx;
            params.intrin.cy = calibrated_camera.intrin.cy;
            params.intrin.coeffs[0] = calibrated_camera.intrin.coeffs[0];
            params.intrin.coeffs[1] = calibrated_camera.intrin.coeffs[1];
            params.intrin.coeffs[2] = calibrated_camera.intrin.coeffs[2];
            params.intrin.coeffs[3] = calibrated_camera.intrin.coeffs[3];
            params.intrin.coeffs[4] = calibrated_camera.intrin.coeffs[4];
            params.width = calibrated_camera.width;
            params.height = calibrated_camera.height;
            parameters->save();
            return true;
          } else if (calibration_panel_view_->calibration_target_index == 2) {
            spdlog::info("Start calibration");

            axis_calib->calibrate();

            spdlog::info("End calibration");
            return true;
          }
          return false;
        });
  }

  void init_reconstruction_panel() {
    reconstruction_panel_view_ = std::make_unique<reconstruction_panel_view>();
    for (const auto& node_info : reconstruction_config->get_node_infos()) {
      std::string path;
      if (node_info.contains_param("address")) {
        path = node_info.get_param<std::string>("address");
      }
      if (node_info.contains_param("db_path")) {
        path = node_info.get_param<std::string>("db_path");
      }
      reconstruction_panel_view_->devices.push_back(
          reconstruction_panel_view::node_info{node_info.name, path});
    }

    reconstruction_panel_view_->is_streaming_changed.push_back(
        [this](const std::vector<reconstruction_panel_view::node_info>& devices,
               bool is_streaming) {
          if (is_streaming) {
            if (reconstruction_panel_view_->source == 0) {
              if (multiview_capture) {
                return false;
              }

              const auto& node_infos = reconstruction_config->get_node_infos();

              if (calibration_panel_view_->is_masking) {
                multiview_capture.reset(new multiview_capture_pipeline(masks));
              } else {
                multiview_capture.reset(new multiview_capture_pipeline());
              }

              multiview_capture->add_marker_received(
                  [this](const std::map<std::string, marker_frame_data>& marker_frame) {
                    std::map<std::string, std::vector<point_data>> frame;
                    for (const auto& [name, markers] : marker_frame) {
                      std::vector<point_data> points;
                      for (const auto& marker : markers.markers) {
                        points.push_back(
                            point_data{glm::vec2(marker.x, marker.y), marker.r, markers.timestamp});
                      }
                      frame.insert(std::make_pair(name, points));
                    }
                    multiview_point_reconstruction_pipeline_->push_frame(frame);
                  });

              for (const auto& node_info : node_infos) {
                if (node_info.is_camera()) {
                  multiview_capture->enable_marker_collecting(node_info.name);
                }
              }

              multiview_capture->run(node_infos);

              for (const auto& node_info : node_infos) {
                if (node_info.is_camera()) {
                  const auto width =
                      static_cast<int>(std::round(node_info.get_param<float>("width")));
                  const auto height =
                      static_cast<int>(std::round(node_info.get_param<float>("height")));
                  const auto stream = std::make_shared<image_tile_view::stream_info>(
                      node_info.name, float2{(float)width, (float)height});
                  image_tile_view_->streams.push_back(stream);
                }
              }
            } else if (reconstruction_panel_view_->source == 1) {
              if (multiview_capture) {
                return false;
              }

              const auto& node_infos = reconstruction_config->get_node_infos();

              if (calibration_panel_view_->is_masking) {
                multiview_capture.reset(new multiview_capture_pipeline(masks));
              } else {
                multiview_capture.reset(new multiview_capture_pipeline());
              }

              multiview_capture->add_image_received(
                  [this](const std::map<std::string, cv::Mat>& image_frame) {
                    std::map<std::string, cv::Mat> color_image_frame;
                    for (const auto& [name, image] : image_frame) {
                      if (image.channels() == 3 && image.depth() == cv::DataType<uchar>::depth) {
                        color_image_frame[name] = image;
                      }
                    }
                    multiview_image_reconstruction_pipeline_->push_frame(color_image_frame);
                  });

              multiview_capture->run(node_infos);

              for (const auto& node_info : node_infos) {
                if (node_info.is_camera()) {
                  const auto width =
                      static_cast<int>(std::round(node_info.get_param<float>("width")));
                  const auto height =
                      static_cast<int>(std::round(node_info.get_param<float>("height")));
                  const auto stream = std::make_shared<image_tile_view::stream_info>(
                      node_info.name, float2{(float)width, (float)height});
                  image_tile_view_->streams.push_back(stream);
                }
              }
            }
          } else {
            if (multiview_capture) {
              multiview_capture->stop();
              multiview_capture.reset();

              const auto& node_infos = reconstruction_config->get_node_infos();

              for (const auto& node_info : node_infos) {
                const auto stream_it =
                    std::find_if(image_tile_view_->streams.begin(), image_tile_view_->streams.end(),
                                 [&](const auto& x) { return x->name == node_info.name; });

                if (stream_it != image_tile_view_->streams.end()) {
                  image_tile_view_->streams.erase(stream_it);
                }
              }
            }
          }
          return true;
        });
  }

  void init_gui() {
    const char* glsl_version = "#version 130";

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    // ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL((GLFWwindow*)get_handle(), true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    static const ImWchar icons_ranges[] = {0xf000, 0xf999, 0};

    {
      const int OVERSAMPLE = true;

      ImFontConfig config_words;
      config_words.OversampleV = OVERSAMPLE;
      config_words.OversampleH = OVERSAMPLE;
      default_font =
          io.Fonts->AddFontFromFileTTF("../fonts/mplus/fonts/ttf/Mplus2-Regular.ttf", 16.0f,
                                       &config_words, io.Fonts->GetGlyphRangesJapanese());

      ImFontConfig config_glyphs;
      config_glyphs.MergeMode = true;
      config_glyphs.OversampleV = OVERSAMPLE;
      config_glyphs.OversampleH = OVERSAMPLE;
      default_font = io.Fonts->AddFontFromMemoryCompressedTTF(font_awesome_compressed_data,
                                                              font_awesome_compressed_size, 14.f,
                                                              &config_glyphs, icons_ranges);
    }
    IM_ASSERT(default_font != NULL);

    {
      const int OVERSAMPLE = true;

      ImFontConfig config_words;
      config_words.OversampleV = OVERSAMPLE;
      config_words.OversampleH = OVERSAMPLE;
      large_font =
          io.Fonts->AddFontFromFileTTF("../fonts/mplus/fonts/ttf/Mplus2-Regular.ttf", 20.0f,
                                       &config_words, io.Fonts->GetGlyphRangesJapanese());

      ImFontConfig config_glyphs;
      config_glyphs.MergeMode = true;
      config_glyphs.OversampleV = OVERSAMPLE;
      config_glyphs.OversampleH = OVERSAMPLE;
      large_font = io.Fonts->AddFontFromMemoryCompressedTTF(font_awesome_compressed_data,
                                                            font_awesome_compressed_size, 20.f,
                                                            &config_glyphs, icons_ranges);
    }
    IM_ASSERT(large_font != NULL);

    {
      ImGuiStyle& style = ImGui::GetStyle();

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

  void term_gui() {
    if (ImGui::GetIO().BackendRendererUserData) {
      ImGui_ImplOpenGL3_Shutdown();
    }
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
  }

 public:
  viewer_app() : window_base("Stargazer", SCREEN_WIDTH, SCREEN_HEIGHT), calib() {}

  virtual void initialize() override {
    gladLoadGL();

    capture_config.reset(new configuration("../config/capture.json"));
    reconstruction_config.reset(new configuration("../config/reconstruction.json"));
    calibration_config.reset(new configuration("../config/calibration.json"));

    parameters = std::make_shared<parameters_t>("../config/parameters.json");
    parameters->load();

    init_gui();

    context = std::make_unique<view_context>();
    context->window = this;
    context->default_font = default_font;
    context->large_font = large_font;

    top_bar_view_ = std::make_unique<top_bar_view>();
    image_tile_view_ = std::make_unique<image_tile_view>();
    contrail_tile_view_ = std::make_unique<image_tile_view>();

    init_capture_panel();
    init_calibration_panel();
    init_reconstruction_panel();

    view_controller =
        std::make_shared<azimuth_elevation>(glm::u32vec2(0, 0), glm::u32vec2(width, height));
    pose_view_ = std::make_unique<pose_view>();

    multiview_point_reconstruction_pipeline_ =
        std::make_unique<multiview_point_reconstruction_pipeline>();
    multiview_point_reconstruction_pipeline_->run();

    {
      const auto& scene = std::get<scene_t>(parameters->at("scene"));
      multiview_point_reconstruction_pipeline_->set_axis(scene.axis);
    }

    for (const auto& device : reconstruction_config->get_node_infos()) {
      if (device.is_camera()) {
        const auto& params =
            std::get<camera_t>(parameters->at(device.get_param<std::string>("id")));
        multiview_point_reconstruction_pipeline_->set_camera(device.name, params);
      }
    }

    multiview_image_reconstruction_pipeline_ =
        std::make_unique<multiview_image_reconstruction_pipeline>();
    multiview_image_reconstruction_pipeline_->run(
        reconstruction_config->get_node_infos("static_pipeline"));

    {
      const auto& scene = std::get<scene_t>(parameters->at("scene"));
      multiview_image_reconstruction_pipeline_->set_axis(scene.axis);
    }

    for (const auto& device : reconstruction_config->get_node_infos()) {
      if (device.is_camera()) {
        const auto& params =
            std::get<camera_t>(parameters->at(device.get_param<std::string>("id")));
        multiview_image_reconstruction_pipeline_->set_camera(device.name, params);
      }
    }

    calib = std::make_unique<calibration_pipeline>();

    calib->add_calibrated([&](const std::unordered_map<std::string, camera_t>& cameras) {
      for (const auto& [name, camera] : cameras) {
        multiview_point_reconstruction_pipeline_->set_camera(name, camera);
      }
    });

    for (const auto& device : calibration_config->get_node_infos()) {
      if (device.is_camera()) {
        if (parameters->contains(device.get_param<std::string>("id"))) {
          const auto& params =
              std::get<camera_t>(parameters->at(device.get_param<std::string>("id")));
          calib->set_camera(device.name, params);
        } else {
          spdlog::error("No camera params found for device: {}", device.name);
        }
      }
    }

    calib->run(calibration_config->get_node_infos("static_pipeline"));

    for (auto& [camera_name, camera] : calib->get_cameras()) {
      camera.extrin.rotation = glm::mat3(1.0);
      camera.extrin.translation = glm::vec3(1.0);
    }

    axis_calib = std::make_unique<axis_calibration_pipeline>(parameters);

    for (const auto& device : calibration_config->get_node_infos()) {
      if (device.is_camera()) {
        if (parameters->contains(device.get_param<std::string>("id"))) {
          const auto& params =
              std::get<camera_t>(parameters->at(device.get_param<std::string>("id")));
          axis_calib->set_camera(device.name, params);
        } else {
          spdlog::error("No camera params found for device: {}", device.name);
        }
      }
    }

    axis_calib->run(calibration_config->get_node_infos("static_pipeline"));

    intrinsic_calib = std::make_unique<intrinsic_calibration_pipeline>();

    intrinsic_calib->run(calibration_config->get_node_infos("static_pipeline"));

    window_base::initialize();
  }

  virtual void finalize() override {
    term_gui();

    calib->stop();
    axis_calib->stop();
    multiview_point_reconstruction_pipeline_->stop();
    multiview_image_reconstruction_pipeline_->stop();

    window_base::finalize();
  }

  virtual void show() override { window_base::show(); }

  virtual void on_close() override {
    window_base::on_close();
    window_manager::get_instance()->exit();
  }

  virtual void on_scroll(double x, double y) override {
    if (view_controller) {
      view_controller->scroll(x, y);
    }
  }

  virtual void update() override {
    if (handle == nullptr) {
      return;
    }
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    top_bar_view_->render(context.get());

    if (top_bar_view_->view_type == top_bar_view::ViewType::Image) {
      if (multiview_capture) {
        const auto frames = multiview_capture->get_frames();
        for (const auto& stream : image_tile_view_->streams) {
          const auto& frame_it = frames.find(stream->name);
          if (frame_it != frames.end()) {
            const auto& frame = frame_it->second;
            if (!frame.empty()) {
              cv::Mat color_image;
              if (frame.channels() == 1) {
                cv::cvtColor(frame, color_image, cv::COLOR_GRAY2RGB);
              } else if (frame.channels() == 3) {
                cv::cvtColor(frame, color_image, cv::COLOR_BGR2RGB);
              }
              stream->texture.upload_image(color_image.cols, color_image.rows, color_image.data,
                                           GL_RGB);
            }
          }
        }
      }

      for (const auto& stream : image_tile_view_->streams) {
        const auto capture_it = captures.find(stream->name);
        if (capture_it != captures.end()) {
          const auto capture = capture_it->second;
          auto frame = capture->get_frame();

          if (!frame.empty()) {
            cv::Mat color_image;
            if (frame.channels() == 1) {
              cv::cvtColor(frame, color_image, cv::COLOR_GRAY2RGB);
            } else if (frame.channels() == 3) {
              cv::cvtColor(frame, color_image, cv::COLOR_BGR2RGB);
            }
            stream->texture.upload_image(color_image.cols, color_image.rows, color_image.data,
                                         GL_RGB);
          }
        }
      }
    }

    if (top_bar_view_->view_mode == top_bar_view::Mode::Calibration) {
      if (calibration_panel_view_->calibration_target_index == 0) {
        for (auto& device : calibration_panel_view_->devices) {
          const auto& node_infos = calibration_config->get_node_infos();
          if (const auto node_info =
                  std::find_if(node_infos.begin(), node_infos.end(),
                               [&](const auto& x) { return x.name == device.name; });
              node_info != node_infos.end()) {
            device.num_points = calib->get_num_frames(device.name);
          }
        }
      } else if (calibration_panel_view_->calibration_target_index == 1) {
        calibration_panel_view_
            ->devices[calibration_panel_view_->intrinsic_calibration_device_index]
            .num_points = intrinsic_calib->get_num_frames();
      } else if (calibration_panel_view_->calibration_target_index == 2) {
        for (auto& device : calibration_panel_view_->devices) {
          const auto& node_infos = calibration_config->get_node_infos();
          if (const auto node_info =
                  std::find_if(node_infos.begin(), node_infos.end(),
                               [&](const auto& x) { return x.name == device.name; });
              node_info != node_infos.end()) {
            device.num_points = axis_calib->get_num_frames(device.name);
          }
        }
      }
    }

    if (top_bar_view_->view_mode == top_bar_view::Mode::Capture) {
      capture_panel_view_->render(context.get());
    } else if (top_bar_view_->view_mode == top_bar_view::Mode::Calibration) {
      calibration_panel_view_->render(context.get());
    } else if (top_bar_view_->view_mode == top_bar_view::Mode::Reconstruction) {
      reconstruction_panel_view_->render(context.get());
    }

    if (top_bar_view_->view_mode == top_bar_view::Mode::Calibration) {
      if (top_bar_view_->view_type == top_bar_view::ViewType::Pose) {
        view_controller->update(mouse_state::get_mouse_state(handle));
        float radius = view_controller->get_radius();
        glm::vec3 forward(0.f, 0.f, 1.f);
        glm::vec3 up(0.0f, 1.0f, 0.0f);
        glm::vec3 view_pos =
            glm::rotate(glm::inverse(view_controller->get_rotation_quaternion()), forward * radius);
        glm::mat4 view = glm::lookAt(view_pos, glm::vec3(0, 0, 0), up);

        context->view = view;

        pose_view_->cameras.clear();

        for (const auto& device : calibration_config->get_node_infos()) {
          const auto& cameras = calib->get_calibrated_cameras();

          if (cameras.find(device.name) != cameras.end()) {
            const auto& camera = cameras.at(device.name);
            pose_view_->cameras[device.name] = pose_view::camera_t{
                (int)camera.width,    (int)camera.height,
                camera.intrin.cx,     camera.intrin.cy,
                camera.intrin.fx,     camera.intrin.fy,
                camera.intrin.coeffs, glm::inverse(camera.extrin.transform_matrix()),
            };
          }
        }

        pose_view_->axis = std::get<scene_t>(parameters->at("scene")).axis;
      } else if (top_bar_view_->view_type == top_bar_view::ViewType::Contrail) {
        for (const auto& node : calibration_config->get_node_infos()) {
          if (!node.is_camera()) {
            continue;
          }

          std::shared_ptr<image_tile_view::stream_info> stream;
          const auto width = static_cast<int>(std::round(node.get_param<float>("width")));
          const auto height = static_cast<int>(std::round(node.get_param<float>("height")));

          const auto found =
              std::find_if(contrail_tile_view_->streams.begin(), contrail_tile_view_->streams.end(),
                           [&](const auto& x) { return x->name == node.name; });
          if (found == contrail_tile_view_->streams.end()) {
            stream = std::make_shared<image_tile_view::stream_info>(
                node.name, float2{(float)width, (float)height});
            contrail_tile_view_->streams.push_back(stream);
          } else {
            stream = *found;
          }

          const auto observed_points = calib->get_observed_points(node.name);
          cv::Mat cloud_image(height, width, CV_8UC3, cv::Scalar::all(0));

          for (const auto& observed_point : observed_points) {
            for (const auto& point : observed_point.points) {
              if (point.x >= 0 && point.x < width && point.y >= 0 && point.y < height) {
                cv::circle(cloud_image, cv::Point(point.x, point.y), 5, cv::Scalar(255, 0, 0));
              }
            }
          }
          stream->texture.upload_image(cloud_image.cols, cloud_image.rows, cloud_image.data,
                                       GL_RGB);
        }
      }
    } else if (top_bar_view_->view_mode == top_bar_view::Mode::Reconstruction) {
      if (top_bar_view_->view_type == top_bar_view::ViewType::Pose) {
        view_controller->update(mouse_state::get_mouse_state(handle));
        float radius = view_controller->get_radius();
        glm::vec3 forward(0.f, 0.f, 1.f);
        glm::vec3 up(0.0f, 1.0f, 0.0f);
        glm::vec3 view_pos =
            glm::rotate(glm::inverse(view_controller->get_rotation_quaternion()), forward * radius);
        glm::mat4 view = glm::lookAt(view_pos, glm::vec3(0, 0, 0), up);

        context->view = view;

        pose_view_->cameras.clear();

        for (const auto& device : reconstruction_config->get_node_infos()) {
          const auto& cameras = multiview_point_reconstruction_pipeline_->get_cameras();

          if (cameras.find(device.name) != cameras.end()) {
            const auto& camera = cameras.at(device.name);
            pose_view_->cameras[device.name] = pose_view::camera_t{
                (int)camera.width,    (int)camera.height,
                camera.intrin.cx,     camera.intrin.cy,
                camera.intrin.fx,     camera.intrin.fy,
                camera.intrin.coeffs, glm::inverse(camera.extrin.transform_matrix()),
            };
          }
        }

        pose_view_->axis = multiview_point_reconstruction_pipeline_->get_axis();

        pose_view_->points.clear();
        for (const auto& point : multiview_point_reconstruction_pipeline_->get_markers()) {
          pose_view_->points.push_back(point);
        }
        for (const auto& point : multiview_image_reconstruction_pipeline_->get_markers()) {
          pose_view_->points.push_back(point);
        }
      } else if (top_bar_view_->view_type == top_bar_view::ViewType::Point) {
        if (multiview_capture) {
          const auto frames = multiview_image_reconstruction_pipeline_->get_features();
          for (const auto& [name, frame] : frames) {
            const auto device_name = name;
            if (!frame.empty()) {
              const auto stream_it =
                  std::find_if(image_tile_view_->streams.begin(), image_tile_view_->streams.end(),
                               [device_name](const auto& x) { return x->name == device_name; });

              if (stream_it != image_tile_view_->streams.end()) {
                cv::Mat color_image;
                if (frame.channels() == 1) {
                  cv::cvtColor(frame, color_image, cv::COLOR_GRAY2RGB);
                } else if (frame.channels() == 3) {
                  cv::cvtColor(frame, color_image, cv::COLOR_BGR2RGB);
                }
                (*stream_it)
                    ->texture.upload_image(color_image.cols, color_image.rows, color_image.data,
                                           GL_RGB);
              }
            }
          }
        }

        for (const auto& device : capture_panel_view_->devices) {
          const auto capture_it = captures.find(device.name);
          if (capture_it != captures.end()) {
            const auto capture = capture_it->second;
            auto frame = capture->get_frame();

            if (!frame.empty()) {
              const auto stream_it =
                  std::find_if(image_tile_view_->streams.begin(), image_tile_view_->streams.end(),
                               [&](const auto& x) { return x->name == device.name; });

              if (stream_it != image_tile_view_->streams.end()) {
                cv::Mat color_image;
                if (frame.channels() == 1) {
                  cv::cvtColor(frame, color_image, cv::COLOR_GRAY2RGB);
                } else if (frame.channels() == 3) {
                  cv::cvtColor(frame, color_image, cv::COLOR_BGR2RGB);
                }
                (*stream_it)
                    ->texture.upload_image(color_image.cols, color_image.rows, color_image.data,
                                           GL_RGB);
              }
            }
          }
        }
      }
    }

    if (top_bar_view_->view_type == top_bar_view::ViewType::Image) {
      image_tile_view_->render(context.get());
    } else if (top_bar_view_->view_type == top_bar_view::ViewType::Contrail) {
      contrail_tile_view_->render(context.get());
    } else if (top_bar_view_->view_type == top_bar_view::ViewType::Point) {
      image_tile_view_->render(context.get());
    } else if (top_bar_view_->view_type == top_bar_view::ViewType::Pose) {
      pose_view_->render(context.get());
    }

    ImGui::Render();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  }

  virtual void on_char(unsigned int codepoint) override {}
};

static void sigint_handler(int) { window_manager::get_instance()->exit(); }

int main() {
  signal(SIGINT, sigint_handler);

  const auto win_mgr = window_manager::get_instance();
  win_mgr->initialize();

  auto window = std::make_shared<viewer_app>();

  window->create();
  auto graphics_ctx = window->create_graphics_context();
  graphics_ctx.attach();
  window->initialize();
  window->show();

  while (!win_mgr->should_close()) {
    win_mgr->handle_event();
    graphics_ctx.clear();
    window->update();
    graphics_ctx.swap_buffer();
  }

  window->finalize();
  graphics_ctx.detach();
  window->destroy();

  window.reset();

  win_mgr->terminate();

  return 0;
}
