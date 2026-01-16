#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>

#include <csignal>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>

#include "../lib/glm_serialize.hpp"
#include "../lib/grpc_server_node.hpp"
#include "../lib/messages.hpp"
#include "gui.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#include "viewer.hpp"
#include "views.hpp"

using namespace stargazer;

namespace {
std::atomic_bool should_stop(false);
std::mutex display_mutex;
std::map<std::string, cv::Mat> latest_frames;
std::map<std::string, std::unique_ptr<texture_buffer>> texture_buffers;

// Decode image data.
cv::Mat decode_image(const stargazer::camera_image& img) {
  if (img.format == stargazer::image_data_format::JPEG) {
    return cv::imdecode(img.image_data, cv::IMREAD_COLOR);
  } else if (img.format == stargazer::image_data_format::PNG) {
    return cv::imdecode(img.image_data, cv::IMREAD_COLOR);
  } else if (img.format == stargazer::image_data_format::RAW) {
    // Infer RAW format from the image size.
    int channels = img.image_data.size() / (img.image_size.x * img.image_size.y);
    int type = (channels == 3) ? CV_8UC3 : CV_8UC1;
    cv::Mat mat(img.image_size.y, img.image_size.x, type);
    std::memcpy(mat.data, img.image_data.data(), img.image_data.size());
    return mat;
  }
  return cv::Mat();
}
}  // namespace

void signal_handler(int signal) {
  if (signal == SIGINT || signal == SIGTERM) {
    spdlog::info("Received signal {}, shutting down...", signal);
    should_stop.store(true);
  }
}

void print_usage(const char* program_name) {
  std::cout << "Usage: " << program_name << " [options]\n"
            << "Options:\n"
            << "  -a, --address <addr>   gRPC server address (default: 0.0.0.0:50051)\n"
            << "  --headless             Run without GUI (images logged only)\n"
            << "  -h, --help             Show this help message\n"
            << std::endl;
}

int main(int argc, char* argv[]) {
  std::string server_address = "0.0.0.0:50051";
  bool headless = false;

  // Parse command-line arguments.
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      return 0;
    } else if ((arg == "-a" || arg == "--address") && i + 1 < argc) {
      server_address = argv[++i];
    } else if (arg == "--headless") {
      headless = true;
    } else {
      std::cerr << "Unknown option: " << arg << std::endl;
      print_usage(argv[0]);
      return 1;
    }
  }

  // Set log level.
  spdlog::set_level(spdlog::level::info);
  spdlog::info("Starting stargazer gRPC server");
  spdlog::info("Server address: {}", server_address);
  spdlog::info("GUI mode: {}", headless ? "disabled" : "enabled");

  // Setup signal handlers.
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  try {
    // Create gRPC server node.
    auto grpc_node = std::make_shared<stargazer::grpc_server_node>();
    grpc_node->set_address(server_address);

    // Register callback for received images.
    auto image_callback =
        std::make_shared<coalsack::graph_message_callback>([](coalsack::graph_message_ptr message) {
          if (const auto msg =
                  std::dynamic_pointer_cast<stargazer::camera_image_list_message>(message)) {
            std::lock_guard<std::mutex> lock(display_mutex);
            const auto& images = msg->get_data();
            for (size_t i = 0; i < images.size(); i++) {
              cv::Mat frame = decode_image(images[i]);
              if (!frame.empty()) {
                std::string window_name = "camera_" + std::to_string(i);
                latest_frames[window_name] = frame.clone();
              } else {
                spdlog::warn("Failed to decode image {}", i);
              }
            }
          }
        });

    grpc_node->get_output()->set_callback(image_callback);

    // Start the server.
    spdlog::info("Starting gRPC server node");
    grpc_node->run();

    spdlog::info("gRPC server is running. Press Ctrl+C to stop.");

    if (headless) {
      // Headless mode: receive images without GUI.
      spdlog::info("Running in headless mode (no GUI)");
      while (!should_stop.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    } else {
      // GUI mode: real-time display with ImGui.
      spdlog::info("Initializing GUI for real-time display...");

      // Initialize GLFW via window_manager (same as viewer_app).
      const auto win_mgr = window_manager::get_instance();
      win_mgr->initialize();

      auto window = std::make_shared<window_base>("gRPC Image Viewer", 1280, 720);
      window->create();

      auto gfx_ctx = window->create_graphics_context();
      gfx_ctx.attach();

      // Create ImGui context first (same as viewer_app).
      IMGUI_CHECKVERSION();
      ImGui::CreateContext();
      ImGuiIO& io = ImGui::GetIO();
      (void)io;
      io.ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;
      ImGui::StyleColorsDark();
      ImGui_ImplGlfw_InitForVulkan(static_cast<GLFWwindow*>(window->get_handle()), true);

      // Initialize Vulkan ImGui backend.
      auto imgui_ctx = std::make_unique<imgui_context>(&gfx_ctx);
      imgui_ctx->initialize();

      window->show();

      // Main loop.
      while (!win_mgr->should_close() && !should_stop.load()) {
        win_mgr->handle_event();

        // Begin frame.
        imgui_ctx->begin_frame();
        gfx_ctx.begin_frame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Render image to the full window.
        {
          std::lock_guard<std::mutex> lock(display_mutex);
          if (!latest_frames.empty()) {
            auto& [name, frame] = *latest_frames.rbegin();
            if (!frame.empty()) {
              // Create or reuse texture buffer.
              if (texture_buffers.find(name) == texture_buffers.end()) {
                texture_buffers[name] = std::make_unique<texture_buffer>();
                texture_buffers[name]->set_context(&gfx_ctx);
              }

              auto& tex_buffer = texture_buffers[name];

              // Convert OpenCV image from BGR to RGB.
              cv::Mat rgb_frame;
              cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);

              // Flip vertically and horizontally to correct orientation.
              cv::flip(rgb_frame, rgb_frame, -1);

              // Upload texture.
              tex_buffer->upload_image(rgb_frame.cols, rgb_frame.rows, rgb_frame.data, 0);

              // Draw fullscreen using an ImGui window.
              ImGuiIO& io = ImGui::GetIO();
              ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
              ImGui::SetNextWindowSize(io.DisplaySize);
              ImGui::Begin("##fullscreen_image", nullptr,
                           ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                               ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings |
                               ImGuiWindowFlags_NoBringToFrontOnFocus);

              ImVec2 pos = ImGui::GetCursorScreenPos();
              ImVec2 size = ImGui::GetContentRegionAvail();
              rect display_rect{pos.x, pos.y, size.x, size.y};
              rect zoom{0, 0, 1, 1};
              tex_buffer->show(display_rect, 1.0f, zoom);

              ImGui::Dummy(size);
              ImGui::End();
            }
          }
        }

        // End frame.
        ImGui::Render();
        imgui_ctx->end_frame();
        gfx_ctx.end_frame();
      }

      // Cleanup.
      gfx_ctx.device->waitIdle();
      texture_buffers.clear();
      imgui_ctx->cleanup();
      imgui_ctx.reset();

      ImGui_ImplGlfw_Shutdown();
      ImGui::DestroyContext();

      gfx_ctx.detach();
      window->destroy();
      window.reset();

      win_mgr->terminate();
    }

    // Stop the server.
    spdlog::info("Stopping gRPC server...");
    grpc_node->stop();

    spdlog::info("Server stopped successfully");
    return 0;

  } catch (const std::exception& e) {
    spdlog::error("Fatal error: {}", e.what());
    return 1;
  }
}
