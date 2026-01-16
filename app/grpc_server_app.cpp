#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include <spdlog/spdlog.h>

#include "../lib/glm_serialize.hpp"
#include "../lib/grpc_server_node.hpp"
#include "../lib/messages.hpp"

namespace {
std::atomic_bool should_stop(false);
}

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
            << "  -h, --help             Show this help message\n"
            << std::endl;
}

int main(int argc, char* argv[]) {
  std::string server_address = "0.0.0.0:50051";

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      return 0;
    } else if ((arg == "-a" || arg == "--address") && i + 1 < argc) {
      server_address = argv[++i];
    } else {
      std::cerr << "Unknown option: " << arg << std::endl;
      print_usage(argv[0]);
      return 1;
    }
  }

  // Set log level
  spdlog::set_level(spdlog::level::info);
  spdlog::info("Starting stargazer gRPC server");
  spdlog::info("Server address: {}", server_address);

  // Setup signal handlers
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  try {
    // Create gRPC server node
    auto grpc_node = std::make_shared<stargazer::grpc_server_node>();
    grpc_node->set_address(server_address);

    // Start the server
    spdlog::info("Starting gRPC server node");
    grpc_node->run();

    spdlog::info("gRPC server is running. Press Ctrl+C to stop.");

    // Main loop - wait for shutdown signal
    while (!should_stop.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Stop the server
    spdlog::info("Stopping gRPC server...");
    grpc_node->stop();

    spdlog::info("Server stopped successfully");
    return 0;

  } catch (const std::exception& e) {
    spdlog::error("Fatal error: {}", e.what());
    return 1;
  }
}
