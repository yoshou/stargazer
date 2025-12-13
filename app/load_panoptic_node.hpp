#pragma once

#include <spdlog/spdlog.h>
#include <sqlite3.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

#include "graph_proc.h"
#include "graph_proc_img.h"

namespace stargazer {
using namespace coalsack;

class load_panoptic_node : public graph_node {
  std::string db_path;
  std::string name;
  stream_type stream;
  stream_format format;
  int fps;

  std::shared_ptr<std::thread> th;
  std::atomic_bool playing;
  graph_edge_ptr output;

  uint64_t start_timestamp;

 public:
  load_panoptic_node()
      : graph_node(),
        stream(stream_type::COLOR),
        format(stream_format::BGR8),
        fps(30),
        output(std::make_shared<graph_edge>(this)),
        start_timestamp(0) {
    set_output(output);
  }

  void set_db_path(std::string value) { db_path = value; }

  void set_topic_name(std::string value) { name = value; }

  void set_stream(stream_type value) { stream = value; }

  void set_format(stream_format value) { format = value; }

  void set_fps(int value) { fps = value; }

  virtual std::string get_proc_name() const override { return "load_panoptic"; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(db_path);
    archive(name);
    archive(stream);
    archive(format);
    archive(fps);
  }

  virtual void initialize() override { start_timestamp = 0; }

  virtual void run() override {
    th.reset(new std::thread([this]() {
      namespace fs = std::filesystem;

      playing = true;
      uint64_t position = 0;
      const auto start_time = std::chrono::system_clock::now();

      spdlog::info("Start playback: {}", name);

      while (playing) {
        const auto timestamp = start_timestamp + position * 1000.0 / fps;
        const auto elapsed_time = timestamp - start_timestamp;

        const auto postfix = cv::format("_%08ld", position);
        const auto image_file = (fs::path(db_path) / name / (name + postfix + ".jpg")).string();

        std::ifstream f;
        f.open(image_file, std::ios::in | std::ios::binary);

        if (!f.is_open()) {
          playing = false;
        } else {
          std::vector<uint8_t> data;
          std::copy(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>(),
                    std::back_inserter(data));

          auto msg = std::make_shared<blob_frame_message>();
          msg->set_data(std::move(data));
          msg->set_frame_number(position);
          msg->set_timestamp(timestamp);
          msg->set_profile(std::make_shared<stream_profile>(stream, 0, format, fps, 0));

          std::this_thread::sleep_until(
              start_time +
              std::chrono::duration<double>(static_cast<double>(elapsed_time) / 1000.0));

          output->send(msg);

          position++;
        }
      }

      spdlog::info("End playback: {}", name);
    }));
  }

  virtual void stop() override {
    playing.store(false);
    if (th && th->joinable()) {
      th->join();
    }
  }
};

}  // namespace stargazer

COALSACK_REGISTER_NODE(stargazer::load_panoptic_node, coalsack::graph_node)
