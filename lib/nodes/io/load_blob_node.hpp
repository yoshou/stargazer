/// @file load_blob_node.hpp
/// @brief SQLite blob frame playback source node.
/// @ingroup io_nodes
#pragma once

#include <spdlog/spdlog.h>
#include <sqlite3.h>

#include <atomic>
#include <chrono>
#include <thread>

#include "coalsack/core/graph_proc.h"
#include "coalsack/image/image_nodes.h"

namespace stargazer {
using namespace coalsack;

/// @brief Plays back `blob` frames from a SQLite database.
/// @details Autonomous source node that reads blob rows from the `messages` table
///          for the configured topic, decodes the timestamp and data, and emits
///          them as `blob_frame_message` on @b "default" at the configured frame rate.
///          Playback runs on a background thread started by `run()`.
///
/// @par Inputs
/// (none — autonomous source node)
///
/// @par Outputs
/// - @b "default" — `blob_frame_message` — decoded blob frame
///
/// @par Properties
/// - `db_path`    (`std::string`, default `""`) — path to the SQLite database file
/// - `topic_name` (`std::string`, default `""`) — topic to replay from the database
///
/// @see dump_blob_node, load_marker_node
class load_blob_node : public graph_node {
  std::string db_path;
  std::string name;
  stream_type stream;
  stream_format format;
  int fps;

  std::vector<uint64_t> timestamps;
  std::shared_ptr<std::thread> th;
  std::atomic_bool playing;
  graph_edge_ptr output;

  uint64_t start_timestamp;
  uint64_t id;

 public:
  load_blob_node()
      : graph_node(),
        stream(stream_type::COLOR),
        format(stream_format::BGR8),
        fps(30),
        timestamps(),
        output(std::make_shared<graph_edge>(this)),
        start_timestamp(0) {
    set_output(output);
  }

  void set_db_path(std::string value) { db_path = value; }

  void set_topic_name(std::string value) { name = value; }

  void set_stream(stream_type value) { stream = value; }

  void set_format(stream_format value) { format = value; }

  void set_fps(int value) { fps = value; }

  virtual std::string get_proc_name() const override { return "load_blob"; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(db_path);
    archive(name);
    archive(stream);
    archive(format);
    archive(fps);
  }

  virtual void initialize() override {
    start_timestamp = std::numeric_limits<uint64_t>::max();

    sqlite3* db;
    if (sqlite3_open_v2(db_path.c_str(), &db, SQLITE_OPEN_READONLY | SQLITE_OPEN_NOMUTEX,
                        nullptr) != SQLITE_OK) {
      throw std::runtime_error("Failed to open database");
    }

    {
      sqlite3_stmt* stmt;
      if (sqlite3_prepare_v2(db, "SELECT timestamp FROM messages ORDER BY timestamp ASC", -1, &stmt,
                             nullptr) != SQLITE_OK) {
        throw std::runtime_error("Failed to prepare statement");
      }

      if (sqlite3_step(stmt) == SQLITE_ROW) {
        const auto timestamp = sqlite3_column_int64(stmt, 0);
        start_timestamp = std::min(start_timestamp, static_cast<uint64_t>(timestamp));
      }

      sqlite3_finalize(stmt);
    }

    {
      sqlite3_stmt* stmt;
      if (sqlite3_prepare_v2(db, "SELECT id, name FROM topics WHERE name = ?", -1, &stmt,
                             nullptr) != SQLITE_OK) {
        throw std::runtime_error("Failed to prepare statement");
      }

      if (sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_STATIC) != SQLITE_OK) {
        throw std::runtime_error("Failed to bind text");
      }

      if (sqlite3_step(stmt) == SQLITE_ROW) {
        id = sqlite3_column_int64(stmt, 0);
      } else {
        throw std::runtime_error("Failed to get topic id");
      }

      sqlite3_finalize(stmt);
    }

    sqlite3_close(db);
  }

  virtual void run() override {
    th.reset(new std::thread([this]() {
      sqlite3* db;
      if (sqlite3_open_v2(db_path.c_str(), &db, SQLITE_OPEN_READONLY | SQLITE_OPEN_NOMUTEX,
                          nullptr) != SQLITE_OK) {
        throw std::runtime_error("Failed to open database");
      }

      sqlite3_stmt* stmt;
      if (sqlite3_prepare_v2(db,
                             "SELECT timestamp, topic_id, data FROM messages WHERE topic_id = ? "
                             "ORDER BY timestamp ASC",
                             -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error("Failed to prepare statement");
      }

      const auto id_str = std::to_string(id);
      if (sqlite3_bind_text(stmt, 1, id_str.c_str(), -1, SQLITE_STATIC) != SQLITE_OK) {
        throw std::runtime_error("Failed to bind text");
      }

      playing = true;
      uint64_t position = 0;
      const auto start_time = std::chrono::system_clock::now();

      spdlog::info("Start playback: {}", name);

      while (playing && sqlite3_step(stmt) == SQLITE_ROW) {
        const auto timestamp = sqlite3_column_int64(stmt, 0);
        const auto elapsed_time = timestamp - start_timestamp;

        const auto data_size = static_cast<size_t>(sqlite3_column_bytes(stmt, 2));
        const auto data_ptr = reinterpret_cast<const uint8_t*>(sqlite3_column_blob(stmt, 2));
        const std::vector<uint8_t> data(data_ptr, data_ptr + data_size);

        auto msg = std::make_shared<blob_frame_message>();
        msg->set_data(std::move(data));
        msg->set_frame_number(position);
        msg->set_timestamp(timestamp);
        msg->set_profile(std::make_shared<stream_profile>(stream, 0, format, fps, 0));

        std::this_thread::sleep_until(
            start_time + std::chrono::duration<double>(static_cast<double>(elapsed_time) / 1000.0));

        output->send(msg);

        position++;
      }

      sqlite3_finalize(stmt);
      stmt = nullptr;

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

COALSACK_REGISTER_NODE(stargazer::load_blob_node, coalsack::graph_node)
