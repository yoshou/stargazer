#pragma once

#include <sqlite3.h>

#include <cereal/types/map.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <deque>
#include <nlohmann/json.hpp>

#include "graph_proc.h"
#include "parameters.hpp"

// Forward declaration (will be included via graph_builder.cpp)
namespace stargazer {
class reconstruction_result_message;
}

namespace stargazer {

using namespace coalsack;

class dump_reconstruction_node : public graph_node {
  std::string db_path;
  std::string name;

  sqlite3* db;
  sqlite3_stmt* stmt;
  int topic_id;

  std::deque<std::tuple<double, std::string>> queue;

 public:
  dump_reconstruction_node() : graph_node(), db(nullptr), stmt(nullptr), topic_id(-1) {}

  void set_db_path(std::string value) { db_path = std::move(value); }

  void set_name(std::string value) { name = std::move(value); }

  virtual std::string get_proc_name() const override { return "dump_reconstruction"; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(db_path, name);
  }

  virtual void initialize() override {
    if (sqlite3_open(db_path.c_str(), &db) != SQLITE_OK) {
      throw std::runtime_error("Failed to open database");
    }

    if (sqlite3_exec(db,
                     "CREATE TABLE IF NOT EXISTS topics(id INTEGER PRIMARY KEY, name TEXT NOT "
                     "NULL, type TEXT NOT NULL)",
                     nullptr, nullptr, nullptr) != SQLITE_OK) {
      throw std::runtime_error("Failed to create table");
    }

    if (sqlite3_exec(db,
                     "CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, topic_id INTEGER "
                     "NOT NULL, timestamp INTEGER NOT NULL, data BLOB NOT NULL)",
                     nullptr, nullptr, nullptr) != SQLITE_OK) {
      throw std::runtime_error("Failed to create table");
    }

    if (sqlite3_exec(db, "CREATE INDEX IF NOT EXISTS timestamp_idx ON messages (timestamp ASC)",
                     nullptr, nullptr, nullptr) != SQLITE_OK) {
      throw std::runtime_error("Failed to create index");
    }

    {
      sqlite3_stmt* qstmt;
      if (sqlite3_prepare_v2(db, "SELECT id FROM topics WHERE name = ?", -1, &qstmt, nullptr) !=
          SQLITE_OK) {
        throw std::runtime_error("Failed to prepare statement");
      }

      if (sqlite3_bind_text(qstmt, 1, name.c_str(), -1, SQLITE_STATIC) != SQLITE_OK) {
        throw std::runtime_error("Failed to bind text");
      }

      if (sqlite3_step(qstmt) == SQLITE_ROW) {
        topic_id = sqlite3_column_int(qstmt, 0);
      } else {
        topic_id = -1;
      }

      sqlite3_finalize(qstmt);
    }

    if (topic_id == -1) {
      sqlite3_stmt* istmt;
      if (sqlite3_prepare_v2(db, "INSERT INTO topics (name, type) VALUES (?, ?)", -1, &istmt,
                             nullptr) != SQLITE_OK) {
        throw std::runtime_error("Failed to prepare statement");
      }

      if (sqlite3_bind_text(istmt, 1, name.c_str(), -1, SQLITE_STATIC) != SQLITE_OK) {
        throw std::runtime_error("Failed to bind text");
      }
      if (sqlite3_bind_text(istmt, 2, "reconstruction", -1, SQLITE_STATIC) != SQLITE_OK) {
        throw std::runtime_error("Failed to bind text");
      }

      auto step_result = sqlite3_step(istmt);
      while (step_result == SQLITE_BUSY) {
        step_result = sqlite3_step(istmt);
      }
      if (step_result != SQLITE_DONE) {
        throw std::runtime_error("Failed to step");
      }
      sqlite3_finalize(istmt);

      sqlite3_stmt* lid_stmt;
      if (sqlite3_prepare_v2(db, "SELECT last_insert_rowid()", -1, &lid_stmt, nullptr) !=
          SQLITE_OK) {
        throw std::runtime_error("Failed to prepare statement");
      }
      if (sqlite3_step(lid_stmt) == SQLITE_ROW) {
        topic_id = sqlite3_column_int(lid_stmt, 0);
      } else {
        throw std::runtime_error("Failed to get last insert id");
      }
      sqlite3_finalize(lid_stmt);
    }

    if (sqlite3_prepare_v2(db, "INSERT INTO messages (topic_id, timestamp, data) VALUES (?, ?, ?)",
                           -1, &stmt, nullptr) != SQLITE_OK) {
      throw std::runtime_error("Failed to prepare statement");
    }
  }

  void flush_queue() {
    if (!db) {
      return;
    }
    if (sqlite3_exec(db, "BEGIN TRANSACTION", nullptr, nullptr, nullptr) != SQLITE_OK) {
      throw std::runtime_error("Failed to begin transaction");
    }

    while (!queue.empty()) {
      const auto& [timestamp, payload] = queue.front();

      if (sqlite3_reset(stmt) != SQLITE_OK) {
        throw std::runtime_error("Failed to reset");
      }

      if (sqlite3_bind_int64(stmt, 1, topic_id) != SQLITE_OK) {
        throw std::runtime_error("Failed to bind int64");
      }

      if (sqlite3_bind_int64(stmt, 2, static_cast<sqlite3_int64>(timestamp)) != SQLITE_OK) {
        throw std::runtime_error("Failed to bind int64");
      }

      if (sqlite3_bind_blob(stmt, 3, payload.data(), static_cast<int>(payload.size()),
                            SQLITE_STATIC) != SQLITE_OK) {
        throw std::runtime_error("Failed to bind blob");
      }

      auto step_result = sqlite3_step(stmt);
      while (step_result == SQLITE_BUSY) {
        step_result = sqlite3_step(stmt);
      }
      if (step_result != SQLITE_DONE) {
        throw std::runtime_error("Failed to step");
      }

      queue.pop_front();
    }

    auto end_result = sqlite3_exec(db, "END TRANSACTION", nullptr, nullptr, nullptr);
    while (end_result == SQLITE_BUSY) {
      end_result = sqlite3_exec(db, "END TRANSACTION", nullptr, nullptr, nullptr);
    }
    if (end_result != SQLITE_OK) {
      throw std::runtime_error("Failed to end transaction");
    }
  }

  virtual void finalize() override {
    if (!db) {
      return;
    }
    flush_queue();
    if (stmt) {
      sqlite3_finalize(stmt);
      stmt = nullptr;
    }
    sqlite3_close(db);
    db = nullptr;
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (auto frame_msg = std::dynamic_pointer_cast<reconstruction_result_message>(message)) {
      const auto timestamp = frame_msg->get_timestamp();
      const auto frame_number = frame_msg->get_frame_number();
      const auto& result = frame_msg->get_result();
      const auto& cameras = frame_msg->get_cameras();
      const auto& axis = frame_msg->get_axis();

      // Extract camera names from views
      std::vector<std::string> camera_names;
      camera_names.reserve(result.views.size());
      for (const auto& view : result.views) {
        camera_names.push_back(view.name);
      }

      nlohmann::json out;
      out["timestamp"] = timestamp;
      out["frame_number"] = static_cast<std::int64_t>(frame_number);
      out["num_keypoints"] = static_cast<std::int64_t>(result.num_keypoints);
      out["camera_names"] = camera_names;
      out["axis"] = axis;
      out["cameras"] = cameras;

      nlohmann::json views = nlohmann::json::array();
      for (const auto& view : result.views) {
        nlohmann::json v;
        v["name"] = view.name;

        nlohmann::json detections = nlohmann::json::array();
        for (const auto& detection : view.detections) {
          nlohmann::json p;
          p["bbox"] = {detection.bbox.left, detection.bbox.top, detection.bbox.right,
                       detection.bbox.bottom};
          p["bbox_score"] = detection.bbox_score;

          nlohmann::json kps = nlohmann::json::array();
          for (size_t j = 0; j < detection.keypoints.size(); ++j) {
            kps.push_back(
                {detection.keypoints[j].x, detection.keypoints[j].y, detection.scores[j]});
          }
          p["keypoints"] = std::move(kps);
          detections.push_back(std::move(p));
        }
        v["detections"] = std::move(detections);
        views.push_back(std::move(v));
      }
      out["views"] = std::move(views);

      nlohmann::json matches = nlohmann::json::array();
      for (const auto& match : result.matches) {
        nlohmann::json m = nlohmann::json::array();
        for (const auto& id : match) {
          m.push_back({{"view", static_cast<std::int64_t>(id.first)},
                       {"detection", static_cast<std::int64_t>(id.second)}});
        }
        matches.push_back(std::move(m));
      }
      out["matches"] = std::move(matches);

      const auto payload = out.dump();
      queue.emplace_back(timestamp, payload);

      if (queue.size() >= 200) {
        flush_queue();
      }
    }
  }
};

}  // namespace stargazer

COALSACK_REGISTER_NODE(stargazer::dump_reconstruction_node, coalsack::graph_node)
