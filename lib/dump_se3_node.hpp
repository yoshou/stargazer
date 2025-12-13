#pragma once

#include <sqlite3.h>

#include <deque>
#include <nlohmann/json.hpp>

#include "graph_proc.h"
#include "messages.hpp"

namespace stargazer {

using namespace coalsack;

class dump_se3_node : public graph_node {
  std::string db_path;
  std::string name;

  sqlite3* db;
  sqlite3_stmt* stmt;
  int topic_id;

  std::deque<std::tuple<double, std::string>> queue;

 public:
  dump_se3_node() : graph_node(), db(nullptr), stmt(nullptr), topic_id(-1) {}

  void set_db_path(std::string value) { db_path = value; }

  void set_name(std::string value) { name = value; }

  virtual std::string get_proc_name() const override { return "dump_se3"; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(db_path);
    archive(name);
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
      sqlite3_stmt* stmt;
      if (sqlite3_prepare_v2(db, "SELECT id FROM topics WHERE name = ?", -1, &stmt, nullptr) !=
          SQLITE_OK) {
        throw std::runtime_error("Failed to prepare statement");
      }

      if (sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_STATIC) != SQLITE_OK) {
        throw std::runtime_error("Failed to bind text");
      }

      if (sqlite3_step(stmt) == SQLITE_ROW) {
        topic_id = sqlite3_column_int(stmt, 0);
      } else {
        topic_id = -1;
      }

      sqlite3_finalize(stmt);
    }

    if (topic_id == -1) {
      sqlite3_stmt* stmt;
      if (sqlite3_prepare_v2(db, "INSERT INTO topics (name, type) VALUES (?, ?)", -1, &stmt,
                             nullptr) != SQLITE_OK) {
        throw std::runtime_error("Failed to prepare statement");
      }

      if (sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_STATIC) != SQLITE_OK) {
        throw std::runtime_error("Failed to bind text");
      }

      if (sqlite3_bind_text(stmt, 2, "se3", -1, SQLITE_STATIC) != SQLITE_OK) {
        throw std::runtime_error("Failed to bind text");
      }

      auto result = sqlite3_step(stmt);

      while (result == SQLITE_BUSY) {
        result = sqlite3_step(stmt);
      }

      if (result != SQLITE_DONE) {
        throw std::runtime_error("Failed to step");
      }

      sqlite3_finalize(stmt);

      if (sqlite3_prepare_v2(db, "SELECT last_insert_rowid()", -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error("Failed to prepare statement");
      }

      if (sqlite3_step(stmt) == SQLITE_ROW) {
        topic_id = sqlite3_column_int(stmt, 0);
      } else {
        throw std::runtime_error("Failed to get last insert id");
      }

      sqlite3_finalize(stmt);
    }

    if (sqlite3_prepare_v2(db, "INSERT INTO messages (topic_id, timestamp, data) VALUES (?, ?, ?)",
                           -1, &stmt, nullptr) != SQLITE_OK) {
      throw std::runtime_error("Failed to prepare statement");
    }
  }

  virtual void finalize() override {
    flush_queue();

    if (stmt) {
      sqlite3_finalize(stmt);
      stmt = nullptr;
    }
    if (db) {
      sqlite3_close(db);
      db = nullptr;
    }
  }

  void flush_queue() {
    if (sqlite3_exec(db, "BEGIN TRANSACTION", nullptr, nullptr, nullptr) != SQLITE_OK) {
      throw std::runtime_error("Failed to begin transaction");
    }

    while (queue.size() > 0) {
      const auto& [timestamp, str] = queue.front();

      if (sqlite3_reset(stmt) != SQLITE_OK) {
        throw std::runtime_error("Failed to reset");
      }

      if (sqlite3_bind_int64(stmt, 1, topic_id) != SQLITE_OK) {
        throw std::runtime_error("Failed to bind int64");
      }

      if (sqlite3_bind_int64(stmt, 2, timestamp) != SQLITE_OK) {
        throw std::runtime_error("Failed to bind int64");
      }

      if (sqlite3_bind_blob(stmt, 3, str.c_str(), str.size(), SQLITE_STATIC) != SQLITE_OK) {
        throw std::runtime_error("Failed to bind text");
      }

      auto result = sqlite3_step(stmt);

      while (result == SQLITE_BUSY) {
        result = sqlite3_step(stmt);
      }

      if (result != SQLITE_DONE) {
        throw std::runtime_error("Failed to step");
      }

      queue.pop_front();
    }

    auto result = sqlite3_exec(db, "END TRANSACTION", nullptr, nullptr, nullptr);

    while (result == SQLITE_BUSY) {
      result = sqlite3_exec(db, "END TRANSACTION", nullptr, nullptr, nullptr);
    }

    if (result != SQLITE_OK) {
      throw std::runtime_error("Failed to end transaction");
    }
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (auto frame_msg = std::dynamic_pointer_cast<se3_list_message>(message)) {
      const auto& data = frame_msg->get_data();

      nlohmann::json j_frame;
      std::vector<nlohmann::json> j_se3_list;
      for (const auto& se3 : data) {
        nlohmann::json j_se3;
        j_se3["position"]["x"] = se3.position.x;
        j_se3["position"]["y"] = se3.position.y;
        j_se3["position"]["z"] = se3.position.z;
        j_se3["rotation"]["x"] = se3.rotation.x;
        j_se3["rotation"]["y"] = se3.rotation.y;
        j_se3["rotation"]["z"] = se3.rotation.z;
        j_se3["rotation"]["w"] = se3.rotation.w;
        j_se3_list.push_back(j_se3);
      }

      j_frame["se3_list"] = j_se3_list;
      j_frame["timestamp"] = frame_msg->get_timestamp();
      j_frame["frame_number"] = frame_msg->get_frame_number();

      const auto j_str = j_frame.dump(2);

      queue.emplace_back(frame_msg->get_timestamp(), j_str);

      if (queue.size() >= 200) {
        flush_queue();
      }
    }
  }
};

}  // namespace stargazer

COALSACK_REGISTER_NODE(stargazer::dump_se3_node, coalsack::graph_node)
