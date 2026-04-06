#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <variant>

namespace stargazer {

// Types of commands that can be dispatched to the pipeline from the gRPC thread
enum class pipeline_command_type {
  start,
  stop,
  enable_collecting,
  disable_collecting,
  dispatch_action,
};

// Result of a command execution: empty on success, error message on failure
using pipeline_command_result = std::optional<std::string>;

struct pipeline_command {
  pipeline_command_type type;
  std::string param;  // action_id for dispatch_action; unused otherwise
  std::promise<pipeline_command_result> promise;

  // Non-copyable because promise is non-copyable
  pipeline_command(pipeline_command_type t, std::string p)
      : type(t), param(std::move(p)) {}
};

// Thread-safe queue of commands.
// Producer: gRPC handler threads (enqueue + wait on future)
// Consumer: GUI/main thread (drain inside update())
class pipeline_command_queue {
  std::mutex mtx_;
  std::queue<std::unique_ptr<pipeline_command>> queue_;

 public:
  // Enqueue a command and return a future for the result.
  // Called from gRPC thread.
  std::future<pipeline_command_result> enqueue(pipeline_command_type type,
                                                std::string param = {}) {
    auto cmd = std::make_unique<pipeline_command>(type, std::move(param));
    auto future = cmd->promise.get_future();
    {
      std::lock_guard<std::mutex> lock(mtx_);
      queue_.push(std::move(cmd));
    }
    return future;
  }

  // Drain all pending commands and execute them via the provided handler.
  // Called from GUI/main thread in update().
  void drain(const std::function<pipeline_command_result(pipeline_command&)>& handler) {
    std::queue<std::unique_ptr<pipeline_command>> local;
    {
      std::lock_guard<std::mutex> lock(mtx_);
      std::swap(local, queue_);
    }
    while (!local.empty()) {
      auto& cmd = *local.front();
      auto result = handler(cmd);
      cmd.promise.set_value(std::move(result));
      local.pop();
    }
  }
};

}  // namespace stargazer
