#pragma once

#include <array>
#include <atomic>
#include <cereal/types/array.hpp>
#include <condition_variable>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <mutex>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <unordered_map>
#include <variant>

#include "glm_json.hpp"

namespace stargazer {
struct camera_intrin_t {
  float fx, fy;
  float cx, cy;
  std::array<float, 5> coeffs = {};

  glm::mat3 get_matrix() const { return glm::mat3(fx, 0, 0, 0, fy, 0, cx, cy, 1); }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(fx, fy, cx, cy, coeffs);
  }
};

struct camera_extrin_t {
  glm::vec3 translation;
  glm::mat3 rotation;

  glm::mat4 transform_matrix() const {
    return glm::translate(glm::mat4(1.0f), translation) * glm::mat4(rotation);
  }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(translation, rotation);
  }
};

struct camera_t {
  camera_intrin_t intrin;
  camera_extrin_t extrin;
  uint32_t width, height;

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(intrin, extrin, width, height);
  }
};

struct scene_t {
  glm::mat4 axis;

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(axis);
  }
};

class parameters_t {
  std::unordered_map<std::string, std::variant<camera_t, scene_t>> parameters;
  std::string path;
  mutable std::mutex mtx;
  std::condition_variable cv;
  std::atomic<uint64_t> version{0};

 public:
  parameters_t(const std::string& path) : path(path) {}

  void load();

  void save() const;

  std::variant<camera_t, scene_t> operator[](const std::string& key) const {
    std::lock_guard lock(mtx);
    return parameters.at(key);
  }
  std::variant<camera_t, scene_t>& operator[](const std::string& key) {
    std::lock_guard lock(mtx);
    return parameters[key];
  }

  std::variant<camera_t, scene_t> at(const std::string& key) const {
    std::lock_guard lock(mtx);
    return parameters.at(key);
  }
  std::variant<camera_t, scene_t>& at(const std::string& key) {
    std::lock_guard lock(mtx);
    return parameters.at(key);
  }

  bool contains(const std::string& key) const {
    std::lock_guard lock(mtx);
    return parameters.find(key) != parameters.end();
  }

  void update_camera(const std::string& key, const camera_t& camera) {
    {
      std::lock_guard lock(mtx);
      parameters[key] = camera;
    }
    ++version;
    cv.notify_all();
  }

  void update_scene(const std::string& key, const scene_t& scene) {
    {
      std::lock_guard lock(mtx);
      parameters[key] = scene;
    }
    ++version;
    cv.notify_all();
  }

  uint64_t get_version() const { return version.load(); }

  // Blocks until version changes from last_version or running is cleared, returns new version
  uint64_t wait_for_change(uint64_t last_version, const std::atomic_bool& running) {
    std::unique_lock lock(mtx);
    cv.wait(lock, [&] { return version.load() != last_version || !running.load(); });
    return version.load();
  }

  void notify_all() { cv.notify_all(); }
};

void get_cv_intrinsic(const camera_intrin_t& intrin, cv::Mat& camera_matrix, cv::Mat& dist_coeffs);

static inline void to_json(nlohmann::json& j, const camera_intrin_t& intrin) {
  j = {
      {"fx", intrin.fx}, {"fy", intrin.fy},         {"cx", intrin.cx},
      {"cy", intrin.cy}, {"coeffs", intrin.coeffs},
  };
}

static inline void from_json(const nlohmann::json& j, camera_intrin_t& intrin) {
  intrin.fx = j["fx"].get<float>();
  intrin.fy = j["fy"].get<float>();
  intrin.cx = j["cx"].get<float>();
  intrin.cy = j["cy"].get<float>();
  intrin.coeffs = j["coeffs"].get<std::array<float, 5>>();
}

static inline void to_json(nlohmann::json& j, const camera_extrin_t& extrin) {
  j = {
      {"rotation", extrin.rotation},
      {"translation", extrin.translation},
  };
}

static inline void from_json(const nlohmann::json& j, camera_extrin_t& extrin) {
  extrin.rotation = j["rotation"].get<glm::mat3>();
  extrin.translation = j["translation"].get<glm::vec3>();
}

static inline void to_json(nlohmann::json& j, const camera_t& camera) {
  j = {
      {"intrin", camera.intrin},
      {"extrin", camera.extrin},
      {"width", camera.width},
      {"height", camera.height},
  };
}

static inline void from_json(const nlohmann::json& j, camera_t& camera) {
  camera.intrin = j["intrin"].get<camera_intrin_t>();
  camera.extrin = j["extrin"].get<camera_extrin_t>();
  camera.width = j["width"].get<uint32_t>();
  camera.height = j["height"].get<uint32_t>();
}

static inline void to_json(nlohmann::json& j, const scene_t& scene) {
  j = {
      {"axis", scene.axis},
  };
}

static inline void from_json(const nlohmann::json& j, scene_t& scene) {
  scene.axis = j["axis"].get<glm::mat4>();
}
}  // namespace stargazer
