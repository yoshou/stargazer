#include "bundle_adjust_data.hpp"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <iostream>
#include <nlohmann/json.hpp>

namespace stargazer::calibration {
void bundle_adjust_data::resize_observations(size_t num_observations) {
  num_observations_ = num_observations;

  point_index_.resize(num_observations_);
  camera_index_.resize(num_observations_);
  observations_.resize(2 * num_observations_);
}
void bundle_adjust_data::resize_parameters(size_t num_cameras, size_t num_points) {
  num_cameras_ = num_cameras;
  num_points_ = num_points;
  num_parameters_ = num_camera_parameters * num_cameras + 3 * num_points;

  parameters_.resize(num_parameters_);
}
bool bundle_adjust_data::load_txt(std::istream &ifs) {
  if (!ifs) {
    return false;
  }

  ifs >> num_cameras_;
  ifs >> num_points_;
  ifs >> num_observations_;
  num_parameters_ = num_camera_parameters * num_cameras_ + 3 * num_points_;

  point_index_.resize(num_observations_);
  camera_index_.resize(num_observations_);
  observations_.resize(2 * num_observations_);
  parameters_.resize(num_parameters_);

  for (int i = 0; i < num_observations_; i++) {
    ifs >> camera_index_[i];
    ifs >> point_index_[i];
    for (int j = 0; j < 2; j++) {
      ifs >> observations_[2 * i + j];
    }
  }
  for (int i = 0; i < num_parameters_; i++) {
    ifs >> parameters_[i];
  }
  return true;
}

bool bundle_adjust_data::save_txt(std::ostream &ofs) {
  if (!ofs) {
    return false;
  }

  ofs << num_cameras_ << " " << num_points_ << " " << num_observations_ << std::endl;

  for (int i = 0; i < num_observations_; i++) {
    ofs << camera_index_[i];
    ofs << " " << point_index_[i];
    for (int j = 0; j < 2; j++) {
      ofs << " " << observations_[2 * i + j];
    }
    ofs << std::endl;
  }
  for (int i = 0; i < num_parameters_; i++) {
    ofs << parameters_[i] << std::endl;
  }
  return true;
}

bool bundle_adjust_data::load_txt(const std::string &filename) {
  std::ifstream ifs;
  ifs.open(filename, std::ios::in);

  return load_txt(ifs);
}

bool bundle_adjust_data::save_txt(const std::string &filename) {
  std::ofstream ofs;
  ofs.open(filename, std::ios::out | std::ios::out);

  return save_txt(ofs);
}

bool bundle_adjust_data::load_json(std::istream &ifs) {
  if (!ifs) {
    return false;
  }

  auto j = nlohmann::json();
  ifs >> j;

  const auto j_observations = j["observations"];
  const auto j_camera_params = j["camera_params"];
  const auto j_points = j["points"];

  num_observations_ = j_observations[j_observations.items().begin().key()].size();
  num_cameras_ = j_camera_params[j_camera_params.items().begin().key()].size();
  num_points_ = j_points[j_points.items().begin().key()].size();

  num_parameters_ = num_camera_parameters * num_cameras_ + 3 * num_points_;

  point_index_.resize(num_observations_);
  camera_index_.resize(num_observations_);
  observations_.resize(2 * num_observations_);
  parameters_.resize(num_parameters_);

  for (int i = 0; i < num_cameras_; i++) {
    for (int j = 0; j < num_camera_parameters; j++) {
      parameters_[i * num_camera_parameters + j] =
          j_camera_params[std::to_string(j)][i].get<double>();
    }
  }
  for (int i = 0; i < num_points_; i++) {
    parameters_[num_camera_parameters * num_cameras_ + i * 3] = j_points["x"][i].get<double>();
    parameters_[num_camera_parameters * num_cameras_ + i * 3 + 1] = j_points["y"][i].get<double>();
    parameters_[num_camera_parameters * num_cameras_ + i * 3 + 2] = j_points["z"][i].get<double>();
  }
  for (int i = 0; i < num_observations_; i++) {
    camera_index_[i] = j_observations["camera"][i].get<int>();
    point_index_[i] = j_observations["index"][i].get<int>();
    observations_[2 * i] = j_observations["x"][i].get<double>();
    observations_[2 * i + 1] = j_observations["y"][i].get<double>();
  }
  return true;
}

bool bundle_adjust_data::save_json(std::ostream &ofs) {
  if (!ofs) {
    return false;
  }

  auto j = nlohmann::json();

  // camera params
  {
    std::unordered_map<std::string, std::vector<double>> camera_params;
    for (int j = 0; j < num_camera_parameters; j++) {
      std::vector<double> values;
      for (int i = 0; i < num_cameras_; i++) {
        values.push_back(parameters_[i * num_camera_parameters + j]);
      }
      camera_params[std::to_string(j)] = values;
    }
    j["camera_params"] = camera_params;
  }

  // points
  {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    for (int i = 0; i < num_points_; i++) {
      x.push_back(parameters_[num_camera_parameters * num_cameras_ + i * 3]);
      y.push_back(parameters_[num_camera_parameters * num_cameras_ + i * 3 + 1]);
      z.push_back(parameters_[num_camera_parameters * num_cameras_ + i * 3 + 2]);
    }
    j["points"]["x"] = x;
    j["points"]["y"] = y;
    j["points"]["z"] = z;
  }

  // observations
  {
    std::vector<int> camera;
    std::vector<int> index;
    std::vector<double> x;
    std::vector<double> y;
    for (int i = 0; i < num_observations_; i++) {
      camera.push_back(camera_index_[i]);
      index.push_back(point_index_[i]);
      x.push_back(observations_[2 * i]);
      y.push_back(observations_[2 * i + 1]);
    }
    j["observations"]["camera"] = camera;
    j["observations"]["index"] = index;
    j["observations"]["x"] = x;
    j["observations"]["y"] = y;
  }

  ofs << j;

  return true;
}

bool bundle_adjust_data::load_json(const std::string &filename) {
  std::ifstream ifs;
  ifs.open(filename, std::ios::in);

  return load_json(ifs);
}

bool bundle_adjust_data::save_json(const std::string &filename) {
  std::ofstream ofs;
  ofs.open(filename, std::ios::out | std::ios::out);

  return save_json(ofs);
}

glm::mat4 bundle_adjust_data::get_camera_extrinsic(std::size_t i) {
  double quat[4];
  ceres::AngleAxisToQuaternion(mutable_camera(i), quat);
  double rot[9];
  ceres::QuaternionToRotation(quat, rot);

  double *trans = &mutable_camera(i)[3];

  glm::mat4 mat(1.0);
  for (size_t j = 0; j < 3; j++) {
    for (size_t k = 0; k < 3; k++) {
      mat[j][k] = rot[k * 3 + j];
    }
  }
  for (size_t k = 0; k < 3; k++) {
    mat[3][k] = trans[k];
  }

  return mat;
}

static constexpr auto num_parameters = bundle_adjust_data::num_camera_parameters;

template <bool is_radial_distortion = false>
struct reprojection_error_functor {
  reprojection_error_functor(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T *const camera, const T *const point, T *residuals) const {
    T p[3];
    ceres::AngleAxisRotatePoint(camera, point, p);

    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    T xp = p[0] / p[2];
    T yp = p[1] / p[2];

    T predicted_x;
    T predicted_y;

    const T &fx = camera[6];
    const T &fy = camera[7];
    const T &cx = camera[8];
    const T &cy = camera[9];
    const T r2 = xp * xp + yp * yp;

    const T &k1 = camera[10];
    const T &k2 = camera[11];
    const T &k3 = camera[12];
    const T &p1 = camera[13];
    const T &p2 = camera[14];

    if constexpr (is_radial_distortion) {
      const T &k4 = camera[15];
      const T &k5 = camera[16];
      const T &k6 = camera[17];

      const T distortion =
          (1.0 + (k1 + (k2 + k3 * r2) * r2) * r2) / (1.0 + (k4 + (k5 + k6 * r2) * r2) * r2);

      predicted_x = fx * (distortion * xp + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp)) + cx;
      predicted_y = fy * (distortion * yp + 2.0 * p2 * xp * yp + p1 * (r2 + 2.0 * yp * yp)) + cy;
    } else {
      const T distortion = (1.0 + (k1 + (k2 + k3 * r2) * r2) * r2);

      predicted_x = fx * (distortion * xp + 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp)) + cx;
      predicted_y = fy * (distortion * yp + 2.0 * p2 * xp * yp + p1 * (r2 + 2.0 * yp * yp)) + cy;
    }

    const T err_x = predicted_x - observed_x;
    const T err_y = predicted_y - observed_y;
    residuals[0] = err_x;
    residuals[1] = err_y;

    return true;
  }

  static ceres::CostFunction *create(const double observed_x, const double observed_y) {
    return (new ceres::AutoDiffCostFunction<reprojection_error_functor, 2, num_parameters, 3>(
        new reprojection_error_functor(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;
};

void bundle_adjustment(stargazer::calibration::bundle_adjust_data &ba_data, bool only_extrinsic,
                       bool robust) {
  const double *observations = ba_data.observations();
  ceres::Problem problem;

  ceres::SubsetManifold *constant_params_manifold = nullptr;
  if (only_extrinsic) {
    std::vector<int> constant_params;

    for (int i = 6; i < num_parameters; i++) {
      constant_params.push_back(i);
    }

    constant_params_manifold = new ceres::SubsetManifold(num_parameters, constant_params);
  }

  for (int i = 0; i < ba_data.num_observations(); ++i) {
    ceres::LossFunction *loss_function = robust ? new ceres::HuberLoss(1.0) : nullptr;

    ceres::CostFunction *cost_function =
        reprojection_error_functor<false>::create(observations[2 * i + 0], observations[2 * i + 1]);
    problem.AddResidualBlock(cost_function, loss_function,
                             ba_data.mutable_camera_for_observation(i),
                             ba_data.mutable_point_for_observation(i));

    if (only_extrinsic) {
      problem.SetManifold(ba_data.mutable_camera_for_observation(i), constant_params_manifold);
    }
  }

  std::cout << "Num cameras: " << ba_data.num_cameras() << std::endl;
  std::cout << "Num points: " << ba_data.num_points() << std::endl;
  std::cout << "Num Observations: " << ba_data.num_observations() << std::endl;

  ceres::Solver::Options options;
  options.use_nonmonotonic_steps = true;
  options.preconditioner_type = ceres::SCHUR_JACOBI;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.use_inner_iterations = true;
  options.max_num_iterations = 100;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";
}
}  // namespace stargazer::calibration
