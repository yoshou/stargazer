#pragma once

#include <array>
#include <glm/glm.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "dust3r_alignment.hpp"
#include "parameters.hpp"

namespace stargazer::dust3r {

// Transform DUSt3R pixel coordinates (512×288 space) to original image pixel coordinates.
// Inverse of: cv::resize(bgr, resized, cv::Size(ONNX_W, ONNX_H), ...)
void dust3r_to_original(float u_512, float v_512, int W_orig, int H_orig, float& u_orig,
                        float& v_orig);

// Transform DUSt3R focal length (512 space) to original image focal lengths (fx, fy).
void dust3r_focal_to_original(float focal_512, int W_orig, int H_orig, float& fx, float& fy);

// A single 2D observation of a 3D point by one camera
struct ba_observation {
  int camera_idx;  // camera index (0-based, into the names vector)
  float u_orig;    // pixel x in original image
  float v_orig;    // pixel y in original image
  float weight;    // sqrt(confidence) for residual weighting
};

// A 3D point with its 2D observations across cameras
struct ba_point {
  float x, y, z;  // world 3D coordinates (DUSt3R global frame)
  float conf;     // accumulated confidence (for merge)
  std::vector<ba_observation> observations;
};

// Build shared 3D point cloud and 2D observations from DUSt3R pair results.
// For each camera, the best pair is selected and per-camera point clouds are merged
// in world frame using a nearest-neighbour search (OpenCV flann).
std::vector<ba_point> merge_shared_points(
    const std::vector<std::string>& camera_names, const std::vector<pair_result>& pair_results,
    const std::unordered_map<std::string, aligned_pose>& poses, int orig_W, int orig_H,
    float conf_threshold = 3.0f, int subsample = 8, float merge_threshold = 0.05f);

// Keep only points observed by at least min_cameras cameras; re-index observations.
void filter_observations(std::vector<ba_point>& points, int min_cameras = 2);

// Remove outlier observations based on reprojection error (median + k*MAD).
// Uses initial camera parameters (from aligned poses) to compute errors.
// Returns number of observations removed.
int filter_outlier_observations(std::vector<ba_point>& points,
                                const std::vector<std::string>& camera_names,
                                const std::unordered_map<std::string, aligned_pose>& poses,
                                int orig_W, int orig_H, float k = 3.0f);

// Run bundle adjustment optimising intrinsics + extrinsics + 3D points.
// Camera 0 extrinsics are held fixed as gauge constraint.
// Initial intrinsics from aligned_pose.focal/cx/cy; distortion initialised to zero.
std::vector<camera_t> run_bundle_adjustment(
    const std::vector<std::string>& camera_names,
    const std::unordered_map<std::string, aligned_pose>& poses, std::vector<ba_point>& points,
    int orig_W, int orig_H, double* out_initial_rmse = nullptr, double* out_final_rmse = nullptr,
    double* out_initial_ceres_cost = nullptr);

}  // namespace stargazer::dust3r
