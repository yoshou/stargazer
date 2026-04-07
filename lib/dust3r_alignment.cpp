#include "dust3r_alignment.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <stdexcept>

namespace stargazer::dust3r {

static constexpr int H = ONNX_H;
static constexpr int W = ONNX_W;
static constexpr int HW = H * W;

struct UnionFind {
  std::vector<int> parent, rank_;
  explicit UnionFind(int n) : parent(n), rank_(n, 0) { std::iota(parent.begin(), parent.end(), 0); }
  int find(int x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }
  bool unite(int a, int b) {
    a = find(a);
    b = find(b);
    if (a == b) return false;
    if (rank_[a] < rank_[b]) std::swap(a, b);
    parent[b] = a;
    if (rank_[a] == rank_[b]) ++rank_[a];
    return true;
  }
};

static bool procrustes(const std::vector<Eigen::Vector3f>& src,
                       const std::vector<Eigen::Vector3f>& dst, const std::vector<float>& weights,
                       Eigen::Matrix3f& R, Eigen::Vector3f& t) {
  assert(src.size() == dst.size());
  assert(src.size() == weights.size());

  float wsum = 0.0f;
  Eigen::Vector3f mu_src = Eigen::Vector3f::Zero();
  Eigen::Vector3f mu_dst = Eigen::Vector3f::Zero();

  for (size_t i = 0; i < src.size(); ++i) {
    wsum += weights[i];
    mu_src += weights[i] * src[i];
    mu_dst += weights[i] * dst[i];
  }
  if (wsum < 1e-8f) return false;
  mu_src /= wsum;
  mu_dst /= wsum;

  Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
  for (size_t i = 0; i < src.size(); ++i) {
    cov += weights[i] * (src[i] - mu_src) * (dst[i] - mu_dst).transpose();
  }

  Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3f U = svd.matrixU();
  Eigen::Matrix3f V = svd.matrixV();

  float diag_last = (V * U.transpose()).determinant() > 0.0f ? 1.0f : -1.0f;
  Eigen::DiagonalMatrix<float, 3> D(1.0f, 1.0f, diag_last);

  R = V * D * U.transpose();
  t = mu_dst - R * mu_src;
  return true;
}

static constexpr int SUBSAMPLE = 8;  // use every 8th pixel → 288*512/64 = 2304 points per pair
std::unordered_map<std::string, aligned_pose> align_global(
    const std::vector<std::string>& camera_names, const std::vector<pair_result>& pair_results) {
  const int N = static_cast<int>(camera_names.size());
  if (N == 0) return {};

  std::unordered_map<std::string, int> name_to_idx;
  for (int i = 0; i < N; ++i) name_to_idx[camera_names[i]] = i;

  // Build lookup for both directed pairs.
  std::map<std::pair<int, int>, const pair_result*> pair_lookup;
  for (const auto& pr : pair_results) {
    pair_lookup[{pr.idx1, pr.idx2}] = &pr;
  }

  struct Edge {
    int u, v;
    float weight;
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
  };
  std::vector<Edge> edges;
  edges.reserve(N * (N - 1) / 2);

  for (int u = 0; u < N; ++u) {
    for (int v = u + 1; v < N; ++v) {
      auto it_uv = pair_lookup.find({u, v});
      auto it_vu = pair_lookup.find({v, u});
      if (it_uv == pair_lookup.end() || it_vu == pair_lookup.end()) continue;

      const pair_result& pr_uv = *it_uv->second;
      const pair_result& pr_vu = *it_vu->second;

      std::vector<Eigen::Vector3f> pts_v_in_v, pts_v_in_u;
      std::vector<float> ws;
      pts_v_in_v.reserve(HW / (SUBSAMPLE * SUBSAMPLE));
      pts_v_in_u.reserve(HW / (SUBSAMPLE * SUBSAMPLE));
      ws.reserve(HW / (SUBSAMPLE * SUBSAMPLE));

      for (int row = 0; row < H; row += SUBSAMPLE) {
        for (int col = 0; col < W; col += SUBSAMPLE) {
          const int px = row * W + col;
          const float cv = pr_vu.view1.conf[px];
          const float cu = pr_uv.view2.conf[px];
          const float w = cv * cu;
          if (w < 1e-6f) continue;

          const float* src = pr_vu.view1.pts3d.data() + px * 3;
          const float* dst = pr_uv.view2.pts3d.data() + px * 3;

          pts_v_in_v.emplace_back(src[0], src[1], src[2]);
          pts_v_in_u.emplace_back(dst[0], dst[1], dst[2]);
          ws.push_back(w);
        }
      }

      if (pts_v_in_v.size() < 6) continue;

      Eigen::Matrix3f R;
      Eigen::Vector3f t;
      if (!procrustes(pts_v_in_v, pts_v_in_u, ws, R, t)) continue;

      float score_uv = 0.0f, score_vu = 0.0f;
      {
        float sum1 = 0, sum2 = 0;
        for (int row = 0; row < H; row += SUBSAMPLE)
          for (int col = 0; col < W; col += SUBSAMPLE) {
            sum1 += pr_uv.view1.conf[row * W + col];
            sum2 += pr_uv.view2.conf[row * W + col];
          }
        const int nsub = (H / SUBSAMPLE) * (W / SUBSAMPLE);
        score_uv = (sum1 / nsub) * (sum2 / nsub);
        sum1 = sum2 = 0;
        for (int row = 0; row < H; row += SUBSAMPLE)
          for (int col = 0; col < W; col += SUBSAMPLE) {
            sum1 += pr_vu.view1.conf[row * W + col];
            sum2 += pr_vu.view2.conf[row * W + col];
          }
        score_vu = (sum1 / nsub) * (sum2 / nsub);
      }
      float weight = std::max(score_uv, score_vu);

      edges.push_back({u, v, weight, R, t});
    }
  }

  if (edges.empty()) {
    throw std::runtime_error("dust3r_alignment: no valid edges found");
  }

  std::sort(edges.begin(), edges.end(),
            [](const Edge& a, const Edge& b) { return a.weight > b.weight; });  // descending

  UnionFind uf(N);
  std::vector<Edge> mst;
  mst.reserve(N - 1);
  for (const auto& e : edges) {
    if (uf.unite(e.u, e.v)) {
      mst.push_back(e);
      if (static_cast<int>(mst.size()) == N - 1) break;
    }
  }

  struct AdjEdge {
    int to;
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
  };
  std::vector<std::vector<AdjEdge>> adj(N);
  for (const auto& e : mst) {
    adj[e.u].push_back({e.v, e.R, e.t});
    adj[e.v].push_back({e.u, e.R.transpose(), -e.R.transpose() * e.t});
  }

  std::vector<Eigen::Matrix3f> R_w(N, Eigen::Matrix3f::Identity());
  std::vector<Eigen::Vector3f> t_w(N, Eigen::Vector3f::Zero());

  std::vector<bool> visited(N, false);
  std::queue<int> bfs;
  bfs.push(0);
  visited[0] = true;

  while (!bfs.empty()) {
    int cur = bfs.front();
    bfs.pop();
    for (const auto& ae : adj[cur]) {
      int nb = ae.to;
      if (visited[nb]) continue;
      visited[nb] = true;

      R_w[nb] = R_w[cur] * ae.R;
      t_w[nb] = R_w[cur] * ae.t + t_w[cur];
      bfs.push(nb);
    }
  }

  for (int i = 0; i < N; ++i) {
    if (!visited[i]) {
      throw std::runtime_error("dust3r_alignment: disconnected MST detected");
    }
  }

  std::unordered_map<std::string, aligned_pose> result;
  for (int i = 0; i < N; ++i) {
    aligned_pose pose;
    const auto& R = R_w[i];
    const auto& t = t_w[i];
    pose.rotation =
        glm::mat3(R(0, 0), R(1, 0), R(2, 0), R(0, 1), R(1, 1), R(2, 1), R(0, 2), R(1, 2), R(2, 2));
    pose.translation = glm::vec3(t(0), t(1), t(2));
    result[camera_names[i]] = pose;
  }
  return result;
}

}  // namespace stargazer::dust3r
