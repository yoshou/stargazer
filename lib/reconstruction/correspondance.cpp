#include <tuple>
#include <map>
#include <vector>
#include <unordered_set>
#include <opencv2/opencv.hpp>
#include <nanoflann.hpp>

#include "correspondance.hpp"
#include "camera_info.hpp"
#include "triangulation.hpp"
#include "utils.hpp"
#include "tuple_hash.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/graph/graphviz.hpp>

#include <chrono>

namespace stargazer::reconstruction
{
    static glm::mat3 calculate_fundametal_matrix(const glm::mat3 &camera_mat1, const glm::mat3 &camera_mat2,
                                          const glm::mat4 &camera_pose1, const glm::mat4 &camera_pose2)
    {
        const auto pose = camera_pose2 * glm::inverse(camera_pose1);

        const auto R = glm::mat3(pose);
        const auto t = glm::vec3(pose[3]);

        const auto Tx = glm::mat3(
            0, -t[2], t[1],
            t[2], 0, -t[0],
            -t[1], t[0], 0);

        const auto E = Tx * R;

        const auto F = glm::inverse(glm::transpose(camera_mat2)) * E * glm::inverse(camera_mat1);

        return F;
    }

    static glm::mat3 calculate_fundametal_matrix(const camera_t &camera1, const camera_t &camera2)
    {
        const auto camera_mat1 = camera1.intrin.get_matrix();
        const auto camera_mat2 = camera2.intrin.get_matrix();

        return calculate_fundametal_matrix(camera_mat1, camera_mat2, camera1.extrin.rotation, camera2.extrin.rotation);
    }

    static glm::vec3 normalize_line(const glm::vec3 &v)
    {
        const auto c = std::sqrt(v.x * v.x + v.y * v.y);
        return v / c;
    }

    static glm::vec3 compute_correspond_epiline(const glm::mat3 &F, const glm::vec2 &p)
    {
        const auto l = F * glm::vec3(p, 1.f);
        // return normalize_line(l);
        return l;
    }

    static glm::vec3 compute_correspond_epiline(const camera_t &camera1, const camera_t &camera2, const glm::vec2 &p)
    {
        const auto F = calculate_fundametal_matrix(camera1, camera2);
        return compute_correspond_epiline(F, p);
    }

    static size_t find_nearest_point(const glm::vec2 &pt, const std::vector<glm::vec2> &pts,
                                    double thresh, float &dist)
    {
        size_t idx = pts.size();
        auto min_dist = std::numeric_limits<float>::max();
        for (size_t i = 0; i < pts.size(); i++)
        {
            auto pt2 = pts[i];

            if (pt2.x < 0 || pt2.y < 0)
            {
                continue;
            }

            const auto dist = glm::dot(pt - pt2, pt - pt2);

            if (dist < min_dist && dist < thresh * thresh)
            {
                min_dist = dist;
                idx = i;
            }
        }

        dist = min_dist;

        return idx;
    }

    struct edge_t
    {
        std::size_t u, v;
        float w;
    };

    template <typename F>
    static void dfs(const adj_list_t &adj, std::size_t v, std::vector<std::uint32_t> &visited, F pred)
    {
        visited[v] = 1;
        pred(v);

        for (auto u : adj[v])
        {
            if (!visited[u])
            {
                dfs(adj, u, visited, pred);
            }
        }
    }

    template <typename V, typename F>
    static void dfs(const adj_list_t &adj, std::size_t v, std::vector<std::uint32_t> &visited, V visit, F pred)
    {
        if (!visit(v))
        {
            return;
        }
        if (visited[v])
        {
            return;
        }

        visited[v] = 1;
        pred(v);

        for (auto u : adj[v])
        {
            dfs(adj, u, visited, visit, pred);
        }
    }

    void compute_observations(const adj_list_t &adj, std::vector<std::vector<std::size_t>> &connected_graphs)
    {
        const auto n = adj.size();
        std::vector<std::uint32_t> visited(n, 0);

        connected_graphs.clear();
        for (std::size_t v = 0; v < n; v++)
        {
            if (!visited[v])
            {
                std::vector<std::size_t> connected_nodes;

                dfs(adj, v, visited, [&](std::size_t u)
                    { connected_nodes.push_back(u); });

                connected_graphs.push_back(connected_nodes);
            }
        }
    }

    bool has_soft_correspondance(const std::vector<node_t> &nodes, const node_index_list_t &graph)
    {
        std::unordered_set<size_t> groups;

        for (const auto v : graph)
        {
            const auto group = nodes[v].camera_idx;
            const auto [it, inserted] = groups.insert(group);
            if (!inserted)
            {
                return true;
            }
        }

        return false;
    }

    static float compute_diff_camera_angle(const camera_t &camera1, const camera_t &camera2)
    {
        const auto r1 = glm::mat3(camera1.extrin.rotation);
        const auto r2 = glm::mat3(camera2.extrin.rotation);

        const auto r = glm::transpose(r1) * r2;
        const auto r_quat = glm::quat_cast(r);
        return glm::angle(r_quat);
    }

    void remove_ambiguous_observations(const std::vector<node_t> &nodes, adj_list_t &adj, const std::vector<camera_t> &cameras, double world_thresh)
    {
        std::vector<std::vector<std::size_t>> connected_graphs;
        compute_observations(adj, connected_graphs);

        // Remove ambiguous edges
        // If there is a path between the points of inconsitency, there is a combination of edges
        // with a large 3D point distance from the 2D point pair in the edge connecting to a node.
        for (const auto &g : connected_graphs)
        {
            if (!has_soft_correspondance(nodes, g))
            {
                continue;
            }

            std::vector<std::pair<std::size_t, std::size_t>> remove_edges;
            for (const auto v : g)
            {
                std::vector<std::pair<std::size_t, std::size_t>> edges;
                for (const auto u : adj[v])
                {
                    edges.push_back(std::make_pair(u, v));
                }

                std::vector<glm::vec3> markers;
                for (std::size_t i = 0; i < edges.size(); i++)
                {
                    const auto get_camera_point = [&nodes, &cameras](const std::size_t v)
                    {
                        return std::make_pair(nodes[v].pt, cameras[nodes[v].camera_idx]);
                    };

                    {
                        const auto [pt1, camera1] = get_camera_point(edges[i].first);
                        const auto [pt2, camera2] = get_camera_point(edges[i].second);
                        const auto marker = triangulate(pt1, pt2, camera1, camera2);
                        markers.push_back(marker);
                    }
                }

                for (std::size_t i = 0; i < edges.size(); i++)
                {
                    for (std::size_t j = i + 1; j < edges.size(); j++)
                    {
                        const auto e1 = edges[i];
                        const auto e2 = edges[j];

                        const auto marker1 = markers[i];
                        const auto marker2 = markers[j];

                        if (glm::distance(marker1, marker2) > world_thresh)
                        {
                            remove_edges.push_back(e1);
                            remove_edges.push_back(e2);
                        }
                    }
                }
            }

            for (const auto &e : remove_edges)
            {
                const auto u = e.first;
                const auto v = e.second;
                {
                    auto iter = std::remove_if(adj[v].begin(), adj[v].end(), [u](const size_t u_)
                                               { return u == u_; });
                    adj[v].erase(iter, adj[v].end());
                }
                {
                    auto iter = std::remove_if(adj[u].begin(), adj[u].end(), [v](const size_t v_)
                                               { return v == v_; });
                    adj[u].erase(iter, adj[u].end());
                }
            }
        }
    }

    void compute_hard_correspondance(const std::vector<node_t> &nodes, adj_list_t &adj, const std::vector<camera_t> &cameras)
    {
        std::vector<size_t> id(nodes.size(), -1);
        std::vector<std::unordered_set<size_t>> camera_list;

        const auto n = adj.size();
        std::vector<std::uint32_t> visited(n, 0);

        for (std::size_t v = 0; v < n; v++)
        {
            if (!visited[v])
            {
                const auto i = camera_list.size();
                camera_list.push_back(std::unordered_set<size_t>());

                auto visit_func = [&, i](std::size_t u)
                {
                    return camera_list[i].find(nodes[u].camera_idx) == camera_list[i].end();
                };

                auto visited_func = [&, i](std::size_t u)
                {
                    id[u] = i;
                    camera_list[i].insert(nodes[u].camera_idx);
                };

                dfs(adj, v, visited, visit_func, visited_func);
            }
        }

        adj_list_t new_adj(adj.size());
        for (std::size_t v = 0; v < adj.size(); v++)
        {
            for (std::size_t u : adj[v])
            {
                assert(id[u] != -1);
                assert(id[v] != -1);
                if (id[u] == id[v])
                {
                    new_adj[v].push_back(u);
                    new_adj[u].push_back(v);
                }
            }
        }

        adj = new_adj;
    }

    static cv::Mat glm2cv_mat3(const glm::mat4 &m)
    {
        cv::Mat ret(3, 3, CV_32F);
        for (std::size_t i = 0; i < 3; i++)
        {
            for (std::size_t j = 0; j < 3; j++)
            {
                ret.at<float>(i, j) = m[j][i];
            }
        }
        return ret;
    }

    static cv::Mat glm2cv_mat4(const glm::mat4 &m)
    {
        cv::Mat ret(4, 4, CV_32F);
        memcpy(ret.data, glm::value_ptr(m), 16 * sizeof(float));
        return ret;
    }

    static glm::vec2 undistort(const glm::vec2 &pt, const camera_t &camera)
    {
        auto pts = std::vector<cv::Point2f>{cv::Point2f(pt.x, pt.y)};
        cv::Mat m = glm2cv_mat3(camera.intrin.get_matrix());
        cv::Mat coeffs(5, 1, CV_32F);
        for (int i = 0; i < 5; i++)
        {
            coeffs.at<float>(i) = camera.intrin.coeffs[i];
        }

        std::vector<cv::Point2f> norm_pts;
        cv::undistortPoints(pts, norm_pts, m, coeffs);

        return glm::vec2(norm_pts[0].x, norm_pts[0].y);
    }

    static glm::vec2 project_undistorted(const glm::vec2 &pt, const camera_t &camera)
    {
        const auto p = camera.intrin.get_matrix() * glm::vec3(pt.x, pt.y, 1.0f);
        return glm::vec2(p.x / p.z, p.y / p.z);
    }

    struct point_cloud_2d
    {
        using point_type = glm::vec2;
        using index_type = std::size_t;
        using distance_type = float;

    private:
        typedef nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<float, point_cloud_2d>,
            point_cloud_2d,
            2, /* dim */
            index_type
            >
            kd_tree_t;

        std::unique_ptr<kd_tree_t> index;

    public:
        std::vector<point_type> points;

        inline std::size_t kdtree_get_point_count() const { return points.size(); }

        inline float kdtree_get_pt(const std::size_t idx, const std::size_t dim) const
        {
            return points[idx][dim];
        }

        template <class BoundingBox>
        inline bool kdtree_get_bbox(BoundingBox & /* bb */) const { return false; }

        point_cloud_2d() {}
        explicit point_cloud_2d(const std::vector<point_type> &points)
            : points(points)
        {
            index = std::make_unique<kd_tree_t>(2 /*dim*/, *this, nanoflann::KDTreeSingleIndexAdaptorParams(1024 /* max leaf */));
        }

        void build_index()
        {
            index->buildIndex();
        }

        inline point_type operator[](std::size_t index) const
        {
            return points[index];
        }

        inline std::size_t size() const
        {
            return points.size();
        }

        std::size_t knn_search(const point_type &query_pt, std::size_t num_points,
                               index_type *result_index, float *result_distsq) const
        {
            return index->knnSearch(&query_pt[0], num_points, &result_index[0], &result_distsq[0]);
        }

        std::size_t radius_search(const point_type &query_pt, float radius,
                                  std::vector<std::pair<index_type, float>> &result) const
        {
#if NANOFLANN_VERSION >= 0x150
        nanoflann::SearchParameters params;
        params.sorted = true;

        std::vector<nanoflann::ResultItem<index_type, distance_type>> founds;
        const auto found_size = index->radiusSearch(&query_pt[0], radius, founds, params);

        for (const auto& found : founds)
        {
            result.push_back(std::make_pair(found.first, found.second));
        }

        return found_size;
#else
        nanoflann::SearchParams params;
        params.sorted = true;
        return index->radiusSearch(&query_pt[0], radius, result, params);
#endif
        }
    };

    static glm::vec2 epipoloar_transfer(const camera_t &c3, const glm::mat3 &f1, const glm::mat3 &f2, const glm::vec2 &pt1, const glm::vec2 &pt2)
    {
        const auto line1 = compute_correspond_epiline(f1, pt1);
        const auto line2 = compute_correspond_epiline(f2, pt2);

        const auto num = line1.x * line2.y - line2.x * line1.y;

        const glm::vec2 pt3(
            (line1.y * line2.z - line2.y * line1.z) / num,
            (line2.x * line1.z - line1.x * line2.z) / num);

        if (pt3.x < 0 || pt3.x >= c3.width || pt3.y < 0 || pt3.y >= c3.height)
        {
            return glm::vec2(-1);
        }

        return pt3;
    }

    void find_correspondance(const std::vector<std::vector<glm::vec2>> &pts,
                             const std::vector<camera_t> &cameras, std::vector<node_t> &nodes, adj_list_t &adj, double screen_thresh)
    {
        const auto start = std::chrono::system_clock::now();
        std::vector<std::pair<std::size_t, std::size_t>> camera_pairs;

        std::vector<std::vector<glm::vec2>> undist_pts(pts.size());
        std::vector<std::vector<glm::vec2>> undist_norm_pts(pts.size());
        for (std::size_t i = 0; i < pts.size(); i++)
        {
            for (std::size_t j = 0; j < pts[i].size(); j++)
            {
                if (pts[i][j].x < 0 || pts[i][j].y < 0)
                {
                    undist_pts[i].push_back(glm::vec2(-1, -1));
                    undist_norm_pts[i].push_back(glm::vec2(-1, -1));
                    continue;
                }
                const auto pt = undistort(pts[i][j], cameras[i]);
                undist_pts[i].push_back(project_undistorted(pt, cameras[i]));
                undist_norm_pts[i].push_back(pt);
            }
        }

#if 0
        std::vector<std::shared_ptr<point_cloud_2d>> clouds;
        for (std::size_t i = 0; i < pts.size(); i++)
        {
            auto cloud = std::make_shared<point_cloud_2d>(undist_pts[i]);
            cloud->build_index();
            clouds.push_back(cloud);
        }
#endif

        const auto num_cameras = cameras.size();

        for (std::size_t c1 = 0; c1 < cameras.size(); c1++)
        {
            for (std::size_t c2 = c1 + 1; c2 < cameras.size(); c2++)
            {
                camera_pairs.push_back(std::make_pair(c1, c2));
            }
        }

        std::vector<std::vector<float>> camera_diff_angles(cameras.size());
        std::vector<std::vector<glm::mat3>> f_mats(cameras.size());

        for (std::size_t c1 = 0; c1 < cameras.size(); c1++)
        {
            camera_diff_angles[c1].resize(cameras.size());
            f_mats[c1].resize(cameras.size());
            for (std::size_t c2 = 0; c2 < cameras.size(); c2++)
            {
                camera_diff_angles[c1][c2] = compute_diff_camera_angle(cameras[c1], cameras[c2]);
                f_mats[c1][c2] = calculate_fundametal_matrix(cameras[c1], cameras[c2]);
            }
        }

        std::vector<edge_t> edges;
        std::vector<std::size_t> node_offsets;
        for (std::size_t i = 0; i < pts.size(); i++)
        {
            node_offsets.push_back(nodes.size());

            for (std::size_t j = 0; j < pts[i].size(); j++)
            {
                nodes.push_back(node_t{
                    pts[i][j], i, j});
            }
        }

        const auto min_angle = 5;

        for (auto [c1, c2] : camera_pairs)
        {
            if (camera_diff_angles[c1][c2] < glm::pi<double>() / 180.0 * min_angle)
            {
                continue;
            }
            const auto F = calculate_fundametal_matrix(cameras[c1], cameras[c2]);

            for (std::size_t i = 0; i < pts[c1].size(); i++)
            {
                const auto pt1 = undist_pts[c1][i];

                if (pt1.x < 0 || pt1.y < 0)
                {
                    continue;
                }

                const auto line = compute_correspond_epiline(F, pt1);
                // const auto line = compute_correspond_epiline(cameras[c1], cameras[c2], pt1);

                for (std::size_t j = 0; j < pts[c2].size(); j++)
                {
                    const auto pt2 = undist_pts[c2][j];

                    if (pt2.x < 0 || pt2.y < 0)
                    {
                        continue;
                    }

                    const auto dist = distance_sq_line_point(line, pt2);

                    if (dist > screen_thresh * screen_thresh)
                    {
                        continue;
                    }

                    // const auto marker = triangulate(pts[c1][i], pts[c2][j], cameras[c1], cameras[c2]);
                    // const auto marker = triangulate_undistorted(undist_norm_pts[c1][i], undist_norm_pts[c2][j], cameras[c1], cameras[c2]);

                    float nearest_pt_dist_acc = dist;
                    size_t nearest_pt_dist_count = 1;

                    std::vector<std::size_t> observed_cameras = {c1, c2};

                    for (size_t c3 = 0; c3 < cameras.size(); c3++)
                    {
                        bool is_dup_camera = false;
                        for (const auto &obs_camera : observed_cameras)
                        {
                            if (obs_camera == c3)
                            {
                                is_dup_camera = true;
                                break;
                            }
                            if (camera_diff_angles[obs_camera][c3] < glm::pi<double>() / 180.0 * min_angle)
                            {
                                is_dup_camera = true;
                                break;
                            }
                        }

                        if (is_dup_camera)
                        {
                            continue;
                        }

                        // const auto pt = project_undist(cameras[c3], marker);
                        const auto pt = epipoloar_transfer(cameras[c3], f_mats[c1][c3], f_mats[c2][c3], undist_pts[c1][i], undist_pts[c2][j]);

                        if (pt.x < 0 || pt.y < 0)
                        {
                            continue;
                        }

                        if (pts[c3].size() == 0)
                        {
                            continue;
                        }

#if 0
                        float nearest_pt_dist_sq = std::numeric_limits<float>::max();
                        const auto nearest_pt_idx = pts[c3].size();
                        clouds[c3]->knn_search(pt, 1, &nearest_pt_idx, &nearest_pt_dist_sq);
#else
                        float nearest_pt_dist_sq = std::numeric_limits<float>::max();
                        const auto nearest_pt_idx = find_nearest_point(pt, undist_pts[c3], screen_thresh, nearest_pt_dist_sq);
#endif

                        if (nearest_pt_dist_sq >= screen_thresh * screen_thresh)
                        {
                            continue;
                        }

                        if (nearest_pt_idx >= pts[c3].size())
                        {
                            continue;
                        }

                        const auto nearest_pt_dist = std::sqrt(nearest_pt_dist_sq);

                        nearest_pt_dist_acc += nearest_pt_dist;
                        nearest_pt_dist_count++;

                        observed_cameras.push_back(c3);
                    }

                    float nearest_pt_dist = nearest_pt_dist_acc / nearest_pt_dist_count;
                    if (nearest_pt_dist_count <= 1)
                    {
                        nearest_pt_dist = std::numeric_limits<float>::max();
                    }

                    if (nearest_pt_dist > screen_thresh)
                    {
                        continue;
                    }

                    const auto u = node_offsets[c1] + i;
                    const auto v = node_offsets[c2] + j;
                    edges.push_back(edge_t{u, v, static_cast<float>(nearest_pt_dist_count)});
                }
            }
        }

        adj.clear();
        adj.resize(nodes.size());
        for (const auto &edge : edges)
        {
            adj[edge.u].push_back(edge.v);
            adj[edge.v].push_back(edge.u);
        }
    }

    template <class Name>
    class label_writer
    {
    public:
        label_writer(Name _name) : name(_name) {}
        template <class VertexOrEdge>
        void operator()(std::ostream &out, const VertexOrEdge &v) const
        {
            out << "[label=\"" << name.at(v) << "\"]";
        }

    private:
        Name name;
    };

    void save_graphs(const std::vector<node_t> &nodes, const adj_list_t &adj, const std::string &prefix)
    {
        std::vector<node_index_list_t> connected_graphs;
        compute_observations(adj, connected_graphs);

        std::size_t count = 0;
        for (std::size_t i = 0; i < connected_graphs.size(); i++)
        {
            const auto &connected_graph = connected_graphs[i];
            if (connected_graph.size() < 2)
            {
                continue;
            }

            using graph_t = boost::adjacency_list<boost::setS, boost::vecS, boost::undirectedS>;
            graph_t g;

            std::map<graph_t::vertex_descriptor, std::string> labels;

            std::map<std::size_t, graph_t::vertex_descriptor> g_nodes;
            for (const auto v : connected_graph)
            {
                const auto node = boost::add_vertex(g);
                g_nodes.insert(std::make_pair(v, node));
                const auto c = nodes[v].camera_idx;
                const auto i = nodes[v].point_idx;
                labels[node] = std::to_string(c) + "," + std::to_string(i);
            }

            for (const auto v : connected_graph)
            {
                for (const auto u : adj[v])
                {
                    boost::add_edge(g_nodes[u], g_nodes[v], g);
                }
            }

            std::ofstream file(prefix + std::to_string(count++) + ".dot");
            boost::write_graphviz(file, g, label_writer<decltype(labels)>(labels));
        }
    }
}
