// File: cluster_graph.h
// Performs Louvain/Leiden graph clustering + query functionality

#ifndef LIBPTS_CLUSTER_GRAPH_H_
#define LIBPTS_CLUSTER_GRAPH_H_

#include <igraph/igraph.h>

#include <spdlog/spdlog.h>

#include <cstdio>
#include <memory>
#include <random>
#include <ranges>
#include <vector>

namespace libpts::clustering {

constexpr double DEFAULT_RESOLUTION = 0.1;

namespace detail {
struct IGraphData {
    IGraphData() = delete;
    IGraphData(const IGraphData &) = delete;
    IGraphData(IGraphData &&) = delete;
    auto operator=(const IGraphData &) -> IGraphData & = delete;
    auto operator=(IGraphData &&) -> IGraphData & = delete;
    IGraphData(int num_vertices, const std::vector<int> &edges);
    ~IGraphData();

    void clear_cluster_graphs();

    igraph_t stg;
    igraph_t d_stg;
    std::vector<igraph_t> cluster_graphs;
    std::vector<std::vector<std::size_t>> cluster_memberships;
};
}    // namespace detail

class ClusterGraphs {
public:
    // Performs clustering on construction
    ClusterGraphs(
        int num_vertices,
        const std::vector<int> &edges,
        std::size_t seed,
        double resolution = DEFAULT_RESOLUTION
    );

    /*
     * Sample undirected paths from a louvain hierarchy
     * @param cluster_level the louvain clustering level
     * @parma rng Source of randomness for the sampling
     * @param max_paths Maximum number of paths to sample
     */
    auto sample_undirected_paths(int cluster_level, std::mt19937_64 &rng, int max_paths) const
        -> std::vector<std::vector<int>> {
        return sample_paths(&igraph_data->stg, cluster_level, rng, max_paths);
    }

    /*
     * Sample directed paths from a louvain hierarchy
     * @param cluster_level the louvain clustering level
     * @parma rng Source of randomness for the sampling
     * @param max_paths Maximum number of paths to sample
     */
    [[nodiscard]] auto sample_directed_paths(int cluster_level, std::mt19937_64 &rng, int max_paths) const
        -> std::vector<std::vector<int>> {
        return sample_paths(&igraph_data->d_stg, cluster_level, rng, max_paths);
    }

    /**
     * Get the color (cluster ID/idx) for the given vertex at a given cluster level
     * @param vertex_idx The vertex index
     * @param cluster_level The cluster level
     * @return The cluster id
     */
    [[nodiscard]] constexpr auto get_cluster_id(int vertex_idx, int cluster_level) const -> std::size_t {
        auto c_idx = static_cast<std::size_t>(cluster_level);
        auto v_idx = static_cast<std::size_t>(vertex_idx);
        return igraph_data->cluster_memberships.at(c_idx).at(v_idx);
    }

    // Number of vertices in the base graph
    [[nodiscard]] auto num_vertices() const -> int {
        return static_cast<int>(igraph_vcount(&igraph_data->stg));
    }

    // Number of edges in the base graph
    [[nodiscard]] auto num_edges() const -> int {
        return static_cast<int>(igraph_ecount(&igraph_data->stg));
    }

    // Number of louvain hierarchy levels
    [[nodiscard]] auto hierarchy_size() const -> int {
        return static_cast<int>(igraph_data->cluster_graphs.size());
    }

    /**
     * Get the number of clusters (nodes) at a given cluster level
     * @param cluster_level The cluster level
     * @return Number of clusters
     */
    [[nodiscard]] auto num_clusters(int cluster_level) const -> int {
        if (cluster_level < 0 || cluster_level >= hierarchy_size()) {
            SPDLOG_ERROR("Unknown cluster level {:d} for range {:d}", cluster_level, hierarchy_size() - 1);
            std::exit(1);
        }
        return static_cast<int>(igraph_vcount(&igraph_data->cluster_graphs[static_cast<std::size_t>(cluster_level)]));
    }

    /**
     * Get the number of clusters (nodes) at all cluster levels
     * @return Number of clusters
     */
    [[nodiscard]] auto num_clusters() const -> std::vector<int> {
        std::vector<int> clusters_sizes;
        for (auto i : std::views::iota(0) | std::views::take(hierarchy_size())) {
            clusters_sizes.push_back(
                static_cast<int>(igraph_vcount(&igraph_data->cluster_graphs[static_cast<std::size_t>(i)]))
            );
        }
        return clusters_sizes;
    }

private:
    void louvain(std::size_t seed, double resolution);

    auto sample_paths(const igraph_t *base_graph, int cluster_level, std::mt19937_64 &rng, int max_paths) const
        -> std::vector<std::vector<int>>;

    std::shared_ptr<detail::IGraphData> igraph_data;
};

}    // namespace libpts::clustering

#endif    // LIBPTS_CLUSTER_GRAPH_H_
