// File: cluster_graph.cpp
// Performs Louvain/Leiden graph clustering + query functionality

#include <libpolicyts/cluster_graph.h>

#include <igraph/igraph.h>
#include <libleidenalg/CPMVertexPartition.h>
#include <libleidenalg/GraphHelper.h>
#include <libleidenalg/Optimiser.h>
#include <libleidenalg/RBConfigurationVertexPartition.h>

#include <spdlog/spdlog.h>

#include <cstdio>
#include <memory>
#include <random>
#include <ranges>
#include <vector>

namespace libpts::clustering {

namespace {
auto aggregate_partition(RBConfigurationVertexPartition &partition) -> std::unique_ptr<RBConfigurationVertexPartition> {
    Graph *collapsed_graph = partition.get_graph()->collapse_graph(&partition);
    RBConfigurationVertexPartition *collapsed_partition = partition.create(collapsed_graph);
    collapsed_partition->destructor_delete_graph = true;
    return std::unique_ptr<RBConfigurationVertexPartition>(collapsed_partition);
}

auto cluster_graph_from_partition(RBConfigurationVertexPartition *partition) -> igraph_t {
    const igraph_t *g = partition->get_graph()->get_igraph();
    igraph_t cluster_graph;
    igraph_copy(&cluster_graph, g);
    igraph_simplify(&cluster_graph, true, true, nullptr);
    return cluster_graph;
}

auto get_path(const igraph_t *g, int vid_from, int vid_to) -> std::vector<int> {
    // First check distance if < INF (otherwise igraph will report warning on non-paths)
    igraph_matrix_t m;
    igraph_matrix_init(&m, 0, 0);
    igraph_vs_t v_from, v_to;
    igraph_vs_1(&v_from, vid_from);
    igraph_vs_1(&v_to, vid_to);
    igraph_distances(g, nullptr, &m, v_from, v_to, IGRAPH_OUT);
    auto distance = MATRIX(m, 0, 0);
    igraph_matrix_destroy(&m);
    igraph_vs_destroy(&v_from);
    igraph_vs_destroy(&v_to);
    if (distance == IGRAPH_INFINITY) {
        return {};
    }
    // Finite distance, so find path
    igraph_vector_int_t vertices;
    igraph_vector_int_init(&vertices, 0);
    igraph_get_shortest_path(g, nullptr, &vertices, nullptr, vid_from, vid_to, IGRAPH_OUT);
    std::vector<int> path(static_cast<std::size_t>(igraph_vector_int_size(&vertices)));
    for (auto i : std::views::iota(0) | std::views::take(path.size())) {
        path[static_cast<std::size_t>(i)] = static_cast<int>(VECTOR(vertices)[i]);
    }
    igraph_vector_int_destroy(&vertices);
    return path;
}

}    // namespace

namespace detail {
IGraphData::IGraphData(int num_vertices, const std::vector<int> &edges) {
    std::vector<igraph_int_t> edge_vec = edges
                                         | std::views::transform([](auto v) { return static_cast<igraph_int_t>(v); })
                                         | std::ranges::to<std::vector>();
    // Create empty graphs with vertices but no edges
    if (auto error_code = igraph_empty(&stg, num_vertices, false)) {
        SPDLOG_ERROR("unknown error: {:d}", static_cast<int>(error_code));
        std::exit(1);
    }
    if (auto error_code = igraph_empty(&d_stg, num_vertices, true)) {
        SPDLOG_ERROR("unknown error: {:d}", static_cast<int>(error_code));
        std::exit(1);
    }
    // Allocate space and copy for the edges
    igraph_vector_int_t e;
    if (auto error_code = igraph_vector_int_init_array(&e, edge_vec.data(), static_cast<igraph_int_t>(edge_vec.size())))
    {
        SPDLOG_ERROR("unknown error: {:d}", static_cast<int>(error_code));
        std::exit(1);
    }
    // Add edges to graphs
    if (auto error_code = igraph_add_edges(&stg, &e, nullptr)) {
        SPDLOG_ERROR("unknown error: {:d}", static_cast<int>(error_code));
        std::exit(1);
    }
    if (auto error_code = igraph_add_edges(&d_stg, &e, nullptr)) {
        SPDLOG_ERROR("unknown error: {:d}", static_cast<int>(error_code));
        std::exit(1);
    }
    igraph_vector_int_destroy(&e);
    igraph_simplify(&stg, true, true, nullptr);
}

IGraphData::~IGraphData() {
    clear_cluster_graphs();
    igraph_destroy(&stg);
    igraph_destroy(&d_stg);
}

void IGraphData::clear_cluster_graphs() {
    for (auto &cluster_graph : cluster_graphs) {
        igraph_destroy(&cluster_graph);
    }
    cluster_graphs.clear();
}
}    // namespace detail

ClusterGraphs::ClusterGraphs(int num_vertices, const std::vector<int> &edges, std::size_t seed, double resolution)
    : igraph_data(std::make_unique<detail::IGraphData>(num_vertices, edges)) {
    louvain(seed, resolution);
}

void ClusterGraphs::louvain(std::size_t seed, double resolution) {
    Graph graph(&igraph_data->stg);
    RBConfigurationVertexPartition partition(&graph, resolution);
    RBConfigurationVertexPartition refined_partition(&graph, resolution);
    Optimiser optimiser;
    optimiser.set_rng_seed(seed);

    igraph_data->clear_cluster_graphs();

    std::vector<std::unique_ptr<RBConfigurationVertexPartition>> allocated_partitions;

    allocated_partitions.emplace_back(aggregate_partition(refined_partition));
    RBConfigurationVertexPartition *partition_agg = allocated_partitions.back().get();

    // Move nodes between neighbouring clusters to improve modularity.
    while (optimiser.move_nodes(partition_agg) > 0) {
        // Get individual membership for partition
        partition.from_coarse_partition(partition_agg, refined_partition.membership());

        // Refine partition
        refined_partition.from_coarse_partition(partition_agg);
        optimiser.merge_nodes_constrained(&refined_partition, &partition);

        // Derive new aggregate graph from new cluster memberships.
        allocated_partitions.emplace_back(aggregate_partition(refined_partition));
        partition_agg = allocated_partitions.back().get();

        // But use membership of actual partition
        std::vector<std::size_t> aggregate_membership =
            std::views::iota(static_cast<std::size_t>(0), refined_partition.n_communities())
            | std::ranges::to<std::vector>();

        partition_agg->set_membership(aggregate_membership);

        // Store this in a vec in the struct
        igraph_data->cluster_graphs.push_back(cluster_graph_from_partition(partition_agg));
        igraph_data->cluster_memberships.push_back(partition.membership());
    }
}

auto ClusterGraphs::sample_paths(
    const igraph_t *base_graph,
    int cluster_level,
    std::mt19937_64 &rng,
    int max_paths
) const -> std::vector<std::vector<int>> {
    if (cluster_level < 0 || cluster_level >= hierarchy_size()) {
        SPDLOG_ERROR("Unknown cluster level {:d} for range {:d}", cluster_level, hierarchy_size() - 1);
        std::exit(1);
    }
    // Get cluster graph
    auto c_level = static_cast<std::size_t>(cluster_level);
    igraph_t g = igraph_data->cluster_graphs[c_level];
    std::unordered_map<std::size_t, std::vector<std::size_t>> vertices_by_cluster;
    for (const auto &[vid, cid] : std::views::enumerate(igraph_data->cluster_memberships.at(c_level))) {
        vertices_by_cluster[cid].push_back(static_cast<std::size_t>(vid));
    }

    auto find_path = [&](const std::vector<std::size_t> &vertices_c1,
                         const std::vector<std::size_t> &vertices_c2) -> std::vector<int> {
        for (const auto &[v1, v2] : std::views::zip(vertices_c1, vertices_c2)) {
            auto path_12 = get_path(base_graph, static_cast<int>(v1), static_cast<int>(v2));
            if (!path_12.empty()) {
                return path_12;
            }
            auto path_21 = get_path(base_graph, static_cast<int>(v1), static_cast<int>(v2));
            if (!path_21.empty()) {
                return path_12;
            }
        }
        return {};
        for (auto v1 : vertices_c1) {
            for (auto v2 : vertices_c2) {
                auto path_12 = get_path(base_graph, static_cast<int>(v1), static_cast<int>(v2));
                if (!path_12.empty()) {
                    return path_12;
                }
                auto path_21 = get_path(base_graph, static_cast<int>(v1), static_cast<int>(v2));
                if (!path_21.empty()) {
                    return path_12;
                }
            }
        }
        return {};
    };
    std::vector<std::vector<int>> paths;

    std::vector<igraph_int_t> edges =
        std::views::iota(static_cast<igraph_int_t>(0), igraph_ecount(&g)) | std::ranges::to<std::vector>();
    std::ranges::shuffle(edges, rng);
    for (auto eid : edges) {
        auto cid_from = static_cast<std::size_t>(IGRAPH_FROM(&g, eid));
        auto cid_to = static_cast<std::size_t>(IGRAPH_TO(&g, eid));
        // Get underlying vertex indices which belong to each cluster
        auto &vertices_c1 = vertices_by_cluster.at(cid_from);
        auto &vertices_c2 = vertices_by_cluster.at(cid_to);
        std::ranges::shuffle(vertices_c1, rng);
        std::ranges::shuffle(vertices_c2, rng);
        // Try to find path between
        auto path = find_path(vertices_c1, vertices_c2);
        if (!path.empty()) {
            paths.push_back(path);
        }
        if (paths.size() >= static_cast<std::size_t>(max_paths)) {
            break;
        }
    }
    return paths;
}

}    // namespace libpts::clustering
