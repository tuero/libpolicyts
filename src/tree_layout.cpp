#include "tree_layout.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <ranges>
#include <unordered_map>
#include <vector>

namespace libpts::treeviz::detail {

constexpr uint64_t MAGIC_HASH = 0x9e3779b97f4a7c15;
constexpr int SHIFT_LEFT = 6;
constexpr int SHIFT_RIGHT = 2;
void hash_combine(uint64_t &seed, uint64_t value)
{
    seed ^= value + MAGIC_HASH + (seed << SHIFT_LEFT) + (seed >> SHIFT_RIGHT);
}

auto compute_layout_fingerprint(std::span<const LayoutInputNode> nodes) -> uint64_t
{
    uint64_t fp = 0xcbf29ce484222325ull;    // NOLINT(*-magic-numbers)

    hash_combine(fp, static_cast<std::uint64_t>(nodes.size()));

    for (const auto &node : nodes) {
        hash_combine(fp, static_cast<uint64_t>(node.id));
        hash_combine(fp, node.parent_id ? static_cast<uint64_t>(*node.parent_id) : 0);
    }

    return fp;
}

void compute_tree_layout(std::span<const LayoutInputNode> nodes, TreeLayoutCache &cache, const TreeLayoutConfig &config)
{
    const uint64_t fp = compute_layout_fingerprint(nodes);

    // Fast path: structure unchanged, keep cached positions
    if (cache.valid && cache.structure_fingerprint == fp) {
        return;
    }

    cache.positions_by_id.clear();
    cache.positions_by_id.reserve(nodes.size());

    if (nodes.empty()) {
        cache.structure_fingerprint = fp;
        cache.valid = true;
        return;
    }

    // Map node id -> input index
    std::unordered_map<int, std::size_t> index_by_id;
    index_by_id.reserve(nodes.size());
    // for (auto &&[i, node] : std::views::enumerate(nodes)) {
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto &node = nodes[i];
        index_by_id.emplace(node.id, i);
    }

    // Build child lists and determine root
    std::optional<std::size_t> root_index;
    std::vector<std::vector<std::size_t>> children_by_index(nodes.size());
    // for (auto &&[i, node] : std::views::enumerate(nodes)) {
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto &node = nodes[i];
        // Find root and ensure only one root is ever found
        if (!node.parent_id) {
            if (root_index) {
                const auto error_msg = "Tree layout requires exactly one root.";
                spdlog::error(error_msg);
                throw std::runtime_error(error_msg);
            }
            root_index = i;
            continue;
        }

        const auto parent_id = *node.parent_id;
        if (index_by_id.contains(parent_id)) {
            children_by_index[index_by_id.at(parent_id)].push_back(static_cast<std::size_t>(i));
        } else {
            const auto error_msg = "Node has missing parent.";
            spdlog::error(error_msg);
            throw std::runtime_error(error_msg);
        }
    }

    // Ensure a root was found
    if (!root_index) {
        const auto error_msg = "Tree layout requires a root.";
        spdlog::error(error_msg);
        throw std::runtime_error(error_msg);
    }

    // Leaf-slot based layout:
    // - leaves are placed left-to-right
    // - internal nodes are centered above their children
    float next_leaf_slot = 0.0f;

    // NOLINTNEXTLINE(*-recursion)
    auto layout_subtree = [&](const auto &self, std::size_t node_index, int depth) -> float {
        const auto &node = nodes[node_index];
        auto &pos = cache.positions_by_id[node.id];

        // y position is its depth level with spacing between
        pos.y = config.margin_y + static_cast<float>(depth) * config.level_gap;

        const auto &children = children_by_index[node_index];

        // Leaf node
        if (children.empty()) {
            pos.x = config.margin_x + next_leaf_slot * config.sibling_gap;
            next_leaf_slot += 1.0f;
            return pos.x;
        }

        // Internal node: place children first, then center over them.
        float min_x = 0.0f;
        float max_x = 0.0f;
        bool first = true;

        for (const auto &child_index : children) {
            const float child_x = self(self, child_index, depth + 1);

            if (first) {
                min_x = max_x = child_x;
                first = false;
            } else {
                min_x = std::min(min_x, child_x);
                max_x = std::max(max_x, child_x);
            }
        }

        pos.x = 0.5f * (min_x + max_x);    // NOLINT(*-magic-numbers)
        return pos.x;
    };

    // root index placement
    layout_subtree(layout_subtree, *root_index, 0);

    cache.structure_fingerprint = fp;
    cache.valid = true;
}

}    // namespace libpts::treeviz::detail
