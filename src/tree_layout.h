// File: tree_layout.h
// Description: Tree layout for drawing nice trees

#ifndef LIBPTS_TREE_LAYOUT_H_
#define LIBPTS_TREE_LAYOUT_H_

#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>

namespace libpts::treeviz::detail {

// Minimal input needed by the layout algorithm.
struct LayoutInputNode {
    int id{};
    std::optional<int> parent_id;
};

// Output position in logical tree space.
// The viewer still applies pan/zoom afterward.
struct LayoutPoint {
    float x = 0.0f;
    float y = 0.0f;
};

constexpr float DEFAULT_LEVEL_GAP = 120.0f;
constexpr float DEFAULT_SIBLING_GAP = 120.0f;
constexpr float DEFAULT_MARGIN_X = 100.0f;
constexpr float DEFAULT_MARGIN_Y = 60.0f;

struct TreeLayoutConfig {
    // Vertical distance between levels in the tree
    float level_gap = DEFAULT_LEVEL_GAP;

    // Horizontal distance between adjacent leaf "slots".
    float sibling_gap = DEFAULT_SIBLING_GAP;

    // Margins for the overall tree drawing.
    float margin_x = DEFAULT_MARGIN_X;
    float margin_y = DEFAULT_MARGIN_Y;

    // Extra leaf slots inserted between disconnected roots in a forest.
    float forest_gap_leaf_slots = 1.0f;
};

// Cache keyed by tree structure
// If the fingerprint matches, the layout can be reused
struct TreeLayoutCache {
    uint64_t structure_fingerprint = 0;
    bool valid = false;

    // Cached positions keyed by node id
    std::unordered_map<int, LayoutPoint> positions_by_id;

    void clear()
    {
        structure_fingerprint = 0;
        valid = false;
        positions_by_id.clear();
    }
};

// Similar to boost hash combine
void hash_combine(uint64_t &seed, uint64_t value);

// Hash only the structure that affects layout
// Since nodes are processed in order, sibling order also affects the hash
auto compute_layout_fingerprint(std::span<const LayoutInputNode> nodes) -> uint64_t;

// Computes and caches positions in cache.positions_by_id
// If the structure fingerprint matches, this is effectively a no-op
void compute_tree_layout(
    std::span<const LayoutInputNode> nodes,
    TreeLayoutCache &cache,
    const TreeLayoutConfig &config
);

}    // namespace libpts::treeviz::detail

#endif    // LIBPTS_TREE_LAYOUT_H_
