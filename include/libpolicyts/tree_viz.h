// File: tree_viz.h
// Description: Tree visualizer

#ifndef LIBPTS_TREE_VIZ_H_
#define LIBPTS_TREE_VIZ_H_

#include <cassert>
#include <concepts>
#include <format>
#include <functional>
#include <memory>
#include <optional>
#include <ranges>
#include <string>
#include <vector>

namespace libpts::treeviz {

// Default viewer config
constexpr int DEFAULT_WIDTH = 1600;
constexpr int DEFAULT_HEIGHT = 900;
constexpr float DEFAULT_WINDOW_SPLIT = 0.3f;    // Percentage of window to right side panel
struct ViewerConfig {
    int width = DEFAULT_WIDTH;
    int height = DEFAULT_HEIGHT;
    bool dark_mode = true;
    float window_split_percentage = DEFAULT_WINDOW_SPLIT;
    std::string title = "Tree Viewer";
};

// Default layout config
constexpr float DEFAULT_LEVEL_GAP = 120.0f;
constexpr float DEFAULT_SIBLING_GAP = 120.0f;
constexpr float DEFAULT_MARGIN_X = 100.0f;
constexpr float DEFAULT_MARGIN_Y = 60.0f;
struct TreeLayoutConfig {
    // Vertical distance between levels in the tree
    float level_gap = DEFAULT_LEVEL_GAP;

    // Horizontal distance between adjacent leaf "slots"
    float sibling_gap = DEFAULT_SIBLING_GAP;

    // Margins for the overall tree drawing.
    float margin_x = DEFAULT_MARGIN_X;
    float margin_y = DEFAULT_MARGIN_Y;

    // Extra leaf slots inserted between disconnected roots in a forest
    float forest_gap_leaf_slots = 1.0f;
};

// User code add fields to the detail UI sidebar for any arbitrary type convertable to str
class DetailUI {
public:
    /**
     * Add text to the node detail UI
     * @param s The string text
     */
    void text(const std::string &s);

    /**
     * Add a separator to the node detail UI
     */
    void separator();

    /**
     * Add a field to the node detail UI
     * @param name The name of the field
     * @param value The value of the field, which must be convertable to string
     */
    template <class T>
    void field(const std::string &name, const T &value)
    {
        text(std::format("{:s}: {:s}", name, to_string_any(value)));
    }

private:
    static auto to_string_any(const std::string &s) -> const std::string &
    {
        return s;
    }
    static auto to_string_any(std::string_view s) -> std::string
    {
        return std::string(s);
    }
    static auto to_string_any(const char *s) -> std::string
    {
        return {s};
    }

    template <class T>
    static auto to_string_any(const T &v) -> std::string
    {
        return std::to_string(v);
    }
};

// Pulls out relevant information from abitrary node types
template <typename Adapter, typename Node>
concept TreeNodeAdapter = requires(const Adapter &a, const Node &n) {
    { a.id(n) } -> std::convertible_to<int>;
    { a.parent_id(n) } -> std::same_as<std::optional<int>>;
    { a.action_taken(n) } -> std::same_as<int>;
    { a.label(n) } -> std::convertible_to<std::string>;
};

struct PreparedNode {
    int id;
    std::optional<int> parent_id;
    int action_taken;
    std::string label;
    std::size_t source_index;
    std::optional<std::size_t> parent_index;
};

struct PreparedTree {
    std::vector<PreparedNode> nodes;
};

// A tree viewer GUI
class TreeViewer {
public:
    explicit TreeViewer(const ViewerConfig &viewer_config = {}, const TreeLayoutConfig &tree_config = {});
    ~TreeViewer();

    // Remove copy/move
    TreeViewer(const TreeViewer &) = delete;
    TreeViewer(TreeViewer &&) = delete;
    auto operator=(const TreeViewer &) -> TreeViewer & = delete;
    auto operator=(TreeViewer &&) -> TreeViewer & = delete;

    /**
     * Check if the GUI window is still open
     */
    [[nodiscard]] auto is_open() const -> bool;

    /**
     * Get the amount of steps the user has asked for to increment the search tree
     */
    [[nodiscard]] auto step_amount() const -> int;

    /**
     * Check if the user requested to reset the search
     */
    [[nodiscard]] auto reset_clicked() const -> bool;

    /**
     * Render loop, called in an external loop
     * We internally cache the nodes and prevent recreating the tree when possible, so this is cheap on most frames
     * @param nodes The search nodes
     * @param adapter The adapter to query info from search nodes
     * @param detail_fn Functor which interacts with the draw call back for displaying node info
     */
    template <typename Node, typename Adapter, typename DetailFn>
        requires TreeNodeAdapter<Adapter, Node> && std::invocable<DetailFn, const Node &, DetailUI &>
    void render(const std::vector<Node> &nodes, const Adapter &adapter, DetailFn detail_fn)
    {
        PreparedTree prepared;
        prepared.nodes.reserve(nodes.size());

        // ID to source index mapping
        std::unordered_map<int, std::size_t> id_to_index_map;
        id_to_index_map.reserve(nodes.size());
#ifdef __GLIBCXX__
        for (auto &&[i, n] : std::views::enumerate(nodes)) {
#else
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            auto &n = nodes[i];
#endif
            id_to_index_map.emplace(adapter.id(n), i);
        }

#ifdef __GLIBCXX__
        for (auto &&[i, n] : std::views::enumerate(nodes)) {
#else
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            auto &n = nodes[i];
#endif
            const auto id = adapter.id(n);
            const auto parent_id = adapter.parent_id(n);
            assert(id_to_index_map.contains(id));
            prepared.nodes.push_back(
                PreparedNode{
                .id = id,
                .parent_id = adapter.parent_id(n),
                .action_taken = adapter.action_taken(n),
                .label = adapter.label(n),
                .source_index = static_cast<std::size_t>(i),
                .parent_index = parent_id.and_then([&](int par_id) -> std::optional<std::size_t> {
                    assert(id_to_index_map.contains(par_id));
                    return id_to_index_map[par_id];
                })
                }
            );
        }

        // Thin templated bridge:
        // convert arbitrary nodes -> PreparedTree,
        // then call the non-templated implementation
        DetailCallback cb = [&, f = std::move(detail_fn)](std::size_t source_index, DetailUI &ui) {
            f(nodes.at(source_index), ui);
        };

        render_prepared(prepared, cb);
    }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    using DetailCallback = std::function<void(std::size_t, DetailUI &)>;

    void render_prepared(const PreparedTree &tree, const DetailCallback &detail_callback);
};

}    // namespace libpts::treeviz

#endif    // LIBPTS_TREE_VIZ_H_
