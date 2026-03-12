#include <libpolicyts/libpolicyts.h>

#include <print>
#include <ranges>
#include <vector>

// Heuristic which satisfies the constraint for bfs
struct Heuristic {
    struct InferenceInput {
        libpts::Observation obs;
    };
    struct InferenceOutput {
        double heuristic;
    };

    using InferenceInputs = std::vector<InferenceInput>;
    [[nodiscard]] auto inference(InferenceInputs &observations) const -> std::vector<InferenceOutput>
    {
        std::vector<InferenceOutput> inference_heuristic;
        inference_heuristic.reserve(observations.size());
        // Heuristic value of 0
        for ([[maybe_unused]] const auto &obs : observations) {
            inference_heuristic.emplace_back(0.0);
        }
        return inference_heuristic;
    }
};

using SokobanHeuristic = Heuristic;
using SokobanState = libpts::env::SokobanState;
namespace bfs = libpts::algorithm::bfs;
using Node = const bfs::Node<SokobanState> *;

struct SearchNodeAdapter {
    [[nodiscard]] auto id(const Node &n) const -> int
    {
        return n->id;
    }

    [[nodiscard]] auto parent_id(const Node &n) const -> std::optional<int>
    {
        return n->parent ? std::optional<int>{n->parent_id} : std::nullopt;
    }

    [[nodiscard]] auto action_taken(const Node &n) const -> int
    {
        return n->action;
    }

    [[nodiscard]] auto label(const Node &n) const -> std::string
    {
        return std::format("{}", n->id);
    }
};

int main()
{
    constexpr auto problem_str =
        "10|10|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|"
        "01|01|01|01|01|01|01|01|01|01|01|01|00|01|01|01|01|01|01|01|01|01|02|01|01|01|01|01|01|01|01|04|04|02|03|01|"
        "01|01|01|01|01|02|03|02|04|01|01|01|01|01|04|03|04|03|04|01|01|01|01|01|01|01|01|01|01|01";
    constexpr int budget = 1e6;

    auto start_state = SokobanState(problem_str);

    std::shared_ptr<libpts::StopToken> stop_token = libpts::signal_installer();

    bfs::SearchInput<SokobanState, SokobanHeuristic> search_input{
        .puzzle_name = "puzzle_0",
        .state = start_state,
        .search_budget = budget,
        .inference_batch_size = 1,
        .weight_g = 1.0,
        .weight_h = 1.0,
        .stop_token = stop_token,
        .model = std::make_shared<SokobanHeuristic>()
    };

    libpts::treeviz::TreeViewer viewer({.width = 1400, .height = 900, .title = "Minimal Tree Viewer"});

    auto step_bfs = bfs::BFS(search_input);
    step_bfs.init();
    std::vector<Node> search_nodes = step_bfs.get_tree();
    while (viewer.is_open()) {
        viewer.render(search_nodes, SearchNodeAdapter{}, [](const Node &n, libpts::treeviz::DetailUI &ui) {
            ui.text("Search Node");
            ui.separator();
            ui.field("id", n->id);
            ui.field("g", n->g);
            ui.field("h", n->h);
            ui.field("f", n->g + n->h);
        });
        if (viewer.step_amount() > 0) {
            for (auto _ : std::views::iota(0, viewer.step_amount())) {
                step_bfs.step();
            }
            std::print("step clicked\n");
            search_nodes = step_bfs.get_tree();
        }
        if (viewer.reset_clicked()) {
            step_bfs.reset();
            step_bfs.init();
            search_nodes = step_bfs.get_tree();
            std::print("reset clicked\n");
        }
    }

    return 0;
}
