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
    [[nodiscard]] auto inference(InferenceInputs &observations) const -> std::vector<InferenceOutput> {
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

int main() {
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

    auto search_result = bfs::search(search_input);
    // solution_path_states includes start and goal state
    // solution_path_actions thus has 1 less item since its the actions between
    if (search_result.solution_found) {
        std::print("Starting state:\n{}\n", search_result.solution_path_states[0]);
        for (auto &&[s, a] : std::views::zip(
                 search_result.solution_path_states | std::views::drop(1),
                 search_result.solution_path_actions
             ))
        {
            std::print("Action taken: {}\n{}\n", a, s);
        }
    }

    return 0;
}
