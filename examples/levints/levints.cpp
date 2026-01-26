#include <libpolicyts/libpolicyts.h>

#include <print>
#include <ranges>
#include <vector>

// Policy which satisfies the constraint for levints
template <int N>
struct Policy {
    static_assert(N >= 1);
    struct InferenceInput {
        libpts::Observation obs;
    };
    struct InferenceOutput {
        std::vector<double> policy;
    };

    using InferenceInputs = std::vector<InferenceInput>;
    [[nodiscard]] auto inference(InferenceInputs &observations) const -> std::vector<InferenceOutput> {
        std::vector<InferenceOutput> inference_policies;
        inference_policies.reserve(observations.size());
        // Uniform policy over each action
        for ([[maybe_unused]] const auto &obs : observations) {
            inference_policies.emplace_back(std::vector<double>(static_cast<std::size_t>(N), 1.0 / N));
        }
        return inference_policies;
    }
};

using SokobanState = libpts::env::SokobanState;
using SokobanPolicy = Policy<SokobanState::num_actions>;
namespace lts = libpts::algorithm::lts;

int main() {
    constexpr auto problem_str =
        "10|10|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|"
        "01|01|01|01|01|01|01|01|01|01|01|01|00|01|01|01|01|01|01|01|01|01|02|01|01|01|01|01|01|01|01|04|04|02|03|01|"
        "01|01|01|01|01|02|03|02|04|01|01|01|01|01|04|03|04|03|04|01|01|01|01|01|01|01|01|01|01|01";
    constexpr int budget = 1e6;

    auto start_state = SokobanState(problem_str);

    std::shared_ptr<libpts::StopToken> stop_token = libpts::signal_installer();

    lts::SearchInput<SokobanState, SokobanPolicy> search_input{
        .puzzle_name = "puzzle_0",
        .state = start_state,
        .search_budget = budget,
        .inference_batch_size = 1,
        .mix_epsilon = 0.0,
        .stop_token = stop_token,
        .model = std::make_shared<SokobanPolicy>()
    };

    auto search_result = lts::search(search_input);
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
