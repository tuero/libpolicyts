#include <libpolicyts/libpolicyts.h>

#include <print>
#include <ranges>
#include <vector>

// Policy + Heuristic which satisfies the constraint for PHS
using SokobanState = libpts::env::SokobanState;
using SokobanPolicy = libpts::model::TwoHeadedConvNetWrapper;
namespace phs = libpts::algorithm::phs;

int main() {
    constexpr auto problem_str =
        "10|10|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|01|"
        "01|01|01|01|01|01|01|01|01|01|01|01|00|01|01|01|01|01|01|01|01|01|02|01|01|01|01|01|01|01|01|04|04|02|03|01|"
        "01|01|01|01|01|02|03|02|04|01|01|01|01|01|04|03|04|03|04|01|01|01|01|01|01|01|01|01|01|01";
    constexpr int budget = 1e6;

    auto start_state = SokobanState(problem_str);

    // Model
    // NOLINTBEGIN(*-magic-numbers)
    auto model_config = SokobanPolicy::get_default_json_config();
    model_config["resnet_channels"] = 16;
    model_config["resnet_blocks"] = 2;
    model_config["policy_channels"] = 2;
    model_config["heuristic_channels"] = 2;
    model_config["policy_mlp_layers"] = std::vector<int>{8, 8};
    model_config["heuristic_mlp_layers"] = std::vector<int>{8, 8};
    model_config["use_batchnorm"] = false;
    model_config["learning_rate"] = 3e-4;
    model_config["l2_weight_decay"] = 1e-4;
    // NOLINTEND(*-magic-numbers)

    // Policy + Heuristic which satisfies the constraint for PHS
    auto model = std::make_shared<SokobanPolicy>(
        model_config,
        start_state.observation_shape(),
        start_state.num_actions,
        "cpu",
        ""
    );
    model->print();

    std::shared_ptr<libpts::StopToken> stop_token = libpts::signal_installer();

    phs::SearchInput<SokobanState, SokobanPolicy> search_input{
        .puzzle_name = "puzzle_0",
        .state = start_state,
        .search_budget = budget,
        .inference_batch_size = 1,
        .mix_epsilon = 0.0,
        .stop_token = stop_token,
        .model = model
    };

    auto search_result = phs::search(search_input);
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
