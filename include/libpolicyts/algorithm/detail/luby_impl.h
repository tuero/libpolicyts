// File: luby_impl.h
// Description: Generic LubyTS and MultiTS implementation
// LubyTS: Orseau, Laurent, et al. "Single-agent policy tree search with guarantees." Advances in Neural Information
//          Processing Systems 31 (2018).
// MultiTS: Orseau, Laurent, et al. "Single-agent policy tree search with guarantees." Advances in Neural Information
//          Processing Systems 31 (2018).

#ifndef LIBPTS_ALGORITHM_DETAIL_LUBY_H_
#define LIBPTS_ALGORITHM_DETAIL_LUBY_H_

#include <libpolicyts/concepts.h>
#include <libpolicyts/stop_token.h>
#include <libpolicyts/timer.h>

#include <spdlog/spdlog.h>

#include <random>
#include <ranges>
#include <string>
#include <vector>

namespace libpts::algorithm::lubyts::detail {

// Concept for simple states for search
template <typename T>
concept IsEnv = std::equality_comparable<T> && IsSTDHashable<T> && requires(T t, const T ct, const std::string &s) {
    { t.apply_action(makeval<int>()) } -> std::same_as<double>;    // apply_action with int action and returns cost
    { ct.get_observation() } -> std::same_as<Observation>;         // Observation for policy/heuristic inference
    { ct.is_solution() } -> std::same_as<bool>;                    // Solution check
    { ct.is_terminal() } -> std::same_as<bool>;                    // Terminal check (both solution + non-solution)
    *(&T::num_actions) == makeval<int>();                          // Number of actions
};

// Either takes an observation as input and returns a type holding a policy
template <typename T>
concept IsLubyModelValue = requires(T t, Observation &obs) {
    { t.inference(obs) } -> HasPolicy;
};

template <typename T>
concept IsLubyModelVector = requires(T t) {
    // Has an inner type called InferenceInput
    typename T::InferenceInput;
    // Which is constructable from an observation
    requires std::is_constructible_v<typename T::InferenceInput, Observation>;
    // Inference takes as input a vector of inference inputs and must return a std::vector<...>
    requires IsSpecialization<
        std::remove_cvref_t<decltype(t.inference(makeval<std::vector<typename T::InferenceInput> &>()))>,
        std::vector>;
    // Returned vector element type must satisfy HasPolicy
    requires HasPolicy<typename std::remove_cvref_t<
        decltype(t.inference(makeval<std::vector<typename T::InferenceInput> &>()))>::value_type>;
};

// Luby model satisfies the following:
// Either takes an observation as input and returns a type holding a policy
// OR takes a vector of observations and returns a vector of types holding a policy
template <typename T>
concept IsLubyModel = IsLubyModelValue<T> || IsLubyModelVector<T>;

// Depth model takes observation and iteration number and returns depth
template <typename T>
concept IsDepthModel = requires(T t, int iter, Observation &obs) {
    { t(iter, obs) } -> std::same_as<int>;
};

// Input to LubyTS search algorithm
template <IsEnv EnvT, IsLubyModel PolicyModelT>
struct SearchInput {
    std::string puzzle_name;
    EnvT state;
    int search_budget = 1;
    double mix_epsilon = 0;
    int seed = 0;
    std::shared_ptr<StopToken> stop_token;
    std::shared_ptr<PolicyModelT> policy_model;
};

// Search algorithm output
template <IsEnv EnvT>
struct SearchOutput {
    std::string puzzle_name;
    bool solution_found = false;
    double solution_cost = -1;
    int num_expanded = 0;
    int num_generated = 0;
    double solution_prob = 0;
    double time = 0;
    std::vector<EnvT> solution_path_states{};
    std::vector<int> solution_path_actions{};
    std::vector<double> solution_path_costs{};
};

// Perform a depth-limited rollout with action sampled from policy
template <IsEnv EnvT, IsLubyModel PolicyModelT>
auto sample_trajectory(
    EnvT state,
    std::shared_ptr<PolicyModelT> policy_model,
    SearchOutput<EnvT> &search_output,
    std::shared_ptr<StopToken> stop_token,
    int depth,
    double epsilon,
    std::mt19937 &rng
) {
    double g = 0;
    double log_p = 0;
    search_output.solution_found = false;
    search_output.solution_path_states.clear();
    search_output.solution_path_actions.clear();
    search_output.solution_path_costs.clear();
    search_output.solution_path_states.push_back(state);

    // Rollout of depth N
    for (auto _ : std::views::iota(0, std::max(depth, 0))) {
        // Goal check
        if (state.is_solution()) {
            search_output.solution_found = true;
            search_output.solution_cost = g;
            search_output.solution_prob = std::exp(log_p);
        }

        // Early stoppage
        if (state.is_terminal() || stop_token->stop_requested()) {
            break;
        }

        // Query policy
        auto policy = [&]() -> std::vector<double> {
            auto obs = state.get_observation();
            if constexpr (IsLubyModelValue<PolicyModelT>) {
                return policy_model->inference(obs).policy;
            } else {
                using InferenceInputT = PolicyModelT::InferenceInput;
                std::vector<InferenceInputT> inference_inputs;
                inference_inputs.emplace_back(std::move(obs));
                return policy_model->inference(inference_inputs)[0].policy;
            }
        }();
        policy_noise_inplace(policy, epsilon);

        // Sample action
        std::discrete_distribution<int> d(policy.begin(), policy.end());
        int action = d(rng);

        // Step
        double step_cost = state.apply_action(action);
        g += step_cost;
        log_p += std::log(policy[static_cast<std::size_t>(action)]);

        search_output.solution_path_states.push_back(state);
        search_output.solution_path_actions.push_back(action);
        search_output.solution_path_costs.push_back(step_cost);

        ++search_output.num_expanded;
        ++search_output.num_generated;
    }

    // Early break
    if (!search_output.solution_found) {
        search_output.solution_path_states.clear();
        search_output.solution_path_actions.clear();
        search_output.solution_path_costs.clear();
    }
}

template <IsEnv EnvT, IsLubyModel PolicyModelT, IsDepthModel DepthModelT>
auto search(const SearchInput<EnvT, PolicyModelT> &input, DepthModelT depth_model) -> SearchOutput<EnvT> {
    SearchOutput<EnvT> output{.puzzle_name = input.puzzle_name};
    std::mt19937 rng(static_cast<std::mt19937::result_type>(input.seed));
    TimerWall timer(-1);
    timer.start();

    // Continue to sample while within budget
    for (int i = 1; output.num_expanded < input.search_budget && !input.stop_token->stop_requested(); ++i) {
        // Query depth
        auto obs = input.state.get_observation();
        auto remaining_budget = std::max(input.search_budget - output.num_expanded, 0);
        auto depth = std::min(depth_model(i, obs), remaining_budget);
        // Rollout
        detail::sample_trajectory(
            input.state,
            input.policy_model,
            output,
            input.stop_token,
            depth,
            input.mix_epsilon,
            rng
        );
        if (output.solution_found) {
            break;
        }
    }

    output.time = timer.get_duration();
    if (output.solution_found) {
        spdlog::info(
            "Solved - name: {:s}, exp: {:d}, gen: {:d}, budget: {:d}, c: {:.0f}",
            input.puzzle_name,
            output.num_expanded,
            output.num_generated,
            input.search_budget,
            output.solution_cost
        );
    } else {
        spdlog::info(
            "Budget timeout - name: {:s}, exp: {:d}, gen: {:d}, budget: {:d}",
            input.puzzle_name,
            output.num_expanded,
            output.num_generated,
            input.search_budget
        );
    }
    return output;
}

}    // namespace libpts::algorithm::lubyts::detail

#endif    // LIBPTS_ALGORITHM_DETAIL_LUBY_H_
