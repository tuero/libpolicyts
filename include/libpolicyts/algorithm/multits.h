// File: multits.h
// Description: MultiTS implementation
// lubyts: Orseau, Laurent, et al. "Single-agent policy tree search with guarantees." Advances in Neural Information
//          Processing Systems 31 (2018).

#ifndef LIBPTS_ALGORITHM_MULTITS_H_
#define LIBPTS_ALGORITHM_MULTITS_H_

#include <libpolicyts/algorithm/detail/luby_impl.h>
#include <libpolicyts/stop_token.h>
#include <libpolicyts/timer.h>

namespace libpts::algorithm::multits {

// Concept for simple states for search
template <typename T>
concept IsEnv = lubyts::detail::IsEnv<T>;

// MultiTS model satisfies the following:
// Either takes an observation as input and returns a type holding a policy
// OR takes a vector of observations and returns a vector of types holding a policy
template <typename T>
concept IsMultiTSModel = lubyts::detail::IsLubyModel<T>;

// Depths follow fixed sequence
struct MultiTSDepthModel {
    int depth;
    auto operator()([[maybe_unused]] int iter, [[maybe_unused]] Observation &obs) -> int {
        return depth;
    }
};

// Input to MultiTS search algorithm
template <IsEnv EnvT, IsMultiTSModel PolicyModelT>
struct SearchInput {
    std::string puzzle_name;
    EnvT state;
    int search_budget = 1;
    int depth = 1;
    double mix_epsilon = 0;
    int seed = 0;
    std::shared_ptr<StopToken> stop_token;
    std::shared_ptr<PolicyModelT> policy_model;
};

// Search algorithm output
template <IsEnv EnvT>
using SearchOutput = lubyts::detail::SearchOutput<EnvT>;

template <IsEnv EnvT, IsMultiTSModel ModelT>
auto search(const SearchInput<EnvT, ModelT> &input) -> SearchOutput<EnvT> {
    lubyts::detail::SearchInput<EnvT, ModelT> search_input{
        .puzzle_name = input.puzzle_name,
        .state = input.state,
        .search_budget = input.search_budget,
        .mix_epsilon = input.mix_epsilon,
        .seed = input.seed,
        .stop_token = input.stop_token,
        .policy_model = input.policy_model,
    };
    return lubyts::detail::search(search_input, MultiTSDepthModel{input.depth});
}

}    // namespace libpts::algorithm::multits

#endif    // LIBPTS_ALGORITHM_MULTITS_H_
