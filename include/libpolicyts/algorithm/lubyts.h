// File: lubyts.h
// Description: Luby Tree Search implementation
// lubyts: Orseau, Laurent, et al. "Single-agent policy tree search with guarantees." Advances in Neural Information
//          Processing Systems 31 (2018).

#ifndef LIBPTS_ALGORITHM_LUBYTS_H_
#define LIBPTS_ALGORITHM_LUBYTS_H_

#include <libpolicyts/algorithm/detail/luby_impl.h>
#include <libpolicyts/stop_token.h>
#include <libpolicyts/timer.h>

namespace libpts::algorithm::lubyts {

// Concept for simple states for search
template <typename T>
concept IsEnv = detail::IsEnv<T>;

// Luby model satisfies the following:
// Either takes an observation as input and returns a type holding a policy
// OR takes a vector of observations and returns a vector of types holding a policy
template <typename T>
concept IsLubyModel = detail::IsLubyModel<T>;

// Depth model
constexpr auto A006519(int n) -> int {
    return ((n ^ (n - 1)) + 1) / 2;
}

// Depths follow A006519 sequence
struct LubyDepthModel {
    auto operator()(int iter, [[maybe_unused]] Observation &obs) -> int {
        return A006519(iter);
    }
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
using SearchOutput = detail::SearchOutput<EnvT>;

template <IsEnv EnvT, IsLubyModel ModelT>
auto search(const SearchInput<EnvT, ModelT> &input) -> SearchOutput<EnvT> {
    detail::SearchInput<EnvT, ModelT> search_input{
        .puzzle_name = input.puzzle_name,
        .state = input.state,
        .search_budget = input.search_budget,
        .mix_epsilon = input.mix_epsilon,
        .seed = input.seed,
        .stop_token = input.stop_token,
        .policy_model = input.policy_model,
    };
    return detail::search(search_input, LubyDepthModel{});
}

}    // namespace libpts::algorithm::lubyts

#endif    // LIBPTS_ALGORITHM_LUBYTS_H_
