// File: levints.h
// Description: LevinTS implementation
// LevinTS: Orseau, Laurent, et al. "Single-agent policy tree search with guarantees." Advances in Neural Information
//          Processing Systems 31 (2018).

#ifndef LIBPTS_ALGORITHM_LTS_H_
#define LIBPTS_ALGORITHM_LTS_H_

#include <libpolicyts/algorithm/detail/phs_impl.h>
#include <libpolicyts/stop_token.h>
#include <libpolicyts/timer.h>

namespace libpts::algorithm::lts {

// Concept for simple states for search
template <typename T>
concept IsEnv = phs::detail::IsEnv<T>;

// LTS model satisfies the following:
// model.observation(std::vector<Observation>) returns a struct which contains
//      a member .policy = std::vector<double>
template <typename T>
concept IsLTSModel = phs::detail::IsLTSModel<T>;

// Input to PHS search algorithm
template <IsEnv EnvT, IsLTSModel ModelT>
using SearchInput = phs::detail::SearchInput<EnvT, ModelT>;

// Search algorithm output
template <IsEnv EnvT>
using SearchOutput = phs::detail::SearchOutput<EnvT>;

using Status = phs::detail::Status;

// Coroutine-like stepable version
template <IsEnv EnvT, IsLTSModel ModelT>
using LTS = phs::detail::PHS<EnvT, ModelT>;

template <IsEnv EnvT, IsLTSModel ModelT>
auto search(const SearchInput<EnvT, ModelT> &input) -> SearchOutput<EnvT> {
    TimerWall timer(-1);
    LTS<EnvT, ModelT> step_lts(input);
    step_lts.init();
    timer.start();
    // Iteratively search until status changes (solved or timeout)
    while (step_lts.get_status() == Status::OK && (!input.stop_token || !input.stop_token->stop_requested())) {
        step_lts.step();
    }
    auto output = step_lts.get_search_output();
    output.time = timer.get_duration();
    return output;
}

}    // namespace libpts::algorithm::lts

#endif    // LIBPTS_ALGORITHM_LTS_H_
