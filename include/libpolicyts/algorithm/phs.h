// File: phs.h
// Description: PHS* implementation
// PHS*: Orseau, Laurent, and Levi HS Lelis. "Policy-guided heuristic search with guarantees." Proceedings of the AAAI
//       Conference on Artificial Intelligence. Vol. 35. No. 14. 2021.

#ifndef LIBPTS_ALGORITHM_PHS_H_
#define LIBPTS_ALGORITHM_PHS_H_

#include <libpolicyts/algorithm/detail/phs_impl.h>
#include <libpolicyts/stop_token.h>
#include <libpolicyts/timer.h>

namespace libpts::algorithm::phs {

// Concept for simple states for search
template <typename T>
concept IsEnv = detail::IsEnv<T>;

// PHS model satisfies the following:
// model.observation(std::vector<Observation>) returns a struct which contains
//      a member .policy = std::vector<double>
//      a member .heurtitic = double
template <typename T>
concept IsPHSModel = detail::IsPHSModel<T>;

// Input to PHS search algorithm
template <IsEnv EnvT, IsPHSModel ModelT>
using SearchInput = detail::SearchInput<EnvT, ModelT>;

// Search algorithm output
template <IsEnv EnvT>
using SearchOutput = detail::SearchOutput<EnvT>;

using Status = detail::Status;

// Coroutine-like stepable version
template <IsEnv EnvT, IsPHSModel ModelT>
using PHS = phs::detail::PHS<EnvT, ModelT>;

template <IsEnv EnvT, IsPHSModel ModelT>
auto search(const SearchInput<EnvT, ModelT> &input) -> SearchOutput<EnvT> {
    TimerWall timer(-1);
    PHS<EnvT, ModelT> step_phs(input);
    step_phs.init();
    timer.start();
    // Iteratively search until status changes (solved or timeout)
    while (step_phs.get_status() == Status::OK && (!input.stop_token || !input.stop_token->stop_requested())) {
        step_phs.step();
    }
    auto output = step_phs.get_search_output();
    output.time = timer.get_duration();
    return output;
}

}    // namespace libpts::algorithm::phs

#endif    // LIBPTS_ALGORITHM_PHS_H_
