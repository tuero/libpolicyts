// File: tsp.h
// Description: Wrapper around tsp_cpp gridworld standalone environment
//              There two environments: TSPGameState and TSPGridlock
//              TSPGameState allows the agent to traverse over previously visited cities
//              TSPDeadlockGameState will deadlock if the agent revisits a city that is not the start
//              https://github.com/tuero/tsp_cpp
#ifndef LIBPTS_ENV_TSP_H_
#define LIBPTS_ENV_TSP_H_

#ifndef LIBPTS_ENVS_FOUND
#error "libpolicyts was built without environment support. Rebuild with environment support enabled."
#endif

#include <libpolicyts/observation.h>
#include <libpolicyts/static_string.h>

#include <tsp/tsp.h>

#include <spdlog/spdlog.h>

#include <concepts>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace libpts::env {

enum TSPEvent : uint64_t {
    kCityVisited = 1 << 0,
};

namespace detail {

template <typename T>
concept IsTSPStateImpl = std::is_same_v<T, tsp::TSPGameState> || std::is_same_v<T, tsp::TSPDeadlockGameState>;

template <IsTSPStateImpl T, StaticString name_str>
class TSPStateImpl {
public:
    inline static const std::string name{name_str.data};
    inline static const int num_actions = 4;

    explicit TSPStateImpl(const std::string &board_str)
        : state(board_str) {}
    ~TSPStateImpl() = default;

    TSPStateImpl(const TSPStateImpl &) noexcept = default;
    TSPStateImpl(TSPStateImpl &&) noexcept = default;
    auto operator=(const TSPStateImpl &) noexcept -> TSPStateImpl & = default;
    auto operator=(TSPStateImpl &&) noexcept -> TSPStateImpl & = default;

    // Apply the action and return step cost
    auto apply_action(int action) -> double {
        if (action < 0 || action >= num_actions) [[unlikely]] {
            const std::string error_msg =
                std::format("Unknown action ({}), expected to be in range[0, {}]", action, num_actions - 1);
            SPDLOG_ERROR(error_msg);
            throw std::invalid_argument(error_msg);
        }
        state.apply_action(static_cast<tsp::Action>(action));
        reward_signal = state.get_reward_signal();
        return 1.0;
    }

    // Get observation, which should be viewed as [C,H,W] = observation_shape()
    [[nodiscard]] auto get_observation() const noexcept -> Observation {
        return state.get_observation();
    }

    // The shape observations should be views as, in [C,H,W] format
    [[nodiscard]] auto observation_shape() const noexcept -> ObservationShape {
        return ObservationShape::from_array(state.observation_shape());
    }

    // Return true if state is a solutions state
    [[nodiscard]] auto is_solution() const noexcept -> bool {
        return state.is_solution();
    }

    // Return true if state is terminal (can be solution or not)
    [[nodiscard]] auto is_terminal() const noexcept -> bool {
        return state.is_solution();
    }

    // Get heuristic (maybe uninformative)
    [[nodiscard]] auto get_heuristic() const noexcept -> double {
        return 0;
    }

    // Get hash of state
    [[nodiscard]] auto get_hash() const noexcept -> uint64_t {
        return state.get_hash();
    }

    // String representation of state
    [[nodiscard]] auto to_str() const noexcept -> std::string {
        std::ostringstream ss;
        ss << state;
        return ss.str();
    }

    // State equality
    [[nodiscard]] auto operator==(const TSPStateImpl &rhs) const -> bool {
        return state == rhs.state;
    }

    // Query if events were completed at this step
    // Use an events mask which is a bitmask of the Event enumeration
    [[nodiscard]] auto query_events(uint64_t events_mask) const noexcept -> bool {
        return (reward_signal & events_mask) > 0;
    }

    friend auto operator<<(std::ostream &os, const TSPStateImpl &s) -> std::ostream & {
        return os << s.state;
    }

    friend struct std::formatter<TSPStateImpl>;

private:
    T state;
    uint64_t reward_signal = 0;
};
}    // namespace detail

// Two versions of TSP
using TSPState = detail::TSPStateImpl<tsp::TSPGameState, "tsp">;
using TSPDeadlockState = detail::TSPStateImpl<tsp::TSPDeadlockGameState, "tsp_deadlock">;

}    // namespace libpts::env

// Hash and format support
template <>
struct std::hash<libpts::env::TSPState> {
    size_t operator()(const libpts::env::TSPState &state) const {
        return state.get_hash();
    }
};
template <>
struct std::hash<libpts::env::TSPDeadlockState> {
    size_t operator()(const libpts::env::TSPDeadlockState &state) const {
        return state.get_hash();
    }
};

template <>
struct std::formatter<libpts::env::TSPState> : std::formatter<std::string> {
    auto format(libpts::env::TSPState s, format_context &ctx) const {
        return formatter<string>::format(std::format("{}", s.state), ctx);
    }
};
template <>
struct std::formatter<libpts::env::TSPDeadlockState> : std::formatter<std::string> {
    auto format(libpts::env::TSPDeadlockState s, format_context &ctx) const {
        return formatter<string>::format(std::format("{}", s.state), ctx);
    }
};

#endif    // LIBPTS_ENV_TSP_H_
