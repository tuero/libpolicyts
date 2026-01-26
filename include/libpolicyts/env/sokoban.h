// File: sokoban.h
// Description: Wrapper around sokoban_cpp standalone environment
//              https://github.com/tuero/sokoban_cpp
#ifndef LIBPTS_ENV_SOKOBAN_H_
#define LIBPTS_ENV_SOKOBAN_H_

#ifndef LIBPTS_ENVS_FOUND
#error "libpolicyts was built without environment support. Rebuild with environment support enabled."
#endif

#include <libpolicyts/observation.h>

#include <sokoban/sokoban.h>

#include <spdlog/spdlog.h>

#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace libpts::env {

// Bitcodes to query events
//     kRewardBoxInGoal = 1 << 0,
//     kRewardAllBoxesInGoal = 1 << 1,
using SokobanEvent = sokoban::RewardCodes;

class SokobanState {
public:
    inline static const std::string name{"sokoban"};
    inline static const int num_actions = 4;

    explicit SokobanState(const std::string &board_str)
        : state(board_str) {}
    ~SokobanState() = default;

    SokobanState(const SokobanState &) noexcept = default;
    SokobanState(SokobanState &&) noexcept = default;
    auto operator=(const SokobanState &) noexcept -> SokobanState & = default;
    auto operator=(SokobanState &&) noexcept -> SokobanState & = default;

    // Apply the action and return step cost
    auto apply_action(int action) -> double {
        if (action < 0 || action >= num_actions) [[unlikely]] {
            const std::string error_msg =
                std::format("Unknown action ({}), expected to be in range[0, {}]", action, num_actions - 1);
            SPDLOG_ERROR(error_msg);
            throw std::invalid_argument(error_msg);
        }
        state.apply_action(static_cast<sokoban::Action>(action));
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
    [[nodiscard]] auto operator==(const SokobanState &rhs) const -> bool {
        return state == rhs.state;
    }

    // Query if events were completed at this step
    // Use an events mask which is a bitmask of the Event enumeration
    [[nodiscard]] auto query_events(uint64_t events_mask) const noexcept -> bool {
        return (reward_signal & events_mask) > 0;
    }

    friend auto operator<<(std::ostream &os, const SokobanState &s) -> std::ostream & {
        return os << s.state;
    }

    friend struct std::formatter<SokobanState>;

private:
    sokoban::SokobanGameState state;
    uint64_t reward_signal = 0;
};

}    // namespace libpts::env

// Hash and format support
template <>
struct std::hash<libpts::env::SokobanState> {
    size_t operator()(const libpts::env::SokobanState &state) const {
        return state.get_hash();
    }
};

template <>
struct std::formatter<libpts::env::SokobanState> : std::formatter<std::string> {
    auto format(libpts::env::SokobanState s, format_context &ctx) const {
        return formatter<string>::format(std::format("{}", s.state), ctx);
    }
};

#endif    // LIBPTS_ENV_SOKOBAN_H_
