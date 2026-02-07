// File: boulderdash.h
// Description: Wrapper around boulderdash_cpp standalone environment
//              https://github.com/tuero/boulderdash_cpp
#ifndef LIBPTS_ENV_BOULDERDASH_H_
#define LIBPTS_ENV_BOULDERDASH_H_

#ifndef LIBPTS_ENVS_FOUND
#error "libpolicyts was built without environment support. Rebuild with environment support enabled."
#endif

#include <libpolicyts/observation.h>

#include <boulderdash/boulderdash.h>

#include <spdlog/spdlog.h>

#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace libpts::env {

// Bitcodes to query events
//     kRewardAgentDies = 1 << 0,
//     kRewardCollectDiamond = 1 << 1,
//     kRewardWalkThroughExit = 1 << 2,
//     kRewardNutToDiamond = 1 << 3,
//     kRewardButterflyToDiamond = 1 << 4,
//     kRewardCollectKey = 1 << 5,
//     kRewardCollectKeyRed = 1 << 6,
//     kRewardCollectKeyBlue = 1 << 7,
//     kRewardCollectKeyGreen = 1 << 8,
//     kRewardCollectKeyYellow = 1 << 9,
//     kRewardWalkThroughGate = 1 << 10,
//     kRewardWalkThroughGateRed = 1 << 11,
//     kRewardWalkThroughGateBlue = 1 << 12,
//     kRewardWalkThroughGateGreen = 1 << 13,
//     kRewardWalkThroughGateYellow = 1 << 14,
using BoulderDashEvent = boulderdash::RewardCodes;

class BoulderDashState {
public:
    inline static const std::string name{"boulderdash"};
    inline static const int num_actions = 4;

    explicit BoulderDashState(const std::string &board_str)
        : state(board_str) {}
    ~BoulderDashState() = default;

    BoulderDashState(const BoulderDashState &) noexcept = default;
    BoulderDashState(BoulderDashState &&) noexcept = default;
    auto operator=(const BoulderDashState &) noexcept -> BoulderDashState & = default;
    auto operator=(BoulderDashState &&) noexcept -> BoulderDashState & = default;

    // Apply the action and return step cost
    auto apply_action(int action) -> double {
        if (action < 0 || action >= num_actions) [[unlikely]] {
            const std::string error_msg =
                std::format("Unknown action ({}), expected to be in range[0, {}]", action, num_actions - 1);
            SPDLOG_ERROR(error_msg);
            throw std::invalid_argument(error_msg);
        }
        state.apply_action(static_cast<boulderdash::Action>(action));
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
    [[nodiscard]] auto operator==(const BoulderDashState &rhs) const -> bool {
        return state == rhs.state;
    }

    // Query if events were completed at this step
    // Use an events mask which is a bitmask of the Event enumeration
    [[nodiscard]] auto query_events(uint64_t events_mask) const noexcept -> bool {
        return (reward_signal & events_mask) > 0;
    }

    friend auto operator<<(std::ostream &os, const BoulderDashState &s) -> std::ostream & {
        return os << s.state;
    }

    friend struct std::formatter<BoulderDashState>;

private:
    boulderdash::BoulderDashGameState state;
    uint64_t reward_signal = 0;
};

}    // namespace libpts::env

// Hash and format support
template <>
struct std::hash<libpts::env::BoulderDashState> {
    size_t operator()(const libpts::env::BoulderDashState &state) const {
        return state.get_hash();
    }
};

template <>
struct std::formatter<libpts::env::BoulderDashState> : std::formatter<std::string> {
    auto format(const libpts::env::BoulderDashState &s, format_context &ctx) const {
        return formatter<string>::format(std::format("{}", s.state), ctx);
    }
};

#endif    // LIBPTS_ENV_BOULDERDASH_H_
