// File: craftworld.h
// Description: Wrapper around craftworld standalone environment
//              https://github.com/tuero/craftworld_cpp_v2
#ifndef LIBPTS_ENV_CRAFTWORLD_H_
#define LIBPTS_ENV_CRAFTWORLD_H_

#ifndef LIBPTS_ENVS_FOUND
#error "libpolicyts was built without environment support. Rebuild with environment support enabled."
#endif

#include <libpolicyts/observation.h>

#include <craftworld/craftworld.h>

#include <spdlog/spdlog.h>

#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace libpts::env {

// Bitcodes to query events
//     kRewardCodeCraftBronzeBar = 1 << 0,
//     kRewardCodeCraftStick = 1 << 1,
//     kRewardCodeCraftPlank = 1 << 2,
//     kRewardCodeCraftRope = 1 << 3,
//     kRewardCodeCraftNails = 1 << 4,
//     kRewardCodeCraftBronzeHammer = 1 << 5,
//     kRewardCodeCraftBronzePick = 1 << 6,
//     kRewardCodeCraftIronPick = 1 << 7,
//     kRewardCodeCraftBridge = 1 << 8,
//     kRewardCodeCraftGoldBar = 1 << 9,
//     kRewardCodeCraftGemRing = 1 << 10,
//     kRewardCodeUseAxe = 1 << 11,
//     kRewardCodeUseBridge = 1 << 12,
//     kRewardCodeCollectTin = 1 << 13,
//     kRewardCodeCollectCopper = 1 << 14,
//     kRewardCodeCollectWood = 1 << 15,
//     kRewardCodeCollectGrass = 1 << 16,
//     kRewardCodeCollectIron = 1 << 17,
//     kRewardCodeCollectGold = 1 << 18,
//     kRewardCodeCollectGem = 1 << 19,
//     kRewardCodeUseAtWorkstation1 = 1 << 20,
//     kRewardCodeUseAtWorkstation2 = 1 << 21,
//     kRewardCodeUseAtWorkstation3 = 1 << 22,
//     kRewardCodeUseAtFurnace = 1 << 23,
using CraftWorldEvent = craftworld::RewardCode;

class CraftWorldState {
public:
    inline static const std::string name{"craftworld"};
    inline static const int num_actions = 5;

    explicit CraftWorldState(const std::string &board_str)
        : state(board_str) {}
    ~CraftWorldState() = default;

    CraftWorldState(const CraftWorldState &) noexcept = default;
    CraftWorldState(CraftWorldState &&) noexcept = default;
    auto operator=(const CraftWorldState &) noexcept -> CraftWorldState & = default;
    auto operator=(CraftWorldState &&) noexcept -> CraftWorldState & = default;

    // Apply the action and return step cost
    auto apply_action(int action) -> double {
        if (action < 0 || action >= num_actions) [[unlikely]] {
            const std::string error_msg =
                std::format("Unknown action ({}), expected to be in range[0, {}]", action, num_actions - 1);
            SPDLOG_ERROR(error_msg);
            throw std::invalid_argument(error_msg);
        }
        state.apply_action(static_cast<craftworld::Action>(action));
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
    [[nodiscard]] auto operator==(const CraftWorldState &rhs) const -> bool {
        return state == rhs.state;
    }

    // Query if events were completed at this step
    // Use an events mask which is a bitmask of the Event enumeration
    [[nodiscard]] auto query_events(uint64_t events_mask) const noexcept -> bool {
        return (reward_signal & events_mask) > 0;
    }

    friend auto operator<<(std::ostream &os, const CraftWorldState &s) -> std::ostream & {
        return os << s.state;
    }

    friend struct std::formatter<CraftWorldState>;

private:
    craftworld::CraftWorldGameState state;
    uint64_t reward_signal = 0;
};

}    // namespace libpts::env

// Hash and format support
template <>
struct std::hash<libpts::env::CraftWorldState> {
    size_t operator()(const libpts::env::CraftWorldState &state) const {
        return state.get_hash();
    }
};

template <>
struct std::formatter<libpts::env::CraftWorldState> : std::formatter<std::string> {
    auto format(libpts::env::CraftWorldState s, format_context &ctx) const {
        return formatter<string>::format(std::format("{}", s.state), ctx);
    }
};

#endif    // LIBPTS_ENV_CRAFTWORLD_H_
