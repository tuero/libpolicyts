// File: utility.h
// Description: Utility helper functions

#ifndef PTS_UTIL_UTILITY_H_
#define PTS_UTIL_UTILITY_H_

#include <spdlog/spdlog.h>

#include <vector>

namespace libpts {

/**
 * Apply uniform mixture to policy
 * @param policy The policy
 * @param epislon Amount of mixing with uniform policy, between 0 and 1.
 * @return Vector of policy with log + uniform mixture applied
 */
[[nodiscard]] auto policy_noise(const std::vector<double> &policy, double epsilon = 0) -> std::vector<double>;

/**
 * Apply uniform mixture to policy in-place
 * @param policy The policy
 * @param epislon Amount of mixing with uniform policy, between 0 and 1.
 */
void policy_noise_inplace(std::vector<double> &policy, double epsilon = 0);

/**
 * Apply log + uniform mixture to policy
 * @param policy The policy
 * @param epislon Amount of mixing with uniform policy, between 0 and 1.
 * @return Vector of policy with log + uniform mixture applied
 */
[[nodiscard]] auto log_policy_noise(const std::vector<double> &policy, double epsilon = 0) -> std::vector<double>;

/**
 * Apply log + uniform mixture to policy in-place
 * @param policy The policy
 * @param epislon Amount of mixing with uniform policy, between 0 and 1.
 */
void log_policy_noise_inplace(std::vector<double> &policy, double epsilon = 0);

/**
 * Compute ceil(x / y) for integral types
 * @param x numerator
 * @param y denominator
 * @return result of the ceiling division
 */
template <typename T>
    requires std::is_integral_v<T>
constexpr auto ceil_div(T x, T y) -> T {
    return (x + (y - 1)) / y;
}

}    // namespace libpts

#endif    // PTS_UTIL_UTILITY_H_
