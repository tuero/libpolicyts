// File: math_util.cpp
// Description: Utility mathematical helper functions

#include <libpolicyts/math_util.h>

#include <cassert>
#include <cmath>

namespace libpts {

constexpr double SMALL_E = 1e-8;    // Ensure log(0) doesn't happen

auto log_policy_noise(const std::vector<double> &policy, double epsilon) -> std::vector<double> {
    std::vector<double> log_policy = policy;
    log_policy_noise_inplace(log_policy, epsilon);
    return log_policy;
}

void log_policy_noise_inplace(std::vector<double> &policy, double epsilon) {
    const double noise = 1.0 / static_cast<double>(policy.size());
    for (auto &p : policy) {
        p = std::log(((1.0 - epsilon) * p) + (epsilon * noise) + SMALL_E);
    }
}

}    // namespace libpts
