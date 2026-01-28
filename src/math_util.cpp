// File: math_util.cpp
// Description: Utility mathematical helper functions

#include <libpolicyts/math_util.h>

#include <cassert>
#include <cmath>

namespace libpts {

constexpr double SMALL_E = 1e-8;    // Ensure log(0) doesn't happen

auto policy_noise(const std::vector<double> &policy, double epsilon) -> std::vector<double> {
    std::vector<double> p = policy;
    policy_noise_inplace(p, epsilon);
    return p;
}

void policy_noise_inplace(std::vector<double> &policy, double epsilon) {
    const double noise = 1.0 / static_cast<double>(policy.size());
    for (auto &p : policy) {
        p = ((1.0 - epsilon) * p) + (epsilon * noise);
    }
}

auto log_policy_noise(const std::vector<double> &policy, double epsilon) -> std::vector<double> {
    std::vector<double> log_policy = policy;
    log_policy_noise_inplace(log_policy, epsilon);
    return log_policy;
}

void log_policy_noise_inplace(std::vector<double> &policy, double epsilon) {
    policy_noise_inplace(policy, epsilon);
    for (auto &p : policy) {
        p = std::log(p + SMALL_E);
    }
}

}    // namespace libpts
