// File: timer.cpp
// Description: Measures user space time and signals a timeout

#include <libpolicyts/timer.h>

namespace libpts {

TimerCPU::TimerCPU(double seconds_limit)
    : seconds_limit_(seconds_limit) {}

void TimerCPU::start() noexcept {
    cpu_start_time_ = std::clock();
}

auto TimerCPU::is_timeout() const noexcept -> bool {
    const std::clock_t cpu_current_time = std::clock();
    const auto current_duration = static_cast<double>(cpu_current_time - cpu_start_time_) / CLOCKS_PER_SEC;
    return seconds_limit_ > 0 && current_duration >= seconds_limit_;
}

auto TimerCPU::get_duration() const noexcept -> double {
    const std::clock_t cpu_current_time = std::clock();
    const auto current_duration = static_cast<double>(cpu_current_time - cpu_start_time_) / CLOCKS_PER_SEC;
    return current_duration;
}

auto TimerCPU::get_time_remaining() const noexcept -> double {
    const std::clock_t cpu_current_time = std::clock();
    const auto current_duration = static_cast<double>(cpu_current_time - cpu_start_time_) / CLOCKS_PER_SEC;
    return seconds_limit_ - current_duration;
}

// -----------------------

constexpr int MILLISECONDS_PER_SECOND = 1000;

TimerWall::TimerWall(double seconds_limit)
    : seconds_limit_(seconds_limit) {}

void TimerWall::start() noexcept {
    wall_start_time_ = std::chrono::high_resolution_clock::now();
}

auto TimerWall::is_timeout() const noexcept -> bool {
    const auto elapsed = std::chrono::high_resolution_clock::now() - wall_start_time_;
    auto duration_count = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    auto current_duration = static_cast<double>(duration_count) / MILLISECONDS_PER_SECOND;
    return seconds_limit_ > 0 && current_duration >= seconds_limit_;
}

auto TimerWall::get_duration() const noexcept -> double {
    const auto elapsed = std::chrono::high_resolution_clock::now() - wall_start_time_;
    auto duration_count = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    auto current_duration = static_cast<double>(duration_count) / MILLISECONDS_PER_SECOND;
    return current_duration;
}

auto TimerWall::get_time_remaining() const noexcept -> double {
    const auto elapsed = std::chrono::high_resolution_clock::now() - wall_start_time_;
    auto duration_count = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    auto current_duration = static_cast<double>(duration_count) / MILLISECONDS_PER_SECOND;
    return seconds_limit_ - current_duration;
}

}    // namespace libpts
