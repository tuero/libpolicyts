// File: timer.h
// Description: Measures user space time and signals a timeout

#ifndef LIBPTS_TIMER_H_
#define LIBPTS_TIMER_H_

#include <chrono>
#include <ctime>
#include <limits>

namespace libpts {

class TimerCPU {
    // Ensure clock_t is a 64 bit value
    // A 32 bit clock_t width will overflow after ~72 minutes which is longer than the expected runtime.
    // A 64 bit clock_t width will overflow after ~300,00 years
    constexpr static int BYTE_CHECK = 8;
    static_assert(sizeof(std::clock_t) == BYTE_CHECK);

public:
    TimerCPU(double seconds_limit = std::numeric_limits<double>::max());

    void start() noexcept;

    [[nodiscard]] auto is_timeout() const noexcept -> bool;

    [[nodiscard]] auto get_duration() const noexcept -> double;

    [[nodiscard]] auto get_time_remaining() const noexcept -> double;

private:
    double seconds_limit_;
    std::clock_t cpu_start_time_ = 0;
};

class TimerWall {
public:
    TimerWall(double seconds_limit = std::numeric_limits<double>::max());

    void start() noexcept;

    [[nodiscard]] auto is_timeout() const noexcept -> bool;

    [[nodiscard]] auto get_duration() const noexcept -> double;

    [[nodiscard]] auto get_time_remaining() const noexcept -> double;

private:
    double seconds_limit_;
    std::chrono::time_point<std::chrono::high_resolution_clock> wall_start_time_;
};

}    // namespace libpts

#endif    // LIBPTS_TIMER_H_
