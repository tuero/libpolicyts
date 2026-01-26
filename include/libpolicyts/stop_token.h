// File: stop_token.h
// Description: std::stop_token like flag class to signal for threads

#ifndef LIBPTS_STOP_TOKEN_H_
#define LIBPTS_STOP_TOKEN_H_

#include <atomic>

namespace libpts {

class StopToken {
public:
    void stop() noexcept {
        flag_ = true;
    }

    [[nodiscard]] auto stop_requested() const noexcept -> bool {
        return flag_;
    }

private:
    std::atomic<bool> flag_{false};
};

}    // namespace libpts

#endif    // LIBPTS_STOP_TOKEN_H_
