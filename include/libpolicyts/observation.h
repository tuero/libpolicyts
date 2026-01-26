// File: observation.h
// Description: A common observation type

#ifndef LIBPTS_OBSERVATION_H_
#define LIBPTS_OBSERVATION_H_

#include <array>
#include <vector>

namespace libpts {

using Observation = std::vector<float>;
struct ObservationShape {
    int c;
    int h;
    int w;
    auto operator==(const ObservationShape &rhs) const -> bool {
        return c == rhs.c && h == rhs.h && w == rhs.w;
    }
    auto operator!=(const ObservationShape &rhs) const -> bool {
        return c != rhs.c || h != rhs.h || w != rhs.w;
    }
    [[nodiscard]] auto flat_size() const -> int {
        return c * h * w;
    }
    static auto from_array(const std::array<int, 3> &obs_shape) -> ObservationShape {
        return {.c = obs_shape[0], .h = obs_shape[1], .w = obs_shape[2]};
    }
};

}    // namespace libpts

#endif    // LIBPTS_OBSERVATION_H_
