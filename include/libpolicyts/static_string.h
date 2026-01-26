// File: static_string.h
// Description: Static string for NTTP

#ifndef LIBPTS_STATIC_STRING_H_
#define LIBPTS_STATIC_STRING_H_

#include <algorithm>
#include <array>

namespace libpts {

template <std::size_t N>
struct StaticString {
    std::array<char, N + 1> data = {};

    constexpr StaticString(const char (&input)[N + 1]) {    // NOLINT(*-avoid-c-arrays)
        std::ranges::copy_n(input, N + 1, data.begin());
    }
};

// Deduction guide
template <std::size_t N>
StaticString(const char (&)[N]) -> StaticString<N - 1>;    // NOLINT(*-avoid-c-arrays)

}    // namespace libpts

#endif    // LIBPTS_STATIC_STRING_H_
