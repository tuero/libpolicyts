// File: resource.cpp
// Description: Resource related queries

#include <libpolicyts/resource.h>

#include <sys/resource.h>

namespace libpts {

long get_mem_usage() {
    struct rusage usage{};
    [[maybe_unused]] int ret{};
    ret = getrusage(RUSAGE_SELF, &usage);
    // NOLINTNEXTLINE
    return usage.ru_maxrss;    // in KB
}

}    // namespace libpts
