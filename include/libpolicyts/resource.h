// File: resource.h
// Description: Resource related queries

#ifndef LIBPTS_RESOURCE_H_
#define LIBPTS_RESOURCE_H_

namespace libpts {

/**
 * Get maximum memory usage used thus far, including spawned threads
 * @return memory usage in KB
 */
long get_mem_usage();

}    // namespace libpts

#endif    // LIBPTS_RESOURCE_H_
