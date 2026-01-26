// File: signaller.h
// Description: Signal handler + install for graceful exiting

#ifndef LIBPTS_COMMON_SIGNALLER_H_
#define LIBPTS_COMMON_SIGNALLER_H_

#include "stop_token.h"

#include <memory>

namespace libpts {

/**
 * Create and install a signal handler
 * On SIGINT, token will request stop, and all objects storing it will call their exit code
 */
std::shared_ptr<StopToken> signal_installer();

/**
 * Create and install a signal handler
 * On SIGINT, token will request stop, and all objects storing it will call their exit code
 */
void signal_installer(std::shared_ptr<StopToken> stop_token);

}    // namespace libpts

#endif    // LIBPTS_COMMON_SIGNALLER_H_
