// File: logging.h
// Description: Various logging utility functions

#ifndef LIBPTS_LOGGING_H_
#define LIBPTS_LOGGING_H_

#include <string>

namespace libpts {

/**
 * Initialize the file and terminal loggers
 * @param console_only Flag to only log to console
 * @param path The directory which the experiment output resides
 * @param postfix Postfix for logger name file
 * @param erase_if_exists Erase if log file already exists
 */
void init_loggers(
    bool console_only = true,
    const std::string &path = "",
    const std::string &postfix = "",
    bool erase_if_exists = true
);

/**
 * Log the invoked command used to run the current program
 * @param argc Number of arguments
 * @param argv char array of params
 */
void log_flags(int argc, char **argv);

/**
 * Flush the logs
 */
void log_flush();

/**
 * Close all loggers
 */
void close_loggers();

}    // namespace libpts

#endif    // LIBPTS_LOGGING_H_
