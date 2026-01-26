// File: env_loaded.h
// Description: Loads the problems and create input for search process

#ifndef LIBPTS_ENV_LOADER_H_
#define LIBPTS_ENV_LOADER_H_

#ifndef LIBPTS_ENVS_FOUND
#error "libpolicyts was built without environment support. Rebuild with environment support enabled."
#endif

#include <libpolicyts/thread_pool.h>

#include <absl/strings/str_format.h>
#include <spdlog/spdlog.h>

#include <fstream>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace libpts::env {

// Concept for environment type requiring constructor with string param
template <typename T>
concept StringConstructable = requires(T t, const std::string &s) { T(s); };

struct LoadProblemOptions {
    auto num_threads(std::size_t _num_threads) -> LoadProblemOptions & {
        num_threads_ = _num_threads;
        return *this;
    }
    auto max_instances(std::size_t _max_instances) -> LoadProblemOptions & {
        max_instances_ = _max_instances;
        return *this;
    }
    auto problem_str_modify_func(const std::function<std::string(std::string)> &_problem_str_modify_func)
        -> LoadProblemOptions & {
        problem_str_modify_func_ = _problem_str_modify_func;
        return *this;
    }
    std::size_t num_threads_ = 1;
    std::size_t max_instances_ = std::numeric_limits<std::size_t>::max();
    std::optional<std::function<std::string(std::string)>> problem_str_modify_func_ = std::nullopt;
};

/**
 * Load states from a given problem file
 * @param path The path to the file containing problem strings
 * @param options The options containing threads to use during loading and max instances
 */
template <StringConstructable T>
[[nodiscard]] auto load_problems(const std::string &path, const LoadProblemOptions &options = LoadProblemOptions())
    -> std::tuple<std::vector<T>, std::vector<std::string>> {
    std::vector<T> problems;
    std::vector<std::string> problem_strs;
    std::size_t problem_counter = 0;

    std::ifstream file(path);
    if (!file.is_open()) {
        SPDLOG_ERROR("Problem file {:s} cannot be opened.", path);
        std::exit(1);
    }

    std::string line;
    while (std::getline(file, line)) {
        // Only grab max_instances if set
        if (problem_counter >= options.max_instances_) {
            break;
        }
        // Line starting with ; is sometimes used as a description for the levelset
        if (line[0] == ';') {
            continue;
        }
        problem_strs.push_back(line);
        ++problem_counter;
    }
    file.close();

    if (problem_counter == 0) {
        SPDLOG_ERROR("No problems found in {:s}.", path);
        std::exit(1);
    }

    ThreadPool<std::string, T> pool(options.num_threads_);
    problems = pool.run(
        [&](const std::string &s) -> T {
            if (options.problem_str_modify_func_) {
                return T(options.problem_str_modify_func_.value()(s));
            } else {
                return T(s);
            }
        },
        problem_strs
    );

    return {problems, problem_strs};
}

}    // namespace libpts::env

#endif    // LIBPTS_ENV_LOADER_H_
