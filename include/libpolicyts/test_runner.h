// File: test_runner.h
// Description: Generic test runner for all supported algorithms
//              This will run the search algorithm over all inputs,
//              using the bootstrap method of iteratively increase the search budget
//              until all problems solved or a timeout occurs

#ifndef LIBPTS_TEST_BOOTSTRAP_H_
#define LIBPTS_TEST_BOOTSTRAP_H_

#include <libpolicyts/concepts.h>
#include <libpolicyts/metrics_tracker.h>
#include <libpolicyts/resource.h>
#include <libpolicyts/stop_token.h>
#include <libpolicyts/thread_pool.h>
#include <libpolicyts/timer.h>

#include <absl/strings/str_cat.h>
#include <spdlog/spdlog.h>

#include <cassert>
#include <exception>
#include <string>
#include <vector>

namespace libpts::test {

constexpr int KB_PER_MB = 1024;

// Search inputs for a supported algorithm must have:
//     puzzle_name: A string puzzle name
//     search_budget: A search budget
template <typename T>
concept IsTestInput = requires(T t, const T ct) {
    { t.puzzle_name } -> std::same_as<std::string &>;
    { t.search_budget } -> std::same_as<int &>;
};

// Search output for a supported algorithm:
//     puzzle_name: A string puzzle name
//     solution_flag: Boolean flag for is a solution is found
//     solution_cost: Cost of the solution path found
//     num_expanded: Number of expanded nodes
//     num_generated: Number of generated nodes
//     solution_prob: Probability of solution path found
//     time: Time spent by the algorithm
template <typename T>
concept IsTestOutput = requires(T t, const T ct) {
    { t.puzzle_name } -> std::same_as<std::string &>;
    { t.solution_found } -> std::same_as<bool &>;
    { t.solution_cost } -> std::same_as<double &>;
    { t.num_expanded } -> std::same_as<int &>;
    { t.num_generated } -> std::same_as<int &>;
    { t.time } -> std::same_as<double &>;
};

// Concept for custom Problem Metrics
template <typename T, typename SearchOutput>
concept IsValidTracker = requires(
    T t,
    const T ct,
    const SearchOutput &output,
    const std::string &export_path,
    const std::string &file_name,
    bool resume,
    int bootstrap_iter,
    int search_budget
) {
    { T(export_path, file_name) };                                     // Constructable with path/file name
    { T(export_path, file_name, resume) };                             // and optional resume
    { t.add_row_by_result(output, bootstrap_iter, search_budget) };    // add output to rows
    { t.save() };                                                      // save
};

template <
    typename SearchInputT,
    typename SearchOutputT,
    typename ProblemMetricsTracker = MetricsTracker<ProblemMetrics>>
    requires IsTestInput<SearchInputT> && IsTestOutput<SearchOutputT>
             && (std::is_same_v<ProblemMetricsTracker, MetricsTracker<ProblemMetrics>>
                 || IsValidTracker<ProblemMetricsTracker, SearchOutputT>)
void test_runner(
    std::vector<SearchInputT> problems,
    std::function<SearchOutputT(const SearchInputT &)> algorithm,
    const std::string &output_dir,
    std::shared_ptr<StopToken> stop_token,
    int initial_search_budget,
    int num_threads,
    int max_iterations,
    double time_budget,
    std::string export_suffix
) {
    assert(stop_token);
    int bootstrap_iter = 0;
    int64_t total_expansions = 0;
    int64_t total_generated = 0;
    double total_cost = 0;
    int search_budget = initial_search_budget;
    std::vector<SearchInputT> outstanding_problems = problems;

    // Create metrics logger + directory
    const std::string metrics_dir = absl::StrCat(output_dir, "/metrics");
    if (export_suffix != "") {
        export_suffix = absl::StrCat("_", export_suffix);
    }
    ProblemMetricsTracker metrics_tracker(metrics_dir, absl::StrCat("test", export_suffix));
    MetricsTracker<TimeMetrics> time_tracker(metrics_dir, absl::StrCat("test_time", export_suffix));
    MetricsTracker<MemoryMetrics> memory_tracker(metrics_dir, absl::StrCat("test_memory", export_suffix));

    ThreadPool<SearchInputT, SearchOutputT> pool(num_threads);

    TimerCPU timer_cpu(time_budget);
    TimerWall timer_wall(time_budget);
    timer_cpu.start();
    timer_wall.start();

    while (!timer_cpu.is_timeout() && bootstrap_iter < max_iterations && !outstanding_problems.empty()) {
        ++bootstrap_iter;
        spdlog::info("Bootstrap iteration: {:d} of {:d}", bootstrap_iter, max_iterations);
        spdlog::info("Remaining unsolved problems: {:d}", outstanding_problems.size());

        // Update problem instance budget
        for (auto &p : outstanding_problems) {
            p.search_budget = search_budget;
        }
        std::vector<SearchInputT> unsolved_problems;

        // Shuffle training and iterate
        for (const auto &[batch_idx, batch_chunk] :
             outstanding_problems | std::views::chunk(num_threads) | std::views::enumerate)
        {
            decltype(problems) batch = batch_chunk | std::ranges::to<std::vector>();
            if (stop_token->stop_requested()) {
                spdlog::info("Stop requested, exiting train batch loop");
                break;
            }
            std::vector<SearchOutputT> results;
            try {
                results = pool.run(algorithm, batch);
            } catch (const std::exception &e) {
                spdlog::error("ThreadPool run failed: {}", e.what());
                throw;
            } catch (...) {
                spdlog::error("ThreadPool run failed with unknown exception.");
                throw;
            }
            if (stop_token->stop_requested()) {
                spdlog::info("Stop requested, exiting train batch loop");
                break;
            }

            for (auto &&[input_problem, result] : std::views::zip(batch, results)) {
                if constexpr (std::is_same_v<ProblemMetricsTracker, MetricsTracker<ProblemMetrics>>) {
                    // We log path probabilities of solution, but we also have non-policy search algorithms
                    // Thus, we need a way to check if a solution_prob is given by the search output, and if not, insert
                    // a dummy value
                    double solution_prob = [&]() -> double {
                        if constexpr (HasSolutionProb<SearchOutputT>) {
                            return result.solution_prob;
                        } else {
                            return 0;
                        }
                    }();
                    metrics_tracker.add_row(
                        ProblemMetrics{
                        .iter = bootstrap_iter,
                        .puzzle_name = result.puzzle_name,
                        .solution_found = result.solution_found,
                        .solution_cost = result.solution_cost,
                        .solution_prob = solution_prob,
                        .expanded = result.num_expanded,
                        .generated = result.num_generated,
                        .time = result.time,
                        .budget = search_budget
                        }
                    );
                } else {
                    metrics_tracker.add_row_by_result(result, bootstrap_iter, search_budget);
                }
                if (result.solution_found) {
                    total_generated += result.num_generated;
                    total_expansions += result.num_expanded;
                    total_cost += result.solution_cost;
                } else {
                    unsolved_problems.push_back(input_problem);
                }
            }
        }

        // Track max memory usage in megabytes
        memory_tracker.add_row({.iter = bootstrap_iter, .max_rss = static_cast<double>(get_mem_usage()) / KB_PER_MB});
        memory_tracker.save();

        double total_time_cpu = timer_cpu.get_duration();
        double total_time_wall = timer_wall.get_duration();
        time_tracker.add_row({.total_time_cpu = total_time_cpu, .total_time_wall = total_time_wall, .outstanding_problems=static_cast<int>(unsolved_problems.size())});

        outstanding_problems = unsolved_problems;

        // Unconditionally double budget
        search_budget *= 2;

        if (stop_token->stop_requested()) {
            spdlog::info("Stop requested, exiting test iteration");
            break;
        }
    }

    // Export
    metrics_tracker.save();
    time_tracker.save();
    memory_tracker.save();

    spdlog::info(
        "Total time cpu: {:.2f}s, total time wall: {:.2f}s total exp: {:d}, total gen: {:d}, total cost: {:.2f}",
        timer_cpu.get_duration(),
        timer_wall.get_duration(),
        total_expansions,
        total_generated,
        total_cost
    );
    spdlog::info("Maximum resident usage: {:.2f}MB", static_cast<double>(get_mem_usage()) / KB_PER_MB);
}

}    // namespace libpts::test

#endif    // LIBPTS_TEST_BOOTSTRAP_H_
