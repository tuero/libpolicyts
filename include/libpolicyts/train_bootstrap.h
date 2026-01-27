// File: train_bootstrap.h
// Description: Generic train runner based on the bootstrap method
// Shahab Jabbari Arfaee, Sandra Zilles, and Robert C. Holte. Learning heuristic functions for large state spaces, 2011.

#ifndef LIBPTS_TRAIN_BOOTSTRAP_H_
#define LIBPTS_TRAIN_BOOTSTRAP_H_

#include <libpolicyts/concepts.h>
#include <libpolicyts/math_util.h>
#include <libpolicyts/metrics_tracker.h>
#include <libpolicyts/resource.h>
#include <libpolicyts/stop_token.h>
#include <libpolicyts/thread_pool.h>
#include <libpolicyts/timer.h>

#include <absl/flags/flag.h>
#include <absl/strings/str_cat.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <random>
#include <ranges>
#include <unordered_set>
#include <utility>
#include <vector>

namespace libpts::train {

/**
 * Split a list of items into a train and validate set
 * @param items The entire list of items to split
 * @param num_train Number of items to place in the train set
 * @param num_validate Number of items to place into the validation set
 * @param seed The seed used on the data shuffling
 * @return Pair of train and validation sets
 */
template <typename T>
auto split_train_validate(std::vector<T> &items, std::size_t num_train, std::size_t num_validate, int seed) {
    if (items.size() < num_train + num_validate) {
        SPDLOG_ERROR(
            "Input items {:d} is less than num_train {:d} + num_validate {:d}",
            items.size(),
            num_train,
            num_validate
        );
        std::exit(1);
    }
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    std::shuffle(items.begin(), items.end(), rng);
    return std::make_pair(
        std::vector<T>(items.begin(), items.begin() + static_cast<std::vector<T>::difference_type>(num_train)),
        std::vector<T>(
            items.begin() + static_cast<std::vector<T>::difference_type>(num_train),
            items.begin() + static_cast<std::vector<T>::difference_type>(num_train + num_validate)
        )
    );
}

constexpr int CHECKPOINT_INTERVAL = 10;
constexpr int KB_PER_MB = 1024;

// Search inputs for a supported algorithm must have:
//     puzzle_name: A string puzzle name
//     search_budget: A search budget
template <typename T>
concept IsTrainInput = requires(T t, const T ct) {
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
concept IsTrainOutput = requires(T t, const T ct) {
    { t.puzzle_name } -> std::same_as<std::string &>;
    { t.solution_found } -> std::same_as<bool &>;
    { t.solution_cost } -> std::same_as<double &>;
    { t.num_expanded } -> std::same_as<int &>;
    { t.num_generated } -> std::same_as<int &>;
    { t.time } -> std::same_as<double &>;
};

// Concept for learning handler to satisfy requirements to interface with training
template <typename T, typename SearchInput, typename SearchOutput>
concept IsLearningHandler = requires(
    T t,
    std::vector<SearchInput> &batch,
    bool is_train,
    std::vector<SearchOutput> &&results,
    std::mt19937 &rng
) {
    { t.preprocess(batch, is_train) } -> std::same_as<void>;
    { t.process_data(std::move(results)) } -> std::same_as<void>;
    { t.learning_step(rng) } -> std::same_as<void>;
    { t.checkpoint() } -> std::same_as<void>;
};

// Concept for custom Problem Metrics
// It must be an child template of MetricsTracker<>,
// and add rows of data through add_row_by_result()
template <typename T, typename SearchOutput>
concept IsValidTracker = requires(T t, const T ct, const SearchOutput &output, int bootstrap_iter, int search_budget) {
    []<typename MetricsItem>(MetricsTracker<MetricsItem> &) {}(t);
    { t.add_row_by_result(output, bootstrap_iter, search_budget) };
};

enum class BootstrapPolicy {
    DOUBLE = 0,    // Doubling if no progress (this can be really slow!)
    LTS_CM = 1     // Uses modfied version using X% percentage of progress
};

// External flags -> enum support
inline auto AbslParseFlag(absl::string_view text, BootstrapPolicy *mode, std::string *error) -> bool {
    if (text == "double") {
        *mode = BootstrapPolicy::DOUBLE;
        return true;
    }
    if (text == "lts_cm") {
        *mode = BootstrapPolicy::LTS_CM;
        return true;
    }
    *error = "unknown value for enumeration";
    return false;
}

inline auto AbslUnparseFlag(BootstrapPolicy mode) -> std::string {
    switch (mode) {
    case BootstrapPolicy::DOUBLE:
        return "double";
    case BootstrapPolicy::LTS_CM:
        return "lts_cm";
    }
    return absl::StrCat(mode);
}

constexpr double default_bootstrap_factor = 0.1;

template <
    typename SearchInputT,
    typename SearchOutputT,
    typename LearnerT,
    typename ProblemMetricsTracker = MetricsTracker<ProblemMetrics>>
    requires IsTrainInput<SearchInputT> && IsTrainOutput<SearchOutputT>
             && IsLearningHandler<LearnerT, SearchInputT, SearchOutputT>
             && (std::is_same_v<ProblemMetricsTracker, MetricsTracker<ProblemMetrics>>
                 || IsValidTracker<ProblemMetricsTracker, SearchOutputT>)
void train_bootstrap(
    std::vector<SearchInputT> problems_train,
    std::vector<SearchInputT> problems_validate,
    std::function<SearchOutputT(const SearchInputT &)> algorithm,
    LearnerT &learner,
    const std::string &output_dir,
    std::mt19937 rng,
    std::shared_ptr<StopToken> stop_token,
    int initial_search_budget,
    double validation_solved_ratio,
    int num_threads,
    int num_problems_per_batch,
    int max_iterations,
    int max_budget,
    double time_budget,
    BootstrapPolicy bootstrap_policy = BootstrapPolicy::DOUBLE,
    double bootstrap_factor = default_bootstrap_factor,
    int extra_iterations = 0,
    bool resume = false
) {
    int bootstrap_iter = 0;
    int search_budget = initial_search_budget;
    int validation_budget = initial_search_budget;
    int64_t total_expansions = 0;
    int n_validate_exit = static_cast<int>(static_cast<double>(problems_validate.size()) * validation_solved_ratio);
    std::unordered_set<std::string> solved_set_train;
    std::unordered_set<std::string> solved_set_validate;
    bool last_iteration = false;

    // If more threads than problems per batch, reduce thread usage to match
    if (num_problems_per_batch < num_threads) {
        num_threads = num_problems_per_batch;
    }

    // Create metrics logger + directory
    const std::string metrics_dir = absl::StrCat(output_dir, "/metrics");
    ProblemMetricsTracker metrics_tracker_train(metrics_dir, "train", resume);
    ProblemMetricsTracker metrics_tracker_validate(metrics_dir, "validate", resume);
    MetricsTracker<MemoryMetrics> memory_tracker(metrics_dir, "memory", resume);
    MetricsTracker<OutstandingMetrics> outstanding_problems_tracker(metrics_dir, "outstanding_problems", resume);
    MetricsTracker<TimeMetrics> time_tracker(metrics_dir, "train_time", resume);

    int bootstrap_to_skip = 0;
    int batch_to_skip = 0;
    if (resume) {
        for (const auto &row : metrics_tracker_train.rows) {
            if (row.solution_found) {
                solved_set_train.insert(row.puzzle_name);
            }
            search_budget = row.budget;
            total_expansions += row.expanded;
        }
        bootstrap_to_skip = metrics_tracker_train.rows.size() / problems_train.size();
        batch_to_skip = (static_cast<int>(metrics_tracker_train.rows.size())
                         - (bootstrap_to_skip * static_cast<int>(problems_train.size())))
                        / num_problems_per_batch;
        for (const auto &row : metrics_tracker_validate.rows) {
            if (row.solution_found) {
                solved_set_validate.insert(row.puzzle_name);
            }
            validation_budget = row.budget;
        }
        SPDLOG_INFO("Skipping Bootstrap iters {:d}, batches {:d}", bootstrap_to_skip, batch_to_skip);
    }

    // Initialize metrics
    if (!resume) {
        outstanding_problems_tracker.add_row(
            {.expansions=total_expansions, .outstanding_problems=static_cast<int>(problems_train.size() - solved_set_train.size())}
        );
    }

    ThreadPool<SearchInputT, SearchOutputT> pool(num_threads);
    learner.checkpoint();

    TimerCPU timer_cpu;
    TimerWall timer_wall;
    timer_cpu.start();
    timer_wall.start();
    double duration = 0;

    while (duration < time_budget && bootstrap_iter < max_iterations) {
        ++bootstrap_iter;
        int n_outstanding_train = problems_train.size() - solved_set_train.size();
        int n_outstanding_validate = problems_validate.size() - solved_set_validate.size();
        std::size_t prev_n_solved_train = solved_set_train.size();
        long long int solved_expansions = 0;    // Expansions required for solved problems this iter
        int curr_n_solved_train = 0;
        SPDLOG_INFO("Bootstrap iteration: {:d} of {:d}", bootstrap_iter, max_iterations);
        SPDLOG_INFO(
            "Remaining unsolved problems: Train = {:d}, Validate = {:d}",
            n_outstanding_train,
            n_outstanding_validate
        );

        // Update problem instance budget
        search_budget = std::min(search_budget, max_budget);
        validation_budget = std::min(validation_budget, max_budget);
        for (auto &p : problems_train) {
            p.search_budget = search_budget;
        }
        for (auto &p : problems_validate) {
            p.search_budget = validation_budget;
        }

        // Shuffle training and iterate
        std::shuffle(problems_train.begin(), problems_train.end(), rng);
        if (bootstrap_to_skip > 0) {
            --bootstrap_to_skip;
            continue;
        }
        for (const auto &[batch_idx, batch_chunk] :
             problems_train | std::views::chunk(num_problems_per_batch) | std::views::enumerate)
        {
            decltype(problems_train) batch = batch_chunk | std::ranges::to<std::vector>();
            SPDLOG_INFO(
                "Iteration {:d}, Batch: {:d} of {:d}, CPU time: {:.2f}, Wall time: {:.2f}",
                bootstrap_iter,
                batch_idx,
                ceil_div(static_cast<int>(problems_train.size()), num_problems_per_batch),
                timer_cpu.get_duration(),
                timer_wall.get_duration()
            );
            if (batch_to_skip > 0) {
                --batch_to_skip;
                continue;
            }

            if (stop_token->stop_requested()) {
                SPDLOG_INFO("Stop requested, exiting train batch loop");
                break;
            }

            learner.preprocess(batch, true);
            std::vector<SearchOutputT> results = pool.run(algorithm, batch);

            if (stop_token->stop_requested()) {
                SPDLOG_INFO("Stop requested, exiting train batch loop");
                break;
            }

            for (const auto &result : results) {
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
                    metrics_tracker_train.add_row({
                        bootstrap_iter,
                        result.puzzle_name,
                        result.solution_found,
                        result.solution_cost,
                        solution_prob,
                        result.num_expanded,
                        result.num_generated,
                        result.time,
                        search_budget,
                    });
                } else {
                    metrics_tracker_train.add_row_by_result(result, bootstrap_iter, search_budget);
                }
                duration += result.time;
                total_expansions += result.num_expanded;
                if (result.solution_found) {
                    solved_set_train.insert(result.puzzle_name);
                    solved_expansions += result.num_expanded;
                    ++curr_n_solved_train;
                }
            }
            // Process results
            learner.process_data(std::move(results));

            // Update model
            learner.learning_step(rng);

            // Metrics for outstanding problems
            outstanding_problems_tracker.add_row(
                {.expansions=total_expansions, .outstanding_problems=static_cast<int>(problems_train.size() - solved_set_train.size())}
            );

            double total_time_cpu = timer_cpu.get_duration();
            double total_time_wall = timer_wall.get_duration();
            time_tracker.add_row(
                {.total_time_cpu=total_time_cpu, .total_time_wall=total_time_wall, .outstanding_problems=static_cast<int>(problems_train.size() - solved_set_train.size())}
            );
            metrics_tracker_train.save();
            outstanding_problems_tracker.save();
            time_tracker.save();

            // Checkpoint model
            if (batch_idx % CHECKPOINT_INTERVAL == 0) {
                learner.checkpoint();
            }

            // Timeout
            if (duration >= time_budget) {
                SPDLOG_INFO("Timeout, exiting train batch loop");
                break;
            }
        }

        // Track max memory usage in megabytes
        memory_tracker.add_row({.iter = bootstrap_iter, .max_rss = static_cast<double>(get_mem_usage()) / KB_PER_MB});
        memory_tracker.save();

        if (stop_token->stop_requested()) {
            SPDLOG_INFO("Stop requested, exiting iteration loop");
            break;
        }

        if (duration < time_budget) {
            SPDLOG_INFO("Running Validation Iteration");
            for (const auto &[batch_idx, batch_chunk] :
                 problems_validate | std::views::chunk(num_problems_per_batch) | std::views::enumerate)
            {
                decltype(problems_validate) batch = batch_chunk | std::ranges::to<std::vector>();
                learner.preprocess(batch, false);
                std::vector<SearchOutputT> results = pool.run(algorithm, batch);

                if (stop_token->stop_requested()) {
                    SPDLOG_INFO("Stop requested, exiting validation batch loop");
                    break;
                }

                for (const auto &result : results) {
                    if constexpr (std::is_same_v<ProblemMetricsTracker, MetricsTracker<ProblemMetrics>>) {
                        // We log path probabilities of solution, but we also have non-policy search algorithms
                        // Thus, we need a way to check if a solution_prob is given by the search output, and if not,
                        // insert a dummy value
                        double solution_prob = [&]() -> double {
                            if constexpr (HasSolutionProb<SearchOutputT>) {
                                return result.solution_prob;
                            } else {
                                return 0;
                            }
                        }();
                        metrics_tracker_validate.add_row({
                            bootstrap_iter,
                            result.puzzle_name,
                            result.solution_found,
                            result.solution_cost,
                            solution_prob,
                            result.num_expanded,
                            result.num_generated,
                            result.time,
                            initial_search_budget,
                        });
                    } else {
                        metrics_tracker_validate.add_row_by_result(result, bootstrap_iter, search_budget);
                    }
                    if (result.solution_found) {
                        solved_set_validate.insert(result.puzzle_name);
                    }
                }
            }
        } else {
            SPDLOG_INFO("Skipping validation due to timeout");
        }
        metrics_tracker_validate.save();

        // Break out if stop requested
        if (stop_token->stop_requested()) {
            SPDLOG_INFO("Stop requested, exiting bootstrap loop");
            break;
        }

        if (last_iteration) {
            if (extra_iterations <= 0) {
                break;
            }
            --extra_iterations;
        }

        if (static_cast<int>(solved_set_validate.size()) >= n_validate_exit && !last_iteration) {
            SPDLOG_INFO("Solved validation set ratio exceeded.");
            SPDLOG_INFO("Running one more pass over training set.");
            last_iteration = true;
            // Don't check if we should modify budget, as this was enough to solve validation set
            continue;
        }

        switch (bootstrap_policy) {
        case BootstrapPolicy::DOUBLE:
            // If no new problems in the training solved, double budget
            if (prev_n_solved_train == solved_set_train.size() && n_outstanding_train > 0) {
                search_budget *= 2;
            }
            break;
        case BootstrapPolicy::LTS_CM:
            // New solved problems is more than X percentage of train size
            bool flag1 = (solved_set_train.size() - prev_n_solved_train) > (bootstrap_factor * problems_train.size());
            // Current iteration solves 1.X more than previous iteration
            bool flag2 = curr_n_solved_train > ((1 + bootstrap_factor) * static_cast<double>(prev_n_solved_train));
            if (!(flag1 && flag2) && n_outstanding_train > 0) {
                search_budget =
                    2 * search_budget
                    + solved_expansions
                          / (static_cast<int>(problems_train.size()) - static_cast<int>(solved_set_train.size()));
            }
            validation_budget = search_budget;
            break;
        }
    }

    double total_time_cpu = timer_cpu.get_duration();
    double total_time_wall = timer_wall.get_duration();

    // Export
    learner.checkpoint();
    metrics_tracker_train.save();
    metrics_tracker_validate.save();
    memory_tracker.save();
    time_tracker.save();

    SPDLOG_INFO("Total cpu time: {:.2f}, wall time: {:.2f}", total_time_cpu, total_time_wall);
    SPDLOG_INFO("Maximum resident usage: {:.2f}MB", static_cast<double>(get_mem_usage()) / 1024);
}

}    // namespace libpts::train

#endif    // LIBPTS_TRAIN_BOOTSTRAP_H_
