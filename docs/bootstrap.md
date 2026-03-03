# Bootstrap

The Bootstrap approach is an automated way to efficiently learn policies and heuristics from search instances, starting from scratch.
It starts with an initial budget, and passes over the training problems with budget-limited search.
Any solutions are used to train the model.
If the search does not solve a sufficient number of new problems, the budget is updated.
The process is completed when a given percentage of the validation problems are solved, which occurs after every sweep of the training set.
- Jabbari Arfaee, S.; Zilles, S.; and Holte, R. C. 2011. Learning heuristic functions for large state spaces. Artificial Intelligence 175(16): 2075–2098

The implementation provides thread support, where multiple searches can occur at once with one in each thread.
This requires shared inputs to be thread safe, which all implemented models are (PyTorch inference is thread-safe by default).
Model updates during learning occurs in the main thread, so there are no data races

## Bootstrap Training

The bootstrap training function `train_bootstrap` in `libpolicyts/train_bootstrap.h` is a templated function which is instantiated on the following types:
- The search input type (each algorithm defines a `SearchInput`)
- The search output type (each algorithm defines a `SearchOutput`)
- A user-defined learner which takes the search output and processes it for updating the internally held model
- An optional metrics tracking struct, which follows `MetricsTracker` in `libpolicyts/metrics_tracker.h`, which you need to define if you want to log custom metrics other than the standard ones

Learning handlers need to satisfy the following:

```cpp
// Concept for learning handler to satisfy requirements to interface with training
template <typename T, typename SearchInput, typename SearchOutput>
concept IsLearningHandler = requires(
    T t,
    std::vector<SearchInput> &batch,
    bool is_train,
    std::vector<SearchOutput> &&results,
    std::mt19937 &rng
) {
    { t.preprocess(batch, is_train) } -> std::same_as<void>;    // Optionally preprocess before passing to the search
    { t.process_data(std::move(results)) } -> std::same_as<void>;    // Process the search results into learning inputs
    { t.learning_step(rng) } -> std::same_as<void>;                  // Perform a learning step
    { t.checkpoint() } -> std::same_as<void>;                        // Checkpoint the model
};
```

There are two policies for budget updating:
- `BootstrapPolicy::DOUBLE` Double the budget if no progress (this can be slow to update budgets)
- `BootstrapPolicy::LTS_CM` (**Recommended**) Uses the modified scheme in Orseau, Laurent, Marcus Hutter, and Levi HS Lelis. "Levin tree search with context models." arXiv preprint arXiv:2305.16945 (2023)

This procedure will also by default store various training metrics for each problem attempted in a `metrics/train.csv` and `metrics/validate.csv`. 
The metrics saved have the following data:
```cpp
struct ProblemMetrics {
    int iter;
    std::string puzzle_name;
    bool solution_found;
    double solution_cost;
    double solution_prob;
    int expanded;
    int generated;
    double time;
    int budget;
};
```

It will also generate a `outstanding_problems.csv` which tracks total expansions incurred and the number of outstanding problems for every iteration of the process.

The training bootstrap has the following signature:
```cpp
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
)
```


## Bootstrap Testing

A similar procedure can be used to test your trained model on a set of problems. 
The calling code should ensure the model is loaded from a checkpoint prior to calling.
The bootstrap testing function `test_runner` in `libpolicyts/test_runner.h` is a templated function which is instantiated on the following types:
- The search input type (each algorithm defines a `SearchInput`)
- The search output type (each algorithm defines a `SearchOutput`)
- An optional metrics tracking struct, which follows `MetricsTracker` in `libpolicyts/metrics_tracker.h`, which you need to define if you want to log custom metrics other than the standard ones

This procedure will also by default store various testing metrics for each problem attempted in a `metrics/test.csv`, which matches the statistics in `train.csv`.

The testing bootstrap has the following signature:
```cpp
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
) 
```
