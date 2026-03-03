// File: rlts.h
// Description: Rerooting Levin Tree Search Implementation
// Orseau, Laurent, Marcus Hutter, and Levi HS Lelis. "Exponential Speedups by Rerooting Levin Tree Search." arXiv
// preprint arXiv:2412.05196 (2024).

#ifndef LIBPTS_ALGORITHM_RLTS_H_
#define LIBPTS_ALGORITHM_RLTS_H_

#include <libpolicyts/concepts.h>
#include <libpolicyts/math_util.h>
#include <libpolicyts/metrics_tracker.h>
#include <libpolicyts/stable_pool.h>
#include <libpolicyts/stop_token.h>
#include <libpolicyts/timer.h>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/flags/flag.h>
#include <absl/hash/hash.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_split.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <optional>
#include <ranges>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace libpts::algorithm::rlts {

// Concept for simple states for flat search
template <typename T>
concept IsEnv = std::equality_comparable<T> && IsSTDHashable<T> && requires(T t, const T ct, const std::string &s) {
    { t.apply_action(makeval<int>()) } -> std::same_as<double>;    // apply_action with int action and returns cost
    { ct.get_observation() } -> std::same_as<Observation>;         // Observation for policy/heuristic inference
    { ct.get_hash() } -> std::same_as<uint64_t>;                   // get hash
    { ct.is_solution() } -> std::same_as<bool>;                    // Solution check
    { ct.is_terminal() } -> std::same_as<bool>;                    // Terminal check (both solution + non-solution)
    *(&T::num_actions) == makeval<int>();                          // Number of actions
};

// Concept for RLTS model
// Must take observation as inference input and policy in inference output
// Optionally can have a heuristic
template <typename T>
concept IsRLTSModel = requires(T t) {
    // Has an inner type called InferenceInput
    typename T::InferenceInput;
    // Which is constructable from an observation
    requires std::is_constructible_v<typename T::InferenceInput, Observation>;
    // Inference takes as input a vector of inference inputs and must return a std::vector<...>
    requires IsSpecialization<
        std::remove_cvref_t<decltype(t.inference(makeval<std::vector<typename T::InferenceInput> &>()))>,
        std::vector>;
    // Returned vector element type must satisfy HasPolicy
    requires HasPolicy<typename std::remove_cvref_t<
        decltype(t.inference(makeval<std::vector<typename T::InferenceInput> &>()))>::value_type>;
};

// Concept for search output rerooter must create
// If you want your rerooter to support metric logging, you must satisfy this constraint
// Single method .to_metric_items(bootstrap_iter, puzzle_name) which creates a vector of valid metric items
// See metrics_tracker.h for the IsMetricsItem concept
template <typename T>
concept IsRerooterSearchOutput = requires(const T ct, int bootstrap_iter, const std::string &puzzle_name) {
    // to_metric_items return a vector of items ...
    requires IsSpecialization<
        std::remove_cvref_t<decltype(ct.to_metric_items(bootstrap_iter, puzzle_name))>,
        std::vector>;
    // ... each of which are valid metric items
    requires IsMetricsItem<
        typename std::remove_cvref_t<decltype(ct.to_metric_items(bootstrap_iter, puzzle_name))>::value_type>;
};

// Helper structs to determine if user provides a rerooter with search output metric logging support
namespace detail {

template <typename T>
concept HasGetSearchOutputMutable = requires(T t) { t.get_search_output(); };

template <typename T>
concept HasGetSearchOutputConst = requires(const T ct) { ct.get_search_output(); };

template <typename T>
concept HasAnyGetSearchOutput = HasGetSearchOutputMutable<T> || HasGetSearchOutputConst<T>;

template <typename T>
concept HasRerooterSearchOutput = requires(const T ct) {
    { ct.get_search_output() } -> IsRerooterSearchOutput;
};

// Dummy rerooter search output type if user does not provide `get_search_output`
struct NoRerooterSearchOutput {};

}    // namespace detail

template <typename T>
concept HasValidRerooterSearchOutput = (!detail::HasAnyGetSearchOutput<T>) || detail::HasRerooterSearchOutput<T>;

template <typename T, typename = void>
struct RerooterSearchOutputType {
    using type = detail::NoRerooterSearchOutput;
};

template <typename T>
struct RerooterSearchOutputType<T, std::void_t<decltype(std::declval<const T &>().get_search_output())>> {
    using type = std::remove_cvref_t<decltype(std::declval<const T &>().get_search_output())>;
};

// Forward Declaration
template <IsEnv EnvT>
struct Node;

// Concept which a rerooter must satisfy
// This are the entry pointers which rerooters may want to use for internal state
// If you want to support logging rerooter metrics, you must satisfy HasRerooterSearchOutput
//   which provides a .get_search_output() which returns a valid type
template <typename T, typename EnvT>
concept IsRerooter =
    IsEnv<EnvT>
    && requires(T t, const T ct, const Node<EnvT> &node, const Node<EnvT> &current_node, const Node<EnvT> &child_node) {
           { t.reset() };                                     // Reset the rerooter state
           { t.init(node) };                                  // Initialize rerooter with root node
           { t.expanded(node) };                              // Node was just expanded
           { t.generated(current_node, child_node) };         // child_node was just generated from current_node
           { t.prev_generated(current_node, child_node) };    // child_node was previously generated and being pruned
           { t(node) } -> std::same_as<double>;               // Get rerooting weight for this node
           { t.batch_inferenced() };                          // Batch inference was just performed for policy
           { t.solution_found(node) };                        // Solution found at the generated node
       } && HasValidRerooterSearchOutput<T>;                  // Optional .get_search_output() support

enum class CostMode {
    Slenderness,
    DPi,
};

// External flags -> enum support
inline auto AbslParseFlag(absl::string_view text, CostMode *cost_mode, std::string *error) -> bool
{
    if (text == "slenderness") {
        *cost_mode = CostMode::Slenderness;
        return true;
    }
    if (text == "d_pi") {
        *cost_mode = CostMode::DPi;
        return true;
    }
    *error = "unknown value for enumeration";
    return false;
}

inline auto AbslUnparseFlag(CostMode cost_mode) -> std::string
{
    switch (cost_mode) {
    case CostMode::Slenderness:
        return "slenderness";
    case CostMode::DPi:
        return "d_pi";
    }
    return absl::StrCat(cost_mode);
}

// Input to PHS search algorithm
template <IsEnv EnvT, IsRLTSModel ModelT, typename RerooterT>
    requires IsRerooter<RerooterT, EnvT>
struct SearchInput {
    using env_t = EnvT;
    std::string puzzle_name;
    EnvT state;
    int search_budget{};
    int inference_batch_size{};
    double mix_epsilon{};
    CostMode cost_mode = CostMode::Slenderness;
    std::shared_ptr<StopToken> stop_token;
    std::shared_ptr<ModelT> model;
    RerooterT rerooter;
};

// Search algorithm output
template <IsEnv EnvT, typename RerooterSearchOutput = detail::NoRerooterSearchOutput>
    requires std::same_as<RerooterSearchOutput, detail::NoRerooterSearchOutput>
             || IsRerooterSearchOutput<RerooterSearchOutput>
struct SearchOutput {
    std::string puzzle_name;
    bool solution_found = false;
    double solution_cost = -1;
    int num_expanded = 0;
    int num_generated = 0;
    double solution_prob = 0;
    double time = 0;
    // Solution path
    std::vector<EnvT> solution_path_states{};
    std::vector<int> solution_path_actions{};
    std::vector<double> solution_path_costs{};
    // Search Statistics
    double w_cumulative = 0;
    std::vector<double> solution_path_w{};
    std::vector<double> solution_path_w_cumulative{};
    double log_slenderness_cost_sol = 0;    // For bound in (16)
    double max_wt = 0;                      // For bound in (15)
    double max_log_slenderness_cost = 0;    // For bound in (15)
    double log_bound_simplified = 0;        // (Corollary 12)
    double log_bound_trivial = 0;
    // Custom weight metrics, or dummy stub type
    RerooterSearchOutput rerooter_search_output{};
};

// General metrics for RLTS
struct RLTSMetrics {
    int iter;
    std::string puzzle_name;
    bool solution_found;
    double solution_cost;
    double solution_prob;
    int expanded;
    int generated;
    double time;
    int budget;
    // Other stats
    double w_cumulative;
    double log_slenderness_cost_sol;    // For bound in (16)
    double max_wt;                      // For bound in (15)
    double max_log_slenderness_cost;    // For bound in (15)
    double log_bound_simplified;        // (Corollary 12)
    double log_bound_trivial;

    static auto make_from_str(const std::string &str) -> RLTSMetrics
    {
        std::vector<std::string> strs = absl::StrSplit(str, ',');
        // NOLINTBEGIN (*-magic-numbers)
        if (strs.size() != 15) {
            spdlog::error("Error reading line {:s}, {:d}", str, strs.size());
            throw std::runtime_error("line does not contain valid data for this metric");
        }
        return {
            .iter = std::stoi(strs[0]),
            .puzzle_name = strs[1],
            .solution_found = static_cast<bool>(std::stoi(strs[2])),
            .solution_cost = std::stod(strs[3]),
            .solution_prob = std::stod(strs[4]),
            .expanded = std::stoi(strs[5]),
            .generated = std::stoi(strs[6]),
            .time = std::stod(strs[7]),
            .budget = std::stoi(strs[8]),
            .w_cumulative = std::stod(strs[9]),
            .log_slenderness_cost_sol = std::stod(strs[10]),
            .max_wt = std::stod(strs[11]),
            .max_log_slenderness_cost = std::stod(strs[12]),
            .log_bound_simplified = std::stod(strs[13]),
            .log_bound_trivial = std::stod(strs[14]),
        };
        // NOLINTEND
    }
    static void dump_header(std::ostream &os)
    {
        os << "iter,";
        os << "puzzle_name,";
        os << "solution_found,";
        os << "solution_cost,";
        os << "solution_prob,";
        os << "expanded,";
        os << "generated,";
        os << "time,";
        os << "budget,";
        os << "w_cumulative,";
        os << "log_slenderness_cost_sol,";
        os << "max_wt,";
        os << "max_log_slenderness_cost,";
        os << "log_bound_simplified,";
        os << "log_bound_trivial\n";
    }
    friend auto operator<<(std::ostream &os, const RLTSMetrics &metrics_item) -> std::ostream &
    {
        os << metrics_item.iter << ",";
        os << metrics_item.puzzle_name << ",";
        os << metrics_item.solution_found << ",";
        os << metrics_item.solution_cost << ",";
        os << metrics_item.solution_prob << ",";
        os << metrics_item.expanded << ",";
        os << metrics_item.generated << ",";
        os << metrics_item.time << ",";
        os << metrics_item.budget << ",";
        os << metrics_item.w_cumulative << ",";
        os << metrics_item.log_slenderness_cost_sol << ",";
        os << metrics_item.max_wt << ",";
        os << metrics_item.max_log_slenderness_cost << ",";
        os << metrics_item.log_bound_simplified << ",";
        os << metrics_item.log_bound_trivial << "\n";
        return os;
    }
};

// Base RLTS metrics tracker for general search statistics
template <typename EnvT>
struct RLTSMetricsTrackerBase {
    std::vector<RLTSMetrics> rows_general;

    RLTSMetricsTrackerBase() = delete;
    RLTSMetricsTrackerBase(const std::string &export_path, const std::string &file_name, bool resume = false)
        : full_path_general((std::format("{:s}/{:s}.csv", export_path, file_name)))
    {
        init_file(full_path_general, rows_general, export_path, resume);
    }

    void save_header_general()
    {
        std::ofstream export_file(full_path_general, std::ofstream::app | std::ofstream::out);
        RLTSMetrics::dump_header(export_file);
    }

    template <typename RerooterSearchOutputT>
    void add_row_general(
        const SearchOutput<EnvT, RerooterSearchOutputT> &search_output,
        int bootstrap_iter,
        int search_budget
    )
    {
        rows_general.push_back({
            .iter = bootstrap_iter,
            .puzzle_name = search_output.puzzle_name,
            .solution_found = search_output.solution_found,
            .solution_cost = search_output.solution_cost,
            .solution_prob = search_output.solution_prob,
            .expanded = search_output.num_expanded,
            .generated = search_output.num_generated,
            .time = search_output.time,
            .budget = search_budget,
            .w_cumulative = search_output.w_cumulative,
            .log_slenderness_cost_sol = search_output.log_slenderness_cost_sol,
            .max_wt = search_output.max_wt,
            .max_log_slenderness_cost = search_output.max_log_slenderness_cost,
            .log_bound_simplified = search_output.log_bound_simplified,
            .log_bound_trivial = search_output.log_bound_trivial,
        });
    }

    void clear_general() noexcept
    {
        rows_general.clear();
    }

    void save_general() noexcept
    {
        if (rows_general.empty()) {
            return;
        }

        std::ofstream export_file(full_path_general, std::ofstream::app | std::ofstream::out);
        spdlog::info("Exporting metrics to {:s}", full_path_general);
        for (auto const &row : rows_general) {
            export_file << row;
        }
        export_file.close();
        rows_general.clear();
    }

    auto get_rows() -> const std::vector<RLTSMetrics> &
    {
        return rows_general;
    }

    auto size() -> int
    {
        return static_cast<int>(rows_general.size());
    }

protected:
    template <typename T>
    static void
        init_file(const std::string &full_path, std::vector<T> &rows, const std::string &export_path, bool resume)
    {
        if (std::filesystem::exists(full_path)) {
            if (resume) {
                std::ifstream export_file(full_path);
                std::string line;
                std::getline(export_file, line);
                while (std::getline(export_file, line)) {
                    rows.push_back(T::make_from_str(line));
                }
                export_file.close();
            }
            if (!resume) {
                std::filesystem::remove(full_path);
            }
        }
        std::filesystem::create_directories(export_path);
    }

    std::string full_path_general;
};

// Support for custom rerooting search output metrics
template <typename EnvT, typename RerooterSearchOutput = detail::NoRerooterSearchOutput>
struct RLTSMetricsTracker;

// Stub for dummy type when not provided
template <typename EnvT>
struct RLTSMetricsTracker<EnvT, detail::NoRerooterSearchOutput> : RLTSMetricsTrackerBase<EnvT> {
    RLTSMetricsTracker() = delete;
    RLTSMetricsTracker(const std::string &export_path, const std::string &file_name, bool resume = false)
        : RLTSMetricsTrackerBase<EnvT>(export_path, file_name, resume)
    {
        this->save_header_general();
    }

    void save_header()
    {
        this->save_header_general();
    }

    void add_row_by_result(
        const SearchOutput<EnvT, detail::NoRerooterSearchOutput> &search_output,
        int bootstrap_iter,
        int search_budget
    )
    {
        this->add_row_general(search_output, bootstrap_iter, search_budget);
    }

    void clear() noexcept
    {
        this->clear_general();
    }

    void save() noexcept
    {
        this->save_general();
    }
};

// Support for user provided rerooter search output statistics
// We log the general metrics + user provided
template <typename EnvT, typename RerooterSearchOutput>
    requires IsRerooterSearchOutput<RerooterSearchOutput>
struct RLTSMetricsTracker<EnvT, RerooterSearchOutput> : RLTSMetricsTrackerBase<EnvT> {
    using RerooterMetrics = decltype(std::declval<RerooterSearchOutput>().to_metric_items(
        std::declval<int>(),
        std::declval<std::string>()
    ))::value_type;
    std::vector<RerooterMetrics> rows_rerooter;

    RLTSMetricsTracker() = delete;
    RLTSMetricsTracker(const std::string &export_path, const std::string &file_name, bool resume = false)
        : RLTSMetricsTrackerBase<EnvT>(export_path, file_name, resume),
          full_path_weight((std::format("{:s}/{:s}_w.csv", export_path, file_name)))
    {
        this->init_file(full_path_weight, rows_rerooter, export_path, resume);
        save_header();
    }

    void save_header()
    {
        this->save_header_general();
        std::ofstream export_file(full_path_weight, std::ofstream::app | std::ofstream::out);
        RerooterMetrics::dump_header(export_file);
    }

    void add_row_by_result(
        const SearchOutput<EnvT, RerooterSearchOutput> &search_output,
        int bootstrap_iter,
        int search_budget
    )
    {
        this->add_row_general(search_output, bootstrap_iter, search_budget);
        rows_rerooter.append_range(
            search_output.rerooter_search_output.to_metric_items(bootstrap_iter, search_output.puzzle_name)
        );
    }

    void clear() noexcept
    {
        this->clear_general();
        rows_rerooter.clear();
    }

    void save() noexcept
    {
        if (this->rows_general.empty() && rows_rerooter.empty()) {
            return;
        }

        this->save_general();

        std::ofstream export_file(full_path_weight, std::ofstream::app | std::ofstream::out);
        spdlog::info("Exporting metrics to {:s}", full_path_weight);
        for (auto const &row : rows_rerooter) {
            export_file << row;
        }
        export_file.close();
        rows_rerooter.clear();
    }

private:
    std::string full_path_weight;
};

constexpr double EPS = 1e-12;

// Create subgrange [idx_start, idx_end) exclusive (like pythons list slicing)
// If idx_start == idx_end, then the subrange is empty
// If idx_end not provided, by default we take until the end
constexpr auto
    make_subrange(const std::vector<double> &items, int idx_start, const std::optional<int> &idx_end = std::nullopt)
{
    int _idx_end = idx_end.value_or(static_cast<int>(items.size()));
    assert(idx_start >= 0 && idx_start <= static_cast<int>(items.size()));
    assert(_idx_end >= 0 && _idx_end <= static_cast<int>(items.size()));
    assert(idx_start <= _idx_end);
    return items | std::views::drop(idx_start) | std::views::take(_idx_end - idx_start);
}

// Sum a range
template <typename T>
    requires std::ranges::range<T>
constexpr auto sum_range(T &&range)
{
    return std::ranges::fold_left(std::forward<T>(range), 0.0, std::plus<>{});
}

// Computes \lambda / \pi (n_idx_end ; n_idx_start) -> Formula (7)
// This uses the Numerical Stability formulation from Remark (43)
constexpr auto log_slenderness_cost_function(
    const std::vector<double> &path_log_probs,
    int idx_start,
    const std::optional<int> &idx_end = std::nullopt
) -> double
{
    double c = 1.0;
    auto sub_path_log_probs_view = make_subrange(path_log_probs, idx_start, idx_end);
    for (double lp : sub_path_log_probs_view) {
        c = c * std::exp(lp) + 1;
    }
    return std::log(c) - sum_range(sub_path_log_probs_view);
}

// log of (d(n) - d(n_k)) / pi(n | n_k)
constexpr auto log_d_pi_cost(const std::vector<double> &path_log_probs, int idx_start) -> double
{
    assert(static_cast<std::size_t>(idx_start) < path_log_probs.size());
    return std::log(static_cast<double>(static_cast<int>(path_log_probs.size()) - idx_start))
           - sum_range(make_subrange(path_log_probs, idx_start));
}

// Node used in search
template <IsEnv EnvT>
struct Node {
    Node() = delete;
    Node(const EnvT &state_)
        : state(state_)
    {}

    void apply_action(const Node<EnvT> *current, double c, int a)
    {
        parent = current;
        state.apply_action(a);
        double local_log_p = current->action_log_prob[static_cast<std::size_t>(a)];
        log_p = current->log_p + local_log_p;
        path_log_probs = parent->path_log_probs;
        path_log_probs.push_back(local_log_p);

        path_weights = parent->path_weights;
        path_weights.push_back(parent->w);
        path_cumulative_weights = parent->path_cumulative_weights;
        path_cumulative_weights.push_back(parent->w_cumulative);

        path_parents = parent->path_parents;
        path_parents.push_back(parent);
        g = current->g + c;
        action = a;
    }

    void set_cost(CostMode cost_mode)
    {
        assert(path_weights.size() == path_cumulative_weights.size());

        // Root base case
        if (parent == nullptr) {
            cost = 0;
            return;
        }

        assert(path_weights.size() >= 1);

        std::vector<double> costs;
        costs.reserve(static_cast<std::size_t>(g));
        for (auto &&[i, rooted_w] : std::views::zip(std::views::iota(static_cast<std::size_t>(0)), path_weights)) {
            double log_cr_cost = [&]() {
                switch (cost_mode) {
                case CostMode::Slenderness:
                    return log_slenderness_cost_function(path_log_probs, static_cast<int>(i));
                case CostMode::DPi:
                    return log_d_pi_cost(path_log_probs, static_cast<int>(i));
                }
                std::unreachable();
            }();

            // LTS rooted at root always has weight of 1
            costs.push_back(log_cr_cost - std::log((i == 0 ? 1.0 : rooted_w) + EPS));
        }

        cost = std::ranges::min(costs);
    }

    struct Hasher {
        using is_transparent = void;
        std::size_t operator()(const Node &node) const
        {
            return node.state.get_hash();
        }
        auto operator()(const Node *node) const -> std::size_t
        {
            return node->state.get_hash();
        }
    };
    struct CompareEqual {
        using is_transparent = void;
        bool operator()(const Node &lhs, const Node &rhs) const
        {
            return lhs.state == rhs.state;
        }
        auto operator()(const Node *lhs, const Node *rhs) const -> bool
        {
            return lhs->state == rhs->state;
        }
    };
    struct CompareOrderedGreater {
        auto operator()(const Node *lhs, const Node *rhs) const -> bool
        {
            return lhs->cost > rhs->cost;
        }
    };

    // NOLINTBEGIN (misc-non-private-member-variables-in-classes)
    EnvT state;
    double log_p = 0;
    double g = 0;
    double h = 1;
    mutable double w = 0;
    mutable double w_cumulative = 0;
    double cost = 0;
    const Node *parent = nullptr;
    int action = -1;
    int id{};
    mutable int expansion_number = -1;
    std::vector<double> action_log_prob;
    std::vector<double> path_weights;
    std::vector<double> path_cumulative_weights;
    std::vector<const Node *> path_parents;
    std::vector<double> path_log_probs;
    // NOLINTEND (misc-non-private-member-variables-in-classes)
};

enum class Status {
    INIT,
    OK,
    ERROR,
    TIMEOUT,
    SOLVED,
};

template <IsEnv EnvT, IsRLTSModel ModelT, typename RerooterT>
    requires IsRerooter<RerooterT, EnvT>
class RLTS {
    using NodeT = Node<EnvT>;
    using InputT = ModelT::InferenceInput;
    using OutputT = ModelT::InferenceOutput;
    using OpenListT = std::vector<const NodeT *>;
    using ClosedListT = absl::flat_hash_set<const NodeT *, typename NodeT::Hasher, typename NodeT::CompareEqual>;
    using RerooterSearchOutputT = typename RerooterSearchOutputType<RerooterT>::type;

public:
    RLTS(const SearchInput<EnvT, ModelT, RerooterT> &input_)
        : input(input_), status(Status::INIT), model(input.model), rerooter(input.rerooter)
    {
        reset();
    }

    // Initialize the search with root node inference output
    void init()
    {
        if (status != Status::INIT) {
            spdlog::error("Coroutine needs to be reset() before calling init()");
            throw std::logic_error("Coroutine needs to be reset() before calling init()");
        }

        const auto root_node_ptr = node_pool.emplace(input.state);
        root_node_ptr->id = ++node_id_counter;
        generated_nodes.insert(root_node_ptr);
        inference_nodes.push_back(root_node_ptr);
        batch_predict();

        rerooter.init(*root_node_ptr);

        status = Status::OK;
    }

    void reset()
    {
        status = Status::INIT;
        timeout = false;
        search_output = SearchOutput<EnvT, RerooterSearchOutputT>{.puzzle_name = input.puzzle_name};
        inference_nodes.clear();
        open.clear();
        node_pool.clear();
        closed.clear();
        generated_nodes.clear();
        node_id_counter = -1;
        cumulative_expansion_weights = 0;
        rerooter.reset();
    }

    void step()
    {
        if (open.empty()) {
            status = Status::ERROR;
            spdlog::error("Exhausted open list - name: {:s}, budget: {:d}.", input.puzzle_name, input.search_budget);
            return;
        }

        // Remove top node from open and put into closed
        ++search_output.num_expanded;
        std::ranges::pop_heap(open, typename NodeT::CompareOrderedGreater{});
        const NodeT *current = open.back();
        open.pop_back();
        closed.insert(current);
        current->expansion_number = search_output.num_expanded;

        rerooter.expanded(*current);

        // Set the weight for the current node, which children nodes will refer to for their cost
        // Ensure that the root always gets a weight of 1
        current->w = rerooter(*current);
        current->w = current->parent ? current->w : 1.0;
        cumulative_expansion_weights += current->w;
        current->w_cumulative = cumulative_expansion_weights;

        // Timeout
        if (input.search_budget >= 0 && search_output.num_expanded >= input.search_budget) {
            timeout = true;
            spdlog::info(
                "Budget timeout - name: {:s}, exp: {:d}, gen: {:d}, budget: {:d}",
                input.puzzle_name,
                search_output.num_expanded,
                search_output.num_generated,
                input.search_budget
            );
            status = Status::TIMEOUT;
            batch_predict();
            return;
        }

        // Consider all children
        for (int a : std::views::iota(0) | std::views::take(current->state.num_actions)) {
            NodeT child(current->state);
            child.apply_action(current, 1, a);

            // Self loop
            if (child.state == current->state) {
                continue;
            }

            // State is not solution but has separate terminal condition, we don't generate
            if (!child.state.is_solution() && child.state.is_terminal()) {
                continue;
            }

            // Previously generated
            if (generated_nodes.contains(&child)) {
                auto prev_generated = generated_nodes.find(&child);
                rerooter.prev_generated(*current, **prev_generated);
                continue;
            }

            child.id = ++node_id_counter;

            // Store in block and get ptr back
            auto child_node = node_pool.emplace(std::move(child));
            generated_nodes.insert(child_node);

            rerooter.generated(*current, *child_node);

            // Solution found, no optimality guarantees so we return on generation instead of expansion
            if (child_node->state.is_solution()) {
                spdlog::info(
                    "Solved - name: {:s}, exp: {:d}, gen: {:d}, budget: {:d}, c: {:.0f}",
                    input.puzzle_name,
                    search_output.num_expanded,
                    search_output.num_generated,
                    input.search_budget,
                    child_node->g
                );
                set_solution(*child_node);
                status = Status::SOLVED;
                rerooter.solution_found(*child_node);
                return;
            }

            inference_nodes.push_back(child_node);
        }

        // Batch inference
        if (open.empty() || inference_nodes.size() >= static_cast<std::size_t>(input.inference_batch_size)) {
            batch_predict();
        }
    }

    [[nodiscard]] Status get_status() const
    {
        return status;
    }

    [[nodiscard]] auto get_search_output() -> SearchOutput<EnvT, RerooterSearchOutputT>
    {
        // If user provided rerooter search output then grab it
        if constexpr (detail::HasRerooterSearchOutput<RerooterT>) {
            search_output.rerooter_search_output = rerooter.get_search_output();
        }
        return search_output;
    }

    void run()
    {
        TimerWall timer(-1);
        timer.start();
        init();
        while (get_status() == Status::OK && !input.stop_token->stop_requested()) {
            step();
        }
    }

private:
    void batch_predict()
    {
        if (inference_nodes.empty()) {
            return;
        }

        // Policy inference
        std::vector<InputT> model_inputs;
        model_inputs.reserve(inference_nodes.size());
        for (const auto &node : inference_nodes) {
            model_inputs.emplace_back(node->state.get_observation());
        }
        std::vector<OutputT> model_outputs = model->inference(model_inputs);

        // Insert current batch into open
        for (const auto &[node, model_output] : std::views::zip(inference_nodes, model_outputs)) {
            node->action_log_prob = log_policy_noise(std::move(model_output.policy), input.mix_epsilon);
            if constexpr (HasHeuristic<OutputT>) {
                node->h = std::max(model_output.heuristic, 0.0);
            }
            node->set_cost(input.cost_mode);
            open.push_back(node);
            ++search_output.num_generated;
        }

        rerooter.batch_inferenced();

        // Heapify open
        std::ranges::make_heap(open, typename NodeT::CompareOrderedGreater{});
        inference_nodes.clear();
    }

    void create_solution_data(const NodeT &node)
    {
        auto current = &node;
        double solution_cost = 0;
        search_output.solution_path_states.push_back(current->state);
        while (current->parent) {
            search_output.solution_path_states.push_back(current->parent->state);
            search_output.solution_path_actions.push_back(current->action);
            solution_cost += (current->g - current->parent->g);
            search_output.solution_path_costs.push_back(solution_cost);
            search_output.solution_path_w.push_back(current->parent->w);
            search_output.solution_path_w_cumulative.push_back(current->parent->w_cumulative);
            current = current->parent;
        }
        // Reverse so its in view of start -> finish
        std::ranges::reverse(search_output.solution_path_states);
        std::ranges::reverse(search_output.solution_path_actions);
        std::ranges::reverse(search_output.solution_path_costs);
        std::ranges::reverse(search_output.solution_path_w);
        std::ranges::reverse(search_output.solution_path_w_cumulative);
    }

    void set_solution(const NodeT &node)
    {
        status = Status::SOLVED;
        search_output.solution_found = true;
        search_output.solution_cost = node.g;
        search_output.solution_prob = std::exp(node.log_p);
        create_solution_data(node);
        // Search statistics
        search_output.w_cumulative = cumulative_expansion_weights;
        search_output.log_slenderness_cost_sol = log_slenderness_cost_function(node.path_log_probs, 0);
        // Corollary 12
        struct CostInfo {
            double wt;
            double log_slenderness_cost;
        };

        // Indices along root -> node path that have non-zero weights i.e. the subtask decomposition
        auto decomposition_indices = node.path_weights | std::views::enumerate
                                     | std::views::filter([](auto &&x) -> bool { return std::get<1>(x) != 0; })
                                     | std::views::transform([](auto &&x) { return static_cast<int>(std::get<0>(x)); })
                                     | std::ranges::to<std::vector>();
        // Guard against all zeros along path, so we get cost up to parent
        assert(decomposition_indices.size() > 0);
        if (decomposition_indices.size() == 1) {
            decomposition_indices.push_back(static_cast<int>(node.path_weights.size()) - 1);
        }

        std::vector<CostInfo> cost_infos;
        for (const auto &[idx_start, idx_end] : decomposition_indices | std::views::adjacent<2>) {
            cost_infos.emplace_back(
                node.path_weights.at(static_cast<std::size_t>(idx_start)),
                log_slenderness_cost_function(node.path_log_probs, idx_start, idx_end)
            );
        }
        const auto max_log_decomposition_cost = std::ranges::max(cost_infos, [](auto &&lhs, auto &&rhs) -> bool {
            return (lhs.log_slenderness_cost - std::log(lhs.wt)) < (rhs.log_slenderness_cost - std::log(rhs.wt));
        });
        search_output.max_wt = max_log_decomposition_cost.wt;
        search_output.max_log_slenderness_cost = max_log_decomposition_cost.log_slenderness_cost;
        search_output.log_bound_simplified =
            std::log(search_output.w_cumulative / search_output.max_wt + EPS) + search_output.max_log_slenderness_cost;
        search_output.log_bound_trivial = std::log(search_output.w_cumulative) + search_output.log_slenderness_cost_sol;
    }

    SearchInput<EnvT, ModelT, RerooterT> input;
    Status status{};             // Current search status
    bool timeout = false;        // Timeout flag on budget
    int node_id_counter = -1;    // Node ID counter
    std::shared_ptr<ModelT> model;
    RerooterT rerooter;
    SearchOutput<EnvT, RerooterSearchOutputT> search_output;    // Search output containing trajectory + stats
    std::vector<NodeT *> inference_nodes;                       // Nodes in queue for batch inference
    OpenListT open;                                             // Open list
    ClosedListT closed;                                         // Closed list
    ClosedListT generated_nodes;                                // open + closed (used for duplication checks)
    double cumulative_expansion_weights = 0;                    // W_<t tracker
    StablePool<NodeT> node_pool;                                // Stable storage + pointers for nodes
};

template <IsEnv EnvT, IsRLTSModel ModelT, typename RerooterT>
    requires IsRerooter<RerooterT, EnvT>
auto search(const SearchInput<EnvT, ModelT, RerooterT> &input)
{
    TimerWall timer(-1);
    RLTS<EnvT, ModelT, RerooterT> step_rlts(input);
    timer.start();
    step_rlts.init();
    // Iteratively search until status changes (solved or timeout)
    while (step_rlts.get_status() == Status::OK && !input.stop_token->stop_requested()) {
        step_rlts.step();
    }
    auto output = step_rlts.get_search_output();
    output.time = timer.get_duration();
    return output;
}

}    // namespace libpts::algorithm::rlts

#endif    // LIBPTS_ALGORITHM_RLTS_H_
