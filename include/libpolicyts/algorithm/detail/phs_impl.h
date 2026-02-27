// File: phs_impl.h
// Description: Generic LevinTS and PHS* implementation
// LevinTS: Orseau, Laurent, et al. "Single-agent policy tree search with guarantees." Advances in Neural Information
//          Processing Systems 31 (2018).
// PHS*: Orseau, Laurent, and Levi HS Lelis. "Policy-guided heuristic search with guarantees." Proceedings of the AAAI
//       Conference on Artificial Intelligence. Vol. 35. No. 14. 2021.

#ifndef LIBPTS_ALGORITHM_DETAIL_PHS_H_
#define LIBPTS_ALGORITHM_DETAIL_PHS_H_

#include <libpolicyts/concepts.h>
#include <libpolicyts/math_util.h>
#include <libpolicyts/stable_pool.h>
#include <libpolicyts/stop_token.h>
#include <libpolicyts/timer.h>

#include <absl/container/flat_hash_set.h>
#include <spdlog/spdlog.h>

#include <cstdint>
#include <memory>
#include <queue>
#include <ranges>
#include <string>

namespace libpts::algorithm::phs::detail {

// Concept for simple states for search
template <typename T>
concept IsEnv = std::equality_comparable<T> && IsSTDHashable<T> && requires(T t, const T ct, const std::string &s) {
    { t.apply_action(makeval<int>()) } -> std::same_as<double>;    // apply_action with int action and returns cost
    { ct.get_observation() } -> std::same_as<Observation>;         // Observation for policy/heuristic inference
    { ct.get_hash() } -> std::same_as<uint64_t>;                   // get hash
    { ct.is_solution() } -> std::same_as<bool>;                    // Solution check
    { ct.is_terminal() } -> std::same_as<bool>;                    // Terminal check (both solution + non-solution)
    *(&T::num_actions) == makeval<int>();                          // Number of actions
};

// LTS model satisfies the following:
template <typename T>
concept IsLTSModel = requires(T t) {
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

// PHS model satisfies the following:
template <typename T>
concept IsPHSModel = requires(T t) {
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
    // Returned vector element type must satisfy HasHeuristic
    requires HasHeuristic<typename std::remove_cvref_t<
        decltype(t.inference(makeval<std::vector<typename T::InferenceInput> &>()))>::value_type>;
};

template <typename T>
concept IsModel = IsLTSModel<T> || IsPHSModel<T>;

// Input to PHS search algorithm
template <IsEnv EnvT, IsModel ModelT>
struct SearchInput {
    std::string puzzle_name;
    EnvT state;
    int search_budget = 1;
    int inference_batch_size = 1;
    double mix_epsilon = 0;
    std::shared_ptr<StopToken> stop_token;
    std::shared_ptr<ModelT> model;
};

// Search algorithm output
template <IsEnv EnvT>
struct SearchOutput {
    std::string puzzle_name;
    bool solution_found = false;
    double solution_cost = -1;
    int num_expanded = 0;
    int num_generated = 0;
    double solution_prob = 0;
    double time = 0;
    std::vector<EnvT> solution_path_states{};
    std::vector<int> solution_path_actions{};
    std::vector<double> solution_path_costs{};
};

constexpr double EPS = 1e-8;
constexpr int INVALID_ACTION = -1;
constexpr double DEFAULT_COST = 0.0;

// Node used in search
template <IsEnv EnvT>
struct Node {
    Node() = delete;
    Node(const EnvT &state_)
        : state(state_) {}

    // Apply action, set parent and costs
    void apply_action(const Node<EnvT> *par, int a) {
        parent = par;
        auto c = state.apply_action(a);
        log_p = parent->log_p + parent->log_policy[static_cast<std::size_t>(a)];
        g = parent->g + c;
        action = a;
    }

    struct Hasher {
        using is_transparent = void;
        std::size_t operator()(const Node &node) const {
            return node.state.get_hash();
        }
        auto operator()(const Node *node) const -> std::size_t {
            return node->state.get_hash();
        }
    };
    struct CompareEqual {
        using is_transparent = void;
        bool operator()(const Node &lhs, const Node &rhs) const {
            return lhs.state == rhs.state;
        }
        auto operator()(const Node *lhs, const Node *rhs) const -> bool {
            return lhs->state == rhs->state;
        }
    };
    struct CompareOrderedLess {
        auto operator()(const Node *lhs, const Node *rhs) const -> bool {
            return lhs->cost < rhs->cost;
        }
    };
    struct CompareOrderedGreater {
        auto operator()(const Node *lhs, const Node *rhs) const -> bool {
            return lhs->cost > rhs->cost;
        }
    };

    // NOLINTBEGIN (misc-non-private-member-variables-in-classes)
    EnvT state;                          // State the node represents
    double log_p = 0;                    // Log path probability from root to this
    double g = 0;                        // Path cost from root to this
    double h = 0;                        // Heuristic value from this to a goal node
    double cost = DEFAULT_COST;          // LTS/PHS cost
    const Node *parent = nullptr;        // Parent node
    int action = INVALID_ACTION;         // Action taken from parent
    std::vector<double> log_policy{};    // Local log policy over child actions
    // NOLINTEND (misc-non-private-member-variables-in-classes)
};

// PHS cost
static constexpr double phs_cost(double log_p, double g, double h) {
    h = (h < 0) ? 0 : h;
    return g == 0 ? 0 : std::log(h + g + EPS) - (log_p * (1.0 + (h / g)));
}

enum class Status {
    INIT,
    OK,
    ERROR,
    TIMEOUT,
    SOLVED,
};

template <IsEnv EnvT, IsModel ModelT>
class PHS {
    using InferenceInputT = ModelT::InferenceInput;
    using NodeT = Node<EnvT>;
    using OpenListT =
        std::priority_queue<const NodeT *, std::vector<const NodeT *>, typename NodeT::CompareOrderedGreater>;
    using ClosedListT = absl::flat_hash_set<const NodeT *, typename NodeT::Hasher, typename NodeT::CompareEqual>;

public:
    PHS(const SearchInput<EnvT, ModelT> &input_problem)
        : input(input_problem), status(Status::INIT), model(input.model) {
        reset();
    }

    // Initialize the search with root node inference output
    void init() {
        if (status != Status::INIT) {
            SPDLOG_ERROR("Coroutine needs to be reset() before calling init()");
            throw std::logic_error("Coroutine needs to be reset() before calling init()");
        }
        const auto root_node_ptr = node_pool.emplace(input.state);
        generated_nodes.insert(root_node_ptr);
        inference_nodes.push_back(root_node_ptr);
        batch_predict();
        status = Status::OK;
    }

    void reset() {
        status = Status::INIT;
        timeout = false;
        search_output = SearchOutput<EnvT>{.puzzle_name = input.puzzle_name};
        inference_nodes.clear();
        {
            decltype(open) empty;
            std::swap(open, empty);
        }
        node_pool.clear();
        closed.clear();
        generated_nodes.clear();
    }

    void step() {
        if (open.empty()) {
            status = Status::ERROR;
            spdlog::error("Exhausted open list - name: {:s}, budget: {:d}.", input.puzzle_name, input.search_budget);
            return;
        }

        // Remove top node from open and put into closed
        const auto current = open.top();
        open.pop();
        closed.insert(current);
        ++search_output.num_expanded;

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
            return;
        }

        // Consider all children
        for (int a : std::views::iota(0) | std::views::take(current->state.num_actions)) {
            NodeT child = *current;
            child.apply_action(current, a);

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
                continue;
            }

            // Store in block and get ptr back
            auto child_node_ptr = node_pool.emplace(std::move(child));
            generated_nodes.insert(child_node_ptr);

            // Solution found, no optimality guarantees so we return on generation instead of expansion
            if (child_node_ptr->state.is_solution()) {
                spdlog::info(
                    "Solved - name: {:s}, exp: {:d}, gen: {:d}, budget: {:d}, c: {:.0f}",
                    input.puzzle_name,
                    search_output.num_expanded,
                    search_output.num_generated,
                    input.search_budget,
                    child_node_ptr->g
                );
                set_solution_trajectory(*child_node_ptr);
                status = Status::SOLVED;
                return;
            }

            inference_nodes.push_back(std::move(child_node_ptr));
        }

        // Batch inference
        if (open.empty() || inference_nodes.size() >= static_cast<std::size_t>(input.inference_batch_size)) {
            batch_predict();
        }
    }

    [[nodiscard]] auto get_status() const -> Status {
        return status;
    }

    [[nodiscard]] auto get_search_output() const -> SearchOutput<EnvT> {
        return search_output;
    }

private:
    void batch_predict() {
        if (inference_nodes.empty()) {
            return;
        }

        // Create inference inputs
        std::vector<InferenceInputT> inference_inputs;
        inference_inputs.reserve(inference_nodes.size());
        for (const auto &node : inference_nodes) {
            inference_inputs.emplace_back(node->state.get_observation());
        }

        auto predictions = model->inference(inference_inputs);
        if (predictions.size() != inference_nodes.size()) [[unlikely]] {
            spdlog::error(
                "Inference returned {} predictions for {} inputs.",
                predictions.size(),
                inference_nodes.size()
            );
            status = Status::ERROR;
            return;
        }
        for (auto &&[child_node, prediction] : std::views::zip(inference_nodes, predictions)) {
            if constexpr (HasHeuristic<decltype(prediction)>) {
                child_node->h = std::max(prediction.heuristic, 0.0);
            }
            if (prediction.policy.size() != EnvT::num_actions) [[unlikely]] {
                spdlog::error(
                    "Received policy of size {} for environment with {} actions.",
                    prediction.policy.size(),
                    EnvT::num_actions
                );
                if (input.stop_token) {
                    input.stop_token->stop();
                }
                status = Status::ERROR;
                return;
            }
            child_node->log_policy = std::move(prediction.policy);
            log_policy_noise_inplace(child_node->log_policy, input.mix_epsilon);
            child_node->cost = phs_cost(child_node->log_p, child_node->g, child_node->h);
            open.push(child_node);
            ++search_output.num_generated;
        }
        inference_nodes.clear();
    }

    // Walk backwards up until the root, setting data
    void set_solution_trajectory(const NodeT &node) {
        double solution_cost = 0;
        auto current = &node;
        search_output.solution_found = true;
        search_output.solution_cost = current->g;
        search_output.solution_prob = std::exp(current->log_p);
        search_output.solution_path_states.push_back(current->state);
        while (current->parent) {
            search_output.solution_path_states.push_back(current->parent->state);
            search_output.solution_path_actions.push_back(current->action);
            solution_cost += (current->g - current->parent->g);
            search_output.solution_path_costs.push_back(solution_cost);
            current = current->parent;
        }
        // Reverse so its in view of start -> finish
        std::ranges::reverse(search_output.solution_path_states);
        std::ranges::reverse(search_output.solution_path_actions);
        std::ranges::reverse(search_output.solution_path_costs);
    }

    SearchInput<EnvT, ModelT> input;         // Search input, containing problem instance, models, budget, etc.
    Status status{};                         // Current search status
    bool timeout = false;                    // Timeout flag on budget
    std::shared_ptr<ModelT> model;           // Policy network with optional heuristic
    SearchOutput<EnvT> search_output;        // Output of the search algorithm, containing trajectory + stats
    std::vector<NodeT *> inference_nodes;    // Nodes in queue for batch inference
    OpenListT open;                          // Open list
    ClosedListT closed;                      // Closed list
    ClosedListT generated_nodes;             // Open + Closed list
    StablePool<NodeT> node_pool;
};

}    // namespace libpts::algorithm::phs::detail

#endif    // LIBPTS_ALGORITHM_DETAIL_PHS_H_
