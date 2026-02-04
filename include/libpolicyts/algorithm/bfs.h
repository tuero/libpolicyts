// File: bfs.h
// Description: Generic Best First Search

#ifndef LIBPTS_ALGORITHM_BFS_H_
#define LIBPTS_ALGORITHM_BFS_H_

#include <libpolicyts/block_allocator.h>
#include <libpolicyts/concepts.h>
#include <libpolicyts/math_util.h>
#include <libpolicyts/stop_token.h>
#include <libpolicyts/timer.h>

#include <absl/container/flat_hash_set.h>
#include <spdlog/spdlog.h>

#include <cstdint>
#include <memory>
#include <queue>
#include <ranges>
#include <string>

namespace libpts::algorithm::bfs {

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

// BFS model satisfies the following:
template <typename T>
concept IsBFSModel = requires(T t) {
    // Has an inner type called InferenceInput
    typename T::InferenceInput;
    // Which is constructable from an observation
    requires std::is_constructible_v<typename T::InferenceInput, Observation>;
    // Inference takes as input a vector of inference inputs and must return a std::vector<...>
    requires IsSpecialization<
        std::remove_cvref_t<decltype(t.inference(makeval<std::vector<typename T::InferenceInput> &>()))>,
        std::vector>;
    // Returned vector element type must satisfy HasHeuristic
    requires HasHeuristic<typename std::remove_cvref_t<
        decltype(t.inference(makeval<std::vector<typename T::InferenceInput> &>()))>::value_type>;
};

// Input to BFS search algorithm
template <IsEnv EnvT, IsBFSModel ModelT>
struct SearchInput {
    std::string puzzle_name;
    EnvT state;
    int search_budget = 1;
    int inference_batch_size = 1;
    double weight_g = 1;
    double weight_h = 0;
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
    double time = 0;
    std::vector<EnvT> solution_path_states{};
    std::vector<int> solution_path_actions{};
    std::vector<double> solution_path_costs{};
};

constexpr int BLOCK_ALLOCATION_SIZE = 2000;
constexpr int INVALID_ACTION = -1;
constexpr double DEFAULT_HEURISTIC = 0.0;
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
    EnvT state;                              // State the node represents
    double g = 0;                            // Path cost from root to this
    mutable double h = DEFAULT_HEURISTIC;    // Heuristic value from this to a goal node
    mutable double cost = DEFAULT_COST;      // BFS cost being g*weight_g + h*weight_h
    const Node *parent = nullptr;            // Parent node
    int action = INVALID_ACTION;             // Action taken from parent
    // NOLINTEND (misc-non-private-member-variables-in-classes)
};

enum class Status {
    INIT,
    OK,
    ERROR,
    TIMEOUT,
    SOLVED,
};

template <IsEnv EnvT, IsBFSModel ModelT>
class BFS {
    using InferenceInputT = ModelT::InferenceInput;
    using NodeT = Node<EnvT>;
    using OpenListT =
        std::priority_queue<const NodeT *, std::vector<const NodeT *>, typename NodeT::CompareOrderedGreater>;
    using ClosedListT = absl::flat_hash_set<const NodeT *, typename NodeT::Hasher, typename NodeT::CompareEqual>;

public:
    BFS(const SearchInput<EnvT, ModelT> &input_problem)
        : input(input_problem),
          status(Status::INIT),
          model(input.model),
          node_allocator(BLOCK_ALLOCATION_SIZE, input.state) {
        reset();
    }

    // Initialize the search with root node inference output
    void init() {
        if (status != Status::INIT) {
            SPDLOG_ERROR("Coroutine needs to be reset() before calling init()");
            throw std::logic_error("Coroutine needs to be reset() before calling init()");
        }
        NodeT root_node(input.state);
        const auto root_node_ptr = node_allocator.add(root_node);
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
        node_allocator.clear();
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
            child.parent = current;
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
            const auto child_node_ptr = node_allocator.add(std::move(child));
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
            child_node->h = std::max(prediction.heuristic, 0.0);
            child_node->cost = (input.weight_g * child_node->g) + (input.weight_h * child_node->h);
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

    SearchInput<EnvT, ModelT> input;               // Search input, containing problem instance, models, budget, etc.
    Status status{};                               // Current search status
    bool timeout = false;                          // Timeout flag on budget
    std::shared_ptr<ModelT> model;                 // Policy network with optional heuristic
    SearchOutput<EnvT> search_output;              // Output of the search algorithm, containing trajectory + stats
    std::vector<const NodeT *> inference_nodes;    // Nodes in queue for batch inference
    OpenListT open;                                // Open list
    ClosedListT closed;                            // Closed list
    ClosedListT generated_nodes;                   // Open + Closed list
    BlockAllocator<NodeT, typename NodeT::Hasher, typename NodeT::CompareEqual> node_allocator;
};

template <IsEnv EnvT, IsBFSModel ModelT>
auto search(const SearchInput<EnvT, ModelT> &input) -> SearchOutput<EnvT> {
    TimerWall timer(-1);
    BFS<EnvT, ModelT> step_bfs(input);
    step_bfs.init();
    timer.start();
    // Iteratively search until status changes (solved or timeout)
    while (step_bfs.get_status() == Status::OK && (!input.stop_token || !input.stop_token->stop_requested())) {
        step_bfs.step();
    }
    auto output = step_bfs.get_search_output();
    output.time = timer.get_duration();
    return output;
}

}    // namespace libpts::algorithm::bfs

#endif    // LIBPTS_ALGORITHM_BFS_H_
