# $\sqrt{\mathrm{LTS}}$

$\sqrt{\mathrm{LTS}}$ is a best-first search algorithm which instantiates an LTS (LevinTS) instance at each node, weighted by the rerooter.
This algorithm can make use of information LTS cannot, which can provide exponential savings in the best case.
- Orseau, Laurent, Marcus Hutter, and Levi HS Lelis. "Exponential Speedups by Rerooting Levin Tree Search." (2024).

$\sqrt{\mathrm{LTS}}$ needs as input a valid environment, model with both policy and heuristic evaluations,
and a rerooter.

## Cost Mode
There are two cost modes which are supported
```cpp
enum class CostMode {
    Slenderness,
    DPi,
};
```

### Slenderness
This uses the slenderness cost function given in Orseau, Laurent, Marcus Hutter, and Levi HS Lelis. "Exponential Speedups by Rerooting Levin Tree Search." (2024).

### DPi
This uses the rerooted Levin cost.

## Pruning Policies
The search input has a `PruningPolicy` field to control the level of pruning:
```cpp
// How aggressive is the pruning
enum class PruningPolicy {
    None, 
    Passive,  
    Eager,
};
```

### None
No state-pruning is done, and solution check is performed at expansion.

### Passive
States are lazily pruned at expansion, and solution checks performed at expansion. 
A node $n'$ is pruned if there is a previously expanded node $n$ representing the same underlying state which has a lower cost than $n'$, and every ancestor of $n'$ is *dominated* by some ancestor of $n$.

For a node $n$ and an ancestor $a$, the **growth** is $\pi(a) / (w_a \pi(n))$, and the base is the cost of the node (either the $d/\pi$ cost or *slenderness* cost).
An ancestor of $a'$ of $n'$ is dominated by an ancestor $a$ of $n$ if $\text{base}(n, a) \le \text{base}(n',a')$ and $\text{growth}(n, a) \le \text{growth}(n',a')$.

> [!IMPORTANT]
> This is an $O(N^2)$ check where $N$ is the depth of the node being considered.


### Eager
A state is pruned during generation if a node representing the same state has been previously generated, and solution checks are done at generation.



## Environment

Environment objects need to support each of the following to satisfy the constraint
- Supports standard equality and hash operators
- `apply_action` updates the internal state by applying the action and returns the step cost
- `get_observation` gets the flat state observation, as a vector of float
- `get_hash` gets the hash of the underlying state
- `is_solution` checks for a solution state
- `is_terminal` checks for terminal non-solution states (this is often just calling `is_solution`)

All the environments in `libpolicyts/env/` satisfies this concept.

```cpp
template <typename T>
concept IsEnv = std::equality_comparable<T> && IsSTDHashable<T> && requires(T t, const T ct, const std::string &s) {
    { t.apply_action(makeval<int>()) } -> std::same_as<double>;    // apply_action with int action and returns cost
    { ct.get_observation() } -> std::same_as<Observation>;         // Observation for policy/heuristic inference
    { ct.get_hash() } -> std::same_as<uint64_t>;                   // get hash
    { ct.is_solution() } -> std::same_as<bool>;                    // Solution check
    { ct.is_terminal() } -> std::same_as<bool>;                    // Terminal check (both solution + non-solution)
    *(&T::num_actions) == makeval<int>();                          // Number of actions
};
```

## Model

Models which evaluate policies and heuristics need to support each of the following to satisfy the constraint.
- Has an inner type `InferenceInput`, which can be constructed from an `Observation`
- `inference` takes a vector of `InferenceInput` and returns a vector of a struct such that it has an inner `std::vector<double> policy` and an optional `double heuristic`. While the cost function doesn't directly use the heuristic, rerooters may want that information.

All of the provided model wrappers satisfies this concept.

```cpp
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
```

## Rerooter

The rerooter provides the weights for which each LTS instance is weighted by, which represents the portion of total time share.
The reooter has several endpoints which the algorithm will use as hooks, and your rerooter may want to take advantage of for internal tracking.
- `reset` called when the search algorithm instance is reset
- `init` signals the search is initialized with the given root node
- `expanded` signals the search has expanded the given node
- `generated` signals the current node has generated the child node
- `prev_generated` signals the current node has previously generated a node representing the child node state
- `batch_inferenced` signals that batch inference for policy evaluation was just performed
- `solution_found` signals that the search found a solution at the given node
- The function call operator for a given node should produce the rerooting weight for that node.

The rerooter can also internally track search statistics related to the weight values. 
An typical use case is that you want to track the weights found along the solution path.
For this to be logged by the search algorithm, you need to implement `get_search_output` which returns a vector of types that 
is a valid metrics item (i.e. satisfy the `IsMetricsItem`).

If you do not need this feature, then simply do not declare a `get_search_output` method.

```cpp
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
```

## Example

Full example in `examples/rlts/`.

```cpp
// Policy + Heuristic which satisfies the constraint for RLTS
template <int N>
struct PolicyAndHeuristic {
    static_assert(N >= 1);
    struct InferenceInput {
        libpts::Observation obs;
    };
    struct InferenceOutput {
        std::vector<double> policy;
        double heuristic;
    };

    using InferenceInputs = std::vector<InferenceInput>;
    [[nodiscard]] auto inference(InferenceInputs &observations) const -> std::vector<InferenceOutput> {
        std::vector<InferenceOutput> inference_policies;
        inference_policies.reserve(observations.size());
        // Uniform policy over each action
        // Heuristic value of 0
        for ([[maybe_unused]] const auto &obs : observations) {
            inference_policies.emplace_back(std::vector<double>(static_cast<std::size_t>(N), 1.0 / N), 0);
        }
        return inference_policies;
    }
};

// Simple rerooter which produces 0 weights everywhere
// The RLTS algorithm will ensure the root gets a weight of 1
struct Rerooter {
    void reset() {}
    void init([[maybe_unused]] const NodeT &node) {}
    void expanded([[maybe_unused]] const NodeT &node) {}
    void generated([[maybe_unused]] const NodeT &current_node, [[maybe_unused]] const NodeT &child_node) {}
    void prev_generated([[maybe_unused]] const NodeT &current_node, [[maybe_unused]] const NodeT &child_node) {}
    auto operator()([[maybe_unused]] const NodeT &node) -> double {
        return 0;
    }
    void batch_inferenced() {}
    void solution_found([[maybe_unused]] const NodeT &node) {}
};

using State = ...
namespace rlts = libpts::algorithm::rlts;
rlts::SearchInput<State, PolicyAndHeuristic<4>, Rerooter> search_input{
    .puzzle_name = "puzzle_0",
    .state = start_state,
    .search_budget = 4000,
    .inference_batch_size = 1,
    .mix_epsilon = 0.0,
    .stop_token = stop_token,
    .model = std::make_shared<PolicyAndHeuristic<4>>(),
    .rerooter = Rerooter{}
};
auto search_result = rlts::search(search_input);
```

For an example using the library policy+heuristic model,
```cpp
using PolicyAndHeuristic = libpts::model::TwoHeadedConvNetWrapper;
auto model_config = PolicyAndHeuristic::get_default_json_config();
model_config["resnet_channels"] = 16;
model_config["resnet_blocks"] = 2;
model_config["policy_channels"] = 2;
model_config["heuristic_channels"] = 2;
model_config["policy_mlp_layers"] = std::vector<int>{8, 8};
model_config["heuristic_mlp_layers"] = std::vector<int>{8, 8};
model_config["use_batchnorm"] = false;
model_config["learning_rate"] = 3e-4;
model_config["l2_weight_decay"] = 1e-4;

// Policy+Heuristic which satisfies the constraint for RLTS
auto model = std::make_shared<PolicyAndHeuristic>(
    model_config,
    start_state.observation_shape(),
    start_state.num_actions,
    "cpu",
    ""
);
rlts::SearchInput<State, PolicyAndHeuristic<4>, Rerooter> search_input{
    .puzzle_name = "puzzle_0",
    .state = start_state,
    .search_budget = 4000,
    .inference_batch_size = 1,
    .mix_epsilon = 0.0,
    .stop_token = stop_token,
    .model = std::make_shared<PolicyAndHeuristic<4>>(),
    .rerooter = Rerooter{}
};
auto search_result = rlts::search(search_input);
```
