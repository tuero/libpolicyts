# Best First Search

Best First Search needs as input a valid environment and model with heuristic evaluations.

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
A node is pruned $n'$ is pruned if there exists a node $n$ previously expanded with the same underlying state.

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

Models which evaluate heuristics need to support each of the following to satisfy the constraint.
- Has an inner type `InferenceInput`, which can be constructed from an `Observation`
- `inference` takes a vector of `InferenceInput` and returns a vector of a struct such that it has an inner `double heuristic`.

All of the provided model wrappers satisfies this concept.

```cpp
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
```

## Example

Full example in `examples/bfs/`.
The search input has arguments for both the weight of the g-cost and weight on the h-cost to produce 
various best-first search algorithms, like greedy best-first search, uniform cost search, and A*.

```cpp
// Heuristic which satisfies the constraint for bfs
struct Heuristic {
    struct InferenceInput {
        libpts::Observation obs;
    };
    struct InferenceOutput {
        double heuristic;
    };

    using InferenceInputs = std::vector<InferenceInput>;
    [[nodiscard]] auto inference(InferenceInputs &observations) const -> std::vector<InferenceOutput> {
        std::vector<InferenceOutput> inference_heuristic;
        inference_heuristic.reserve(observations.size());
        // Heuristic value of 0
        for ([[maybe_unused]] const auto &obs : observations) {
            inference_heuristic.emplace_back(0.0);
        }
        return inference_heuristic;
    }
};

using State = ...
namespace bfs = libpts::algorithm::bfs;
bfs::SearchInput<State, Heuristic> search_input{
    .puzzle_name = "puzzle_0",
    .state = start_state,
    .search_budget = 4000,
    .inference_batch_size = 1,
    .weight_g = 1.0,
    .weight_h = 1.0,
    .stop_token = stop_token,
    .model = std::make_shared<Heuristic>()
};
auto search_result = bfs::search(search_input);
```

For an example using the library heuristic model,
```cpp
using Heuristic = libpts::model::HeuristicConvNetWrapper;
auto model_config = Heuristic::get_default_json_config();
model_config["resnet_channels"] = 16;
model_config["resnet_blocks"] = 2;
model_config["heuristic_channels"] = 2;
model_config["heuristic_mlp_layers"] = std::vector<int>{8, 8};
model_config["use_batchnorm"] = false;
model_config["learning_rate"] = 3e-4;
model_config["l2_weight_decay"] = 1e-4;

// Heuristic which satisfies the constraint for BFS
auto model = std::make_shared<Heuristic>(
    model_config,
    start_state.observation_shape(),
    "cpu",
    ""
);
bfs::SearchInput<State, Heuristic> search_input{
    .puzzle_name = "puzzle_0",
    .state = start_state,
    .search_budget = 4000,
    .inference_batch_size = 1,
    .weight_g = 1.0,
    .weight_h = 1.0,
    .stop_token = stop_token,
    .model = model 
};
auto search_result = bfs::search(search_input);
```

