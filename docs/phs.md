# PHS

PHS* is a best-first search algorithm which uses a modification of the LevinTS evaluation function $d/\pi$,
where $d$ is the depth and $\pi$ is the path probability,
by including heuristic information. 
The evaluation function provides a bound on the search effort required to reach a goal that goes through the given node.
- Orseau, Laurent, and Levi HS Lelis. "Policy-guided heuristic search with guarantees." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 35. No. 14. 2021.

PHS* needs as input a valid environment and model with both policy and heuristic evaluations.

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
- `inference` takes a vector of `InferenceInput` and returns a vector of a struct such that it has an inner `std::vector<double> policy` and `double heuristic`.

All of the provided model wrappers satisfies this concept.

```cpp
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
```

## Example

Full example in `examples/phs/`.

```cpp
// Policy + Heuristic which satisfies the constraint for PHS
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

using State = ...
namespace phs = libpts::algorithm::phs;
phs::SearchInput<State, PolicyAndHeuristic<4>> search_input{
    .puzzle_name = "puzzle_0",
    .state = start_state,
    .search_budget = 4000,
    .inference_batch_size = 1,
    .mix_epsilon = 0.0,
    .stop_token = stop_token,
    .model = std::make_shared<PolicyAndHeuristic<4>>()
};
auto search_result = phs::search(search_input);
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

// Policy+Heuristic which satisfies the constraint for PHS
auto model = std::make_shared<PolicyAndHeuristic>(
    model_config,
    start_state.observation_shape(),
    start_state.num_actions,
    "cpu",
    ""
);
phs::SearchInput<State, PolicyAndHeuristic> search_input{
    .puzzle_name = "puzzle_0",
    .state = start_state,
    .search_budget = 4000,
    .inference_batch_size = 1,
    .mix_epsilon = 0.0,
    .stop_token = stop_token,
    .model = std::make_shared<PolicyAndHeuristic>()
};
auto search_result = phs::search(search_input);
```
