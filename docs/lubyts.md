# LubyTS

LubyTS is a rollout sampling algorithm, which samples actions for rollouts of depth following the Luby universal sequence, 
then reset and performs another rollout if a solution is found.
This continues until the budget is exhausted.
- Orseau, Laurent, et al. "Single-agent policy tree search with guarantees." Advances in Neural Information Processing Systems 31 (2018).

LubyTS needs as input a valid environment and model with policy evaluations.

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

Models which evaluate policies need to support each of the following to satisfy the constraint.
- Has an inner type `InferenceInput`, which can be constructed from an `Observation`
- `inference` takes either a vector of `InferenceInput` and returns a vector of a struct such that it has an inner `std::vector<double> policy`, OR takes a single `InferenceInput` and returns a struct with an inner `std::vector<double> policy`.

All of the provided model wrappers satisfies this concept.

```cpp
template <typename T>
concept IsLubyModelValue = requires(T t, Observation &obs) {
    { t.inference(obs) } -> HasPolicy;
};

template <typename T>
concept IsLubyModelVector = requires(T t) {
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

// Luby model satisfies the following:
// Either takes an observation as input and returns a type holding a policy
// OR takes a vector of observations and returns a vector of types holding a policy
template <typename T>
concept IsLubyModel = IsLubyModelValue<T> || IsLubyModelVector<T>;
```

## Example

Full example in `examples/lubyts/`.

```cpp
// Policy which satisfies the constraint for LubyTS
template <int N>
struct Policy {
    static_assert(N >= 1);
    struct InferenceOutput {
        std::vector<double> policy;
    };

    [[nodiscard]] auto inference([[maybe_unused]] libpts::Observation &observations) const -> InferenceOutput {
        return {std::vector<double>(static_cast<std::size_t>(N), 1.0 / N)};
    }
};

using State = ...
namespace luby = libpts::algorithm::luby;
luby::SearchInput<State, Policy<4>> search_input{
    .puzzle_name = "puzzle_0",
    .state = start_state,
    .search_budget = budget,
    .mix_epsilon = 0.0,
    .seed = 0,
    .stop_token = stop_token,
    .policy_model = std::make_shared<SokobanPolicy>()
};
auto search_result = luby::search(search_input);
```

For an example using the library policy model,
```cpp
using Policy = libpts::model::PolicyConvNetWrapper;
auto model_config = Policy::get_default_json_config();
model_config["resnet_channels"] = 16;
model_config["resnet_blocks"] = 2;
model_config["policy_channels"] = 2;
model_config["policy_mlp_layers"] = std::vector<int>{8, 8};
model_config["use_batchnorm"] = false;
model_config["learning_rate"] = 3e-4;
model_config["l2_weight_decay"] = 1e-4;

// Policy which satisfies the constraint for LubyTS
auto model = std::make_shared<Policy>(
    model_config,
    start_state.observation_shape(),
    start_state.num_actions,
    "cpu",
    ""
);
luby::SearchInput<State, Policy> search_input{
    .puzzle_name = "puzzle_0",
    .state = start_state,
    .search_budget = budget,
    .mix_epsilon = 0.0,
    .seed = 0,
    .stop_token = stop_token,
    .policy_model = std::make_shared<SokobanPolicy>()
};
auto search_result = luby::search(search_input);
```
