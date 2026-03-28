# Torch Model Wrappers

To enable this feature, set the `LIBPOLICYTS_BUILD_TORCH` cmake variable to `ON` if building directly,
or ensure you add with the `torch` feature if installing through vcpkg.

Model wrappers are wrappers around pytorch models, 
and provides an interface to perform inference and learn, 
which is compatible with all algorithms provided.

The following model wrappers are included:
- `binary_classifier_convnet_wrapper`: A ResNet-based convnet binary classifier
- `heuristic_convnet_wrapper`: A ResNet-based convnet with single policy head
- `policy_convnet_wrapper`: A ResNet-based convnet with single heuristic head
- `twoheaded_convnet_wrapper`: A ResNet-based convnet with a policy and heuristic head

Each model wrapper has the following inner structs:
- `Config`: Necessary info to construct
- `InferenceInput`: Input type for inference
- `InferenceOutput`: Output type for inference
- `LearningInput`: Input type for learning

## Usage

### Construction
Model wrappers can be constructed from a `Config` or from a `json` containing the same fields as the config.
A valid `json` type struct can be gotten from the method `.get_default_json_config()`, and then you can edit the entries.
Model wrappers will also have additional arguments for construction, as the number of actions for a policy network,
and the device.
An optional last argument is the `checkpoint_base_name`, used to differentiate multiple of the same model type when saving/loading model checkpoints.

```cpp
using PolicyModel = libpts::model::TwoHeadedConvNetWrapper;
auto model_config = PolicyModel::get_default_json_config();
model_config["resnet_channels"] = 16;
model_config["resnet_blocks"] = 2;
model_config["policy_channels"] = 2;
model_config["heuristic_channels"] = 2;
model_config["policy_mlp_layers"] = std::vector<int>{8, 8};
model_config["heuristic_mlp_layers"] = std::vector<int>{8, 8};
model_config["use_batchnorm"] = false;
model_config["learning_rate"] = 3e-4;
model_config["l2_weight_decay"] = 1e-4;

auto model = std::make_shared<PolicyModel>(
    model_config,
    start_state.observation_shape(),
    start_state.num_actions,
    "cuda:0",
    ""
);
```

The models provided are inference thread-safe, so its generally recommended in multi-threaded scenarios 
such as using the multi-threaded bootstrap training process to use a `shared_ptr` of the model wrapper.

### Saving and Loading

Saving and loading has an optional step parameter, useful for if you want to checkpoint various model steps.
```cpp
model->save_checkpoint();
model->load_checkpoint();
```


### Inference
Model inference requires a `std::vector<InferenceInput>` of input by non-const reference.
This is due to how torch requires mutable pointers to the underlying data to efficiently load into a tensor,
without manually assigning each individual value one-at-a-time.

```cpp
using InferenceInput = PolicyModel::InferenceInput;
using InferenceOutput = PolicyModel::InferenceInput;
std::vector<InferenceInput> inputs;
std::vector<InferenceOutput> outputs = model->inference(inputs);
```

### Learning
Learning should only be performed by a single thread, as this is not thread safe. 
The bootstrap training procedure calls `.learn()` after the current batch of problems are finished by the search algorithm.
If you want to train models using standard supervised learning, you can manually call learn in a training loop.

```cpp
using LearningInput = PolicyModel::LearningInput;
std::vector<LearningInput> inputs;
double loss = model->learn(inputs);
```
