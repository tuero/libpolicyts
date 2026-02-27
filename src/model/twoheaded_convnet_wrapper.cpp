// File: twoheaded_convnet_wrapper.cpp
// Description: Convnet wrapper for policy + heuristic net

#include <libpolicyts/model/torch_util.h>
#include <libpolicyts/model/twoheaded_convnet_wrapper.h>

// NOLINTBEGIN
#include <absl/strings/str_cat.h>
// NOLINTEND

#include <spdlog/spdlog.h>

#include <filesystem>
#include <format>
#include <ostream>
#include <ranges>
#include <sstream>

namespace libpts::model {
namespace {
void check_config_key_exits(const nlohmann::json &model_config, const std::string &key) {
    if (!model_config.contains(key)) {
        spdlog::error("model config json should contain an entry '{}'", key);
        std::exit(1);
    }
}

auto config_from_json(const nlohmann::json &model_config, const ObservationShape &obs_shape, int num_actions)
    -> TwoHeadedConvNetWrapper::Config {
    // Check for valid json
    if (!model_config.contains("model_type")
        || model_config["model_type"].get<std::string>() != TwoHeadedConvNetWrapper::name)
    {
        spdlog::error("model config json should contain an entry 'model_type': '{}'", TwoHeadedConvNetWrapper::name);
        std::exit(1);
    }
    check_config_key_exits(model_config, "resnet_channels");
    check_config_key_exits(model_config, "resnet_blocks");
    check_config_key_exits(model_config, "policy_channels");
    check_config_key_exits(model_config, "heuristic_channels");
    check_config_key_exits(model_config, "policy_mlp_layers");
    check_config_key_exits(model_config, "heuristic_mlp_layers");
    check_config_key_exits(model_config, "use_batchnorm");
    check_config_key_exits(model_config, "learning_rate");
    check_config_key_exits(model_config, "l2_weight_decay");

    return {
        .observation_shape = obs_shape,
        .num_actions = num_actions,
        .resnet_channels = model_config["resnet_channels"].template get<int>(),
        .resnet_blocks = model_config["resnet_blocks"].template get<int>(),
        .policy_channels = model_config["policy_channels"].template get<int>(),
        .heuristic_channels = model_config["heuristic_channels"].template get<int>(),
        .policy_mlp_layers = model_config["policy_mlp_layers"].template get<std::vector<int>>(),
        .heuristic_mlp_layers = model_config["heuristic_mlp_layers"].template get<std::vector<int>>(),
        .use_batchnorm = model_config["use_batchnorm"].template get<bool>(),
        .learning_rate = model_config["learning_rate"].template get<double>(),
        .l2_weight_decay = model_config["l2_weight_decay"].template get<double>(),
    };
}
}    // namespace

TwoHeadedConvNetWrapper::TwoHeadedConvNetWrapper(
    Config config_,
    const std::string &device,
    const std::string &output_path,
    const std::string &checkpoint_base_name
)
    : BaseModelWrapper(device, output_path, checkpoint_base_name),
      config(std::move(config_)),
      model_(
          config.observation_shape,
          config.num_actions,
          config.resnet_channels,
          config.resnet_blocks,
          config.policy_channels,
          config.heuristic_channels,
          config.policy_mlp_layers,
          config.heuristic_mlp_layers,
          config.use_batchnorm
      ),
      model_optimizer_(
          model_->parameters(),
          torch::optim::AdamOptions(config.learning_rate).weight_decay(config.l2_weight_decay)
      ),
      input_flat_size(config.observation_shape.flat_size()),
      num_actions(config.num_actions) {
    model_->to(torch_device_);
};

TwoHeadedConvNetWrapper::TwoHeadedConvNetWrapper(
    const nlohmann::json &model_config_json,
    const ObservationShape &obs_shape,
    int _num_actions,
    const std::string &device,
    const std::string &output_path,
    const std::string &checkpoint_base_name
)
    : BaseModelWrapper(device, output_path, checkpoint_base_name),
      config(config_from_json(model_config_json, obs_shape, _num_actions)),
      model_(
          config.observation_shape,
          config.num_actions,
          config.resnet_channels,
          config.resnet_blocks,
          config.policy_channels,
          config.heuristic_channels,
          config.policy_mlp_layers,
          config.heuristic_mlp_layers,
          config.use_batchnorm
      ),
      model_optimizer_(
          model_->parameters(),
          torch::optim::AdamOptions(config.learning_rate).weight_decay(config.l2_weight_decay)
      ),
      input_flat_size(config.observation_shape.flat_size()),
      num_actions(config.num_actions) {
    model_->to(torch_device_);
};

void TwoHeadedConvNetWrapper::print() const {
    std::ostringstream oss;
    std::ostream &os = oss;
    os << *model_;
    spdlog::info("{:s}", oss.str());
    std::size_t num_params = 0;
    for (const auto &p : model_->parameters()) {
        num_params += static_cast<std::size_t>(p.numel());
    }
    spdlog::info("Number of parameters: {:d}", num_params);
}

auto TwoHeadedConvNetWrapper::save_checkpoint(long long int step) -> std::string {
    // create directory for model
    std::filesystem::create_directories(path_);
    std::string full_path = absl::StrCat(path_, checkpoint_base_name_, "checkpoint-", step);
    spdlog::info("Checkpointing model to {:s}.pt", full_path);
    torch::save(model_, absl::StrCat(full_path, ".pt"));
    torch::save(model_optimizer_, absl::StrCat(full_path, "-optimizer.pt"));
    return full_path;
}
auto TwoHeadedConvNetWrapper::save_checkpoint_without_optimizer(long long int step) -> std::string {
    // create directory for model
    std::filesystem::create_directories(path_);
    std::string full_path = absl::StrCat(path_, checkpoint_base_name_, "checkpoint-", step);
    spdlog::info("Checkpointing model to {:s}.pt", full_path);
    torch::save(model_, absl::StrCat(full_path, ".pt"));
    return full_path;
}

void TwoHeadedConvNetWrapper::load_checkpoint(const std::string &path) {
    if (!std::filesystem::exists(absl::StrCat(path, ".pt"))
        || !std::filesystem::exists(absl::StrCat(path, "-optimizer.pt")))
    {
        const auto error_msg = std::format("path {:s} does not contain model and/or optimizer", path);
        spdlog::error(error_msg);
        throw std::filesystem::filesystem_error(error_msg, std::error_code());
    }
    torch::load(model_, absl::StrCat(path, ".pt"), torch_device_);
    torch::load(model_optimizer_, absl::StrCat(path, "-optimizer.pt"), torch_device_);
}
void TwoHeadedConvNetWrapper::load_checkpoint_without_optimizer(const std::string &path) {
    if (!std::filesystem::exists(absl::StrCat(path, ".pt"))) {
        const auto error_msg = std::format("path {:s} does not contain model", path);
        spdlog::error(error_msg);
        throw std::filesystem::filesystem_error(error_msg, std::error_code());
    }
    torch::load(model_, absl::StrCat(path, ".pt"), torch_device_);
}

auto TwoHeadedConvNetWrapper::inference(std::vector<InferenceInput> &batch) -> std::vector<InferenceOutput> {
    const int batch_size = static_cast<int>(batch.size());

    // Check for bad input
    for (const auto &batch_item : batch) {
        if (static_cast<int>(batch_item.observation.size()) != input_flat_size) [[unlikely]] {
            const auto error_msg = std::format(
                "Input observation of size {:d} unexpected for size {:d}",
                batch_item.observation.size(),
                input_flat_size
            );
            spdlog::error(error_msg);
            throw std::logic_error(error_msg);
        }
    }

    // Create tensor from raw flat array
    // torch::from_blob requires a pointer to non-const and doesn't take ownership
    auto options = torch::TensorOptions().dtype(torch::kFloat);
    torch::Tensor input_observations = torch::empty({batch_size, input_flat_size}, options);
    for (const auto &[idx, batch_item] : std::views::enumerate(batch)) {
        const auto i = static_cast<int>(idx);    // stop torch from complaining about narrowing conversions
        assert(static_cast<int>(batch_item.observation.size()) == input_flat_size);
        input_observations[i] = torch::from_blob(batch_item.observation.data(), {input_flat_size}, options);
    }

    // Reshape to expected size for network (batch_size, flat) -> (batch_size, c, h, w)
    input_observations = input_observations.to(torch_device_);
    input_observations = input_observations.reshape(
        {batch_size, config.observation_shape.c, config.observation_shape.h, config.observation_shape.w}
    );

    // Put model in eval mode for inference + scoped no_grad
    model_->eval();
    const torch::NoGradGuard no_grad;

    // Run inference
    const auto model_output = model_->forward(input_observations);
    const auto logits_output = model_output.logits.to(torch::kCPU);
    const auto policy_output = model_output.policy.to(torch::kCPU);
    const auto log_policy_output = model_output.log_policy.to(torch::kCPU);
    const auto heuristic_output = model_output.heuristic.to(torch::kCPU);
    std::vector<InferenceOutput> inference_output;
    for (int i = 0; i < batch_size; ++i) {
        inference_output.emplace_back(
            tensor_to_vec<double, float>(logits_output[i]),
            tensor_to_vec<double, float>(policy_output[i]),
            tensor_to_vec<double, float>(log_policy_output[i]),
            heuristic_output[i].item<double>()
        );
    }
    return inference_output;
}

auto TwoHeadedConvNetWrapper::learn(std::vector<LearningInput> &batch) -> double {
    const int batch_size = static_cast<int>(batch.size());

    // Check for bad input
    for (const auto &batch_item : batch) {
        if (static_cast<int>(batch_item.observation.size()) != input_flat_size) [[unlikely]] {
            const auto error_msg = std::format(
                "Input observation of size {:d} unexpected for size {:d}",
                batch_item.observation.size(),
                input_flat_size
            );
            spdlog::error(error_msg);
            throw std::logic_error(error_msg);
        }
        if (batch_item.target_action < 0 || batch_item.target_action >= num_actions) [[unlikely]] {
            const auto error_msg = std::format(
                "Input action {:d} unexpected for number of action {:d}",
                batch_item.target_action,
                num_actions
            );
            spdlog::error(error_msg);
            throw std::logic_error(error_msg);
        }
    }

    const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
    const auto options_long = torch::TensorOptions().dtype(torch::kLong);

    // Create tensor from raw flat array
    // torch::from_blob requires a pointer to non-const and doesn't take ownership
    torch::Tensor input_observations = torch::empty({batch_size, input_flat_size}, options_float);
    torch::Tensor target_actions = torch::empty({batch_size, 1}, options_long);
    torch::Tensor target_costs = torch::empty({batch_size, 1}, options_float);
    torch::Tensor expandeds = torch::empty({batch_size, 1}, options_float);

    for (const auto &[idx, batch_item] : std::views::enumerate(batch)) {
        const auto i = static_cast<int>(idx);    // stop torch from complaining about narrowing conversions
        assert(static_cast<int>(batch_item.observation.size()) == input_flat_size);
        input_observations[i] = torch::from_blob(batch_item.observation.data(), {input_flat_size}, options_float);
        target_actions[i] = batch_item.target_action;
        assert(batch_item.target_action >= 0 && batch_item.target_action < num_actions);
        target_costs[i] = static_cast<float>(batch_item.target_cost_to_goal);
        expandeds[i] = static_cast<float>(batch_item.solution_expanded);
    }

    // Reshape to expected size for network (batch_size, flat) -> (batch_size, c, h, w)
    input_observations = input_observations.to(torch_device_);
    target_actions = target_actions.to(torch_device_);
    target_costs = target_costs.to(torch_device_);
    expandeds = expandeds.to(torch_device_);
    input_observations = input_observations.reshape(
        {batch_size, config.observation_shape.c, config.observation_shape.h, config.observation_shape.w}
    );

    // Put model in train mode for learning
    model_->train();
    model_->zero_grad();

    // Get model output
    auto model_output = model_->forward(input_observations);

    const torch::Tensor loss =
        (expandeds * cross_entropy_loss(model_output.logits, target_actions, false).reshape({batch_size, -1})
         + mean_squared_error_loss(model_output.heuristic, target_costs, false))
            .mean();
    auto loss_value = loss.item<double>();

    // Optimize model
    loss.backward();
    model_optimizer_.step();

    return loss_value;
}

}    // namespace libpts::model
