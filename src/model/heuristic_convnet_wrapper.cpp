// File: heuristic_convnet_wrapper.cpp
// Description: Convnet wrapper for heuristic net

#include <libpolicyts/model/heuristic_convnet_wrapper.h>
#include <libpolicyts/model/torch_util.h>

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
        SPDLOG_ERROR("model config json should contain an entry '{}'", key);
        std::exit(1);
    }
}

auto config_from_json(const nlohmann::json &model_config, const ObservationShape &obs_shape)
    -> HeuristicConvNetWrapper::Config {
    // Check for valid json
    if (!model_config.contains("model_type")
        || model_config["model_type"].get<std::string>() != HeuristicConvNetWrapper::name)
    {
        SPDLOG_ERROR("model config json should contain an entry 'model_type': '{}'", HeuristicConvNetWrapper::name);
        std::exit(1);
    }
    check_config_key_exits(model_config, "resnet_channels");
    check_config_key_exits(model_config, "resnet_blocks");
    check_config_key_exits(model_config, "heuristic_channels");
    check_config_key_exits(model_config, "heuristic_mlp_layers");
    check_config_key_exits(model_config, "use_batchnorm");
    check_config_key_exits(model_config, "learning_rate");
    check_config_key_exits(model_config, "l2_weight_decay");

    return {
        .observation_shape = obs_shape,
        .resnet_channels = model_config["resnet_channels"].template get<int>(),
        .resnet_blocks = model_config["resnet_blocks"].template get<int>(),
        .heuristic_channels = model_config["heuristic_channels"].template get<int>(),
        .heuristic_mlp_layers = model_config["heuristic_mlp_layers"].template get<std::vector<int>>(),
        .use_batchnorm = model_config["use_batchnorm"].template get<bool>(),
        .learning_rate = model_config["learning_rate"].template get<double>(),
        .l2_weight_decay = model_config["l2_weight_decay"].template get<double>(),
    };
}
}    // namespace

HeuristicConvNetWrapper::HeuristicConvNetWrapper(
    Config config_,
    const std::string &device,
    const std::string &output_path,
    const std::string &checkpoint_base_name
)
    : BaseModelWrapper(device, output_path, checkpoint_base_name),
      config(std::move(config_)),
      model_(
          config.observation_shape,
          config.resnet_channels,
          config.resnet_blocks,
          config.heuristic_channels,
          config.heuristic_mlp_layers,
          config.use_batchnorm
      ),
      model_optimizer_(
          model_->parameters(),
          torch::optim::AdamOptions(config.learning_rate).weight_decay(config.l2_weight_decay)
      ),
      input_flat_size(config.observation_shape.flat_size()) {
    model_->to(torch_device_);
};

HeuristicConvNetWrapper::HeuristicConvNetWrapper(
    const nlohmann::json &model_config_json,
    const ObservationShape &obs_shape,
    const std::string &device,
    const std::string &output_path,
    const std::string &checkpoint_base_name
)
    : BaseModelWrapper(device, output_path, checkpoint_base_name),
      config(config_from_json(model_config_json, obs_shape)),
      model_(
          config.observation_shape,
          config.resnet_channels,
          config.resnet_blocks,
          config.heuristic_channels,
          config.heuristic_mlp_layers,
          config.use_batchnorm
      ),
      model_optimizer_(
          model_->parameters(),
          torch::optim::AdamOptions(config.learning_rate).weight_decay(config.l2_weight_decay)
      ),
      input_flat_size(config.observation_shape.flat_size()) {
    model_->to(torch_device_);
};

void HeuristicConvNetWrapper::print() const {
    std::ostringstream oss;
    std::ostream &os = oss;
    os << *model_;
    SPDLOG_INFO("{:s}", oss.str());
    std::size_t num_params = 0;
    for (const auto &p : model_->parameters()) {
        num_params += static_cast<std::size_t>(p.numel());
    }
    SPDLOG_INFO("Number of parameters: {:d}", num_params);
}

auto HeuristicConvNetWrapper::save_checkpoint(long long int step) -> std::string {
    // create directory for model
    std::filesystem::create_directories(path_);
    std::string full_path = absl::StrCat(path_, checkpoint_base_name_, "checkpoint-", step);
    SPDLOG_INFO("Checkpointing model to {:s}.pt", full_path);
    torch::save(model_, absl::StrCat(full_path, ".pt"));
    torch::save(model_optimizer_, absl::StrCat(full_path, "-optimizer.pt"));
    return full_path;
}
auto HeuristicConvNetWrapper::save_checkpoint_without_optimizer(long long int step) -> std::string {
    // create directory for model
    std::filesystem::create_directories(path_);
    std::string full_path = absl::StrCat(path_, checkpoint_base_name_, "checkpoint-", step);
    SPDLOG_INFO("Checkpointing model to {:s}.pt", full_path);
    torch::save(model_, absl::StrCat(full_path, ".pt"));
    return full_path;
}

void HeuristicConvNetWrapper::load_checkpoint(const std::string &path) {
    if (!std::filesystem::exists(absl::StrCat(path, ".pt"))
        || !std::filesystem::exists(absl::StrCat(path, "-optimizer.pt")))
    {
        SPDLOG_ERROR("path {:s} does not contain model and/or optimizer", path);
        throw std::filesystem::filesystem_error(
            std::format("path {:s} does not contain model and/or optimizer", path),
            std::error_code()
        );
    }
    torch::load(model_, absl::StrCat(path, ".pt"), torch_device_);
    torch::load(model_optimizer_, absl::StrCat(path, "-optimizer.pt"), torch_device_);
}
void HeuristicConvNetWrapper::load_checkpoint_without_optimizer(const std::string &path) {
    if (!std::filesystem::exists(absl::StrCat(path, ".pt"))) {
        SPDLOG_ERROR("path {:s} does not contain model", path);
        throw std::filesystem::filesystem_error(
            std::format("path {:s} does not contain model", path),
            std::error_code()
        );
    }
    torch::load(model_, absl::StrCat(path, ".pt"), torch_device_);
}

auto HeuristicConvNetWrapper::inference(std::vector<InferenceInput> &batch) -> std::vector<InferenceOutput> {
    const int batch_size = static_cast<int>(batch.size());

    // Create tensor from raw flat array
    // torch::from_blob requires a pointer to non-const and doesn't take ownership
    auto options = torch::TensorOptions().dtype(torch::kFloat);
    torch::Tensor input_observations = torch::empty({batch_size, input_flat_size}, options);
    for (auto &&[idx, batch_item] : std::views::enumerate(batch)) {
        const auto i = static_cast<int>(idx);    // stop torch from complaining about narrowing conversions
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
    const auto heuristic_output = model_output.heuristic.to(torch::kCPU);
    std::vector<InferenceOutput> inference_output;
    for (int i = 0; i < batch_size; ++i) {
        inference_output.emplace_back(heuristic_output[i].item<double>());
    }
    return inference_output;
}

auto HeuristicConvNetWrapper::learn(std::vector<LearningInput> &batch) -> double {
    const int batch_size = static_cast<int>(batch.size());
    const auto options_float = torch::TensorOptions().dtype(torch::kFloat);

    // Create tensor from raw flat array
    // torch::from_blob requires a pointer to non-const and doesn't take ownership
    torch::Tensor input_observations = torch::empty({batch_size, input_flat_size}, options_float);
    torch::Tensor target_costs = torch::empty({batch_size, 1}, options_float);

    for (auto &&[idx, batch_item] : std::views::enumerate(batch)) {
        const auto i = static_cast<int>(idx);    // stop torch from complaining about narrowing conversions
        input_observations[i] = torch::from_blob(batch_item.observation.data(), {input_flat_size}, options_float);
        target_costs[i] = static_cast<float>(batch_item.target_cost_to_goal);
    }

    // Reshape to expected size for network (batch_size, flat) -> (batch_size, c, h, w)
    input_observations = input_observations.to(torch_device_);
    target_costs = target_costs.to(torch_device_);
    input_observations = input_observations.reshape(
        {batch_size, config.observation_shape.c, config.observation_shape.h, config.observation_shape.w}
    );

    // Put model in train mode for learning
    model_->train();
    model_->zero_grad();

    // Get model output
    auto model_output = model_->forward(input_observations);

    const torch::Tensor loss = mean_squared_error_loss(model_output.heuristic, target_costs, true);
    auto loss_value = loss.item<double>();

    // Optimize model
    loss.backward();
    model_optimizer_.step();

    return loss_value;
}

}    // namespace libpts::model
