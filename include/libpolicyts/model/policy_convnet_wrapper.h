// File: policy_convnet_wrapper.h
// Description: Convnet wrapper for policy

#ifndef LIBPTS_WRAPPER_POLICY_CONVNET_H_
#define LIBPTS_WRAPPER_POLICY_CONVNET_H_

#ifndef LIBPTS_TORCH_FOUND
#error "libpolicyts was built without torch support. Rebuild with torch support enabled."
#endif

#include <libpolicyts/model/base_model_wrapper.h>
#include <libpolicyts/model/detail/policy_convnet.h>
#include <libpolicyts/observation.h>

#include <nlohmann/json.hpp>

namespace libpts::model {

class PolicyConvNetWrapper : public BaseModelWrapper {
public:
    inline static const std::string name = "policy_convnet";

    struct Config {
        ObservationShape observation_shape;
        int num_actions;
        int resnet_channels;
        int resnet_blocks;
        int policy_channels;
        std::vector<int> policy_mlp_layers;
        bool use_batchnorm;
        double learning_rate;
        double l2_weight_decay;
    };

    struct InferenceInput {
        Observation observation;
    };

    struct InferenceOutput {
        std::vector<double> logits;
        std::vector<double> policy;
        std::vector<double> log_policy;
    };

    struct LearningInput {
        Observation observation;
        int target_action;
        int solution_expanded;
    };

    PolicyConvNetWrapper(
        Config config,
        const std::string &device,
        const std::string &output_path,
        const std::string &checkpoint_base_name = ""
    );
    PolicyConvNetWrapper(
        const nlohmann::json &model_config_json,
        const ObservationShape &obs_shape,
        int num_actions,
        const std::string &device,
        const std::string &output_path,
        const std::string &checkpoint_base_name = ""
    );

    // Get the default model config json with all required fields set
    static auto get_default_json_config() -> nlohmann::json {
        // NOLINTBEGIN(*-magic-numbers)
        return {
            {"model_type", name},
            {"resnet_channels", 16},
            {"resnet_blocks", 2},
            {"policy_channels", 2},
            {"policy_mlp_layers", std::vector<int>{8}},
            {"use_batchnorm", false},
            {"learning_rate", 3e-4},
            {"l2_weight_decay", 1e-4},
        };
        // NOLINTEND(*-magic-numbers)
    }

    void print() const override;

    auto save_checkpoint(long long int step = -1) -> std::string override;
    auto save_checkpoint_without_optimizer(long long int step = -1) -> std::string override;

    using BaseModelWrapper::load_checkpoint;
    using BaseModelWrapper::load_checkpoint_without_optimizer;
    void load_checkpoint(const std::string &path) override;
    void load_checkpoint_without_optimizer(const std::string &path) override;

    /**
     * Perform inference
     * @param inputs Batched observations (implementation defined)
     * @returns Implementation defined output
     */
    [[nodiscard]] auto inference(std::vector<InferenceInput> &batch) -> std::vector<InferenceOutput>;

    /**
     * Perform a model update learning step
     * @param batch Batched learning input
     * @returns Loss for current batch
     */
    auto learn(std::vector<LearningInput> &batch) -> double;

protected:
    Config config;
    network::PolicyConvNet model_;
    torch::optim::Adam model_optimizer_;
    int input_flat_size;
    int num_actions;
};

}    // namespace libpts::model

#endif    // LIBPTS_WRAPPER_POLICY_CONVNET_H_
