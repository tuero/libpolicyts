// File: heuristic_convnet_wrapper.h
// Description: Convnet wrapper for heuristic net

#ifndef LIBPTS_WRAPPER_HEURISTIC_CONVNET_H_
#define LIBPTS_WRAPPER_HEURISTIC_CONVNET_H_

#ifndef LIBPTS_TORCH_FOUND
#error "libpolicyts was built without torch support. Rebuild with torch support enabled."
#endif

#include <libpolicyts/model/base_model_wrapper.h>
#include <libpolicyts/model/detail/heuristic_convnet.h>
#include <libpolicyts/observation.h>

#include <nlohmann/json.hpp>

namespace libpts::model {

class HeuristicConvNetWrapper : public BaseModelWrapper {
public:
    inline static const std::string name = "heuristic_convnet";

    struct Config {
        ObservationShape observation_shape;
        int resnet_channels;
        int resnet_blocks;
        int heuristic_channels;
        std::vector<int> heuristic_mlp_layers;
        bool use_batchnorm;
        double learning_rate;
        double l2_weight_decay;
    };

    struct InferenceInput {
        Observation observation;
    };

    struct InferenceOutput {
        double heuristic;
    };

    struct LearningInput {
        Observation observation;
        double target_cost_to_goal;
    };

    HeuristicConvNetWrapper(
        Config config,
        const std::string &device,
        const std::string &output_path,
        const std::string &checkpoint_base_name = ""
    );
    HeuristicConvNetWrapper(
        const nlohmann::json &model_config_json,
        const ObservationShape &obs_shape,
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
            {"heuristic_channels", 2},
            {"heuristic_mlp_layers", std::vector<int>{8}},
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
    network::HeuristicConvNet model_;
    torch::optim::Adam model_optimizer_;
    int input_flat_size;
};

}    // namespace libpts::model

#endif    // LIBPTS_WRAPPER_HEURISTIC_CONVNET_H_
