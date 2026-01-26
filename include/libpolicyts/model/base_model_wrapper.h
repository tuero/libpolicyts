// File: base_model_wrapper.h
// Description: Holds model + optimizer to directly interface with nn::Module for inference + learning

#ifndef LIBPTS_MODEL_WRAPPER_H_
#define LIBPTS_MODEL_WRAPPER_H_

#ifndef LIBPTS_TORCH_FOUND
#error "libpolicyts was built without torch support. Rebuild with torch support enabled."
#endif

#include <libpolicyts/concepts.h>

#include <string>

// NOLINTBEGIN
#include <torch/torch.h>
// NOLINTEND

namespace libpts::model {

class BaseModelWrapper;

// Concept which model wrappers must implement
template <typename T>
concept IsModelWrapper = std::is_base_of_v<BaseModelWrapper, T> && requires(T t) {
    typename T::InferenceInput;
    typename T::InferenceOutput;
    typename T::LearningInput;
    typename T::Config;
    { t.name } -> std::same_as<const std::string &>;
    {
        t.inference(makeval<std::vector<typename T::InferenceInput> &>())
    } -> std::same_as<std::vector<typename T::InferenceOutput>>;
    { t.learn(makeval<std::vector<typename T::LearningInput> &>()) } -> std::same_as<double>;
};

// Base model wrapper which specific models will inherit and implement specific learn/inference methods
class BaseModelWrapper {
public:
    BaseModelWrapper(
        const std::string &device,
        const std::string &output_path,
        const std::string &checkpoint_base_name = ""
    );
    virtual ~BaseModelWrapper() = default;

    // Doesn't make sense to allow copies
    BaseModelWrapper(const BaseModelWrapper &) = delete;
    BaseModelWrapper(BaseModelWrapper &&) = delete;
    BaseModelWrapper &operator=(const BaseModelWrapper &) = delete;
    BaseModelWrapper &operator=(BaseModelWrapper &&) = delete;

    /**
     * Log model pretty print to log file
     */
    virtual void print() const = 0;

    /**
     * Checkpoint model + optimizer to file
     * @param step Step number of checkpoint (only used if wanting to differentiate checkpoint steps)
     * @return Checkpoint path, used for loading models back to sync if using multiple models
     */
    virtual auto save_checkpoint(long long int step = -1) -> std::string = 0;

    /**
     * Checkpoint model to file
     * @param step Step number of checkpoint (only used if wanting to differentiate checkpoint steps)
     * @return Checkpoint path, used for loading models back to sync if using multiple models
     */
    virtual auto save_checkpoint_without_optimizer(long long int step = -1) -> std::string = 0;

    /**
     * Load model + checkpoint from checkpoint step
     * @param step Step number of checkpoint (only used if wanting to differentiate checkpoint steps)
     */
    virtual void load_checkpoint(long long int step);

    /**
     * Load model from checkpoint step
     * @param step Step number of checkpoint (only used if wanting to differentiate checkpoint steps)
     */
    virtual void load_checkpoint_without_optimizer(long long int step);

    /**
     * Load model + optimizer from checkpoint path
     * @param path Base path directory of model + optimizer
     */
    virtual void load_checkpoint(const std::string &path) = 0;

    /**
     * Load model from checkpoint path
     * @param path Base path directory of model
     */
    virtual void load_checkpoint_without_optimizer(const std::string &path) = 0;

protected:
    std::string device_;
    std::string path_;
    std::string checkpoint_base_name_;
    torch::Device torch_device_;
};

}    // namespace libpts::model

#endif    // LIBPTS_MODEL_WRAPPER_H_
