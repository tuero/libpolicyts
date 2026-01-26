// File: base_model_wrapper.cpp
// Description: Holds model + optimizer to directly interface with nn::Module for inference + learning

#include <libpolicyts/model/base_model_wrapper.h>

#include <absl/strings/str_cat.h>

namespace libpts::model {

BaseModelWrapper::BaseModelWrapper(
    const std::string &device,
    const std::string &output_path,
    const std::string &checkpoint_base_name
)
    : device_(device),
      path_(absl::StrCat(output_path, "/checkpoints/")),
      checkpoint_base_name_(
          checkpoint_base_name.empty() ? checkpoint_base_name : absl::StrCat(checkpoint_base_name, "-")
      ),
      torch_device_(device) {}

void BaseModelWrapper::load_checkpoint(long long int step) {
    load_checkpoint(absl::StrCat(path_, checkpoint_base_name_, "checkpoint-", step));
}

void BaseModelWrapper::load_checkpoint_without_optimizer(long long int step) {
    load_checkpoint_without_optimizer(absl::StrCat(path_, checkpoint_base_name_, "checkpoint-", step));
}

}    // namespace libpts::model
