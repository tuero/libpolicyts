// File: twoheaded_convent.h
// Description: Convnet for policy + heuristic predictions

#ifndef LIBPTS_MODEL_TWOHEADED_CONVNET_H_
#define LIBPTS_MODEL_TWOHEADED_CONVNET_H_

#ifndef LIBPTS_TORCH_FOUND
#error "libpolicyts was built without torch support. Rebuild with torch support enabled."
#endif

#include <libpolicyts/model/detail/layers.h>
#include <libpolicyts/observation.h>

#include <vector>

// NOLINTBEGIN
#include <torch/torch.h>
// NOLINTEND

namespace libpts::model::network {

struct TwoHeadedConvNetOutput {
    torch::Tensor logits;
    torch::Tensor policy;
    torch::Tensor log_policy;
    torch::Tensor heuristic;
};

class TwoHeadedConvNetImpl : public torch::nn::Module {
public:
    /**
     * ResNet style heuristic + policy convnet
     * @param observation_shape Input observation shape to the network
     * @param num_actions Number of actions for the policy output
     * @param resnet_channels Number of channels for each resenet block
     * @param resnet_blocks Number of resnet blocks
     * @param policy_channels Number of channels in the policy reduce head
     * @param heuristic_channels Number of channels in the heuristic reduce head
     * @param policy_mlp_layers Hidden layer sizes for the policy head MLP
     * @param heuristic_mlp_layers Hidden layer sizes for the heuristic head MLP
     * @param use_batchnorm Flag to use batchnorm in the resnet layers
     */
    TwoHeadedConvNetImpl(
        const ObservationShape &observation_shape,
        int num_actions,
        int resnet_channels,
        int resnet_blocks,
        int policy_channels,
        int heuristic_channels,
        const std::vector<int> &policy_mlp_layers,
        const std::vector<int> &heuristic_mlp_layers,
        bool use_batchnorm
    );
    [[nodiscard]] auto forward(torch::Tensor x) -> TwoHeadedConvNetOutput;

private:
    int input_channels_;
    int input_height_;
    int input_width_;
    int resnet_channels_;
    int policy_channels_;
    int heuristic_channels_;
    int policy_mlp_input_size_;
    int heuristic_mlp_input_size_;
    ResidualHead resnet_head_;
    torch::nn::Conv2d conv1x1_policy_;       // Conv pass before passing to policy mlp
    torch::nn::Conv2d conv1x1_heuristic_;    // Conv pass before passing to heuristic mlp
    MLP policy_mlp_;
    MLP heuristic_mlp_;
    torch::nn::ModuleList resnet_layers_;
};
TORCH_MODULE(TwoHeadedConvNet);

}    // namespace libpts::model::network

#endif    // LIBPTS_MODEL_TWOHEADED_CONVNET_H_
