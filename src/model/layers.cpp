// File: layers.cpp
// Description: Model layers/subnets

// NOLINTBEGIN
#include <torch/torch.h>
// NOLINTEND

#include <libpolicyts/model/detail/layers.h>

#include <cassert>

namespace libpts::model {

// Create a conv1x1 layer using pytorch defaults
torch::nn::Conv2dOptions conv1x1(int in_channels, int out_channels, int groups) {
    return torch::nn::Conv2dOptions(in_channels, out_channels, 1)
        .stride(1)
        .padding(0)
        .bias(true)
        .dilation(1)
        .groups(groups)
        .padding_mode(torch::kZeros);
}

torch::nn::Conv1dOptions conv1x1_1d(int in_channels, int out_channels, int groups) {
    return torch::nn::Conv1dOptions(in_channels, out_channels, 1)
        .stride(1)
        .padding(0)
        .bias(true)
        .dilation(1)
        .groups(groups)
        .padding_mode(torch::kZeros);
}

// Create a conv3x3 layer using pytorch defaults
torch::nn::Conv2dOptions conv3x3(int in_channels, int out_channels, int stride, int padding, bool bias, int groups) {
    return torch::nn::Conv2dOptions(in_channels, out_channels, 3)
        .stride(stride)
        .padding(padding)
        .bias(bias)
        .dilation(1)
        .groups(groups)
        .padding_mode(torch::kZeros);
}

// Create a average pooling layer using pytorch defaults
torch::nn::AvgPool2dOptions avg_pool3x3(int kernel_size, int stride, int padding) {
    return torch::nn::AvgPool2dOptions(kernel_size).stride(stride).padding(padding);
}

// Create a batchnorm2d layer using pytorch defaults
torch::nn::BatchNorm2dOptions bn(int num_filters) {
    return {num_filters};
}

// ------------------------------- MLP Network ------------------------------
// MLP
MLPImpl::MLPImpl(int input_size, const std::vector<int> &layer_sizes, int output_size, const std::string &name) {
    std::vector<int> sizes = layer_sizes;
    sizes.insert(sizes.begin(), input_size);
    sizes.push_back(output_size);

    // Walk through adding layers
    for (std::size_t i = 0; i < sizes.size() - 1; ++i) {
        layers->push_back("linear_" + std::to_string(i), torch::nn::Linear(sizes[i], sizes[i + 1]));
        if (i < sizes.size() - 2) {
            layers->push_back("activation_" + std::to_string(i), torch::nn::ReLU());
        }
    }
    register_module(name + "mlp", layers);
}

auto MLPImpl::forward(torch::Tensor x) -> torch::Tensor {
    torch::Tensor output = layers->forward(x);
    return output;
}
// ------------------------------- MLP Network ------------------------------

// ------------------------------ ResNet Block ------------------------------
// Main ResNet style residual block
ResidualBlockImpl::ResidualBlockImpl(int num_channels, int layer_num, bool use_batchnorm_, int groups)
    : conv1(conv3x3(num_channels, num_channels, 1, 1, true, groups)),
      conv2(conv3x3(num_channels, num_channels, 1, 1, true, groups)),
      batch_norm1(bn(num_channels)),
      batch_norm2(bn(num_channels)),
      use_batchnorm(use_batchnorm_) {
    register_module("resnet_" + std::to_string(layer_num) + "_conv1", conv1);
    register_module("resnet_" + std::to_string(layer_num) + "_conv2", conv2);
    if (use_batchnorm) {
        register_module("resnet_" + std::to_string(layer_num) + "_bn1", batch_norm1);
        register_module("resnet_" + std::to_string(layer_num) + "_bn2", batch_norm2);
    }
}

auto ResidualBlockImpl::forward(torch::Tensor x) -> torch::Tensor {
    const torch::Tensor residual = x;
    torch::Tensor output = conv1(x);
    if (use_batchnorm) {
        output = batch_norm1(output);
    }
    output = torch::relu(output);
    output = conv2(output);
    if (use_batchnorm) {
        output = batch_norm2(output);
    }
    output += residual;
    output = torch::relu(output);
    return output;
}
// ------------------------------ ResNet Block ------------------------------

// ------------------------------ ResNet Head -------------------------------
// Initial input convolutional before ResNet residual blocks
// Primary use is to take N channels and set to the expected number
//   of channels for the rest of the resnet body
ResidualHeadImpl::ResidualHeadImpl(
    int input_channels,
    int output_channels,
    bool use_batchnorm_,
    const std::string &name_prefix
)
    : conv(conv3x3(input_channels, output_channels)), batch_norm(bn(output_channels)), use_batchnorm(use_batchnorm_) {
    register_module(name_prefix + "resnet_head_conv", conv);
    if (use_batchnorm) {
        register_module(name_prefix + "resnet_head_bn", batch_norm);
    }
}

auto ResidualHeadImpl::forward(torch::Tensor x) -> torch::Tensor {
    torch::Tensor output = conv(x);
    if (use_batchnorm) {
        output = batch_norm(output);
    }
    output = torch::relu(output);
    return output;
}

// Shape doesn't change
ObservationShape ResidualHeadImpl::encoded_state_shape(ObservationShape observation_shape) {
    return observation_shape;
}
// ------------------------------ ResNet Head -------------------------------

// ------------------------------ ResNet Body -------------------------------
ResnetBodyImpl::ResnetBodyImpl(int input_channels, int resnet_channels, int resnet_blocks, bool use_batchnorm)
    : resnet_head_(ResidualHead(input_channels, resnet_channels, use_batchnorm, "head")) {
    // ResNet body
    for (int i = 0; i < resnet_blocks; ++i) {
        resnet_layers_->push_back(ResidualBlock(resnet_channels, i, use_batchnorm));
    }
    register_module("representation_head", resnet_head_);
    register_module("representation_layers", resnet_layers_);
}

auto ResnetBodyImpl::forward(torch::Tensor x) -> torch::Tensor {
    auto output = resnet_head_->forward(x);
    for (int i = 0; i < static_cast<int>(resnet_layers_->size()); ++i) {
        output = resnet_layers_[i]->as<ResidualBlock>()->forward(output);
    }
    return output;
}

// ------------------------------ ResNet Body -------------------------------

}    // namespace libpts::model
