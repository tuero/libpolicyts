// File: torch_init.h
// Description: initialization for torch

#ifndef LIBPTS_TORCH_INIT_H_
#define LIBPTS_TORCH_INIT_H_

#ifndef LIBPTS_TORCH_FOUND
#error "libpolicyts was built without torch support. Rebuild with torch support enabled."
#endif

// NOLINTBEGIN
#include <torch/torch.h>
// NOLINTEND

#include <cstdint>

namespace libpts::model {

/**
 * Initialize torch reproducibility
 * @param seed The seed to initialize torch rngs
 */
inline void init_torch(uint64_t seed) {
    // Set torch seed
    torch::manual_seed(seed);
    torch::globalContext().setDeterministicAlgorithms(true, false);
    if (torch::cuda::is_available()) {
        torch::cuda::manual_seed_all(seed);
        torch::globalContext().setDeterministicCuDNN(true);
        torch::globalContext().setBenchmarkCuDNN(false);
    }
    if (torch::mps::is_available()) {
        torch::mps::manual_seed(seed);
    }
}

}    // namespace libpts::model

#endif    // LIBPTS_TORCH_INIT_H_
