#pragma once

#include "core/tensor.h"

#include <cstdint>
#include <span>
#include <string_view>

namespace ninfer::test::linear_swiglu {

enum class ActivationCompute : std::uint8_t {
    A16,
    A8,
    A4,
};

struct Profile {
    QType qtype;
    std::int32_t gate_up_rows;
    std::int32_t input_rows;
    std::int32_t output_rows;
    std::uint32_t seed;
    ActivationCompute activation_compute;
    // NVFP4 only: prepack the weight into the Volta QPN layout the production loader uses, and
    // override the patterned weight's scale divisor (0 keeps the default).
    bool nvfp4_prepack           = false;
    float weight_scale_divisor   = 0.0F;
    bool nvfp4_swiglu_interleave = true; // with nvfp4_prepack: the production gate/up layout
    bool fp8_swiglu_interleave   = true; // FP8 is always prepacked on Volta; production layout
    // FP8 only: checkpoint-like weights (make_checkpoint_like_fp8_weight) and dense Gaussian
    // activations instead of the patterned weight and sparse activations.
    bool fp8_checkpoint_like = false;
};

int run_profile(std::string_view label, const Profile& profile,
                std::span<const std::int32_t> token_cases,
                std::span<const std::int32_t> graph_cases = {});

} // namespace ninfer::test::linear_swiglu
