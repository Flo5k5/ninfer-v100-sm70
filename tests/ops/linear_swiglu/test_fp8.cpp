#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
#ifdef NINFER_VOLTA_BUILD
        // QPN split covers up to 32 tokens per pass: 5 is the K=4 verify width, 9/17/32 the
        // two- and four-tile buckets. The CUTLASS widths (from 33) run on checkpoint-like data
        // below.
        constexpr std::array<std::int32_t, 8> kA16Cases{1, 2, 4, 5, 8, 9, 17, 32};
#else
        constexpr std::array<std::int32_t, 2> kA16Cases{1, 2};
#endif
        constexpr std::array<std::int32_t, 6> kA8Cases{1, 2, 3, 48, 65, 1024};
        int failures = 0;
        failures += run_profile(
            "LinearSwiGLU FP8_A16",
            {QType::FP8_E4M3FN_ROW_BF16S, 34816, 5120, 17408, 1811U, ActivationCompute::A16},
            kA16Cases);
#ifdef NINFER_VOLTA_BUILD
        // The generic (non-interleaved) prepacked layout keeps its two-launch fp32-scratch route.
        constexpr std::array<std::int32_t, 8> kPlainCases{1, 2, 4, 5, 8, 9, 17, 32};
        Profile plain{QType::FP8_E4M3FN_ROW_BF16S, 34816, 5120, 17408, 1815U,
                      ActivationCompute::A16};
        plain.fp8_swiglu_interleave = false;
        failures += run_profile("LinearSwiGLU FP8_A16 plain prepacked", plain, kPlainCases);
        // The CUTLASS route (T >= 33) materializes gate and up in BF16, each rounded once from
        // the row-scaled FP32 accumulator, before SiLU*up. Its output error is then three BF16
        // roundings deep, and on the patterned fixture's sparse tokens (four products with few
        // significant bits) the gate and up rounding errors are correlated: correctly rounded
        // gate/up give relative L2 4.1e-3 there against 2.6e-3 for a doubly rounded pair, the
        // opposite of their own errors. Checkpoint-like weights with dense activations are
        // representative: 3.0e-3 with one rounding, 3.9e-3 with two.
        constexpr std::array<std::int32_t, 3> kCheckpointCases{5, 32, 33};
        Profile checkpoint{QType::FP8_E4M3FN_ROW_BF16S, 34816, 5120, 17408, 1817U,
                           ActivationCompute::A16};
        checkpoint.fp8_checkpoint_like = true;
        failures += run_profile("LinearSwiGLU FP8_A16 checkpoint-like", checkpoint,
                                kCheckpointCases);
#endif
#ifndef NINFER_VOLTA_BUILD
        failures += run_profile(
            "LinearSwiGLU FP8_A8",
            {QType::FP8_E4M3FN_ROW_BF16S, 34816, 5120, 17408, 1813U, ActivationCompute::A8},
            kA8Cases);
#endif
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU FP8 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU FP8 test failed: " << error.what() << '\n';
        return 1;
    }
}
