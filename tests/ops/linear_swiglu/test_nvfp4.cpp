#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        constexpr std::array<std::int32_t, 4> kA16Cases{1, 4, 8, 16};
        constexpr std::array<std::int32_t, 6> kA4Cases{5, 48, 49, 128, 256, 1024};
        int failures = 0;
        failures += run_profile("LinearSwiGLU NVFP4_A16",
                                {QType::NVFP4, 34816, 5120, 17408, 1801U, ActivationCompute::A16},
                                kA16Cases);
#ifdef NINFER_VOLTA_BUILD
        // Production prepacks NVFP4 gate/up at load (SwiGLU-interleaved), so the QPN2 prepacked
        // kernel is the route every decode and verify width takes. The second divisor is
        // checkpoint-like and keeps the kernel's folded scale in fp16 range; the default one forces
        // its fp32 fallback. The last profile keeps the plain (non-interleaved) prepacked layout on
        // its fp32-scratch route. T=48 is CUTLASS dequantizing the interleaved layout; that route
        // rounds gate and up to BF16 before SiLU*up (see nvfp4_linear_swiglu_qpn_split.cuh), which
        // sits at the edge of this tolerance with the default divisor's patterned weights, so it
        // is exercised with the checkpoint-like one.
        constexpr std::array<std::int32_t, 7> kPrepackedCases{1, 4, 5, 8, 9, 16, 32};
        constexpr std::array<std::int32_t, 8> kPrepackedWideCases{1, 4, 5, 8, 9, 16, 32, 48};
        failures += run_profile("LinearSwiGLU NVFP4_A16 prepacked",
                                {QType::NVFP4, 34816, 5120, 17408, 1805U, ActivationCompute::A16,
                                 true},
                                kPrepackedCases);
        failures += run_profile("LinearSwiGLU NVFP4_A16 prepacked real-divisor",
                                {QType::NVFP4, 34816, 5120, 17408, 1807U, ActivationCompute::A16,
                                 true, 2688.0F},
                                kPrepackedWideCases);
        failures += run_profile("LinearSwiGLU NVFP4_A16 prepacked plain",
                                {QType::NVFP4, 34816, 5120, 17408, 1809U, ActivationCompute::A16,
                                 true, 0.0F, false},
                                kPrepackedCases);
#else
        failures +=
            run_profile("LinearSwiGLU NVFP4_A4",
                        {QType::NVFP4, 34816, 5120, 17408, 1803U, ActivationCompute::A4}, kA4Cases);
#endif
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU NVFP4 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU NVFP4 test failed: " << error.what() << '\n';
        return 1;
    }
}
