#include "core/device.h"
#include "core/tensor.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_qpn_split.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {
constexpr std::int32_t kIntermediate = 17408; // Nvfp4MlpGateUpGeometry::kOutputRows / 2

__global__ void bf16_to_fp16_kernel(const __nv_bfloat16* __restrict__ input,
                                    half* __restrict__ output, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { output[i] = __float2half(__bfloat162float(input[i])); }
}
} // namespace

bool nvfp4_linear_swiglu_qpn_split_supported(std::int32_t k, std::int32_t t) noexcept {
    return nvfp4_volta_qpn_supported(kIntermediate, k, t);
}

void nvfp4_linear_swiglu_qpn_split_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                          float* gate_scratch, float* up_scratch,
                                          void* activation_scratch,
                                          cudaStream_t stream) {
    const std::int32_t k = x.ne[0];
    const std::int32_t t = x.ne[1];
    const float inverse_weight_divisor = 1.0F / weight.weight_scale_divisor;
    auto* x_fp16 = static_cast<half*>(activation_scratch);
    const std::int64_t activation_count = static_cast<std::int64_t>(k) * t;
    bf16_to_fp16_kernel<<<static_cast<int>((activation_count + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), x_fp16, activation_count);

    // Gate and up are one contiguous weight in the QPN-prepacked layout (32-row tiles, up starting
    // at tile kIntermediate/32 in both the code and the scale plane), so a single launch covers
    // both halves: 1088 CTAs = 6.8 Volta waves instead of two 544-CTA launches of 3.4 waves each.
    launch_nvfp4_volta_qpn_with_fp16_activation(
        x, weight, x_fp16,
        Nvfp4Fp32SplitContiguousOutput{gate_scratch, up_scratch, kIntermediate},
        2 * kIntermediate, inverse_weight_divisor, stream);

    const std::int64_t elements = static_cast<std::int64_t>(kIntermediate) * t;
    const int threads           = 256;
    const int blocks = static_cast<int>(std::min<std::int64_t>((elements + threads - 1) / threads, 4096));
    nvfp4_swiglu_fp32_combine_kernel<<<blocks, threads, 0, stream>>>(
        gate_scratch, up_scratch, static_cast<__nv_bfloat16*>(out.data), elements);
    CUDA_CHECK(cudaGetLastError());
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
