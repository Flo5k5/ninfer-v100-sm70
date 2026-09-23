// ninfer::ops - fused per-head QK RMSNorm + RoPE for the fixed Qwen3.6 text geometries.
#include "ops/launcher/qk_rmsnorm_rope.h"

#include "core/device.h"
#include "ops/kernel/rmsnorm.cuh"
#include "ops/kernel/rope.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kHeadDim = 256;
constexpr int kHalf    = 32; // rotary_dim / 2

// One CTA per token, one warp per head row (queries first). The row normalization is
// rmsnorm_warp_row with the exact template arguments the standalone Op uses for these rows
// (offset epilogue, prefetched gain, D256), and the rotation is apply_rope_head with the
// coefficients rope_fixed_kernel computes, so every value matches the three separate launches.
template <RopeKernelMode Mode, int QHeads, int KHeads>
__global__ void __launch_bounds__((QHeads + KHeads) * kWarpSize)
    qk_rmsnorm_rope_kernel(const std::int32_t* __restrict__ positions,
                           const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                           const __nv_bfloat16* __restrict__ q_norm,
                           const __nv_bfloat16* __restrict__ k_norm, __nv_bfloat16* q_out,
                           __nv_bfloat16* k_out, std::int32_t tokens, float eps) {
    const int token = static_cast<int>(blockIdx.x);
    __shared__ float cos_cache[kHalf];
    __shared__ float sin_cache[kHalf];
    if (threadIdx.x < kHalf) {
        const int pair = static_cast<int>(threadIdx.x);
        fixed_sincos<Mode>(positions, tokens, token, pair, &sin_cache[pair], &cos_cache[pair]);
    }

    const int lane         = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp         = static_cast<int>(threadIdx.x) / kWarpSize;
    const bool query       = warp < QHeads;
    const int head         = query ? warp : warp - QHeads;
    const int heads        = query ? QHeads : KHeads;
    __nv_bfloat16* out     = query ? q_out : k_out;
    const std::int64_t row = static_cast<std::int64_t>(token) * heads + head;
    rmsnorm_warp_row<RmsEpilogue::Offset, true, kHeadDim, __nv_bfloat162>(
        reinterpret_cast<const __nv_bfloat162*>(query ? q : k),
        reinterpret_cast<const __nv_bfloat162*>(query ? q_norm : k_norm), nullptr,
        reinterpret_cast<__nv_bfloat162*>(out), kHeadDim, row, lane, eps);
    // Publishes the coefficient cache and every warp's normalized row to the rotation below.
    __syncthreads();

    float c0 = 0.0F, c1 = 0.0F, s0 = 0.0F, s1 = 0.0F;
    if (lane < kHalf / 2) {
        const int pair = lane * 2;
        c0             = cos_cache[pair];
        c1             = cos_cache[pair + 1];
        s0             = sin_cache[pair];
        s1             = sin_cache[pair + 1];
    }
    apply_rope_head<kHeadDim, kHalf>(out, static_cast<std::int64_t>(heads) * kHeadDim, head, token,
                                     lane, c0, c1, s0, s1);
}

template <RopeKernelMode Mode, int QHeads, int KHeads>
void launch(const Tensor& positions, const Tensor& q, const Tensor& k, const Tensor& q_norm,
            const Tensor& k_norm, Tensor& q_out, Tensor& k_out, float eps, cudaStream_t stream) {
    const std::int32_t tokens = positions.ne[0];
    qk_rmsnorm_rope_kernel<Mode, QHeads, KHeads>
        <<<static_cast<unsigned>(tokens), (QHeads + KHeads) * kWarpSize, 0, stream>>>(
            static_cast<const std::int32_t*>(positions.data),
            static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
            static_cast<const __nv_bfloat16*>(q_norm.data),
            static_cast<const __nv_bfloat16*>(k_norm.data), static_cast<__nv_bfloat16*>(q_out.data),
            static_cast<__nv_bfloat16*>(k_out.data), tokens, eps);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void qk_rmsnorm_rope_fused_launch(const Tensor& positions, const Tensor& q, const Tensor& k,
                                  const Tensor& q_norm, const Tensor& k_norm, Tensor& q_out,
                                  Tensor& k_out, float eps, cudaStream_t stream) {
    const bool mrope = positions.ne[1] == 3;
    if (q.ne[1] == 24) {
        if (mrope) {
            launch<RopeKernelMode::TextMrope, 24, 4>(positions, q, k, q_norm, k_norm, q_out, k_out,
                                                     eps, stream);
        } else {
            launch<RopeKernelMode::Text1D, 24, 4>(positions, q, k, q_norm, k_norm, q_out, k_out,
                                                  eps, stream);
        }
    } else {
        if (mrope) {
            launch<RopeKernelMode::TextMrope, 16, 2>(positions, q, k, q_norm, k_norm, q_out, k_out,
                                                     eps, stream);
        } else {
            launch<RopeKernelMode::Text1D, 16, 2>(positions, q, k, q_norm, k_norm, q_out, k_out,
                                                  eps, stream);
        }
    }
}

} // namespace ninfer::ops::detail
