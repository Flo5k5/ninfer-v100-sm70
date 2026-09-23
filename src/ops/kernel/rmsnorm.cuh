#pragma once

// ninfer::ops - RMSNorm kernels over contiguous BF16 rows.

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

enum class RmsEpilogue {
    Offset,
    Plain,
    Gated,
};

template <RmsEpilogue Epilogue>
__device__ __forceinline__ float rmsnorm_epilogue(float x, float inv, float weight, float z) {
    if constexpr (Epilogue == RmsEpilogue::Offset) { weight += 1.0f; }
    float value = x * inv * weight;
    if constexpr (Epilogue == RmsEpilogue::Gated) { value *= silu(z); }
    return value;
}

// Output pair store. An FP16 output carries the BF16-rounded value: it is the fp16 activation copy
// the Volta QPN GEMVs read, bit-identical to writing BF16 and staging that to fp16 afterwards.
template <class OutPair>
__device__ __forceinline__ OutPair rmsnorm_pack(float a, float b);

template <>
__device__ __forceinline__ __nv_bfloat162 rmsnorm_pack<__nv_bfloat162>(float a, float b) {
    return __floats2bfloat162_rn(a, b);
}

template <>
__device__ __forceinline__ __half2 rmsnorm_pack<__half2>(float a, float b) {
    const float2 rounded = __bfloat1622float2(__floats2bfloat162_rn(a, b));
    return __floats2half2_rn(rounded.x, rounded.y);
}

// Fast geometry for D in {64, 128, 192, 256}. One warp owns one row, keeps the input in
// registers, and uses only warp shuffles for the reduction. Block is a scheduling choice rather
// than part of the row geometry.
// One row of the warp geometry, shared with the fused QK-norm + RoPE kernel (qk_rmsnorm_rope),
// which must reproduce this kernel's output bit for bit.
template <RmsEpilogue Epilogue, bool Prefetch, int FixedD, class OutPair>
__device__ __forceinline__ void rmsnorm_warp_row(const __nv_bfloat162* x,
                                                 const __nv_bfloat162* weight,
                                                 const __nv_bfloat162* z, OutPair* out,
                                                 std::int32_t input_d, std::int64_t row, int lane,
                                                 float eps) {
    const int d                    = FixedD ? FixedD : input_d;
    constexpr int kMaxPairsPerLane = 4;
    const int pairs             = d / 2;
    const std::int64_t row_base = row * static_cast<std::int64_t>(pairs);
    __nv_bfloat162 values[kMaxPairsPerLane];
    __nv_bfloat162 weights[kMaxPairsPerLane];
    __nv_bfloat162 gates[kMaxPairsPerLane];
    float sum = 0.0f;

#pragma unroll
    for (int k = 0; k < kMaxPairsPerLane; ++k) {
        const int pair = lane + k * kWarpSize;
        if (pair < pairs) {
            values[k] = x[row_base + pair];
            if constexpr (Prefetch) {
                weights[k] = weight[pair];
                if constexpr (Epilogue == RmsEpilogue::Gated) { gates[k] = z[row_base + pair]; }
            }
            const float2 xf = __bfloat1622float2(values[k]);
            sum += xf.x * xf.x + xf.y * xf.y;
        }
    }

    sum       = warp_reduce_sum(sum);
    float inv = lane == 0 ? rsqrtf(sum / static_cast<float>(d) + eps) : 0.0f;
    inv       = __shfl_sync(kFullWarpMask, inv, 0);

#pragma unroll
    for (int k = 0; k < kMaxPairsPerLane; ++k) {
        const int pair = lane + k * kWarpSize;
        if (pair < pairs) {
            const float2 xf = __bfloat1622float2(values[k]);
            __nv_bfloat162 w_pair;
            __nv_bfloat162 z_pair{};
            if constexpr (Prefetch) {
                w_pair = weights[k];
                if constexpr (Epilogue == RmsEpilogue::Gated) { z_pair = gates[k]; }
            } else {
                w_pair = weight[pair];
                if constexpr (Epilogue == RmsEpilogue::Gated) { z_pair = z[row_base + pair]; }
            }
            const float2 wf = __bfloat1622float2(w_pair);
            float2 zf{0.0f, 0.0f};
            if constexpr (Epilogue == RmsEpilogue::Gated) { zf = __bfloat1622float2(z_pair); }
            out[row_base + pair] =
                rmsnorm_pack<OutPair>(rmsnorm_epilogue<Epilogue>(xf.x, inv, wf.x, zf.x),
                                      rmsnorm_epilogue<Epilogue>(xf.y, inv, wf.y, zf.y));
        }
    }
}

template <RmsEpilogue Epilogue, int Block, bool Prefetch, int FixedD = 0,
          class OutPair = __nv_bfloat162>
__launch_bounds__(Block) __global__
    void rmsnorm_warp_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                    const __nv_bfloat162* z, OutPair* out,
                                    std::int32_t input_d, std::int64_t rows, float eps) {
    static_assert(Block % kWarpSize == 0);
    constexpr int kWarpsPerBlock = Block / kWarpSize;
    const int lane               = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp               = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::int64_t row       = static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) { return; }
    rmsnorm_warp_row<Epilogue, Prefetch, FixedD, OutPair>(x, weight, z, out, input_d, row, lane,
                                                          eps);
}

// Implements: include/ninfer/ops/rmsnorm.h
// Match: aligned contiguous BF16, plain epilogue, D=128, sm_120a.
// Algorithm assumptions: exactly two BF16x2 values per lane; one warp owns one logical row.
template <RmsEpilogue Epilogue, int Block>
__launch_bounds__(Block) __global__
    void rmsnorm_d128_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                    const __nv_bfloat162* z, __nv_bfloat162* out, std::int64_t rows,
                                    float eps) {
    static_assert(Block % kWarpSize == 0);
    constexpr int kWarpsPerBlock = Block / kWarpSize;
    constexpr int kPairsPerRow   = 64;
    const int lane               = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp               = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::int64_t row       = static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) { return; }

    const std::int64_t row_base = row * kPairsPerRow;
    const int pair0             = lane;
    const int pair1             = lane + kWarpSize;
    const __nv_bfloat162 value0 = x[row_base + pair0];
    const __nv_bfloat162 value1 = x[row_base + pair1];
    const float2 x0             = __bfloat1622float2(value0);
    const float2 x1             = __bfloat1622float2(value1);
    float sum                   = x0.x * x0.x + x0.y * x0.y + x1.x * x1.x + x1.y * x1.y;
    sum                         = warp_reduce_sum(sum);
    float inv                   = lane == 0 ? rsqrtf(sum * (1.0f / 128.0f) + eps) : 0.0f;
    inv                         = __shfl_sync(kFullWarpMask, inv, 0);

    const float2 w0 = __bfloat1622float2(weight[pair0]);
    const float2 w1 = __bfloat1622float2(weight[pair1]);
    float2 z0{0.0f, 0.0f};
    float2 z1{0.0f, 0.0f};
    if constexpr (Epilogue == RmsEpilogue::Gated) {
        z0 = __bfloat1622float2(z[row_base + pair0]);
        z1 = __bfloat1622float2(z[row_base + pair1]);
    }
    out[row_base + pair0] =
        __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(x0.x, inv, w0.x, z0.x),
                              rmsnorm_epilogue<Epilogue>(x0.y, inv, w0.y, z0.y));
    out[row_base + pair1] =
        __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(x1.x, inv, w1.x, z1.x),
                              rmsnorm_epilogue<Epilogue>(x1.y, inv, w1.y, z1.y));
}

// The 5120-wide hidden row in one vector pass: 640 threads, eight BF16 per thread through one
// 16-byte load of x and one of the gain, no loops. Small on purpose. A decode step runs this ~150
// times, each between weight streams that evict its code from L2, and ncu shows the unrolled CTA
// kernel below (536 SASS instructions at D=5120) spending ~60% of its warp cycles stalled on
// no_instruction: every launch paid a cold instruction fetch. Plain and offset epilogues only;
// the launcher requires 16-byte aligned x, weight and out.
template <RmsEpilogue Epilogue, class OutPair>
__launch_bounds__(640) __global__
    void rmsnorm_row5120_vec8_kernel(const uint4* __restrict__ x, const uint4* __restrict__ weight,
                                     uint4* __restrict__ out, float eps) {
    static_assert(Epilogue != RmsEpilogue::Gated);
    static_assert(sizeof(OutPair) == 4);
    constexpr int kThreads = 640;
    constexpr int kWarps   = kThreads / kWarpSize;
    const int tid          = static_cast<int>(threadIdx.x);
    const std::int64_t row = blockIdx.x;

    const uint4 x_vec = x[row * kThreads + tid];
    const uint4 w_vec = weight[tid];
    const auto* x_pairs = reinterpret_cast<const __nv_bfloat162*>(&x_vec);
    float2 xf[4];
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        xf[j] = __bfloat1622float2(x_pairs[j]);
        sum += xf[j].x * xf[j].x + xf[j].y * xf[j].y;
    }

    __shared__ float warp_sums[kWarps];
    __shared__ float inv_shared;
    sum = warp_reduce_sum(sum);
    if ((tid & (kWarpSize - 1)) == 0) { warp_sums[tid / kWarpSize] = sum; }
    __syncthreads();
    if (tid < kWarpSize) {
        float total = tid < kWarps ? warp_sums[tid] : 0.0f;
        total       = warp_reduce_sum(total);
        if (tid == 0) { inv_shared = rsqrtf(total * (1.0f / 5120.0f) + eps); }
    }
    __syncthreads();
    const float inv = inv_shared;

    const auto* w_pairs = reinterpret_cast<const __nv_bfloat162*>(&w_vec);
    OutPair result[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 wf = __bfloat1622float2(w_pairs[j]);
        result[j]       = rmsnorm_pack<OutPair>(rmsnorm_epilogue<Epilogue>(xf[j].x, inv, wf.x, 0.0f),
                                                rmsnorm_epilogue<Epilogue>(xf[j].y, inv, wf.y, 0.0f));
    }
    out[row * kThreads + tid] = *reinterpret_cast<const uint4*>(result);
}

// Fast geometry for wide rows. One CTA owns one row and keeps up to MaxPairsPerThread BF16x2
// values per lane. The launcher admits only widths evenly divisible by the CTA vector span.
template <RmsEpilogue Epilogue, int Block, int MaxPairsPerThread, bool Prefetch, int FixedD = 0,
          class OutPair = __nv_bfloat162>
__launch_bounds__(Block) __global__
    void rmsnorm_cta_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                   const __nv_bfloat162* z, OutPair* out,
                                   std::int32_t input_d, std::int64_t rows, float eps) {
    const int d = FixedD ? FixedD : input_d;
    static_assert(Block % kWarpSize == 0);
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }

    const int pairs             = d / 2;
    const int pairs_per_thread  = pairs / Block;
    const std::int64_t row_base = row * static_cast<std::int64_t>(pairs);
    __nv_bfloat162 values[MaxPairsPerThread];
    // Neither the weight nor the gate depends on the sum of squares, so under Prefetch both
    // are issued in the same pass as x and the two trips to memory overlap instead of running
    // back to back across the reduction. They cost registers, so the launcher turns this off
    // once the grid is wide enough that occupancy is worth more than the overlap.
    __nv_bfloat162 weights[MaxPairsPerThread];
    __nv_bfloat162 gates[MaxPairsPerThread];
    float sum = 0.0f;

#pragma unroll
    for (int k = 0; k < MaxPairsPerThread; ++k) {
        if (k < pairs_per_thread) {
            const int pair = static_cast<int>(threadIdx.x) + k * Block;
            values[k]      = x[row_base + pair];
            if constexpr (Prefetch) {
                weights[k] = weight[pair];
                if constexpr (Epilogue == RmsEpilogue::Gated) { gates[k] = z[row_base + pair]; }
            }
            const float2 xf = __bfloat1622float2(values[k]);
            sum += xf.x * xf.x + xf.y * xf.y;
        }
    }

    __shared__ float warp_sums[Block / kWarpSize];
    __shared__ float inv_shared;
    const float block_sum = block_reduce_sum<Block>(sum, warp_sums);
    if (threadIdx.x == 0) { inv_shared = rsqrtf(block_sum / static_cast<float>(d) + eps); }
    __syncthreads();
    const float inv = inv_shared;

#pragma unroll
    for (int k = 0; k < MaxPairsPerThread; ++k) {
        if (k < pairs_per_thread) {
            const int pair  = static_cast<int>(threadIdx.x) + k * Block;
            const float2 xf = __bfloat1622float2(values[k]);
            __nv_bfloat162 w_pair;
            __nv_bfloat162 z_pair{};
            if constexpr (Prefetch) {
                w_pair = weights[k];
                if constexpr (Epilogue == RmsEpilogue::Gated) { z_pair = gates[k]; }
            } else {
                w_pair = weight[pair];
                if constexpr (Epilogue == RmsEpilogue::Gated) { z_pair = z[row_base + pair]; }
            }
            const float2 wf = __bfloat1622float2(w_pair);
            float2 zf{0.0f, 0.0f};
            if constexpr (Epilogue == RmsEpilogue::Gated) { zf = __bfloat1622float2(z_pair); }
            out[row_base + pair] =
                rmsnorm_pack<OutPair>(rmsnorm_epilogue<Epilogue>(xf.x, inv, wf.x, zf.x),
                                      rmsnorm_epilogue<Epilogue>(xf.y, inv, wf.y, zf.y));
        }
    }
}

// Implements: include/ninfer/ops/rmsnorm.h
// Match: aligned contiguous BF16, plain epilogue, D=2048, sm_120a.
// Algorithm assumptions: exactly two BF16x2 values per thread; one 512-thread CTA owns one row.
template <RmsEpilogue Epilogue>
__launch_bounds__(512) __global__
    void rmsnorm_d2048_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                     const __nv_bfloat162* z, __nv_bfloat162* out,
                                     std::int64_t rows, float eps) {
    constexpr int kBlock       = 512;
    constexpr int kPairsPerRow = 1024;
    const std::int64_t row     = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }

    const std::int64_t row_base = row * kPairsPerRow;
    const int pair0             = static_cast<int>(threadIdx.x);
    const int pair1             = pair0 + kBlock;
    const __nv_bfloat162 value0 = x[row_base + pair0];
    const __nv_bfloat162 value1 = x[row_base + pair1];
    const float2 x0             = __bfloat1622float2(value0);
    const float2 x1             = __bfloat1622float2(value1);
    const float local_sum       = x0.x * x0.x + x0.y * x0.y + x1.x * x1.x + x1.y * x1.y;

    __shared__ float warp_sums[kBlock / kWarpSize];
    __shared__ float inv_shared;
    const float sum = block_reduce_sum<kBlock>(local_sum, warp_sums);
    if (threadIdx.x == 0) { inv_shared = rsqrtf(sum * (1.0f / 2048.0f) + eps); }
    __syncthreads();
    const float inv = inv_shared;

    const float2 w0 = __bfloat1622float2(weight[pair0]);
    const float2 w1 = __bfloat1622float2(weight[pair1]);
    float2 z0{0.0f, 0.0f};
    float2 z1{0.0f, 0.0f};
    if constexpr (Epilogue == RmsEpilogue::Gated) {
        z0 = __bfloat1622float2(z[row_base + pair0]);
        z1 = __bfloat1622float2(z[row_base + pair1]);
    }
    out[row_base + pair0] =
        __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(x0.x, inv, w0.x, z0.x),
                              rmsnorm_epilogue<Epilogue>(x0.y, inv, w0.y, z0.y));
    out[row_base + pair1] =
        __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(x1.x, inv, w1.x, z1.x),
                              rmsnorm_epilogue<Epilogue>(x1.y, inv, w1.y, z1.y));
}

// Functional fallback outside the aligned fast domains. It intentionally favors a simple complete
// implementation over another family of shape-specific paths.
template <RmsEpilogue Epilogue>
__launch_bounds__(256) __global__
    void rmsnorm_generic_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                const __nv_bfloat16* z, __nv_bfloat16* out, std::int32_t d,
                                std::int64_t rows, float eps) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }

    const std::int64_t base = row * static_cast<std::int64_t>(d);
    float sum               = 0.0f;
    for (std::int64_t i = threadIdx.x; i < static_cast<std::int64_t>(d); i += blockDim.x) {
        const float xv = __bfloat162float(x[base + i]);
        sum += xv * xv;
    }

    __shared__ float scratch[256];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) { scratch[threadIdx.x] += scratch[threadIdx.x + stride]; }
        __syncthreads();
    }

    const float inv = rsqrtf(scratch[0] / static_cast<float>(d) + eps);
    for (std::int64_t i = threadIdx.x; i < static_cast<std::int64_t>(d); i += blockDim.x) {
        const std::int64_t index = base + i;
        const float xv           = __bfloat162float(x[index]);
        const float wv           = __bfloat162float(weight[i]);
        float zv                 = 0.0f;
        if constexpr (Epilogue == RmsEpilogue::Gated) { zv = __bfloat162float(z[index]); }
        out[index] = __float2bfloat16_rn(rmsnorm_epilogue<Epilogue>(xv, inv, wv, zv));
    }
}

} // namespace ninfer::ops
