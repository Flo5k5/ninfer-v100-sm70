#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

struct Nvfp4IdentityEpilogue {
    __device__ __forceinline__ float apply(std::int32_t, std::int32_t, float value) const {
        return value;
    }
};

struct Nvfp4ContiguousOutput {
    __nv_bfloat16* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        data[static_cast<std::int64_t>(token) * rows + parent_row] = __float2bfloat16_rn(value);
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        auto* destination = data + static_cast<std::int64_t>(token) * rows + parent_row;
        store_vec(destination, values);
    }
};

// Adds the projection into a BF16 residual stream in place: one fp32 add and a single BF16
// round, where materializing the projection first and running residual_add rounds twice and costs
// a second launch (the FP8 sibling's Fp8ResidualOutput). Each (row, token) is owned by exactly one
// thread of one CTA, so the read-modify-write needs no atomics.
struct Nvfp4ResidualOutput {
    __nv_bfloat16* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        const std::int64_t index = static_cast<std::int64_t>(token) * rows + parent_row;
        data[index]              = __float2bfloat16_rn(value + __bfloat162float(data[index]));
    }
};

// SwiGLU straight out of the QPN2 epilogue, for a gate/up weight prepacked with
// QuantLayout::VoltaQpnPrepackedSwiGlu: every CTA's 32 columns are 16 gate features and the same
// 16 up features, so the kernel hands both halves of a feature to store_pair and the fp32
// SiLU(gate) * up gets its single BF16 round here -- no fp32 scratch planes, no combine launch.
struct Nvfp4SwiGluPairOutput {
    static constexpr bool kSwiGluPairs = true;
    __nv_bfloat16* data;
    std::int32_t features;

    __device__ __forceinline__ void store_pair(std::int32_t feature, std::int32_t token, float gate,
                                               float up) const {
        data[static_cast<std::int64_t>(token) * features + feature] =
            __float2bfloat16_rn(silu(gate) * up);
    }
};

template <class Policy, class = void>
struct nvfp4_swiglu_pairs : std::false_type {};
template <class Policy>
struct nvfp4_swiglu_pairs<Policy, std::void_t<decltype(Policy::kSwiGluPairs)>>
    : std::bool_constant<Policy::kSwiGluPairs> {};
template <class Policy>
inline constexpr bool nvfp4_swiglu_pairs_v = nvfp4_swiglu_pairs<Policy>::value;

// Writes the mma accumulator straight through, no BF16 round. Exists for split-projection SwiGLU
// (see nvfp4_linear_swiglu_qpn_split.cuh): two independent QPN2 launches -- one per weight half,
// unmodified, at whatever schedule QPN2 already measured fastest for this shape -- write gate and
// up into two fp32 scratch planes, and a small combine kernel applies silu(gate)*up in fp32 before
// the single BF16 round. Splitting keeps each launch at QPN2's own tuned register/occupancy
// profile instead of the fused kernel's doubled one; see the plan.cpp route for the measured
// comparison.
struct Nvfp4Fp32ContiguousOutput {
    float* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        data[static_cast<std::int64_t>(token) * rows + parent_row] = value;
    }
};

// One launch over a fused [gate; up] weight whose halves land in two separate FP32 planes, so the
// SwiGLU combine still sees the per-half layout Nvfp4Fp32ContiguousOutput gives it.
struct Nvfp4Fp32SplitContiguousOutput {
    float* lower;
    float* upper;
    std::int32_t half_rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        const bool is_upper = parent_row >= half_rows;
        float* data         = is_upper ? upper : lower;
        const std::int32_t row = is_upper ? parent_row - half_rows : parent_row;
        data[static_cast<std::int64_t>(token) * half_rows + row] = value;
    }
};

} // namespace ninfer::ops::detail
