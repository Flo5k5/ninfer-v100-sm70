#pragma once

#include "ops/common/fp16_activation.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

struct Fp8IdentityEpilogue {
    __device__ __forceinline__ float apply(std::int32_t, std::int32_t, float value) const {
        return value;
    }
};

struct Fp8ContiguousOutput {
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

struct Fp8ResidualOutput {
    __nv_bfloat16* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        const std::int64_t index = static_cast<std::int64_t>(token) * rows + parent_row;
        data[index] = __float2bfloat16_rn(value + __bfloat162float(data[index]));
    }
};

// SwiGLU straight out of the QPN epilogue, for a gate/up weight prepacked with
// QuantLayout::VoltaQpnPrepackedSwiGlu: every CTA's 32 columns are 16 gate features and the same
// 16 up features, already row-scaled, and the fp32 SiLU(gate) * up gets its single rounding here.
// `Out` is BF16, or FP16 carrying the BF16-rounded value (the fp16 activation domain).
template <class Out>
struct Fp8SwiGluPairOutputT {
    static constexpr bool kSwiGluPairs = true;
    Out* data;
    std::int32_t features;

    __device__ __forceinline__ void store_pair(std::int32_t feature, std::int32_t token, float gate,
                                               float up) const {
        data[static_cast<std::int64_t>(token) * features + feature] =
            round_activation<Out>(silu(gate) * up);
    }
};

template <class Policy, class = void>
struct fp8_swiglu_pairs : std::false_type {};
template <class Policy>
struct fp8_swiglu_pairs<Policy, std::void_t<decltype(Policy::kSwiGluPairs)>>
    : std::bool_constant<Policy::kSwiGluPairs> {};
template <class Policy>
inline constexpr bool fp8_swiglu_pairs_v = fp8_swiglu_pairs<Policy>::value;

// Split-projection SwiGLU keeps both projection results in FP32 until the fused
// SiLU/multiply epilogue performs the single observable BF16 rounding.
struct Fp8Fp32ContiguousOutput {
    float* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        data[static_cast<std::int64_t>(token) * rows + parent_row] = value;
    }
};

} // namespace ninfer::ops::detail
