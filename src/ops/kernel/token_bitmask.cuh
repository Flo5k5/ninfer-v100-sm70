#pragma once

// Implements: include/ninfer/ops/token_bitmask.h
// Match: contiguous BF16 logits. The vector route needs every column to start on a 16-byte
// boundary (an aligned base and physical_rows a multiple of eight); other shapes take the
// one-token route.
// Algorithm assumptions: a vector thread owns eight consecutive tokens of one column, which lie in
// one mask word. Fully allowed groups are skipped, fully excluded groups are stored without a load,
// and mixed groups are blended. Unconstrained columns exit before touching memory.

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kTokenBitmaskBlock              = 256;
inline constexpr int kTokenBitmaskGroup              = 8;
inline constexpr std::uint32_t kBf16NegativeInfinity = 0xff80u;

struct TokenBitmaskGeometry {
    std::int32_t physical_rows = 0;
    std::int32_t token_domain  = 0;
    std::int32_t words         = 0;
    std::int32_t width         = 0;
    // Column capacity C of the [words,C,R] mask buffer.
    std::int32_t mask_columns_capacity = 0;
    // First logits column and row of this launch; launches split only beyond grid limits.
    std::int32_t column_base = 0;
    std::int32_t row_base    = 0;
};

__device__ __forceinline__ const std::uint32_t* token_bitmask_column(const std::int32_t* bitmask,
                                                                     const TokenBitmaskGeometry& g,
                                                                     std::int32_t row,
                                                                     std::int32_t column) {
    return reinterpret_cast<const std::uint32_t*>(bitmask) +
           (static_cast<std::int64_t>(row) * g.mask_columns_capacity + column) * g.words;
}

// Allowed bits of tokens [first, first + 8), bit k for token first + k. Tokens at or beyond the
// token domain are never allowed; `first` is a multiple of eight, so the group lies in one word.
__device__ __forceinline__ std::uint32_t
token_bitmask_group(const std::uint32_t* mask, std::int32_t first, std::int32_t token_domain) {
    if (first >= token_domain) { return 0u; }
    std::uint32_t bits           = (mask[first >> 5] >> (first & 31)) & 0xffu;
    const std::int32_t in_domain = token_domain - first;
    if (in_domain < kTokenBitmaskGroup) { bits &= (1u << in_domain) - 1u; }
    return bits;
}

__launch_bounds__(kTokenBitmaskBlock) __global__
    void apply_token_bitmask_vector_kernel(__nv_bfloat16* logits, const std::int32_t* bitmask,
                                           const std::int32_t* mask_columns,
                                           TokenBitmaskGeometry g) {
    const std::int32_t column = g.column_base + static_cast<std::int32_t>(blockIdx.y);
    const std::int32_t row    = g.row_base + static_cast<std::int32_t>(blockIdx.z);
    if (column >= mask_columns[row]) { return; }
    const std::int32_t first = (static_cast<std::int32_t>(blockIdx.x) * kTokenBitmaskBlock +
                                static_cast<std::int32_t>(threadIdx.x)) *
                               kTokenBitmaskGroup;
    if (first >= g.physical_rows) { return; }

    const std::uint32_t allowed =
        token_bitmask_group(token_bitmask_column(bitmask, g, row, column), first, g.token_domain);
    if (allowed == 0xffu) { return; }
    auto* group = reinterpret_cast<uint4*>(
        logits + (static_cast<std::int64_t>(row) * g.width + column) * g.physical_rows + first);
    constexpr std::uint32_t kPair = (kBf16NegativeInfinity << 16) | kBf16NegativeInfinity;
    uint4 values                  = make_uint4(kPair, kPair, kPair, kPair);
    if (allowed != 0u) {
        values               = *group;
        std::uint32_t* pairs = reinterpret_cast<std::uint32_t*>(&values);
#pragma unroll
        for (int pair = 0; pair < kTokenBitmaskGroup / 2; ++pair) {
            // Little-endian: the even token of a pair is the low half of its word.
            std::uint32_t bits = pairs[pair];
            if (((allowed >> (2 * pair)) & 1u) == 0u) {
                bits = (bits & 0xffff0000u) | kBf16NegativeInfinity;
            }
            if (((allowed >> (2 * pair + 1)) & 1u) == 0u) {
                bits = (bits & 0x0000ffffu) | (kBf16NegativeInfinity << 16);
            }
            pairs[pair] = bits;
        }
    }
    *group = values;
}

__launch_bounds__(kTokenBitmaskBlock) __global__
    void apply_token_bitmask_scalar_kernel(__nv_bfloat16* logits, const std::int32_t* bitmask,
                                           const std::int32_t* mask_columns,
                                           TokenBitmaskGeometry g) {
    const std::int32_t column = g.column_base + static_cast<std::int32_t>(blockIdx.y);
    const std::int32_t row    = g.row_base + static_cast<std::int32_t>(blockIdx.z);
    if (column >= mask_columns[row]) { return; }
    const std::int32_t token = static_cast<std::int32_t>(blockIdx.x) * kTokenBitmaskBlock +
                               static_cast<std::int32_t>(threadIdx.x);
    if (token >= g.physical_rows) { return; }
    const std::uint32_t* mask = token_bitmask_column(bitmask, g, row, column);
    const bool allowed = token < g.token_domain && ((mask[token >> 5] >> (token & 31)) & 1u) != 0u;
    if (allowed) { return; }
    logits[(static_cast<std::int64_t>(row) * g.width + column) * g.physical_rows + token] =
        __ushort_as_bfloat16(static_cast<unsigned short>(kBf16NegativeInfinity));
}

} // namespace ninfer::ops
