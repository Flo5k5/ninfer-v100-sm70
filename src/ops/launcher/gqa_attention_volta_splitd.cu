// ninfer::ops::detail - Volta (sm_70) FP32-accumulating flash-attention prefill route.
//
// Drives the vendored 1CatAI "Split-D N32" D256 kernel (third_party/sm70_flash_d256): 64 query
// rows per CTA, FP16 tensor-core operands, and FP32 accumulators for both Q.K^T and P.V. The
// llama.cpp MMA kernel used by the `flash` route keeps its P.V accumulators in FP16 on Volta
// (register budget of its D256 tile), so their rounding grows with the number of keys a row
// attends to; this route has no such term.
//
// Staging mirrors the flash route: the width's K/V are appended to the paged cache and the
// visible key range is gathered once per layer into contiguous FP16 (dequantized for INT8). Each
// Q-block stages Q to FP16 in the kernel's [head][row][256] layout (normalized-Hadamard rotated
// for INT8, whose persistent K lives in the rotated basis), runs the kernel (three-way KV split
// with an FP32 merge for long prefixes), and converts the FP32 output to BF16.
//
// This translation unit is the only one that sees the vendored headers (CuTe-based); the vendored
// include directory is scoped to it in CMake.

#include "fattn-sm70-d256-kernel.cuh" // vendored

#include "core/device.h"
#include "core/tensor.h"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using SplitDTraits = FLASH_NAMESPACE::Sm70D256SplitDTraits;

constexpr int kHeadDim = 256;
static_assert(SplitDTraits::kHeadDim == kHeadDim);
static_assert(SplitDTraits::kBlockM == kVoltaSplitDBlockRows);

constexpr float kLog2E = 1.4426950408889634f;

// BF16 q [256, QHeads, T] (d fastest) -> FP16 [QHeads][q_pad][256]. One warp owns one
// (row, head) pair; rows at or past `tokens` are written as zero (the kernel's Q tile copy is
// unguarded, and their outputs are never read back).
template <bool Rotate>
__launch_bounds__(256) __global__ void volta_splitd_stage_q_kernel(
        const __nv_bfloat16* __restrict__ q, __half* __restrict__ staged, int q_heads,
        int tokens, int q_pad) {
    constexpr int kWarps = 8;
    const int unit       = static_cast<int>(blockIdx.x) * kWarps + (static_cast<int>(threadIdx.x) >> 5);
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    if (unit >= q_pad * q_heads) { return; }
    const int head = unit / q_pad;
    const int row  = unit % q_pad;
    __half* destination =
        staged + (static_cast<std::int64_t>(head) * q_pad + row) * kHeadDim;
    if (row >= tokens) {
#pragma unroll
        for (int item = 0; item < 8; ++item) { destination[lane + 32 * item] = __float2half(0.0f); }
        return;
    }
    const __nv_bfloat16* source =
        q + (static_cast<std::int64_t>(row) * q_heads + head) * kHeadDim;
    float values[8];
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        values[item] = __bfloat162float(source[lane + 32 * item]);
    }
    if constexpr (Rotate) { normalized_hadamard_d256_inplace(values, lane); }
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        destination[lane + 32 * item] = __float2half_rn(values[item]);
    }
}

// FP32 [q_pad][QHeads][256] -> BF16 [T][QHeads][256]: the first `count` elements share the order.
__global__ void volta_splitd_convert_out_kernel(const float* __restrict__ in,
                                                __nv_bfloat16* __restrict__ out,
                                                std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    out[i] = __float2bfloat16_rn(in[i]);
}

using Element = cutlass::half_t;

const auto kDenseKernel =
    FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<Element, false, float, false, false, false>;
const auto kSplitKernel =
    FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<Element, false, float, true, false, false>;

void raise_shared_memory_limit() {
    static const cudaError_t dense = cudaFuncSetAttribute(
        kDenseKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SplitDTraits::kSmemBytes);
    static const cudaError_t split = cudaFuncSetAttribute(
        kSplitKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SplitDTraits::kSmemBytes);
    CUDA_CHECK(dense);
    CUDA_CHECK(split);
}

std::int32_t padded_rows(std::int32_t tokens) {
    return (tokens + kVoltaSplitDBlockRows - 1) / kVoltaSplitDBlockRows * kVoltaSplitDBlockRows;
}

} // namespace

VoltaSplitDWorkspaceShape causal_attention_volta_splitd_workspace_shape(std::int32_t q_heads,
                                                                        std::int32_t tokens) {
    const std::int32_t q_pad = padded_rows(std::min(tokens, kVoltaFlashQBlockTokens));
    const std::int64_t rows  = static_cast<std::int64_t>(q_pad) * q_heads;
    return {
        .staged_q_halves   = rows * kHeadDim,
        .output_floats     = rows * kHeadDim,
        .partial_floats    = 3 * rows * (kHeadDim + 2),
    };
}

void causal_attention_volta_splitd_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                          const Tensor& positions, const Tensor& table_rows,
                                          float scale, PagedKVBatchLayerView cache,
                                          CausalAttentionExecutionEnvelope envelope,
                                          std::int32_t kv_heads, Tensor& k_gathered,
                                          Tensor& v_gathered, Tensor& staged_q,
                                          Tensor& staged_out, Tensor& partials, Tensor& out,
                                          cudaStream_t stream) {
    const std::int32_t q_heads    = q.ne[1];
    const std::int32_t width      = q.ne[2];
    const std::int32_t n_kv_total = static_cast<std::int32_t>(envelope.max_visible_keys);
    if (q_heads % kv_heads != 0) {
        throw std::invalid_argument("gqa_attention volta split-D: invalid head geometry");
    }
    raise_shared_memory_limit();

    // 1-2. Append this call's K/V and gather the visible key range to contiguous FP16
    //      [key][kv_head][256], zero past the visible keys.
    causal_attention_volta_stage_kv(k, v, positions, table_rows, cache, envelope, kv_heads, width,
                                    k_gathered, v_gathered, stream);

    const bool rotate_q    = cache.storage == KvCacheStorage::Int8Group64;
    const float scale_log2 = scale * kLog2E;
    const auto* keys       = static_cast<const Element*>(k_gathered.data);
    const auto* values     = static_cast<const Element*>(v_gathered.data);
    auto* q_staged         = static_cast<__half*>(staged_q.data);
    auto* o_staged         = static_cast<float*>(staged_out.data);

    // 3. Q-blocks. Prefill positions are contiguous: token t of this call sits at absolute
    //    position n_kv_total - width + t, so a block's rows see keys [0, base + begin + row].
    const std::int32_t base = n_kv_total - width;
    for (std::int32_t begin = 0; begin < width; begin += kVoltaFlashQBlockTokens) {
        const std::int32_t tokens = std::min(kVoltaFlashQBlockTokens, width - begin);
        const std::int32_t q_pad  = padded_rows(tokens);
        const std::int32_t kv_len = base + begin + tokens;
        const std::int64_t rows   = static_cast<std::int64_t>(q_pad) * q_heads;

        const auto* q_begin = static_cast<const __nv_bfloat16*>(q.data) +
                              static_cast<std::int64_t>(begin) * q_heads * kHeadDim;
        constexpr int kStageWarps = 8;
        const int stage_blocks    = static_cast<int>((rows + kStageWarps - 1) / kStageWarps);
        if (rotate_q) {
            volta_splitd_stage_q_kernel<true><<<stage_blocks, kStageWarps * 32, 0, stream>>>(
                q_begin, q_staged, q_heads, tokens, q_pad);
        } else {
            volta_splitd_stage_q_kernel<false><<<stage_blocks, kStageWarps * 32, 0, stream>>>(
                q_begin, q_staged, q_heads, tokens, q_pad);
        }
        CUDA_CHECK(cudaGetLastError());

        const bool split = kv_len >= kVoltaSplitDKeySplitMinimum && kv_len > tokens;
        float* partial_out = static_cast<float*>(partials.data);
        float* partial_max = partial_out + 3 * rows * kHeadDim;
        float* partial_sum = partial_max + 3 * rows;
        const dim3 grid(static_cast<unsigned>(q_pad / kVoltaSplitDBlockRows), split ? 3u : 1u,
                        static_cast<unsigned>(q_heads));
        const dim3 block(SplitDTraits::kNThreads);
        (split ? kSplitKernel : kDenseKernel)<<<grid, block, SplitDTraits::kSmemBytes, stream>>>(
            reinterpret_cast<const Element*>(q_staged), keys, values, o_staged,
            /*q_batch_stride=*/static_cast<int>(rows * kHeadDim),
            /*q_row_stride=*/kHeadDim,
            /*q_head_stride=*/q_pad * kHeadDim,
            /*k_outer_stride=*/0,
            /*k_row_stride=*/kv_heads * kHeadDim,
            /*k_head_stride=*/kHeadDim,
            /*v_outer_stride=*/0,
            /*v_row_stride=*/kv_heads * kHeadDim,
            /*v_head_stride=*/kHeadDim,
            /*query_len=*/q_pad, kv_len, q_heads, kv_heads,
            /*kv_offset=*/kv_len - tokens, scale_log2,
            /*block_table=*/nullptr, /*page_size=*/0, /*block_table_batch_stride=*/0,
            split ? partial_out : nullptr, split ? partial_max : nullptr,
            split ? partial_sum : nullptr);
        CUDA_CHECK(cudaGetLastError());
        if (split) {
            FLASH_NAMESPACE::sm70_d256_splitkv3_merge_kernel<<<static_cast<unsigned>(rows),
                                                              kHeadDim, 0, stream>>>(
                partial_out, partial_max, partial_sum, o_staged, rows, scale_log2);
            CUDA_CHECK(cudaGetLastError());
        }

        const std::int64_t count = static_cast<std::int64_t>(tokens) * q_heads * kHeadDim;
        constexpr int kConvertThreads = 256;
        volta_splitd_convert_out_kernel<<<static_cast<int>((count + kConvertThreads - 1) /
                                                           kConvertThreads),
                                          kConvertThreads, 0, stream>>>(
            o_staged,
            static_cast<__nv_bfloat16*>(out.data) +
                static_cast<std::int64_t>(begin) * q_heads * kHeadDim,
            count);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops::detail
