#pragma once

// ninfer::ops::detail - private launch prototypes for causal_softmax_attention policies.

#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/ops/softmax_attention.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class CausalAttentionRoute { SmallT, ChunkedSmallT, Prompt };

struct CausalSmallTInvocation {
    const Tensor* valid_columns = nullptr;
    const Tensor* table_rows    = nullptr;
    std::int32_t full_width     = 0;
    std::int32_t column_begin   = 0;
    std::int32_t width          = 0;
    std::int32_t batch_size     = 1;
};

std::int32_t causal_attention_split_capacity(std::int32_t q_heads, std::int32_t tokens,
                                             KvCacheStorage cache_storage,
                                             CausalAttentionExecutionEnvelope envelope,
                                             std::int32_t batch_size = 1);

CausalAttentionRoute causal_attention_resolve_route(std::int32_t q_heads, std::int32_t width,
                                                    std::int32_t batch_size, KvCacheStorage storage,
                                                    CausalAttentionExecutionEnvelope envelope);

const char* causal_attention_route_name(CausalAttentionRoute route);

void causal_attention_small_t_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream);

void causal_attention_cached_small_t_launch(const Tensor& q, const Tensor& positions, float scale,
                                            const PagedKVLayerView& cache,
                                            CausalAttentionExecutionEnvelope envelope,
                                            Tensor& partial_acc, Tensor& partial_m,
                                            Tensor& partial_l, Tensor& out, cudaStream_t stream);

void causal_attention_small_t_fp8_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream);

void causal_attention_cached_small_t_fp8_launch(const Tensor& q, const Tensor& positions,
                                                float scale, const PagedKVLayerView& cache,
                                                CausalAttentionExecutionEnvelope envelope,
                                                Tensor& partial_acc, Tensor& partial_m,
                                                Tensor& partial_l, Tensor& out,
                                                cudaStream_t stream);

void causal_attention_small_t_nvfp4_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream);

void causal_attention_cached_small_t_nvfp4_launch(const Tensor& q, const Tensor& positions,
                                                  float scale, const PagedKVLayerView& cache,
                                                  CausalAttentionExecutionEnvelope envelope,
                                                  Tensor& partial_acc, Tensor& partial_m,
                                                  Tensor& partial_l, Tensor& out,
                                                  cudaStream_t stream);

void causal_attention_small_t_k8v4_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream);

void causal_attention_cached_small_t_k8v4_launch(const Tensor& q, const Tensor& positions,
                                                 float scale, const PagedKVLayerView& cache,
                                                 CausalAttentionExecutionEnvelope envelope,
                                                 Tensor& partial_acc, Tensor& partial_m,
                                                 Tensor& partial_l, Tensor& out,
                                                 cudaStream_t stream);

void causal_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& positions, const Tensor& valid_columns,
                                    const Tensor& table_rows, float scale,
                                    PagedKVBatchLayerView cache, Tensor& out, cudaStream_t stream);

void causal_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                              const PagedKVLayerView& cache, Tensor& out,
                                              cudaStream_t stream);

void causal_attention_prompt_fp8_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                        const Tensor& positions, const Tensor& valid_columns,
                                        const Tensor& table_rows, float scale,
                                        PagedKVBatchLayerView cache, Tensor& out,
                                        cudaStream_t stream);

void causal_attention_prompt_fp8_attention_launch(const Tensor& q, const Tensor& positions,
                                                  float scale, const PagedKVLayerView& cache,
                                                  Tensor& out, cudaStream_t stream);

void causal_attention_prompt_nvfp4_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                          const Tensor& positions, const Tensor& valid_columns,
                                          const Tensor& table_rows, float scale,
                                          PagedKVBatchLayerView cache, Tensor& out,
                                          cudaStream_t stream);

void causal_attention_prompt_nvfp4_attention_launch(const Tensor& q, const Tensor& positions,
                                                    float scale, const PagedKVLayerView& cache,
                                                    Tensor& out, cudaStream_t stream);

void causal_attention_prompt_k8v4_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const Tensor& positions, const Tensor& valid_columns,
                                         const Tensor& table_rows, float scale,
                                         PagedKVBatchLayerView cache, Tensor& out,
                                         cudaStream_t stream);

void causal_attention_prompt_k8v4_attention_launch(const Tensor& q, const Tensor& positions,
                                                   float scale, const PagedKVLayerView& cache,
                                                   Tensor& out, cudaStream_t stream);


#ifdef NINFER_VOLTA_BUILD
inline constexpr std::int32_t kVoltaFlashQBlockTokens = 1024;
inline constexpr std::int32_t kVoltaFlashMinimumWidth = 64;
inline constexpr std::int32_t kVoltaFlashMaskRowPad   = 64;
inline constexpr std::int32_t kVoltaFlashKeyPad       = 256;

inline constexpr std::int32_t kVoltaSplitDBlockRows       = 64;
inline constexpr std::int32_t kVoltaSplitDKeySplitMinimum = 2048;

// Kernel behind the wide BF16/INT8 prompt route, selected once per process by
// NINFER_VOLTA_PREFILL_ATTENTION (any other value is rejected):
//   splitd    (default) vendored Split-D kernel, FP32 Q.K^T and P.V accumulators; an inexact
//             envelope keeps the direct kernel;
//   flash     vendored llama.cpp MMA kernel, FP32 Q.K^T but FP16 P.V accumulators;
//   reference the direct FP32 kernel (slow; for numerical A/B only).
enum class VoltaPrefillAttention : std::uint8_t { SplitD, Flash, Reference };
VoltaPrefillAttention volta_prefill_attention();

struct VoltaSplitDWorkspaceShape {
    std::int64_t staged_q_halves;
    std::int64_t output_floats;
    std::int64_t partial_floats;
};

VoltaSplitDWorkspaceShape causal_attention_volta_splitd_workspace_shape(std::int32_t q_heads,
                                                                        std::int32_t tokens);

// Appends a prompt width's K/V to the paged cache and gathers the visible key range into
// contiguous FP16 [key][kv_head][256] (dequantized for INT8, zero past the visible keys).
void causal_attention_volta_stage_kv(const Tensor& k, const Tensor& v, const Tensor& positions,
                                     const Tensor& table_rows, PagedKVBatchLayerView cache,
                                     CausalAttentionExecutionEnvelope envelope,
                                     std::int32_t kv_heads, std::int32_t width, Tensor& k_gathered,
                                     Tensor& v_gathered, cudaStream_t stream);

// Split-D prompt attention. Precondition: an exact envelope (min == max visible keys) and
// sequential positions, token t of the width at max_visible_keys - width + t; every row's causal
// limit is derived from that layout rather than read from `positions`.
void causal_attention_volta_splitd_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                          const Tensor& positions, const Tensor& table_rows,
                                          float scale, PagedKVBatchLayerView cache,
                                          CausalAttentionExecutionEnvelope envelope,
                                          std::int32_t kv_heads, Tensor& k_gathered,
                                          Tensor& v_gathered, Tensor& staged_q,
                                          Tensor& staged_out, Tensor& partials, Tensor& out,
                                          cudaStream_t stream);

std::size_t causal_attention_volta_flash_meta_elements(std::int32_t q_heads, std::int32_t tokens);

void causal_attention_volta_flash_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t q_block_tokens, Tensor& k_gathered,
    Tensor& v_gathered, Tensor& mask, Tensor& q_f32, Tensor& out_f32, Tensor& dst_meta, Tensor& out,
    cudaStream_t stream);
#endif

} // namespace ninfer::ops::detail
