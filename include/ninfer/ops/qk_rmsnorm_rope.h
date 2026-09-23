#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * The attention query/key preparation of the Qwen3.6 text layers as one call:
 *
 *   rmsnorm(q, q_norm_weight, eps, unit_offset=true, q_out);
 *   rmsnorm(k, k_norm_weight, eps, unit_offset=true, k_out);
 *   rope(positions, rotary_dim, theta, q_out, k_out);
 *
 * with exactly those three Ops' contracts, domains and results (bit for bit). For the fixed text
 * geometries rope() specializes -- BF16 D256 heads, 24Q/4K or 16Q/2K, rotary_dim 64, theta 1e7,
 * 1-D or 3-D MRoPE positions -- one kernel does all three per token: each warp normalizes one
 * head row with RMSNorm's own row routine and the CTA then rotates it with RoPE's own head
 * routine, instead of three latency-bound launches. Any other input takes the composed calls.
 */
void qk_rmsnorm_rope(const Tensor& q, const Tensor& k, const Tensor& q_norm_weight,
                     const Tensor& k_norm_weight, float eps, const Tensor& positions,
                     int rotary_dim, float theta, Tensor& q_out, Tensor& k_out,
                     cudaStream_t stream);

} // namespace ninfer::ops
