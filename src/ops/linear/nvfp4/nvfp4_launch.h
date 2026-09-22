#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

void launch_nvfp4_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void launch_nvfp4_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

#ifdef NINFER_VOLTA_BUILD
// Quadpair-split-N form (nvfp4_volta_qpn_gemm.cuh), the fourth sibling of the q4/w8/fp8 QPN
// kernels. The tile height is mirrored here because dispatch is host code and cannot see the
// device header; the launcher static_asserts the two agree.
inline constexpr std::int32_t kNvfp4VoltaQpnRowsPerTile = 8;
inline constexpr std::int32_t kNvfp4VoltaQpnMaxTokens   = 32;
void launch_nvfp4_volta_qpn(const Tensor&, const Weight&, Tensor&, cudaStream_t);
// Same GEMM on an fp16 copy of x staged once by the caller (fp8_stage_bf16_activation_sm70).
// Volta's mma takes fp16 only, so the bf16 form converts every activation in the inner loop, once
// per CTA and per quadpair, on the quarter-rate F2F pipe: that conversion, not DRAM, capped the
// 5120x17408 down projection (114 us with in-loop conversion, 88 us without).
void launch_nvfp4_volta_qpn_fp16(const Tensor& x, const Weight& w, const void* x_fp16, Tensor& out,
                                 cudaStream_t stream);
[[nodiscard]] bool nvfp4_volta_qpn_supported(std::int32_t n, std::int32_t k,
                                             std::int32_t t) noexcept;

// Wide-T form (nvfp4_volta_mma_gemm.cuh): T on the mma A axis, dequantizing into shared memory
// once per K-pass instead of QPN2's small-T-chunked re-streaming. Prefill's route.
void launch_nvfp4_volta_mma(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                            cudaStream_t stream, int splits_override = 0);
[[nodiscard]] bool nvfp4_volta_mma_supported(std::int32_t n, std::int32_t k,
                                             std::int32_t t) noexcept;
[[nodiscard]] std::size_t nvfp4_volta_mma_workspace_bytes(std::int32_t n, std::int32_t k,
                                                          std::int32_t t) noexcept;
[[nodiscard]] int nvfp4_volta_mma_splits(std::int32_t n, std::int32_t k, std::int32_t t) noexcept;
#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
