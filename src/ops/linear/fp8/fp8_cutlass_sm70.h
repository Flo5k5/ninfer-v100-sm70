#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t fp8_cutlass_sm70_workspace_bytes(std::int32_t n, std::int32_t k,
                                                            std::int32_t cols);

// Row-scaled E4M3 Weight x BF16 activations -> contiguous BF16 [n, cols]. The row scale
// multiplies the FP32 accumulator, so every output is rounded once. Intended for wide-T Volta
// routes; narrow-T paths should keep the packed QPN/GEMV implementations.
void fp8_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                             cudaStream_t stream);

// Same projection accumulated onto an FP32 residual stream [n, cols]: residual += x W^T from the
// row-scaled FP32 accumulators, rounded once to FP32. Same workspace as fp8_cutlass_sm70_launch.
void fp8_cutlass_sm70_residual_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                      WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops::detail
