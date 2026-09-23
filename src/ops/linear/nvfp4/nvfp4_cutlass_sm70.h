#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_cutlass_sm70_workspace_bytes(std::int32_t n, std::int32_t k,
                                                              std::int32_t cols);
void nvfp4_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                               cudaStream_t stream);

// Same projection accumulated onto an FP32 residual stream [n, cols]: residual += x W^T from the
// FP32 accumulators, with no intermediate rounding. Same workspace as nvfp4_cutlass_sm70_launch.
void nvfp4_cutlass_sm70_residual_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                        WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops::detail
