#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                                    std::int32_t input_rows,
                                                                    LinearPolicy policy,
                                                                    std::int32_t min_tokens,
                                                                    std::int32_t max_tokens);

void nvfp4_linear_add_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                    cudaStream_t stream);

void nvfp4_linear_add_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                     cudaStream_t stream);

void nvfp4_linear_add_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  Nvfp4W4a4Workspace workspace, cudaStream_t stream);

#ifdef NINFER_VOLTA_BUILD
[[nodiscard]] bool nvfp4_linear_add_fp16_activation_supported(const Weight& weight,
                                                              LinearPolicy policy,
                                                              std::int32_t tokens);
// True when the resolved route accepts an FP32 residual: the Volta linear-then-add route, whose
// QPN epilogue (T <= 32) and CUTLASS GEMM (T >= 33) update it from FP32 accumulators with one
// rounding. The W4A4 route writes BF16 and does not.
[[nodiscard]] bool nvfp4_linear_add_fp32_residual_supported(const Weight& weight,
                                                            LinearPolicy policy,
                                                            std::int32_t tokens);
#endif
void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                               LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
