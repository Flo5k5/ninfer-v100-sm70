#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t fp8_gdn_snapshot_workspace_capacity_bytes(LinearPolicy policy,
                                                                    std::int32_t batch_size,
                                                                    std::int32_t min_width,
                                                                    std::int32_t max_width);

[[nodiscard]] std::size_t fp8_gdn_record_workspace_capacity_bytes(LinearPolicy policy,
                                                                  std::int32_t batch_size,
                                                                  std::int32_t min_width,
                                                                  std::int32_t max_width);

void fp8_gdn_snapshot_fused_launch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                   Tensor& conv_states, const Tensor& valid_columns,
                                   const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                                   Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                   cudaStream_t stream);

void fp8_gdn_record_fused_launch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                 const Tensor& conv_states, const Tensor& valid_columns,
                                 const Tensor& initial_slot, Tensor& conv_record, Tensor& query,
                                 Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream);

#ifdef NINFER_VOLTA_BUILD
// Record and snapshot take an FP16 x (the staged copy) where their route is one QPN pass.
[[nodiscard]] bool fp8_gdn_conv_fp16_activation_supported(LinearPolicy policy, std::int32_t width,
                                                          std::int32_t batch) noexcept;
#endif
void fp8_gdn_snapshot_dispatch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                               Tensor& conv_states, const Tensor& valid_columns,
                               const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                               Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                               LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream);

void fp8_gdn_record_dispatch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                             const Tensor& conv_states, const Tensor& valid_columns,
                             const Tensor& initial_slot, Tensor& conv_record, Tensor& query,
                             Tensor& key, Tensor& value, Tensor& z, LinearPolicy policy,
                             WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
