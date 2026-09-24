#pragma once

// ninfer::ops::detail - private launch prototype for apply_token_bitmask.

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void apply_token_bitmask_launch(Tensor& logits, const Tensor& bitmask, const Tensor& mask_columns,
                                std::int32_t token_domain, cudaStream_t stream);

} // namespace ninfer::ops::detail
