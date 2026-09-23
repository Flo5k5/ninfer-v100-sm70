#pragma once

// ninfer::ops::detail - private launch prototype for qk_rmsnorm_rope's fused route. The wrapper
// admits only the fixed geometries this kernel implements.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void qk_rmsnorm_rope_fused_launch(const Tensor& positions, const Tensor& q, const Tensor& k,
                                  const Tensor& q_norm, const Tensor& k_norm, Tensor& q_out,
                                  Tensor& k_out, float eps, cudaStream_t stream);

} // namespace ninfer::ops::detail
