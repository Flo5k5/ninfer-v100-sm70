#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// In-place load-time permutation from row-major FP8 codes to the coalesced Volta QPN8 stream.
// Row scales remain in their registered order and the payload size is unchanged.
// `swiglu_interleave` is for a fused [gate; up] weight consumed only by linear_swiglu: each
// 32-row tile holds 16 gate rows then the matching 16 up rows (QuantLayout::
// VoltaQpnPrepackedSwiGlu) so one QPN CTA applies SwiGLU in its epilogue.
void fp8_prepack_qpn_sm70(Weight& weight, cudaStream_t stream = nullptr,
                          bool swiglu_interleave = false);

} // namespace ninfer::ops::detail
