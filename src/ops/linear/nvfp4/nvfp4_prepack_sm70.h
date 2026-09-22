#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// In-place, load-time permutation from the portable NVFP4 artifact layout to the fragment order
// consumed by Volta QPN2. The payload size is unchanged and no transformed copy is retained.
// `swiglu_interleave` is for a fused [gate; up] weight consumed only by linear_swiglu: it pairs
// gate row f with up row f inside every 32-row tile (QuantLayout::VoltaQpnPrepackedSwiGlu) so
// the QPN2 epilogue applies SiLU(gate) * up itself.
void nvfp4_prepack_qpn_sm70(Weight& weight, cudaStream_t stream = nullptr,
                            bool swiglu_interleave = false);

} // namespace ninfer::ops::detail
