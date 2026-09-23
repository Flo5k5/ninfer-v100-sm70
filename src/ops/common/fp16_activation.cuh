#pragma once

// The Volta fp16 activation domain. Volta's mma takes fp16 operands only, so every QPN GEMV
// stages a BF16 activation to fp16 in a separate pass before reading it. A producer that knows its
// consumer is such a GEMV can write that copy itself: an FP16 activation tensor holds the
// BF16-rounded value converted to fp16, which is bit-identical to what the staging pass would
// have produced from the BF16 tensor (BF16 -> fp16 is exact across fp16's range).

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ninfer::ops {

template <class Out>
__device__ __forceinline__ Out round_activation(float value);

template <>
__device__ __forceinline__ __nv_bfloat16 round_activation<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <>
__device__ __forceinline__ __half round_activation<__half>(float value) {
    return __float2half_rn(__bfloat162float(__float2bfloat16_rn(value)));
}

} // namespace ninfer::ops
