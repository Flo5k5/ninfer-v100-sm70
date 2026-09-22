#include "ops/linear/nvfp4/nvfp4_prepack_sm70.h"

#include "core/arena.h"
#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_format.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

__device__ __forceinline__ std::uint8_t native_nibble(const std::uint8_t* row, int k) {
    const std::uint8_t byte = row[k >> 1];
    return (k & 1) == 0 ? byte & 0x0fu : byte >> 4;
}

__global__ void prepack_qpn_kernel(const std::uint8_t* __restrict__ input_codes,
                                   const std::uint8_t* __restrict__ input_scales,
                                   std::uint8_t* __restrict__ output_codes,
                                   std::uint8_t* __restrict__ output_scales, int n, int k,
                                   bool swiglu_interleave) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int groups         = k / 16;
    const std::int64_t count = static_cast<std::int64_t>(n / 32) * groups * 32;
    if (index >= count) { return; }

    const int lane  = static_cast<int>(index % 32);
    const int group = static_cast<int>((index / 32) % groups);
    const int tile  = static_cast<int>(index / (static_cast<std::int64_t>(groups) * 32));
    const int physical_row =
        tile * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
    // SwiGLU interleave: tile j = gate rows [16j, 16j+16) then up rows [n/2 + 16j, n/2 + 16j + 16).
    const int local = physical_row & 31;
    const int row   = !swiglu_interleave ? physical_row
                      : local < 16      ? tile * 16 + local
                                        : n / 2 + tile * 16 + (local - 16);
    const std::uint8_t* source_row = input_codes + static_cast<std::int64_t>(row) * (k / 2);
    constexpr int order[16] = {0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15};
    std::uint8_t* destination = output_codes + index * 8;
#pragma unroll
    for (int byte = 0; byte < 8; ++byte) {
        const int low_k  = group * 16 + order[2 * byte];
        const int high_k = group * 16 + order[2 * byte + 1];
        destination[byte] = static_cast<std::uint8_t>(native_nibble(source_row, low_k) |
                                                      (native_nibble(source_row, high_k) << 4));
    }

    const int scale_tile      = group / 4;
    const int scale_lane      = group & 3;
    const int row_inner       = row & 127;
    const int scales_per_m128 = k / 64;
    const std::int64_t scale_offset =
        static_cast<std::int64_t>((row / 128) * scales_per_m128 + scale_tile) * 512 +
        (row_inner & 31) * 16 + (row_inner >> 5) * 4 + scale_lane;
    output_scales[index] = input_scales[scale_offset];
}

} // namespace

void nvfp4_prepack_qpn_sm70(Weight& weight, cudaStream_t stream, bool swiglu_interleave) {
    const Nvfp4WeightGeometry geometry = validate_nvfp4_weight(weight, "NVFP4 QPN prepack");
    if (weight.layout != QuantLayout::BlockScaleK16M128x4) {
        throw std::invalid_argument("NVFP4 QPN prepack: weight is already prepacked");
    }
    if (swiglu_interleave && weight.n % 64 != 0) {
        throw std::invalid_argument("NVFP4 QPN SwiGLU prepack: gate/up halves must be 32-aligned");
    }
    DeviceBuffer scratch(static_cast<std::size_t>(geometry.code_plane_bytes +
                                                  geometry.scale_plane_bytes));
    auto* packed_codes  = static_cast<std::uint8_t*>(scratch.p);
    auto* packed_scales = packed_codes + geometry.code_plane_bytes;
    const std::int64_t tuples = static_cast<std::int64_t>(weight.n / 32) * (weight.k / 16) * 32;
    prepack_qpn_kernel<<<static_cast<int>((tuples + 255) / 256), 256, 0, stream>>>(
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), packed_codes, packed_scales, weight.n,
        weight.k, swiglu_interleave);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(const_cast<void*>(weight.qdata), packed_codes,
                               static_cast<std::size_t>(geometry.code_plane_bytes),
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(const_cast<void*>(weight.scales), packed_scales,
                               static_cast<std::size_t>(geometry.scale_plane_bytes),
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    weight.layout = swiglu_interleave ? QuantLayout::VoltaQpnPrepackedSwiGlu
                                      : QuantLayout::VoltaQpnPrepacked;
}

} // namespace ninfer::ops::detail
