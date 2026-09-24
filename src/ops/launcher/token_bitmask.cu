// Implements: include/ninfer/ops/token_bitmask.h
// Match: validated contiguous BF16 logits, I32 bitmask and I32 per-row column counts.
// Algorithm assumptions: grid y/z cover columns and rows and split only past the CUDA grid limit.
#include "ops/launcher/token_bitmask.h"

#include "ops/common/math.h"
#include "ops/kernel/token_bitmask.cuh"
#include "core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kMaximumGridYZ = 65535;

bool vector_route(const Tensor& logits) {
    const auto base = reinterpret_cast<std::uintptr_t>(logits.data);
    return logits.ne[0] % kTokenBitmaskGroup == 0 && base % sizeof(uint4) == 0;
}

} // namespace

void apply_token_bitmask_launch(Tensor& logits, const Tensor& bitmask, const Tensor& mask_columns,
                                std::int32_t token_domain, cudaStream_t stream) {
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t width         = logits.ne[1];
    const std::int32_t batch         = logits.ne[2];
    const bool vector                = vector_route(logits);
    const std::int32_t tokens_per_block =
        vector ? kTokenBitmaskBlock * kTokenBitmaskGroup : kTokenBitmaskBlock;
    const auto x_blocks = static_cast<unsigned int>(div_up(physical_rows, tokens_per_block));

    TokenBitmaskGeometry geometry{
        .physical_rows         = physical_rows,
        .token_domain          = token_domain,
        .words                 = bitmask.ne[0],
        .width                 = width,
        .mask_columns_capacity = bitmask.ne[1],
    };
    auto* logits_data        = static_cast<__nv_bfloat16*>(logits.data);
    const auto* mask_data    = static_cast<const std::int32_t*>(bitmask.data);
    const auto* columns_data = static_cast<const std::int32_t*>(mask_columns.data);
    for (std::int32_t row = 0; row < batch; row += kMaximumGridYZ) {
        for (std::int32_t column = 0; column < width; column += kMaximumGridYZ) {
            geometry.row_base    = row;
            geometry.column_base = column;
            const dim3 grid(x_blocks,
                            static_cast<unsigned int>(std::min(kMaximumGridYZ, width - column)),
                            static_cast<unsigned int>(std::min(kMaximumGridYZ, batch - row)));
            if (vector) {
                apply_token_bitmask_vector_kernel<<<grid, kTokenBitmaskBlock, 0, stream>>>(
                    logits_data, mask_data, columns_data, geometry);
            } else {
                apply_token_bitmask_scalar_kernel<<<grid, kTokenBitmaskBlock, 0, stream>>>(
                    logits_data, mask_data, columns_data, geometry);
            }
            CUDA_CHECK(cudaGetLastError());
        }
    }
}

} // namespace ninfer::ops::detail
