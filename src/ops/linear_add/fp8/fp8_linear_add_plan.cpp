#include "ops/linear_add/fp8/fp8_linear_add_plan.h"

#include "core/layout.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_config.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/fp8/fp8_cutlass_sm70.h"
#include "ops/linear/fp8/fp8_launch.h"
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Fp8LinearAddRoute : std::uint8_t {
    A16,
#ifdef NINFER_VOLTA_BUILD
    QpnResidual,
#endif
    A8,
};

Fp8LinearAddRoute resolve_route(std::int32_t output_rows, std::int32_t input_rows,
                                LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0 || output_rows != Fp8Residual6144Geometry::kOutputRows ||
        (input_rows != Fp8Residual6144Geometry::kInputRows &&
         input_rows != Fp8Residual17408Geometry::kInputRows)) {
        throw std::invalid_argument("fp8 linear_add: unsupported shape");
    }
    if (policy == LinearPolicy::A16Only) {
#ifdef NINFER_VOLTA_BUILD
        return Fp8LinearAddRoute::QpnResidual;
#else
        return Fp8LinearAddRoute::A16;
#endif
    }
    if (policy != LinearPolicy::AllowA8) {
        throw std::invalid_argument("fp8 linear_add: unsupported policy");
    }
    const std::int32_t first_a8 = input_rows == Fp8Residual6144Geometry::kInputRows ? 22 : 25;
    if (tokens >= first_a8) { return Fp8LinearAddRoute::A8; }
#ifdef NINFER_VOLTA_BUILD
    return Fp8LinearAddRoute::QpnResidual;
#else
    return Fp8LinearAddRoute::A16;
#endif
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kFp8LastSmallT) {
        const std::int32_t active = std::min(kFp8LastSmallT, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* output = static_cast<std::uint8_t*>(residual.data) +
                       static_cast<std::int64_t>(token_begin) * weight.n * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor residual_chunk(output, DType::BF16, {weight.n, active});
        if (active == 1) {
            fp8_linear_add_decode_launch(input_chunk, weight, residual_chunk, stream);
        } else {
            fp8_linear_add_small_t_launch(input_chunk, weight, residual_chunk, stream);
        }
    }
}

#ifdef NINFER_VOLTA_BUILD
template <class Allocator>
Tensor allocate_projected(Allocator& allocator, std::int32_t output_rows, std::int32_t tokens) {
    return allocator.alloc(DType::BF16, {output_rows, tokens}, 256);
}

// The QPN kernel's bf16 form converts every activation in its inner loop, once per CTA and per
// quadpair, on Volta's quarter-rate F2F pipe; staging one fp16 copy per call lifts these residual
// projections from ~455 to the ~745 GB/s the fp16-fed FP8 kernels reach.
std::size_t qpn_activation_bytes(std::int32_t input_rows, std::int32_t tokens) {
    return static_cast<std::size_t>(input_rows) * std::min(tokens, kFp8VoltaQpnMaxTokens) *
           sizeof(std::uint16_t);
}

bool workspace_fits(const WorkspaceArena& workspace, std::size_t bytes) {
    constexpr std::size_t kAlign = 256;
    const std::size_t start      = (workspace.used() + kAlign - 1) / kAlign * kAlign;
    return start + bytes <= workspace.capacity();
}

void launch_qpn_residual(const Tensor& x, const Weight& weight, Tensor& residual,
                         WorkspaceArena& workspace, cudaStream_t stream) {
    if (x.ne[1] <= kFp8VoltaQpnMaxTokens) {
        if (x.dtype == DType::FP16) {
            fp8_linear_add_qpn_launch(x, weight, x.data, residual, stream);
            return;
        }
        const std::size_t activation_bytes = qpn_activation_bytes(weight.k, x.ne[1]);
        if (!workspace_fits(workspace, activation_bytes)) {
            fp8_linear_add_qpn_launch(x, weight, nullptr, residual, stream);
            return;
        }
        auto scope            = workspace.scope();
        DeviceSpan activation = workspace.alloc_bytes(activation_bytes, 256);
        fp8_stage_bf16_activation_sm70(x, activation.data, stream);
        fp8_linear_add_qpn_launch(x, weight, activation.data, residual, stream);
        return;
    }
    auto scope       = workspace.scope();
    Tensor projected = allocate_projected(workspace, weight.n, x.ne[1]);
    fp8_cutlass_sm70_launch(x, weight, projected, workspace, stream);
    residual_add(projected, residual, stream);
}

std::size_t qpn_residual_workspace_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                         std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    if (tokens <= kFp8VoltaQpnMaxTokens) {
        (void)layout.alloc_bytes(qpn_activation_bytes(input_rows, tokens), 256);
        return layout.peak_bytes(1);
    }
    (void)allocate_projected(layout, output_rows, tokens);
    const std::size_t linear_bytes =
        fp8_cutlass_sm70_workspace_bytes(output_rows, input_rows, tokens);
    (void)layout.alloc_bytes(linear_bytes, 256);
    return layout.peak_bytes(1);
}
#endif

} // namespace

std::size_t fp8_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                    std::int32_t input_rows, LinearPolicy policy,
                                                    std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 linear_add workspace: invalid token interval");
    }
    const Fp8LinearAddRoute narrow_route = resolve_route(output_rows, input_rows, policy, min_tokens);
    const Fp8LinearAddRoute route        = resolve_route(output_rows, input_rows, policy, max_tokens);
    std::size_t capacity                 = 0;
    if (route == Fp8LinearAddRoute::A8) {
        capacity = fp8_a8_workspace_capacity_bytes(max_tokens, input_rows);
    }
#ifdef NINFER_VOLTA_BUILD
    // Calls anywhere in [min_tokens, max_tokens] must fit. Under AllowA8 the narrow end stays on
    // the QPN route while the wide end is A8, and within the QPN route the small-T staging buffer
    // and the wide-T CUTLASS scratch are separate, so size for the largest.
    if (narrow_route == Fp8LinearAddRoute::QpnResidual) {
        capacity = std::max(capacity, qpn_residual_workspace_bytes(
                                          output_rows, input_rows,
                                          std::min(max_tokens, kFp8VoltaQpnMaxTokens)));
    }
    if (route == Fp8LinearAddRoute::QpnResidual) {
        capacity = std::max(capacity,
                            qpn_residual_workspace_bytes(output_rows, input_rows, max_tokens));
    }
#else
    (void)narrow_route;
#endif
    return capacity;
}

#ifdef NINFER_VOLTA_BUILD
bool fp8_linear_add_fp16_activation_supported(const Weight& weight, LinearPolicy policy,
                                              std::int32_t tokens) {
    return resolve_route(weight.n, weight.k, policy, tokens) == Fp8LinearAddRoute::QpnResidual &&
           tokens <= kFp8VoltaQpnMaxTokens;
}
#endif

void fp8_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                             LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream) {
    const Fp8LinearAddRoute route = resolve_route(weight.n, weight.k, policy, x.ne[1]);
    if (route == Fp8LinearAddRoute::A16) {
        launch_a16(x, weight, residual, stream);
        return;
    }
#ifdef NINFER_VOLTA_BUILD
    if (route == Fp8LinearAddRoute::QpnResidual) {
        launch_qpn_residual(x, weight, residual, workspace, stream);
        return;
    }
#endif
    fp8_linear_add_a8_launch(x, weight, residual, workspace, stream);
}

} // namespace ninfer::ops::detail
