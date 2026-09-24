#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include "core/layout.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/fp8/fp8_launch.h"
#include "ops/linear/nvfp4/nvfp4_cutlass_sm70.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4LinearAddRoute : std::uint8_t {
    A16,
#ifdef NINFER_VOLTA_BUILD
    LinearThenAdd,
#endif
    W4A4,
};

Nvfp4LinearAddRoute resolve_route(std::int32_t output_rows, std::int32_t input_rows,
                                  LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0 || output_rows != 5120 || (input_rows != 6144 && input_rows != 17408)) {
        throw std::invalid_argument("nvfp4 linear_add: unsupported shape");
    }
    if (policy == LinearPolicy::A16Only) {
#ifdef NINFER_VOLTA_BUILD
        // Prepacked down projections are consumed by QPN2 for every decode width. Materialize the
        // projection and add the residual separately because the row-major fused kernels cannot
        // read that load-time layout.
        return Nvfp4LinearAddRoute::LinearThenAdd;
#else
        return Nvfp4LinearAddRoute::A16;
#endif
    }
    if (policy != LinearPolicy::AllowA4) {
        throw std::invalid_argument("nvfp4 linear_add: unsupported policy");
    }
    const std::int32_t first_w4a4 = input_rows == 6144 ? 7 : 8;
    if (tokens >= first_w4a4) { return Nvfp4LinearAddRoute::W4A4; }
#ifdef NINFER_VOLTA_BUILD
    return Nvfp4LinearAddRoute::LinearThenAdd;
#else
    return Nvfp4LinearAddRoute::A16;
#endif
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    constexpr std::int32_t kChunk = kNvfp4LastSmallT;
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* output = static_cast<std::uint8_t*>(residual.data) +
                       static_cast<std::int64_t>(token_begin) * weight.n * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor residual_chunk(output, DType::BF16, {weight.n, active});
        if (active == 1) {
            nvfp4_linear_add_decode_launch(input_chunk, weight, residual_chunk, stream);
        } else {
            nvfp4_linear_add_small_t_launch(input_chunk, weight, residual_chunk, stream);
        }
    }
}

#ifdef NINFER_VOLTA_BUILD
template <class Allocator>
Tensor allocate_projected(Allocator& allocator, std::int32_t output_rows, std::int32_t tokens) {
    return allocator.alloc(DType::BF16, {output_rows, tokens}, 256);
}

// Verify and decode widths fold the residual add into the QPN2 epilogue on a staged fp16 copy of
// x: 57 of the 27B's 64 down projections are NVFP4, and each paid a separate residual_add launch
// plus a materialized BF16 projection.
bool qpn_residual(const Weight& weight, std::int32_t tokens) {
    return tokens <= kNvfp4VoltaQpnMaxTokens &&
           nvfp4_volta_qpn_supported(weight.n, weight.k, tokens);
}

std::size_t qpn_activation_bytes(std::int32_t input_rows, std::int32_t tokens) {
    return static_cast<std::size_t>(input_rows) * tokens * sizeof(std::uint16_t);
}

void launch_linear_then_add(const Tensor& x, const Weight& weight, Tensor& residual,
                            WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope         = workspace.scope();
    if (qpn_residual(weight, x.ne[1])) {
        if (x.dtype == DType::FP16) {
            launch_nvfp4_volta_qpn_residual(x, weight, x.data, residual, stream);
            return;
        }
        constexpr std::size_t kAlign = 256;
        const std::size_t bytes      = qpn_activation_bytes(weight.k, x.ne[1]);
        const std::size_t start      = (workspace.used() + kAlign - 1) / kAlign * kAlign;
        if (start + bytes > workspace.capacity()) {
            launch_nvfp4_volta_qpn_residual(x, weight, nullptr, residual, stream);
            return;
        }
        DeviceSpan activation = workspace.alloc_bytes(bytes, kAlign);
        fp8_stage_bf16_activation_sm70(x, activation.data, stream);
        launch_nvfp4_volta_qpn_residual(x, weight, activation.data, residual, stream);
        return;
    }
    if (residual.dtype == DType::FP32 && x.ne[1] >= 33) {
        // An FP32 residual stream takes the GEMM's FP32 accumulators directly.
        nvfp4_cutlass_sm70_residual_launch(x, weight, residual, workspace, stream);
        return;
    }
    Tensor projected    = allocate_projected(workspace, weight.n, x.ne[1]);
    if (x.ne[1] >= 33) {
        nvfp4_cutlass_sm70_launch(x, weight, projected, workspace, stream);
    } else {
        const std::size_t linear_bytes = std::max<std::size_t>(
            linear_workspace_capacity_bytes(QType::NVFP4, weight.n, weight.k,
                                            kNvfp4InternalPolicy, x.ne[1], x.ne[1]),
            256);
        WorkspaceArena linear_workspace(workspace.alloc_bytes(linear_bytes, 256));
        linear(x, weight, projected, kNvfp4InternalPolicy, linear_workspace, stream);
    }
    residual_add(projected, residual, stream);
}

std::size_t linear_then_add_workspace_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                            std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    if (tokens <= kNvfp4VoltaQpnMaxTokens &&
        nvfp4_volta_qpn_supported(output_rows, input_rows, tokens)) {
        (void)layout.alloc_bytes(qpn_activation_bytes(input_rows, tokens), 256);
        return layout.peak_bytes(1);
    }
    (void)allocate_projected(layout, output_rows, tokens);
    const std::size_t linear_bytes = tokens >= 33
        ? nvfp4_cutlass_sm70_workspace_bytes(output_rows, input_rows, tokens)
        : std::max<std::size_t>(
              linear_workspace_capacity_bytes(QType::NVFP4, output_rows, input_rows,
                                              kNvfp4InternalPolicy, tokens, tokens),
              256);
    (void)layout.alloc_bytes(linear_bytes, 256);
    return layout.peak_bytes(1);
}
#endif // NINFER_VOLTA_BUILD

} // namespace

std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                      std::int32_t input_rows, LinearPolicy policy,
                                                      std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear_add workspace: invalid token interval");
    }
    (void)resolve_route(output_rows, input_rows, policy, min_tokens);
    const Nvfp4LinearAddRoute route = resolve_route(output_rows, input_rows, policy, max_tokens);
    if (route == Nvfp4LinearAddRoute::W4A4) {
        return nvfp4_w4a4_workspace_capacity_bytes(max_tokens, input_rows);
    }
#ifdef NINFER_VOLTA_BUILD
    if (route == Nvfp4LinearAddRoute::LinearThenAdd) {
        return linear_then_add_workspace_bytes(output_rows, input_rows, max_tokens);
    }
#endif
    return 0;
}

#ifdef NINFER_VOLTA_BUILD
bool nvfp4_linear_add_fp16_activation_supported(const Weight& weight, LinearPolicy policy,
                                                std::int32_t tokens) {
    return resolve_route(weight.n, weight.k, policy, tokens) ==
               Nvfp4LinearAddRoute::LinearThenAdd &&
           qpn_residual(weight, tokens);
}

bool nvfp4_linear_add_fp32_residual_supported(const Weight& weight, LinearPolicy policy,
                                              std::int32_t tokens) {
    return resolve_route(weight.n, weight.k, policy, tokens) == Nvfp4LinearAddRoute::LinearThenAdd;
}
#endif

void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                               LinearPolicy policy, WorkspaceArena& workspace,
                               cudaStream_t stream) {
    const Nvfp4LinearAddRoute route = resolve_route(weight.n, weight.k, policy, x.ne[1]);
    if (route == Nvfp4LinearAddRoute::A16) {
        launch_a16(x, weight, residual, stream);
        return;
    }
#ifdef NINFER_VOLTA_BUILD
    if (route == Nvfp4LinearAddRoute::LinearThenAdd) {
        launch_linear_then_add(x, weight, residual, workspace, stream);
        return;
    }
#endif
    auto scope                       = workspace.scope();
    const Nvfp4W4a4Workspace scratch = allocate_nvfp4_w4a4_workspace(workspace, x.ne[1], weight.k);
    nvfp4_linear_add_w4a4_launch(x, weight, residual, scratch, stream);
}

} // namespace ninfer::ops::detail
