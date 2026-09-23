#include "ops/linear/nvfp4/nvfp4_dispatch.h"

#include "core/layout.h"

#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/fp8/fp8_launch.h"
#endif

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4LinearRoute : std::uint8_t {
    A16,
    W4A4,
};

Nvfp4LinearRoute resolve_route(std::int32_t output_rows, std::int32_t input_rows,
                               LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0 || !is_nvfp4_dispatch_problem(output_rows, input_rows)) {
        throw std::invalid_argument("nvfp4 linear: unsupported shape");
    }
    if (policy == LinearPolicy::A16Only) { return Nvfp4LinearRoute::A16; }
    if (is_nvfp4_vocabulary_problem(output_rows, input_rows)) {
        throw std::invalid_argument("nvfp4 linear: the vocabulary head has an A16 route only");
    }
    if (policy != LinearPolicy::AllowA4) {
        throw std::invalid_argument("nvfp4 linear: unsupported policy");
    }

    switch (resolve_nvfp4_problem(output_rows, input_rows)) {
    case Nvfp4Problem::AttnInput:
        return tokens >= 4 ? Nvfp4LinearRoute::W4A4 : Nvfp4LinearRoute::A16;
    case Nvfp4Problem::GdnInput:
        return Nvfp4LinearRoute::W4A4;
    case Nvfp4Problem::MlpGateUp:
        return tokens >= 5 ? Nvfp4LinearRoute::W4A4 : Nvfp4LinearRoute::A16;
    case Nvfp4Problem::Residual6144:
    case Nvfp4Problem::Residual17408:
        return tokens >= 8 ? Nvfp4LinearRoute::W4A4 : Nvfp4LinearRoute::A16;
    }
    throw std::logic_error("unreachable NVFP4 linear problem");
}

#ifdef NINFER_VOLTA_BUILD
// fp16 staging buffer for the QPN route's activations (see launch_nvfp4_volta_qpn_fp16).
std::size_t qpn_activation_bytes(std::int32_t input_rows, std::int32_t tokens) {
    return static_cast<std::size_t>(input_rows) *
           std::min(tokens, kNvfp4VoltaQpnMaxTokens) * sizeof(std::uint16_t);
}

bool workspace_fits(const WorkspaceArena& workspace, std::size_t bytes) {
    constexpr std::size_t kAlign = 256;
    const std::size_t start      = (workspace.used() + kAlign - 1) / kAlign * kAlign;
    return start + bytes <= workspace.capacity();
}
#endif

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out,
#ifdef NINFER_VOLTA_BUILD
                WorkspaceArena* workspace,
#endif
                cudaStream_t stream) {
    const std::int32_t total_t = x.ne[1];
#ifdef NINFER_VOLTA_BUILD
    // Above QPN2's own range, a single wide-T MMA pass (nvfp4_volta_mma_gemm.cuh) beats
    // chunking through QPN2 in kNvfp4VoltaQpnMaxTokens pieces -- QPN2's whole design assumes T is
    // small enough that decoding the weight once per ~32 tokens is cheap; at prefill widths that
    // means re-decoding the entire weight dozens of times. Measured directly against the
    // 64-chunk QPN2 path at T=2048 on the gate_up shape: 45.5ms -> 31.3ms, 1.45x. Needs a real
    // workspace only when split-K applies (rare at production shapes -- both registered NVFP4
    // shapes measured splits=1 at prefill width); fall back to the chunked route rather than
    // fault if a caller genuinely has none. See docs/v100.md.
    // The wide-T kernel reads the checkpoint-native code and scale planes; a QPN-prepacked weight
    // stays on the chunked QPN2 route, which reads both layouts.
    if (workspace != nullptr && total_t > kNvfp4VoltaQpnMaxTokens &&
        weight.layout == QuantLayout::BlockScaleK16M128x4 &&
        nvfp4_volta_mma_supported(weight.n, weight.k, total_t)) {
        const std::size_t need = nvfp4_volta_mma_workspace_bytes(weight.n, weight.k, total_t);
        if (need == 0 || workspace->capacity() - workspace->used() >= need) {
            launch_nvfp4_volta_mma(x, weight, out, *workspace, stream);
            return;
        }
    }
    // Where the quadpair route is available the chunk drops to its tile width, matching the FP8
    // sibling's reasoning: several QPN passes over the weights beat one wider SIMT pass, because
    // SIMT throughput decays with T while QPN's is flat across the tile.
    const bool qpn            = nvfp4_volta_qpn_supported(weight.n, weight.k, kNvfp4VoltaQpnMaxTokens);
    const std::int32_t kChunk = qpn ? kNvfp4VoltaQpnMaxTokens : kNvfp4LastSmallT;
    // Stage the fp16 activation once per chunk when the caller's workspace has room; without it
    // the QPN kernel converts in its inner loop (correct, slower).
    std::optional<WorkspaceArena::Scope> scope;
    DeviceSpan activation;
    const std::size_t activation_bytes = qpn_activation_bytes(weight.k, total_t);
    if (qpn && workspace != nullptr && workspace_fits(*workspace, activation_bytes)) {
        scope.emplace(workspace->scope());
        activation = workspace->alloc_bytes(activation_bytes, 256);
    }
#else
    constexpr std::int32_t kChunk = kNvfp4LastSmallT;
#endif
    for (std::int32_t token_begin = 0; token_begin < total_t; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* output = static_cast<std::uint8_t*>(out.data) +
                       static_cast<std::int64_t>(token_begin) * weight.n * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor output_chunk(output, DType::BF16, {weight.n, active});
#ifdef NINFER_VOLTA_BUILD
        if (nvfp4_volta_qpn_supported(weight.n, weight.k, active)) {
            if (activation.data != nullptr) {
                fp8_stage_bf16_activation_sm70(input_chunk, activation.data, stream);
                launch_nvfp4_volta_qpn_fp16(input_chunk, weight, activation.data, output_chunk,
                                            stream);
            } else {
                launch_nvfp4_volta_qpn(input_chunk, weight, output_chunk, stream);
            }
            continue;
        }
#endif
        if (active == 1) {
            launch_nvfp4_decode(input_chunk, weight, output_chunk, stream);
        } else {
            launch_nvfp4_small_t(input_chunk, weight, output_chunk, stream);
        }
    }
}

} // namespace

std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                                  LinearPolicy policy, std::int32_t min_tokens,
                                                  std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear workspace: invalid token interval");
    }
    (void)resolve_route(output_rows, input_rows, policy, min_tokens);
    std::size_t capacity = 0;
    if (resolve_route(output_rows, input_rows, policy, max_tokens) == Nvfp4LinearRoute::W4A4) {
        capacity = nvfp4_w4a4_workspace_capacity_bytes(max_tokens, input_rows);
    }
#ifdef NINFER_VOLTA_BUILD
    // The wide-T MMA route in launch_a16 only needs workspace when split-K applies; report that
    // so a caller sizing for the widest T this interval reaches has it available. A caller that
    // doesn't (the zero-workspace linear() overload) still works -- launch_a16 falls back to the
    // chunked route rather than fault.
    if (policy == LinearPolicy::A16Only && max_tokens > kNvfp4VoltaQpnMaxTokens &&
        nvfp4_volta_mma_supported(output_rows, input_rows, max_tokens)) {
        capacity = std::max(capacity,
                            nvfp4_volta_mma_workspace_bytes(output_rows, input_rows, max_tokens));
    }
    // Every A16 call stages its activation for the QPN route. Under AllowA4 the narrow end of the
    // interval can be A16 while the wide end is W4A4, so size for both.
    if (resolve_route(output_rows, input_rows, policy, min_tokens) == Nvfp4LinearRoute::A16 &&
        nvfp4_volta_qpn_supported(output_rows, input_rows, kNvfp4VoltaQpnMaxTokens)) {
        WorkspaceLayoutBuilder layout;
        (void)layout.alloc_bytes(qpn_activation_bytes(input_rows, max_tokens), 256);
        capacity = std::max(capacity, layout.peak_bytes(1));
    }
#endif
    return capacity;
}

void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    WorkspaceArena* workspace, cudaStream_t stream) {
    validate_nvfp4_weight(weight, "nvfp4 linear");
    if (weight.layout == QuantLayout::VoltaQpnPrepackedSwiGlu) {
        throw std::invalid_argument("nvfp4 linear: SwiGLU-interleaved weights are linear_swiglu-only");
    }
    if (!is_nvfp4_dispatch_problem(weight.n, weight.k) || x.ne[1] <= 0) {
        throw std::invalid_argument("nvfp4 linear: unsupported shape");
    }

    if (resolve_route(weight.n, weight.k, policy, x.ne[1]) == Nvfp4LinearRoute::A16) {
#ifdef NINFER_VOLTA_BUILD
        launch_a16(x, weight, out, workspace, stream);
#else
        launch_a16(x, weight, out, stream);
#endif
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 linear requires caller workspace");
    }
    auto scope                       = workspace->scope();
    const Nvfp4W4a4Workspace scratch = allocate_nvfp4_w4a4_workspace(*workspace, x.ne[1], weight.k);
    launch_nvfp4_w4a4(x, weight, out, scratch, stream);
}

} // namespace ninfer::ops::detail
