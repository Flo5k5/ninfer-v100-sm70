#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"

#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_launch.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_cutlass_sm70.h"
#endif

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

#ifdef NINFER_VOLTA_BUILD
// Volta A16 width frontier. One QPN pass streams the 80 MiB of FP8 codes once for up to 32
// columns, and wider calls run one pass per 32 columns through a single FP16 activation staging
// buffer. The CUTLASS route stages the whole weight in FP16 on every call instead: NCU counts
// about 500 MiB of DRAM traffic per call (its dequant kernel alone takes as long as its GEMM) and
// it needs over 160 MiB of workspace. Measured on a V100-PCIE (cold L2): a pass costs 115 us at
// one column and 255 us at 32, the CUTLASS route 1.12-1.15 ms up to 128 columns and 1.25-1.27 ms
// from 129, where its GEMM takes a second 128-row tile. Five passes (1.26 ms at 160 columns) still
// match it and cover every record width (at most eight rows of 16 columns); a sixth does not.
constexpr std::int32_t kVoltaQpnMaxPasses = 5;
constexpr std::int32_t kVoltaCutlassMinT  = kVoltaQpnMaxPasses * kFp8VoltaQpnMaxTokens + 1;
#endif

enum class Fp8GdnInputRoute : std::uint8_t {
    A16,
    A8,
};

Fp8GdnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("fp8 gdn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only) { return Fp8GdnInputRoute::A16; }
    if (policy != LinearPolicy::AllowA8) {
        throw std::invalid_argument("fp8 gdn_input_proj: unsupported policy");
    }
    return tokens >= 8 ? Fp8GdnInputRoute::A8 : Fp8GdnInputRoute::A16;
}

} // namespace

std::size_t fp8_gdn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 gdn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    std::size_t capacity = resolve_route(policy, max_tokens) == Fp8GdnInputRoute::A8
                               ? fp8_a8_workspace_capacity_bytes(max_tokens,
                                                                 Fp8GdnInputGeometry::kInputRows)
                               : 0;
#ifdef NINFER_VOLTA_BUILD
    if (max_tokens >= kVoltaCutlassMinT) {
        capacity = std::max(capacity, fp8_gdn_input_cutlass_workspace_bytes(max_tokens));
    }
    capacity = std::max(capacity, static_cast<std::size_t>(Fp8GdnInputGeometry::kInputRows) *
                                      std::min(max_tokens, kFp8VoltaQpnMaxTokens) *
                                      sizeof(std::uint16_t));
#endif
    return capacity;
}

void fp8_gdn_input_a16_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                WorkspaceArena* workspace, cudaStream_t stream) {
    constexpr std::int32_t kQkvRows = 10240;
    constexpr std::int32_t kZRows   = 6144;
#ifdef NINFER_VOLTA_BUILD
    const bool qpn = fp8_volta_qpn_supported(weight.n, weight.k, kFp8VoltaQpnMaxTokens);
    if (x.dtype == DType::FP16) {
        // The fp16 activation domain: x is already the staged copy (callers admit it only up to
        // one QPN pass). Checked before the width frontier: the CUTLASS route reads x as BF16.
        if (!qpn || x.ne[1] > kFp8VoltaQpnMaxTokens) {
            throw std::logic_error("fp8 GDN input: FP16 x needs a single QPN pass");
        }
        launch_fp8_gdn_input_volta_qpn(x, weight, qkv, z, x.data, stream);
        return;
    }
    if (x.ne[1] >= kVoltaCutlassMinT) {
        if (workspace == nullptr) {
            throw std::invalid_argument("fp8 Volta GDN prefill requires caller workspace");
        }
        fp8_gdn_input_cutlass_sm70_launch(x, weight, qkv, z, *workspace, stream);
        return;
    }
    const std::int32_t kChunk =
        qpn ? kFp8VoltaQpnMaxTokens : kFp8LinearSmallTMax<Fp8GdnInputGeometry>;
    if (!qpn) { throw std::logic_error("fp8 Volta GDN problem has no QPN route"); }
    std::optional<WorkspaceArena::Scope> scope;
    DeviceSpan activation;
    if (workspace != nullptr) {
        scope.emplace(workspace->scope());
        activation = workspace->alloc_bytes(
            static_cast<std::size_t>(weight.k) * std::min(x.ne[1], kChunk) * sizeof(std::uint16_t),
            256);
    }
#else
    constexpr std::int32_t kChunk   = kFp8LinearSmallTMax<Fp8GdnInputGeometry>;
#endif
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* qkv_output =
            static_cast<std::uint8_t*>(qkv.data) +
            static_cast<std::int64_t>(token_begin) * kQkvRows * sizeof(std::uint16_t);
        auto* z_output = static_cast<std::uint8_t*>(z.data) +
                         static_cast<std::int64_t>(token_begin) * kZRows * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor qkv_chunk(qkv_output, DType::BF16, {kQkvRows, active});
        Tensor z_chunk(z_output, DType::BF16, {kZRows, active});
#ifdef NINFER_VOLTA_BUILD
        if (fp8_volta_qpn_supported(weight.n, weight.k, active)) {
            if (activation.data != nullptr) {
                fp8_stage_bf16_activation_sm70(input_chunk, activation.data, stream);
            }
            launch_fp8_gdn_input_volta_qpn(input_chunk, weight, qkv_chunk, z_chunk,
                                           activation.data, stream);
            continue;
        }
#endif
        if (active == 1) {
            fp8_gdn_input_decode_launch(input_chunk, weight, qkv_chunk, z_chunk, stream);
        } else {
            fp8_gdn_input_small_t_launch(input_chunk, weight, qkv_chunk, z_chunk, stream);
        }
    }
}

void fp8_gdn_input_a8_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                               WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope                   = workspace.scope();
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(workspace, x.ne[1], weight.k);
    fp8_gdn_input_a8_launch(x, weight, qkv, z, scratch, stream);
}

void fp8_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                            LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    if (resolve_route(policy, x.ne[1]) == Fp8GdnInputRoute::A16) {
        fp8_gdn_input_a16_dispatch(x, weight, qkv, z, workspace, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("fp8 A8 gdn_input_proj requires caller workspace");
    }
    fp8_gdn_input_a8_dispatch(x, weight, qkv, z, *workspace, stream);
}

} // namespace ninfer::ops::detail
