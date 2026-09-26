#pragma once

#include "core/device.h"
#include "targets/qwen3_6_27b/impl/config.h"
#include "ninfer/ops/sparse_moe.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include <ninfer/targets/qwen3_6/runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::targets::qwen3_6_27b::detail {

using GraphExecutionProfile = qwen3_6::GraphExecutionProfile;

// Compile-time data and the three closed execution leaves supplied to the Qwen3.6 family runtime.
// It owns no request state, execution phase, graph object, or schedule callback.
struct Variant {
    using WeightsProfile                 = detail::WeightsProfile;
    using TextConfig                     = detail::TextConfig;
    using VisionConfig                   = detail::VisionConfig;
    using DFlashConfig                   = detail::DFlashConfig;
    using ModelView                      = detail::RuntimeModelView;
    using FullAttentionProjectionWeights = detail::FullAttentionProjectionPayload;
    using GdnProjectionWeights           = detail::GdnProjectionPayload;
    using PostMixerWeights               = detail::DensePostMixerPayload;
    using MtpAttentionProjectionWeights  = detail::MtpAttentionPayload;
    using MtpPostMixerWeights            = detail::DensePostMixerPayload;
    using VisionWeights                  = qwen3_6::VisionWeights;
    using GraphExecutionProfile          = detail::GraphExecutionProfile;

    // The dense post-mixer streams no MoE weights, so this target names no prefetch span.
    static ::ninfer::ops::SparseMoeHints
    projection_prefetch_hints(const FullAttentionProjectionWeights&) {
        return {};
    }

    static ::ninfer::ops::SparseMoeHints projection_prefetch_hints(const GdnProjectionWeights&) {
        return {};
    }

    static constexpr float attention_scale                     = kAttentionScale;
    static constexpr float gdn_scale                           = kGdnScale;
    static constexpr std::uint32_t prefill_chunk_alignment     = kPrefillChunkAlignment;
    static constexpr std::uint32_t maximum_mtp_draft_tokens    = kMaximumMtpDraftTokens;
    static constexpr std::uint32_t maximum_dflash_draft_tokens = kMaximumDFlashDraftTokens;
    static constexpr std::uint32_t maximum_context             = kNativeContext;
    static constexpr bool supports_dflash                      = DFlashConfig::supported;
    static constexpr std::int32_t draft_head_rows              = 131072;

    // True when every op that reads or writes the text residual stream has an FP32 form for this
    // weights profile: an FP8 token embedding and FP8/NVFP4 residual projections. The four Qwen3.8
    // NVFP4 profiles meet it: the NVFP4 down projections of MLP 56-63 (full-a, full-b and full-c)
    // take the route of layers 0-55, their NVFP4 output head reads the normed BF16 hidden state,
    // not the stream, and the NVFP4 attention and GDN input projections (full-b and full-c) read
    // that normed hidden state too. The NVFP4 attention and GDN output projections of full-c update
    // the stream through the linear_add routes of the down projections, which take an FP32
    // residual on the [5120,6144] shape as well: the QPN2 epilogue up to 32 columns, the CUTLASS
    // FP32 accumulators from 33.
    [[nodiscard]] static constexpr bool fp32_residual_supported(WeightsProfile profile) {
        return profile == WeightsProfile::Qwen38Nvfp4 ||
               profile == WeightsProfile::Qwen38Nvfp4FullA ||
               profile == WeightsProfile::Qwen38Nvfp4FullB ||
               profile == WeightsProfile::Qwen38Nvfp4FullC;
    }

    static void attention_projection(const Tensor& hidden,
                                     const FullAttentionProjectionWeights& weights, Tensor& query,
                                     Tensor& gate, Tensor& key, Tensor& value,
                                     qwen3_6::TextPhase phase, WorkspaceArena& workspace,
                                     cudaStream_t stream);
    static void attention_output_projection(const Tensor& attention, const Weight& weight,
                                            Tensor& residual, qwen3_6::TextPhase phase,
                                            WorkspaceArena& workspace, cudaStream_t stream);
    static void mtp_attention_projection(const Tensor& hidden,
                                         const MtpAttentionProjectionWeights& weights,
                                         Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                                         WorkspaceArena& workspace, cudaStream_t stream);
    static void mtp_kv_projection(const Tensor& hidden,
                                  const MtpAttentionProjectionWeights& weights, Tensor& key,
                                  Tensor& value, WorkspaceArena& workspace, cudaStream_t stream);
    static void mtp_q_gate_projection(const Tensor& hidden,
                                      const MtpAttentionProjectionWeights& weights, Tensor& query,
                                      Tensor& gate, WorkspaceArena& workspace, cudaStream_t stream);
    static void gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                     Tensor& qkv, Tensor& output_gate, qwen3_6::TextPhase phase,
                                     WorkspaceArena& workspace, cudaStream_t stream);
    static void
    gdn_input_projection_snapshot(const Tensor& hidden, const GdnProjectionWeights& weights,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_slot,
                                  const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& output_gate, qwen3_6::TextPhase phase,
                                  WorkspaceArena& workspace, cudaStream_t stream);
    static void gdn_input_projection_record(
        const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
        const Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slots,
        Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value, Tensor& output_gate,
        qwen3_6::TextPhase phase, WorkspaceArena& workspace, cudaStream_t stream);
    static void gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                      qwen3_6::TextPhase phase, WorkspaceArena& workspace,
                                      cudaStream_t stream);
    static void gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                            float eps, const GdnProjectionWeights& weights,
                                            Tensor& hidden, Tensor& g, Tensor& beta,
                                            WorkspaceArena& workspace,
                                            DeviceExecutionView execution);
    static void post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                           qwen3_6::TextPhase phase, const ::ninfer::ops::SparseMoeHints& hints,
                           WorkspaceArena& workspace, cudaStream_t stream);
    // True when post_mixer takes an FP16 `hidden` at this width: the Volta fp16 activation domain,
    // in which the caller's RMSNorm writes the copy the QPN GEMVs would otherwise stage.
    [[nodiscard]] static bool post_mixer_takes_fp16(const PostMixerWeights& weights,
                                                    std::int32_t tokens);
    // True when gdn_output_projection takes an FP16 `hidden` at this width (the caller's gated
    // RMSNorm then writes the QPN GEMV's fp16 copy itself).
    [[nodiscard]] static bool gdn_output_takes_fp16(const Weight& weight, std::int32_t tokens);
    // True when attention_projection takes an FP16 `hidden` at this width (the caller's RMSNorm
    // then writes the QPN GEMV's fp16 copy itself).
    [[nodiscard]] static bool attention_projection_takes_fp16(
        const FullAttentionProjectionWeights& weights, std::int32_t tokens);
    // True when the verify-phase GDN norm/control and conv-record/snapshot input projection keep
    // the normalized hidden in fp16 (the fused norm kernel writes the QPN GEMV's copy itself).
    [[nodiscard]] static bool gdn_input_takes_fp16(const GdnProjectionWeights& weights,
                                                   std::int32_t width, std::int32_t batch);
    static void mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                               Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream);
    [[nodiscard]] static std::size_t
    mtp_attention_projection_workspace_capacity_bytes(std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t mtp_kv_projection_workspace_capacity_bytes(std::int32_t first,
                                                                                std::int32_t last);
    [[nodiscard]] static std::size_t
    mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    attention_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                  qwen3_6::TextPhase phase, std::int32_t first,
                                                  std::int32_t last);
    [[nodiscard]] static std::size_t
    attention_output_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                         qwen3_6::TextPhase phase,
                                                         std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    gdn_input_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                  qwen3_6::TextPhase phase, std::int32_t first,
                                                  std::int32_t last);
    [[nodiscard]] static std::size_t gdn_input_projection_snapshot_workspace_capacity_bytes(
        WeightsProfile weights_profile, qwen3_6::TextPhase phase, std::int32_t batch_size,
        std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t gdn_input_projection_record_workspace_capacity_bytes(
        WeightsProfile weights_profile, qwen3_6::TextPhase phase, std::int32_t batch_size,
        std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    gdn_output_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                   qwen3_6::TextPhase phase, std::int32_t first,
                                                   std::int32_t last);
    [[nodiscard]] static std::size_t
    gdn_norm_control_projection_workspace_capacity_bytes(std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    post_mixer_workspace_capacity_bytes(WeightsProfile weights_profile, qwen3_6::TextPhase phase,
                                        std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t mtp_post_mixer_workspace_capacity_bytes(std::int32_t first,
                                                                             std::int32_t last);

    [[nodiscard]] static std::vector<GraphExecutionProfile>
    ordinary_graph_profiles(std::uint32_t capacity);
    [[nodiscard]] static std::vector<GraphExecutionProfile>
    mtp_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window);
    [[nodiscard]] static std::vector<GraphExecutionProfile>
    dflash_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window,
                          std::uint32_t batch_size);
};

} // namespace ninfer::targets::qwen3_6_27b::detail
