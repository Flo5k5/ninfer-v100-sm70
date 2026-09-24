#pragma once

#include "ninfer/types.h"

namespace ninfer::targets::qwen3_6 {

[[nodiscard]] constexpr bool is_masked_draft_backend(SpeculativeBackend backend) noexcept {
    return backend == SpeculativeBackend::DFlash || backend == SpeculativeBackend::DFlash2;
}

struct StartupFeatures {
    bool vision                    = false;
    SpeculativeBackend speculative = SpeculativeBackend::None;
    ProposalHead proposal_head     = ProposalHead::Full;

    bool operator==(const StartupFeatures&) const = default;

    [[nodiscard]] bool speculative_enabled() const noexcept {
        return speculative != SpeculativeBackend::None;
    }

    [[nodiscard]] bool mtp() const noexcept { return speculative == SpeculativeBackend::Mtp; }

    [[nodiscard]] bool dflash() const noexcept { return speculative == SpeculativeBackend::DFlash; }

    [[nodiscard]] bool dflash2() const noexcept {
        return speculative == SpeculativeBackend::DFlash2;
    }

    [[nodiscard]] bool masked_draft() const noexcept {
        return is_masked_draft_backend(speculative);
    }

    [[nodiscard]] bool optimized_proposal() const noexcept {
        return speculative_enabled() && proposal_head == ProposalHead::Optimized;
    }
};

// Whether the host can mask every position a backend samples. MTP drafts are known before the
// round that verifies them; DFlash and DFlash2 draft and verify inside one graph.
[[nodiscard]] constexpr bool supports_token_masks(SpeculativeBackend backend) noexcept {
    return !is_masked_draft_backend(backend);
}

// Structured output of an Engine built with these options. Causal-scoring Engines never generate
// and build neither the grammar compiler nor the token masks.
[[nodiscard]] inline StructuredOutputOptions
structured_output_options(const EngineOptions& options) noexcept {
    StructuredOutputOptions out = options.structured_output;
    if (options.purpose != EnginePurpose::Generation) { out.enabled = false; }
    return out;
}

[[nodiscard]] inline StartupFeatures startup_features(const EngineOptions& options) noexcept {
    return StartupFeatures{
        .vision        = options.enable_vision,
        .speculative   = options.speculative.backend,
        .proposal_head = options.speculative.proposal_head,
    };
}

} // namespace ninfer::targets::qwen3_6
