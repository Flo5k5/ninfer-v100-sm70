#pragma once

#include "ninfer/types.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

[[nodiscard]] inline SpeculativeBackend parse_speculative_backend(std::string_view value) {
    if (value == "mtp") { return SpeculativeBackend::Mtp; }
    if (value == "dflash") { return SpeculativeBackend::DFlash; }
    if (value == "dflash2") { return SpeculativeBackend::DFlash2; }
    throw std::invalid_argument("invalid speculative backend: " + std::string(value));
}

[[nodiscard]] inline const char* speculative_backend_name(SpeculativeBackend backend) noexcept {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash:
        return "dflash";
    case SpeculativeBackend::DFlash2:
        return "dflash2";
    }
    return "unknown";
}

[[nodiscard]] inline ContextLookupPolicy parse_context_lookup_policy(std::string_view value) {
    if (value == "off") { return ContextLookupPolicy::Off; }
    if (value == "fixed") { return ContextLookupPolicy::Fixed; }
    if (value == "adaptive") { return ContextLookupPolicy::Adaptive; }
    throw std::invalid_argument("invalid lookup policy: " + std::string(value));
}

[[nodiscard]] inline const char* context_lookup_policy_name(ContextLookupPolicy policy) noexcept {
    switch (policy) {
    case ContextLookupPolicy::Off:
        return "off";
    case ContextLookupPolicy::Fixed:
        return "fixed";
    case ContextLookupPolicy::Adaptive:
        return "adaptive";
    }
    return "unknown";
}

// The MTP context-lookup flags a command line gave explicitly. Validation depends on which flags
// were given, not on whether a value happens to equal its default.
struct ContextLookupFlags {
    bool policy       = false;
    bool min_suffix   = false;
    bool max_proposal = false;

    [[nodiscard]] bool any() const noexcept { return policy || min_suffix || max_proposal; }
};

// Shared by the CLI and the server: both expose the same MTP context-lookup flags.
inline void validate_context_lookup_cli_options(const SpeculativeOptions& options,
                                                const ContextLookupFlags& given) {
    const ContextLookupOptions& lookup = options.context_lookup;
    if (options.backend != SpeculativeBackend::Mtp) {
        if (given.any()) {
            throw std::invalid_argument("--lookup-policy, --lookup-min-suffix and "
                                        "--lookup-max-proposal require --spec mtp");
        }
        return;
    }
    if (lookup.policy == ContextLookupPolicy::Off) {
        // Disabled lookup plans no frame, so a suffix or proposal size would have no effect.
        if (given.min_suffix || given.max_proposal) {
            throw std::invalid_argument("--lookup-min-suffix and --lookup-max-proposal require "
                                        "--lookup-policy fixed or adaptive");
        }
        return;
    }
    if (lookup.min_suffix < kContextLookupMinimumSuffix ||
        lookup.min_suffix > kContextLookupMaximumSuffix) {
        throw std::invalid_argument("--lookup-min-suffix must be in [" +
                                    std::to_string(kContextLookupMinimumSuffix) + "," +
                                    std::to_string(kContextLookupMaximumSuffix) + "]");
    }
    if (lookup.max_proposal <= options.draft_tokens ||
        lookup.max_proposal > kContextLookupMaximumProposal) {
        throw std::invalid_argument(
            "--lookup-max-proposal must exceed --draft-tokens and be at most " +
            std::to_string(kContextLookupMaximumProposal));
    }
}

// `lookup_flags` names the context-lookup flags the command line gave; a command without those
// flags passes none.
inline void validate_speculative_cli_options(const SpeculativeOptions& options,
                                             const ContextLookupFlags& lookup_flags = {}) {
    validate_context_lookup_cli_options(options, lookup_flags);
    switch (options.backend) {
    case SpeculativeBackend::None:
        if (options.draft_tokens != 0 || options.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "--draft-tokens and --lm-head-draft require --spec mtp|dflash|dflash2");
        }
        return;
    case SpeculativeBackend::Mtp:
        // The sm_70 width-6+ target-verify regression (draft window >= 5 drifting off the
        // greedy argmax) tracked back to the dedicated Volta small_t_i8/bf16 verify kernels
        // dropped during the DFlash2 merge; both are restored (small_t_i8_volta_kt.cuh, whose
        // per-row arithmetic is width-invariant, and small_t_bf16_volta.cuh) and every draft
        // window through 7 is bit-exact against --spec none again, so sm_70 no longer needs a
        // narrower cap than upstream.
        if (options.draft_tokens == 0 || options.draft_tokens > 7) {
            throw std::invalid_argument("--spec mtp requires --draft-tokens in [1,7]");
        }
        return;
    case SpeculativeBackend::DFlash:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash requires --draft-tokens in [1,15]");
        }
        return;
    case SpeculativeBackend::DFlash2:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash2 requires --draft-tokens in [1,15]");
        }
        return;
    }
    throw std::invalid_argument("invalid speculative backend");
}

} // namespace ninfer::product
