#pragma once

// Host-side proposal policy for MTP context lookup. The target verifies every copied token, so
// nothing here can change emitted text; it only decides when a round pays for the wider
// verification block. The header is CUDA-free so the policy is testable in isolation.

#include "ninfer/types.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace ninfer::targets::qwen3_6::detail {

// Single-token delimiters that open and close one tool call in generated text.
struct ToolCallDelimiters {
    TokenId open  = -1;
    TokenId close = -1;
};

// A copy whose source shares at least this many tokens with the current suffix is treated as
// established: the fixed rule's evidence.
inline constexpr std::uint32_t kContextLookupEstablishedSuffix = 16;

struct ContextLookupCandidate {
    std::array<TokenId, kContextLookupMaximumProposal> tokens{};
    // History index of the first copied token.
    std::size_t source = 0;
    // Tokens shared by the source occurrence and the current suffix, measured up to
    // kContextLookupEstablishedSuffix.
    std::uint32_t matched_suffix = 0;
};

namespace context_lookup_detail {

// Copies the continuation that starts at `source`. source < history.size(), so a position past
// the history repeats a token already copied (a source that runs into the current suffix is
// continued periodically).
inline void copy_continuation(std::span<const TokenId> history, std::size_t source,
                              std::uint32_t extent, ContextLookupCandidate& result) {
    const std::size_t size = history.size();
    for (std::size_t i = 0; i < extent; ++i) {
        const std::size_t index = source + i;
        result.tokens[i]        = index < size ? history[index] : result.tokens[index - size];
    }
    result.source = source;
}

inline std::uint32_t shared_suffix(std::span<const TokenId> history, std::size_t source,
                                   std::uint32_t known) {
    const std::size_t size = history.size();
    std::uint32_t shared   = known;
    while (shared < kContextLookupEstablishedSuffix && source > shared &&
           history[source - shared - 1U] == history[size - shared - 1U]) {
        ++shared;
    }
    return shared;
}

inline bool suffix_recurs_at(std::span<const TokenId> history, std::size_t source,
                             std::uint32_t min_suffix) {
    const std::size_t size = history.size();
    return history[source - 1U] == history[size - 1U] &&
           std::equal(history.begin() + static_cast<std::ptrdiff_t>(source - min_suffix),
                      history.begin() + static_cast<std::ptrdiff_t>(source - 1U),
                      history.begin() + static_cast<std::ptrdiff_t>(size - min_suffix));
}

} // namespace context_lookup_detail

enum class ContextLookupSearch : std::uint8_t {
    // The most recent occurrence with a complete continuation, whether or not it agrees.
    Nearest,
    // The most recent occurrence whose continuation begins with the required prefix.
    NearestAgreeing,
};

// Finds an earlier occurrence of the last `min_suffix` tokens of `history` and copies the
// `extent` tokens that followed it. A source whose continuation runs into the current suffix is
// continued periodically from the copied prefix. The most recent qualifying occurrence wins so
// repeated edits follow the freshest copy.
[[nodiscard]] inline std::optional<ContextLookupCandidate>
find_context_continuation(std::span<const TokenId> history, std::uint32_t min_suffix,
                          std::uint32_t extent, std::span<const TokenId> required_prefix,
                          ContextLookupSearch search) {
    namespace impl = context_lookup_detail;
    if (min_suffix == 0 || extent == 0 || extent > kContextLookupMaximumProposal ||
        required_prefix.size() > extent || history.size() <= min_suffix) {
        return std::nullopt;
    }
    // A source index s copies history[s...] after an occurrence ending at s; the current suffix
    // itself ends at history.size(), so every earlier occurrence has s < history.size().
    for (std::size_t source = history.size() - 1U; source >= min_suffix; --source) {
        if (!impl::suffix_recurs_at(history, source, min_suffix)) { continue; }
        ContextLookupCandidate result;
        impl::copy_continuation(history, source, extent, result);
        if (search == ContextLookupSearch::NearestAgreeing &&
            !std::equal(required_prefix.begin(), required_prefix.end(), result.tokens.begin())) {
            continue;
        }
        result.matched_suffix = impl::shared_suffix(history, source, min_suffix);
        return result;
    }
    return std::nullopt;
}

// Looks for the continuation of an interrupted copy: an occurrence of the last `min_suffix`
// tokens whose source lies within [resume - before, resume + after] and whose continuation
// begins with `required_prefix`. The source closest to `resume` wins. After a local edit the
// copy usually picks up just past the replaced text, so a short suffix is enough evidence here
// while a search of the whole history would need a long one.
[[nodiscard]] inline std::optional<ContextLookupCandidate>
find_resumed_continuation(std::span<const TokenId> history, std::uint32_t min_suffix,
                          std::uint32_t extent, std::span<const TokenId> required_prefix,
                          std::size_t resume, std::size_t before, std::size_t after) {
    namespace impl = context_lookup_detail;
    if (min_suffix == 0 || extent == 0 || extent > kContextLookupMaximumProposal ||
        required_prefix.size() > extent || history.size() <= min_suffix) {
        return std::nullopt;
    }
    const std::size_t first =
        std::max<std::size_t>(min_suffix, resume > before ? resume - before : 0);
    const std::size_t last = std::min(history.size() - 1U, resume + after);
    std::optional<ContextLookupCandidate> best;
    std::size_t best_distance = 0;
    for (std::size_t source = first; source <= last; ++source) {
        const std::size_t distance = source > resume ? source - resume : resume - source;
        if ((best && distance >= best_distance) ||
            !impl::suffix_recurs_at(history, source, min_suffix)) {
            continue;
        }
        ContextLookupCandidate result;
        impl::copy_continuation(history, source, extent, result);
        if (!std::equal(required_prefix.begin(), required_prefix.end(), result.tokens.begin())) {
            continue;
        }
        result.matched_suffix = impl::shared_suffix(history, source, min_suffix);
        best          = result;
        best_distance = distance;
    }
    return best;
}

// Tracks whether generation is inside a tool call. Tool-call arguments (edit targets, paths,
// commands) are the text most often copied from context, so the adaptive policy treats them as
// their own region.
class ToolCallRegion {
public:
    // Generation starts outside a tool call; only tokens appended after `generated_begin` count.
    void begin(ToolCallDelimiters delimiters, std::size_t generated_begin) noexcept {
        delimiters_ = delimiters;
        scanned_    = generated_begin;
        inside_     = false;
    }

    bool observe(std::span<const TokenId> ledger) noexcept {
        if (ledger.size() < scanned_) {
            // A shorter ledger means the continuation was rebuilt; restart conservatively.
            scanned_ = ledger.size();
            inside_  = false;
        }
        for (; scanned_ < ledger.size(); ++scanned_) {
            if (ledger[scanned_] == delimiters_.open) {
                inside_ = true;
            } else if (ledger[scanned_] == delimiters_.close) {
                inside_ = false;
            }
        }
        return inside_;
    }

    [[nodiscard]] bool inside() const noexcept { return inside_; }

private:
    ToolCallDelimiters delimiters_;
    std::size_t scanned_ = 0;
    bool inside_         = false;
};

// Per-request lookup policy state.
//
// Fixed reproduces the original rule: the nearest occurrence of an exact `min_suffix` suffix,
// used only when the learned MTP drafts agree with it.
//
// Adaptive adds three things, all judged against the learned MTP drafts (which already copy well,
// so a lookup round only pays when the copy runs well past them):
//  - Tiers: when an entry tier is planned, a copy uses the full lookup width only once it is
//    established (its previous round accepted every copied token, or it matches 16 tokens of
//    context like the fixed rule); other copies are probed with the narrower entry tier.
//  - Resume: after a lookup round, the copy it followed is remembered. When generation returns
//    to that source just past the point where it diverged (a local edit), `min_suffix` matching
//    tokens are enough to continue it.
//  - Region thresholds: elsewhere a new copy needs a longer recurring suffix, tracked separately
//    for tool-call arguments (which start at `min_suffix`) and other text (which starts at the
//    fixed rule's 16). A paying round halves its region's threshold toward `min_suffix`; a round
//    that gained nothing doubles it toward kContextLookupMaximumSuffix, so a region full of
//    incidental repetitions stops widening while a copy-heavy region widens after a few matching
//    tokens.
class ContextLookupController {
public:
    static constexpr std::uint32_t kTextInitialSuffix = 16;
    // Tokens a lookup round must add beyond its agreed learned drafts to count as paying, and
    // below which it counts as wasted.
    static constexpr std::uint32_t kPayingExcess = 4;
    static constexpr std::uint32_t kWastedExcess = 2;
    // Resume window around the divergence point, and how long an interrupted copy stays
    // resumable, in generated tokens.
    static constexpr std::size_t kResumeBefore  = 4;
    static constexpr std::size_t kResumeAfter   = 96;
    static constexpr std::size_t kResumeHorizon = 256;

    ContextLookupController() = default;

    ContextLookupController(const ContextLookupOptions& options, ToolCallDelimiters delimiters,
                            std::size_t generated_begin)
        : policy_(options.policy), floor_(options.min_suffix) {
        const std::uint32_t text_initial = std::max(options.min_suffix, kTextInitialSuffix);
        thresholds_                      = {text_initial, options.min_suffix};
        region_.begin(delimiters, generated_begin);
    }

    // Advances region tracking over newly committed tokens.
    void observe(std::span<const TokenId> ledger) noexcept {
        if (policy_ == ContextLookupPolicy::Adaptive) { (void)region_.observe(ledger); }
    }

    [[nodiscard]] bool in_tool_call() const noexcept { return region_.inside(); }

    [[nodiscard]] std::uint32_t suffix_threshold() const noexcept {
        return policy_ == ContextLookupPolicy::Adaptive ? thresholds_[region_index()] : floor_;
    }

    // True when the previous round was a lookup round that accepted every copied token, so the
    // copy it followed is still running at `ledger_size`.
    [[nodiscard]] bool continues_copy(std::size_t ledger_size) const noexcept {
        return continuing_at_ != 0 && continuing_at_ == ledger_size;
    }

    // Whether `candidate` has earned the full lookup width: its copy is still running, or it
    // matches as much context as the fixed rule requires. Other copies enter narrow.
    [[nodiscard]] bool established(const ContextLookupCandidate& candidate,
                                   std::size_t ledger_size) const noexcept {
        return continues_copy(ledger_size) ||
               candidate.matched_suffix >= kContextLookupEstablishedSuffix;
    }

    // Proposes a copied continuation of `extent` tokens that begins with the learned drafts, and
    // remembers its source for record().
    [[nodiscard]] std::optional<ContextLookupCandidate>
    propose(std::span<const TokenId> ledger, std::span<const TokenId> learned_drafts,
            std::uint32_t extent) {
        if (policy_ == ContextLookupPolicy::Off || learned_drafts.empty()) { return std::nullopt; }
        std::optional<ContextLookupCandidate> found;
        if (policy_ == ContextLookupPolicy::Adaptive) {
            if (resume_ && ledger.size() <= resume_->ledger_size + kResumeHorizon) {
                found = find_resumed_continuation(ledger, floor_, extent, learned_drafts,
                                                  resume_->source, kResumeBefore, kResumeAfter);
            }
            if (!found) {
                found =
                    find_context_continuation(ledger, thresholds_[region_index()], extent,
                                              learned_drafts, ContextLookupSearch::NearestAgreeing);
            }
        } else {
            found = find_context_continuation(ledger, floor_, extent, learned_drafts,
                                              ContextLookupSearch::Nearest);
            if (found &&
                !std::equal(learned_drafts.begin(), learned_drafts.end(), found->tokens.begin())) {
                found.reset();
            }
        }
        if (found) { proposed_ = Resume{.source = found->source, .ledger_size = ledger.size()}; }
        return found;
    }

    // Records the verified outcome of the last proposal: `agreed` learned drafts preceded the
    // copied tail, `proposed` tokens were verified and the target accepted `accepted` of them.
    void record(std::uint32_t agreed, std::uint32_t accepted, std::uint32_t proposed) noexcept {
        if (policy_ != ContextLookupPolicy::Adaptive || !proposed_) { return; }
        // The accepted prefix plus the target's own next token were committed.
        resume_ = Resume{.source      = proposed_->source + accepted,
                         .ledger_size = proposed_->ledger_size + accepted + 1U};
        proposed_.reset();
        const bool complete        = accepted >= proposed;
        continuing_at_             = complete ? resume_->ledger_size : 0U;
        const std::uint32_t excess = accepted > agreed ? accepted - agreed : 0U;
        std::uint32_t& threshold   = thresholds_[region_index()];
        if (complete || excess >= kPayingExcess) {
            threshold = std::max(floor_, threshold / 2U);
        } else if (excess < kWastedExcess) {
            threshold = std::min(kContextLookupMaximumSuffix, threshold * 2U);
        }
    }

private:
    struct Resume {
        std::size_t source      = 0;
        std::size_t ledger_size = 0;
    };

    [[nodiscard]] std::size_t region_index() const noexcept { return region_.inside() ? 1U : 0U; }

    ContextLookupPolicy policy_ = ContextLookupPolicy::Off;
    std::uint32_t floor_        = 16;
    std::array<std::uint32_t, 2> thresholds_{16, 16};
    ToolCallRegion region_;
    std::optional<Resume> proposed_;
    std::optional<Resume> resume_;
    // Ledger size at which the last fully accepted lookup round's copy continues; zero if none.
    std::size_t continuing_at_ = 0;
};

} // namespace ninfer::targets::qwen3_6::detail
