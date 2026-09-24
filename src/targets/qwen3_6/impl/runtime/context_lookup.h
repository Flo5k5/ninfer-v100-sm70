#pragma once

// Host-side policy for MTP context lookup: which verification frames an engine plans, which copy
// a request proposes, and which frame each round uses. The target verifies every copied token, so
// the policy only decides when a round pays for a wider verification block; every emitted token
// is still the target's own choice at the verified width. The header is CUDA-free so the policy
// is testable in isolation.

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

// Which frame verifies an MTP round: the learned-draft window, the adaptive entry tier, or the
// full lookup width. Planning keeps the planned widths distinct, so the block width of a verified
// row identifies its frame.
enum class MtpVerification : std::uint8_t {
    Drafts,
    LookupEntry,
    Lookup,
};

// Startup-fixed MTP verification frames. Each planned lookup frame costs one more CUDA-graph
// family, frame and ReplaySSM record set, so `off` plans none and only adaptive plans the entry
// tier.
struct ContextLookupPlan {
    // Learned MTP drafts verified by an ordinary round.
    std::uint32_t draft_window = 0;
    // Copied tokens verified by a full lookup round; zero when no lookup frame is planned.
    std::uint32_t lookup_window = 0;
    // Copied tokens verified by the adaptive entry tier; zero when no entry frame is planned.
    std::uint32_t entry_window = 0;

    [[nodiscard]] constexpr bool lookup_planned() const noexcept { return lookup_window != 0; }

    // Drafts one round verifies; its target block holds one more column.
    [[nodiscard]] constexpr std::uint32_t
    verify_window(MtpVerification verification) const noexcept {
        switch (verification) {
        case MtpVerification::Lookup:
            return lookup_window;
        case MtpVerification::LookupEntry:
            return entry_window;
        case MtpVerification::Drafts:
            break;
        }
        return draft_window;
    }

    // Learned drafts a round proposes for the next round. The full tier only runs while a copy
    // continues and keeps one draft as the next agreement guard; the entry tier regenerates the
    // configured window, since the copy it probes often ends inside the round.
    [[nodiscard]] constexpr std::uint32_t
    proposal_window(MtpVerification verification) const noexcept {
        return verification == MtpVerification::Lookup ? 1U : draft_window;
    }

    // The widest planned verification; it bounds the MTP workspace and graph profiles.
    [[nodiscard]] constexpr std::uint32_t widest_window() const noexcept {
        return std::max(draft_window, lookup_window);
    }

    // CUDA-graph families of the MTP backend: the learned-draft family and one per lookup frame.
    [[nodiscard]] constexpr std::uint32_t mtp_graph_families() const noexcept {
        return 1U + (lookup_window != 0 ? 1U : 0U) + (entry_window != 0 ? 1U : 0U);
    }

    // The frame that verified a row whose block holds `row_stride` columns.
    [[nodiscard]] constexpr MtpVerification
    verification_for_stride(std::uint32_t row_stride) const noexcept {
        if (lookup_window != 0 && row_stride == lookup_window + 1U) {
            return MtpVerification::Lookup;
        }
        if (entry_window != 0 && row_stride == entry_window + 1U) {
            return MtpVerification::LookupEntry;
        }
        return MtpVerification::Drafts;
    }
};

// Plans the lookup frames of validated engine options. Only MTP looks up context, and `off` plans
// no lookup frame. Adaptive lookup enters copies through the narrower entry tier when it fits
// strictly between the learned-draft window and the full lookup width; otherwise every copy is
// verified at the full width.
[[nodiscard]] constexpr ContextLookupPlan plan_context_lookup(SpeculativeBackend backend,
                                                              std::uint32_t draft_window,
                                                              const ContextLookupOptions& options) {
    ContextLookupPlan plan{.draft_window = draft_window};
    if (backend != SpeculativeBackend::Mtp || options.policy == ContextLookupPolicy::Off) {
        return plan;
    }
    plan.lookup_window = options.max_proposal;
    if (options.policy == ContextLookupPolicy::Adaptive &&
        kContextLookupEntryProposal > draft_window &&
        kContextLookupEntryProposal < plan.lookup_window) {
        plan.entry_window = kContextLookupEntryProposal;
    }
    return plan;
}

// One batch row's lookup outcome before an MTP round.
struct ContextLookupRow {
    // Drafts the row may still verify: its output budget beyond the target's own next token,
    // bounded by the remaining context capacity. It may exceed every frame; row_extent clamps it.
    std::uint32_t room = 0;
    // The row proposed a copied continuation that agrees with its learned drafts.
    bool proposed = false;
    // The proposed copy has earned the full lookup width (ContextLookupController::established).
    bool established = false;
};

// The verification one MTP round uses for every row of its compact batch.
struct ContextLookupRound {
    MtpVerification verification = MtpVerification::Drafts;
    // Drafts verified per row; the target block holds one more column.
    std::uint32_t verify = 0;
    // Learned drafts proposed for the next round.
    std::uint32_t proposal = 0;

    [[nodiscard]] constexpr bool lookup() const noexcept {
        return verification != MtpVerification::Drafts;
    }

    // Drafts one row verifies: its copy in a lookup round, the learned drafts it holds otherwise.
    // Both stop at the row's room and never exceed the round's frame.
    [[nodiscard]] constexpr std::uint32_t row_extent(std::uint32_t room,
                                                     std::uint32_t learned_drafts) const noexcept {
        return std::min(lookup() ? verify : std::min(learned_drafts, verify), room);
    }
};

// Chooses the verification of one MTP round. Long verification is one batch topology, so the batch
// widens only when every row proposed a copy and has room for more drafts than the learned window
// verifies. It uses the full lookup width when every row's copy is established and otherwise
// probes with the entry tier, if one is planned.
[[nodiscard]] constexpr ContextLookupRound
select_context_lookup_round(const ContextLookupPlan& plan,
                            std::span<const ContextLookupRow> rows) noexcept {
    const ContextLookupRound learned{.verification = MtpVerification::Drafts,
                                     .verify       = plan.draft_window,
                                     .proposal     = plan.draft_window};
    if (!plan.lookup_planned() || rows.empty()) { return learned; }
    bool entry = false;
    for (const ContextLookupRow& row : rows) {
        if (!row.proposed) { return learned; }
        entry = entry || (plan.entry_window != 0 && !row.established);
    }
    const MtpVerification verification =
        entry ? MtpVerification::LookupEntry : MtpVerification::Lookup;
    const std::uint32_t verify = plan.verify_window(verification);
    for (const ContextLookupRow& row : rows) {
        if (std::min(verify, row.room) <= plan.draft_window) { return learned; }
    }
    return ContextLookupRound{.verification = verification,
                              .verify       = verify,
                              .proposal     = plan.proposal_window(verification)};
}

} // namespace ninfer::targets::qwen3_6::detail
