#include "targets/qwen3_6/impl/runtime/context_lookup.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <random>
#include <span>
#include <string_view>
#include <vector>

namespace {

namespace q36 = ninfer::targets::qwen3_6::detail;
using ninfer::TokenId;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

constexpr q36::ToolCallDelimiters kDelimiters{.open = 900, .close = 901};

std::vector<TokenId> tokens(std::initializer_list<TokenId> values) { return values; }

std::optional<q36::ContextLookupCandidate> nearest(const std::vector<TokenId>& history,
                                                   std::uint32_t suffix, std::uint32_t extent) {
    return q36::find_context_continuation(history, suffix, extent, {},
                                          q36::ContextLookupSearch::Nearest);
}

bool continuation_is(const std::optional<q36::ContextLookupCandidate>& candidate,
                     std::initializer_list<TokenId> expected) {
    return candidate.has_value() &&
           std::equal(expected.begin(), expected.end(), candidate->tokens.begin());
}

// The rule shipped before configurable lookup: a fixed 16-token suffix and a 15-token
// continuation from the nearest complete occurrence. Fixed lookup with the default options must
// select exactly the same continuation.
std::optional<std::array<TokenId, 15>> original_rule(const std::vector<TokenId>& history) {
    constexpr std::size_t match  = 16;
    constexpr std::size_t extent = 15;
    if (history.size() <= match) { return std::nullopt; }
    const std::size_t suffix = history.size() - match;
    for (std::size_t candidate = suffix; candidate-- > 0;) {
        if (!std::equal(history.begin() + static_cast<std::ptrdiff_t>(candidate),
                        history.begin() + static_cast<std::ptrdiff_t>(candidate + match),
                        history.begin() + static_cast<std::ptrdiff_t>(suffix))) {
            continue;
        }
        const std::size_t source = candidate + match;
        std::array<TokenId, extent> result{};
        bool valid = true;
        for (std::size_t i = 0; i < extent; ++i) {
            const std::size_t index = source + i;
            if (index < history.size()) {
                result[i] = history[index];
            } else {
                const std::size_t generated = index - history.size();
                if (generated >= i) {
                    valid = false;
                    break;
                }
                result[i] = result[generated];
            }
        }
        if (valid) { return result; }
    }
    return std::nullopt;
}

void test_search() {
    // The nearest occurrence wins, so a repeated edit follows the freshest copy.
    const auto history = tokens({1, 2, 3, 10, 11, 1, 2, 3, 20, 21, 5, 1, 2, 3});
    expect(continuation_is(nearest(history, 3, 2), {20, 21}),
           "search did not prefer the nearest occurrence");
    expect(!nearest(tokens({1, 2, 3, 4, 5, 6}), 2, 1).has_value(),
           "search proposed a continuation for a suffix that never recurs");
    expect(!nearest(tokens({1, 2}), 2, 1).has_value(),
           "search proposed a continuation without an earlier occurrence");

    // A source that runs into the current suffix continues periodically from the copied prefix.
    const auto periodic = tokens({9, 1, 2, 1, 2});
    expect(continuation_is(nearest(periodic, 2, 5), {1, 2, 1, 2, 1}),
           "periodic source was not continued from its copied prefix");

    // Agreement-aware search looks past a nearer source that contradicts the learned drafts.
    const auto agreeing_history = tokens({4, 5, 60, 61, 62, 4, 5, 70, 71, 72, 3, 4, 5});
    const std::array<TokenId, 2> drafts{60, 61};
    const auto plain    = q36::find_context_continuation(agreeing_history, 2, 3, drafts,
                                                         q36::ContextLookupSearch::Nearest);
    const auto agreeing = q36::find_context_continuation(agreeing_history, 2, 3, drafts,
                                                         q36::ContextLookupSearch::NearestAgreeing);
    expect(continuation_is(plain, {70, 71, 72}), "nearest search consulted the learned drafts");
    expect(continuation_is(agreeing, {60, 61, 62}),
           "agreement-aware search did not reach the agreeing source");

    // The source index points at the first copied token; the shared suffix extends past the
    // minimum and stops at the first differing token.
    const auto sourced = nearest(tokens({50, 1, 2, 3, 4, 99, 51, 1, 2, 3, 4}), 2, 1);
    expect(sourced.has_value() && sourced->source == 5 && sourced->tokens[0] == 99 &&
               sourced->matched_suffix == 4,
           "search did not report the copied source and shared suffix");
    std::vector<TokenId> long_copy(200);
    for (std::size_t i = 0; i < long_copy.size(); ++i) {
        long_copy[i] = static_cast<TokenId>(i % 97);
    }
    const auto established = nearest(long_copy, 4, 15);
    expect(established.has_value() &&
               established->matched_suffix == q36::kContextLookupEstablishedSuffix,
           "shared suffix was not measured up to the established length");
}

// Histories built from random text and copies of earlier spans (including overlapping, hence
// periodic, copies) so 16-token recurrences are common.
std::vector<TokenId> copy_heavy_history(std::mt19937& generator) {
    std::uniform_int_distribution<int> token(0, 5);
    std::uniform_int_distribution<int> run(1, 24);
    std::uniform_int_distribution<int> pieces(1, 8);
    std::vector<TokenId> history;
    const int count = pieces(generator);
    for (int piece = 0; piece < count; ++piece) {
        const int length = run(generator);
        if (history.size() < 4 || generator() % 2 == 0) {
            for (int i = 0; i < length; ++i) { history.push_back(token(generator)); }
            continue;
        }
        std::uniform_int_distribution<std::size_t> start(0, history.size() - 1);
        const std::size_t from = start(generator);
        for (int i = 0; i < length * 2; ++i) {
            const TokenId copied = history[from + static_cast<std::size_t>(i)];
            history.push_back(copied);
        }
    }
    return history;
}

void test_fixed_rule_is_unchanged() {
    std::mt19937 generator(20260923);
    int compared = 0;
    int found    = 0;
    for (int trial = 0; trial < 4000; ++trial) {
        const std::vector<TokenId> history = copy_heavy_history(generator);
        const auto expected                = original_rule(history);
        const auto actual                  = nearest(history, 16, 15);
        ++compared;
        if (expected.has_value() != actual.has_value() ||
            (expected && !std::equal(expected->begin(), expected->end(), actual->tokens.begin()))) {
            expect(false, "fixed lookup diverged from the original 16/15 rule");
            return;
        }
        found += expected.has_value() ? 1 : 0;
    }
    expect(compared == 4000 && found > 500, "fixed-rule comparison did not exercise matches");
}

void test_tool_call_region() {
    q36::ToolCallRegion region;
    // Delimiters in the prompt (tool instructions, earlier turns) do not open a region.
    region.begin(kDelimiters, 3);
    std::vector<TokenId> ledger = tokens({900, 1, 2});
    expect(!region.observe(ledger), "prompt tokens opened a tool-call region");
    ledger.insert(ledger.end(), {5, 900, 6});
    expect(region.observe(ledger), "generated tool-call opener was not tracked");
    ledger.insert(ledger.end(), {7, 901});
    expect(!region.observe(ledger), "generated tool-call closer was not tracked");
    ledger.insert(ledger.end(), {900, 8});
    expect(region.observe(ledger), "second tool call was not tracked");
    ledger.resize(4);
    expect(!region.observe(ledger), "a rebuilt shorter ledger kept a stale region");
}

q36::ContextLookupController controller(ninfer::ContextLookupPolicy policy,
                                        std::uint32_t min_suffix, std::size_t generated_begin) {
    return q36::ContextLookupController(ninfer::ContextLookupOptions{.policy       = policy,
                                                                     .min_suffix   = min_suffix,
                                                                     .max_proposal = 15},
                                        kDelimiters, generated_begin);
}

void test_fixed_and_off_controllers() {
    auto fixed = controller(ninfer::ContextLookupPolicy::Fixed, 3, 0);
    expect(fixed.suffix_threshold() == 3, "fixed lookup did not use the configured suffix");
    fixed.record(4, 0, 15);
    expect(fixed.suffix_threshold() == 3, "fixed lookup adapted its threshold");

    const auto history = tokens({1, 2, 3, 10, 11, 12, 1, 2, 3, 20, 21, 22, 1, 2, 3});
    const std::array<TokenId, 1> nearest_draft{20};
    const std::array<TokenId, 1> older_draft{10};
    expect(fixed.propose(history, nearest_draft, 3).has_value(),
           "fixed lookup rejected an agreeing nearest source");
    expect(!fixed.propose(history, older_draft, 3).has_value(),
           "fixed lookup reached past a disagreeing nearest source");
    expect(!fixed.propose(history, {}, 3).has_value(),
           "lookup was proposed without learned drafts to agree with");

    auto off = controller(ninfer::ContextLookupPolicy::Off, 3, 0);
    expect(!off.propose(history, nearest_draft, 3).has_value(), "disabled lookup proposed");
}

// A 80-token source span, a separator, then a copy of the span's first `copied` tokens: any
// suffix of up to `copied` tokens recurs, and its continuation starts at history index `copied`.
std::vector<TokenId> copied_ledger(std::size_t copied) {
    std::vector<TokenId> ledger;
    for (TokenId token = 0; token < 80; ++token) { ledger.push_back(1000 + token); }
    ledger.insert(ledger.end(), {5, 6});
    for (std::size_t i = 0; i < copied; ++i) { ledger.push_back(1000 + static_cast<TokenId>(i)); }
    return ledger;
}

void propose_and_record(q36::ContextLookupController& controller,
                        const std::vector<TokenId>& ledger, TokenId next, std::uint32_t agreed,
                        std::uint32_t accepted) {
    const std::array<TokenId, 1> draft{next};
    const auto found = controller.propose(ledger, draft, 15);
    expect(found.has_value() && found->tokens[0] == next, "copy was not proposed");
    controller.record(agreed, accepted, 15);
}

void test_adaptive_thresholds() {
    auto adaptive = controller(ninfer::ContextLookupPolicy::Adaptive, 4, 0);
    expect(adaptive.suffix_threshold() == 16,
           "adaptive text threshold did not start from the fixed rule");
    const std::vector<TokenId> ledger = copied_ledger(70);
    // Rounds that add nothing beyond the agreed learned drafts back the region off.
    propose_and_record(adaptive, ledger, 1070, 4, 4);
    expect(adaptive.suffix_threshold() == 32, "a wasted round did not raise the threshold");
    propose_and_record(adaptive, ledger, 1070, 4, 5);
    propose_and_record(adaptive, ledger, 1070, 4, 3);
    expect(adaptive.suffix_threshold() == 64, "the threshold was not capped at the match cap");
    // A round that only nearly pays leaves the threshold alone; a paying one halves it.
    propose_and_record(adaptive, ledger, 1070, 4, 7);
    expect(adaptive.suffix_threshold() == 64, "a marginal round moved the threshold");
    propose_and_record(adaptive, ledger, 1070, 1, 12);
    expect(adaptive.suffix_threshold() == 32, "a paying round did not halve the threshold");

    // Tool-call arguments keep their own threshold, starting at the configured floor.
    auto regions              = controller(ninfer::ContextLookupPolicy::Adaptive, 4, 0);
    std::vector<TokenId> tool = copied_ledger(70);
    tool.insert(tool.begin(), 900);
    regions.observe(tool);
    expect(regions.in_tool_call() && regions.suffix_threshold() == 4,
           "tool-call region did not start at the configured floor");
    propose_and_record(regions, tool, 1070, 1, 1);
    expect(regions.suffix_threshold() == 8, "tool-call region did not back off");
    propose_and_record(regions, tool, 1070, 1, 15);
    propose_and_record(regions, tool, 1070, 4, 15);
    expect(regions.suffix_threshold() == 4, "a paying round lowered the threshold below its floor");
    tool.push_back(901);
    regions.observe(tool);
    expect(!regions.in_tool_call() && regions.suffix_threshold() == 16,
           "tool-call rounds changed the text threshold");

    // Accepting a whole narrow proposal pays even though it adds few tokens beyond the drafts.
    auto narrow = controller(ninfer::ContextLookupPolicy::Adaptive, 4, 0);
    const std::array<TokenId, 4> drafts{1070, 1071, 1072, 1073};
    expect(narrow.propose(ledger, drafts, 15).has_value(), "narrow copy was not proposed");
    narrow.record(4, 7, 7);
    expect(narrow.suffix_threshold() == 8, "a fully accepted narrow round did not pay");

    // A floor above the fixed default also raises the text starting point.
    const auto strict = controller(ninfer::ContextLookupPolicy::Adaptive, 24, 0);
    expect(strict.suffix_threshold() == 24, "text threshold started below the configured floor");
}

void test_adaptive_resume() {
    auto adaptive = controller(ninfer::ContextLookupPolicy::Adaptive, 4, 0);
    // Back the text threshold off to the cap so only a resumed copy can match a short suffix.
    std::vector<TokenId> ledger = copied_ledger(20);
    const std::array<TokenId, 1> draft{1020};
    for (int round = 0; round < 3; ++round) {
        expect(adaptive.propose(ledger, draft, 15).has_value(), "the first copy was not proposed");
        adaptive.record(1, 1, 15);
    }
    expect(adaptive.suffix_threshold() == 64, "text threshold did not reach the cap");

    // The last round accepted 1020 then diverged; generation inserts one token and resumes the
    // source two tokens later.
    ledger.insert(ledger.end(), {1020, 77, 1022, 1023, 1024, 1025});
    const std::array<TokenId, 2> resumed_drafts{1026, 1027};
    const auto resumed = adaptive.propose(ledger, resumed_drafts, 15);
    expect(resumed.has_value() && resumed->source == 26 && resumed->tokens[0] == 1026,
           "an interrupted copy was not resumed from a short suffix");

    // Far from the interrupted copy, the same short suffix is not enough.
    auto expired                 = controller(ninfer::ContextLookupPolicy::Adaptive, 4, 0);
    std::vector<TokenId> distant = copied_ledger(20);
    for (int round = 0; round < 3; ++round) {
        expect(expired.propose(distant, draft, 15).has_value(), "the first copy was not proposed");
        expired.record(1, 1, 15);
    }
    for (std::size_t i = 0; i < q36::ContextLookupController::kResumeHorizon; ++i) {
        distant.push_back(2000 + static_cast<TokenId>(i));
    }
    distant.insert(distant.end(), {1022, 1023, 1024, 1025});
    expect(!expired.propose(distant, resumed_drafts, 15).has_value(),
           "a copy was resumed after its horizon");
}

void test_copy_continuation() {
    auto adaptive                     = controller(ninfer::ContextLookupPolicy::Adaptive, 4, 0);
    const std::vector<TokenId> ledger = copied_ledger(20);
    const std::array<TokenId, 1> draft{1020};
    expect(!adaptive.continues_copy(ledger.size()), "a copy continued before any lookup round");
    expect(adaptive.propose(ledger, draft, 15).has_value(), "copy was not proposed");
    adaptive.record(1, 7, 7);
    // Seven copied tokens and the target's own next token were committed.
    expect(adaptive.continues_copy(ledger.size() + 8), "a fully accepted copy did not continue");
    expect(!adaptive.continues_copy(ledger.size() + 9),
           "a copy continued past the round that followed it");
    expect(adaptive.propose(ledger, draft, 15).has_value(), "copy was not proposed");
    adaptive.record(1, 3, 7);
    expect(!adaptive.continues_copy(ledger.size() + 4), "a partially accepted copy continued");
}

void test_established_copy() {
    // Inside a tool call the floor applies, so a six-token copy is found but only probed narrow.
    auto adaptive = controller(ninfer::ContextLookupPolicy::Adaptive, 4, 0);
    std::vector<TokenId> short_copy = copied_ledger(6);
    short_copy.insert(short_copy.begin(), 900);
    adaptive.observe(short_copy);
    const std::array<TokenId, 1> short_draft{1006};
    const auto probe = adaptive.propose(short_copy, short_draft, 15);
    expect(probe.has_value() && probe->matched_suffix == 6 &&
               !adaptive.established(*probe, short_copy.size()),
           "a six-token match was treated as established");

    // A copy that shares the fixed rule's sixteen tokens is established immediately.
    std::vector<TokenId> long_copy = copied_ledger(20);
    long_copy.insert(long_copy.begin(), 900);
    const std::array<TokenId, 1> long_draft{1020};
    const auto match = adaptive.propose(long_copy, long_draft, 15);
    expect(match.has_value() && adaptive.established(*match, long_copy.size()),
           "a sixteen-token match was not established");

    // A short copy whose whole narrow proposal was accepted is established for the next round.
    expect(adaptive.propose(short_copy, short_draft, 15).has_value(), "copy was not proposed");
    adaptive.record(1, 7, 7);
    expect(adaptive.established(*probe, short_copy.size() + 8),
           "a fully accepted copy was not established for the next round");
}

q36::ContextLookupPlan plan(ninfer::SpeculativeBackend backend, std::uint32_t draft_window,
                            ninfer::ContextLookupPolicy policy, std::uint32_t max_proposal = 15) {
    return q36::plan_context_lookup(backend, draft_window,
                                    ninfer::ContextLookupOptions{.policy       = policy,
                                                                 .min_suffix   = 16,
                                                                 .max_proposal = max_proposal});
}

bool plans(const q36::ContextLookupPlan& actual, std::uint32_t lookup, std::uint32_t entry,
           std::uint32_t families) {
    return actual.lookup_window == lookup && actual.entry_window == entry &&
           actual.mtp_graph_families() == families;
}

void test_planning() {
    using ninfer::ContextLookupPolicy;
    using ninfer::SpeculativeBackend;
    // Off plans no lookup frame and no graph family beyond the learned-draft one.
    const auto off = plan(SpeculativeBackend::Mtp, 4, ContextLookupPolicy::Off);
    expect(plans(off, 0, 0, 1) && !off.lookup_planned() && off.widest_window() == 4,
           "disabled lookup planned a lookup frame or graph family");
    // Only MTP looks up context, whatever the (default fixed) policy says.
    for (const auto backend :
         {SpeculativeBackend::None, SpeculativeBackend::DFlash, SpeculativeBackend::DFlash2}) {
        const auto other = q36::plan_context_lookup(backend, 7, ninfer::ContextLookupOptions{});
        expect(!other.lookup_planned() && other.entry_window == 0,
               "a non-MTP backend planned a lookup frame");
    }

    // Fixed verifies the configured proposal and never plans the entry tier.
    expect(plans(plan(SpeculativeBackend::Mtp, 4, ContextLookupPolicy::Fixed), 15, 0, 2),
           "fixed lookup did not plan one fifteen-token frame");
    expect(plans(plan(SpeculativeBackend::Mtp, 4, ContextLookupPolicy::Fixed, 8), 8, 0, 2),
           "fixed lookup did not follow the configured proposal");
    // Adaptive adds the seven-token entry tier only strictly between the draft window and the
    // full proposal.
    expect(plans(plan(SpeculativeBackend::Mtp, 4, ContextLookupPolicy::Adaptive), 15, 7, 3),
           "adaptive lookup did not plan the entry tier");
    expect(plans(plan(SpeculativeBackend::Mtp, 6, ContextLookupPolicy::Adaptive, 8), 8, 7, 3),
           "adaptive lookup did not plan the entry tier between six drafts and eight copies");
    expect(plans(plan(SpeculativeBackend::Mtp, 7, ContextLookupPolicy::Adaptive), 15, 0, 2),
           "adaptive lookup planned an entry tier no wider than the draft window");
    expect(plans(plan(SpeculativeBackend::Mtp, 4, ContextLookupPolicy::Adaptive, 7), 7, 0, 2),
           "adaptive lookup planned an entry tier as wide as the full proposal");

    // Every accepted combination keeps the planned widths distinct and inside the lookup frame,
    // so a verified row's block width identifies its frame.
    for (std::uint32_t drafts = 1; drafts <= 7; ++drafts) {
        for (std::uint32_t proposal = drafts + 1; proposal <= ninfer::kContextLookupMaximumProposal;
             ++proposal) {
            for (const auto policy : {ContextLookupPolicy::Fixed, ContextLookupPolicy::Adaptive}) {
                const auto planned = plan(SpeculativeBackend::Mtp, drafts, policy, proposal);
                bool valid         = planned.lookup_window == proposal &&
                             planned.widest_window() == proposal &&
                             (planned.entry_window == 0 ||
                              (drafts < planned.entry_window && planned.entry_window < proposal));
                for (const auto verification :
                     {q36::MtpVerification::Drafts, q36::MtpVerification::LookupEntry,
                      q36::MtpVerification::Lookup}) {
                    const std::uint32_t width = planned.verify_window(verification);
                    if (width != 0) {
                        valid =
                            valid && planned.verification_for_stride(width + 1U) == verification;
                    }
                }
                expect(valid, "a planned lookup frame is ambiguous or outside the frame domain");
            }
        }
    }
}

q36::ContextLookupRound select_round(const q36::ContextLookupPlan& planned,
                                     std::initializer_list<q36::ContextLookupRow> rows) {
    const std::vector<q36::ContextLookupRow> batch(rows);
    return q36::select_context_lookup_round(planned, batch);
}

bool selects(const q36::ContextLookupRound& round, q36::MtpVerification verification,
             std::uint32_t verify, std::uint32_t proposal) {
    return round.verification == verification && round.verify == verify &&
           round.proposal == proposal;
}

void test_round_selection() {
    using ninfer::ContextLookupPolicy;
    using ninfer::SpeculativeBackend;
    using q36::MtpVerification;
    const auto off      = plan(SpeculativeBackend::Mtp, 4, ContextLookupPolicy::Off);
    const auto fixed    = plan(SpeculativeBackend::Mtp, 4, ContextLookupPolicy::Fixed);
    const auto adaptive = plan(SpeculativeBackend::Mtp, 4, ContextLookupPolicy::Adaptive);
    constexpr q36::ContextLookupRow established{.room = 100, .proposed = true, .established = true};
    constexpr q36::ContextLookupRow probe{.room = 100, .proposed = true, .established = false};
    constexpr q36::ContextLookupRow none{.room = 100, .proposed = false, .established = false};

    expect(selects(select_round(off, {established}), MtpVerification::Drafts, 4, 4),
           "disabled lookup widened a round");
    expect(selects(select_round(fixed, {}), MtpVerification::Drafts, 4, 4),
           "an empty batch widened a round");
    // Fixed widens every agreeing copy to the full width and keeps one learned draft as the next
    // agreement guard.
    expect(selects(select_round(fixed, {probe}), MtpVerification::Lookup, 15, 1),
           "fixed lookup did not verify a copy at the full width");
    // Adaptive probes a copy that is not established yet with the entry tier, which regenerates
    // the learned window; an established copy uses the full width.
    expect(selects(select_round(adaptive, {probe}), MtpVerification::LookupEntry, 7, 4),
           "adaptive lookup did not enter a new copy through the entry tier");
    expect(selects(select_round(adaptive, {established}), MtpVerification::Lookup, 15, 1),
           "adaptive lookup did not widen an established copy fully");
    // Without an entry tier, adaptive verifies a new copy at the full width.
    expect(selects(select_round(plan(SpeculativeBackend::Mtp, 7, ContextLookupPolicy::Adaptive),
                                {probe}),
                   MtpVerification::Lookup, 15, 1),
           "adaptive lookup without an entry tier did not use the full width");

    // Long verification is one batch topology: one row without a copy keeps the batch on the
    // learned window, one unestablished row keeps it on the entry tier.
    expect(
        selects(select_round(adaptive, {established, probe}), MtpVerification::LookupEntry, 7, 4),
        "an unestablished row did not hold the batch on the entry tier");
    expect(selects(select_round(fixed, {established, none}), MtpVerification::Drafts, 4, 4),
           "a row without a copy did not hold the batch on the learned window");
    // A row whose room does not exceed the learned window keeps the batch narrow; a little more
    // room widens it, and that row verifies only what fits.
    const q36::ContextLookupRow full_budget{.room = 4, .proposed = true, .established = true};
    const q36::ContextLookupRow tight_budget{.room = 5, .proposed = true, .established = true};
    expect(selects(select_round(fixed, {established, full_budget}), MtpVerification::Drafts, 4, 4),
           "a row without room beyond the learned window widened the batch");
    const auto tight = select_round(fixed, {established, tight_budget});
    expect(selects(tight, MtpVerification::Lookup, 15, 1) && tight.row_extent(5, 1) == 5,
           "a row with little room did not verify only what fits");

    // Room beyond a frame is clamped to it; a learned-draft row verifies the drafts it holds.
    const auto full    = select_round(fixed, {established});
    const auto entry   = select_round(adaptive, {probe});
    const auto learned = select_round(off, {established});
    expect(full.row_extent(1000, 1) == 15 && entry.row_extent(1000, 4) == 7 &&
               learned.row_extent(1000, 4) == 4 && learned.row_extent(1000, 2) == 2 &&
               learned.row_extent(1, 4) == 1,
           "a round extent was not clamped to its frame, room and learned drafts");
}

void test_empty_history() {
    // No history, or one no longer than the suffix, proposes nothing and plans a learned round.
    const std::vector<TokenId> empty;
    expect(!nearest(empty, 2, 15).has_value() && !nearest(tokens({1, 2}), 2, 15).has_value(),
           "an empty or suffix-only history proposed a continuation");
    expect(!q36::find_resumed_continuation(empty, 2, 15, {}, 0, 4, 96).has_value(),
           "an empty history resumed a copy");
    const std::array<TokenId, 1> draft{7};
    for (const auto policy :
         {ninfer::ContextLookupPolicy::Fixed, ninfer::ContextLookupPolicy::Adaptive}) {
        auto lookup = controller(policy, 2, 0);
        lookup.observe(empty);
        const auto found = lookup.propose(empty, draft, 15);
        const auto round =
            select_round(plan(ninfer::SpeculativeBackend::Mtp, 4, policy),
                         {q36::ContextLookupRow{.room = 100, .proposed = found.has_value()}});
        expect(!found && !lookup.in_tool_call() &&
                   round.verification == q36::MtpVerification::Drafts,
               "an empty history widened a round");
    }
}

void test_first_generated_tool_call() {
    // The prompt is a source span and a separator; generation opens a tool call with its first
    // token and then copies four source tokens.
    const std::vector<TokenId> prompt = copied_ledger(0);
    std::vector<TokenId> ledger       = prompt;
    ledger.push_back(kDelimiters.open);
    ledger.insert(ledger.end(), {1000, 1001, 1002, 1003});
    const std::array<TokenId, 1> draft{1004};
    const auto adaptive_plan =
        plan(ninfer::SpeculativeBackend::Mtp, 4, ninfer::ContextLookupPolicy::Adaptive);

    auto opened = controller(ninfer::ContextLookupPolicy::Adaptive, 4, prompt.size());
    opened.observe(ledger);
    expect(opened.in_tool_call() && opened.suffix_threshold() == 4,
           "a tool call opened by the first generated token was not tracked");
    const auto found = opened.propose(ledger, draft, 15);
    expect(found.has_value() && found->tokens[0] == 1004 &&
               !opened.established(*found, ledger.size()),
           "a short copy inside the first tool call was not proposed as a probe");
    const auto round = select_round(
        adaptive_plan,
        {q36::ContextLookupRow{.room = 100, .proposed = found.has_value(), .established = false}});
    expect(round.verification == q36::MtpVerification::LookupEntry,
           "a short copy inside the first tool call did not use the entry tier");

    // The same delimiter as the last prompt token is not generated text, so the copy needs the
    // plain-text suffix and is not proposed.
    std::vector<TokenId> quoted = prompt;
    quoted.push_back(kDelimiters.open);
    const std::size_t quoted_prompt = quoted.size();
    quoted.insert(quoted.end(), {1000, 1001, 1002, 1003});
    auto text = controller(ninfer::ContextLookupPolicy::Adaptive, 4, quoted_prompt);
    text.observe(quoted);
    expect(!text.in_tool_call() && text.suffix_threshold() == 16 &&
               !text.propose(quoted, draft, 15).has_value(),
           "a prompt delimiter opened a tool-call region");
}

void test_resumed_search() {
    // Two sources of the same suffix; the one closest to the resume point wins.
    const auto history    = tokens({1, 2, 3, 40, 41, 9, 9, 1, 2, 3, 50, 51, 8, 1, 2, 3});
    const auto near_first = q36::find_resumed_continuation(history, 3, 2, {}, 4, 2, 20);
    expect(continuation_is(near_first, {40, 41}) && near_first->source == 3,
           "resumed search did not prefer the source closest to the resume point");
    const auto near_second = q36::find_resumed_continuation(history, 3, 2, {}, 11, 2, 20);
    expect(continuation_is(near_second, {50, 51}), "resumed search ignored the nearer source");
    const std::array<TokenId, 1> second_draft{50};
    const auto agreeing = q36::find_resumed_continuation(history, 3, 2, second_draft, 4, 2, 20);
    expect(continuation_is(agreeing, {50, 51}),
           "resumed search did not skip a source that contradicts the learned drafts");
    expect(!q36::find_resumed_continuation(history, 3, 2, {}, 4, 0, 0).has_value(),
           "resumed search looked outside its window");
    expect(!q36::find_resumed_continuation(history, 3, 2, {}, 14, 1, 1).has_value(),
           "resumed search matched the current suffix itself");
}

} // namespace

int main() {
    test_search();
    test_fixed_rule_is_unchanged();
    test_tool_call_region();
    test_fixed_and_off_controllers();
    test_adaptive_thresholds();
    test_adaptive_resume();
    test_copy_continuation();
    test_established_copy();
    test_resumed_search();
    test_planning();
    test_round_selection();
    test_empty_history();
    test_first_generated_tool_call();
    if (failures != 0) {
        std::cerr << failures << " context lookup check(s) failed\n";
        return 1;
    }
    std::cout << "context lookup tests passed\n";
    return 0;
}
