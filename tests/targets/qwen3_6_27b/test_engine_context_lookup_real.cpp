#include "ninfer/engine.h"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::EngineOptions engine_options(const char* artifact, ninfer::ContextLookupPolicy policy) {
    ninfer::EngineOptions options;
    options.artifact_path              = artifact;
    options.max_context                = 4096;
    options.kv_capacity                = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk              = 1024;
    options.speculative.backend        = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens   = 4;
    options.speculative.proposal_head  = ninfer::ProposalHead::Optimized;
    // A disabled lookup has no frame to size, so the Engine must accept sizes that fixed and
    // adaptive reject.
    const bool off                     = policy == ninfer::ContextLookupPolicy::Off;
    options.speculative.context_lookup = ninfer::ContextLookupOptions{
        .policy       = policy,
        .min_suffix   = off ? 0U : 4U,
        .max_proposal = off ? 0U : ninfer::kContextLookupMaximumProposal};
    options.max_concurrency      = 1;
    options.max_pending_requests = 1;
    return options;
}

// A copy with one edit near the start of every line. The rest of each line is long enough for the
// fixed rule's sixteen-token match; after each edit, adaptive lookup resumes the copy from a few
// matching tokens and verifies it with its entry tier first.
ninfer::PromptInput copy_prompt() {
    static constexpr const char* kText =
        "Rewrite the list below exactly, but replace the word \"red\" with \"blue\" on every "
        "line. Output only the rewritten list.\n\n"
        "1. The red kite drifted slowly over the harbor while the morning ferry left the pier for "
        "the northern islands, carrying mail, fresh bread and a crate of oranges for the "
        "lighthouse keeper.\n"
        "2. A red lantern hung above the bakery door on the narrow street behind the market, where "
        "the baker stacked warm loaves on wooden shelves long before the first customers "
        "arrived.\n"
        "3. Her red notebook held every measurement from the survey of the old stone bridge, "
        "including the width of each arch, the depth of the river and the date of every repair.\n"
        "4. The red tractor pulled a trailer of hay across the lower field before the evening "
        "rain, while two dogs ran along the hedge and the farmer counted the remaining bales.\n"
        "5. Two red chairs stood on the porch facing the orchard, where the apple trees had been "
        "pruned in March and the first blossoms appeared a week after the frost had passed.\n"
        "6. A red scarf was left on the bench at the station after the last train departed, and "
        "the night guard folded it carefully and placed it in the box of lost property.\n"
        "7. The red door of the library opened at nine every morning, and the librarian sorted the "
        "returned books into three carts before the reading room filled with students.\n"
        "8. His red bicycle leaned against the fence next to the garden of tall sunflowers, with a "
        "basket of tools on the back and a map of the coastal path folded under the seat.\n";
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = kText, .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

ninfer::SpeculativeStats generate_copy(const char* artifact, ninfer::ContextLookupPolicy policy,
                                       const char* label) {
    ninfer::Engine engine(engine_options(artifact, policy));
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens    = 768;
    request.execution.sampling.temperature       = 0.0F;
    request.execution.sampling.presence_penalty  = 0.0F;
    request.execution.sampling.frequency_penalty = 0.0F;
    request.execution.allow_prefix_reuse         = false;
    const ninfer::GenerationResult result = engine.generate(engine.prepare(copy_prompt()), request);
    const ninfer::SpeculativeStats& stats = result.speculative;
    std::cout << label << ": " << result.generated_token_ids.size() << " tokens, "
              << stats.rounds + stats.fallback_steps << " rounds, " << stats.lookup_rounds
              << " lookup (" << stats.lookup_entry_rounds << " entry), "
              << stats.lookup_accepted_tokens << "/" << stats.lookup_drafted_tokens
              << " lookup tokens accepted\n";
    require(result.generated_token_ids.size() > 100 && stats.rounds != 0,
            std::string(label) + " did not decode the copy through MTP rounds: " + result.content);
    return stats;
}

} // namespace

// Opt-in: loads the artifact three times (off, fixed, adaptive) on one GPU.
int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_QWEN3_6_27B_WEIGHTS is not set\n";
        return 77;
    }
    try {
        const auto off = generate_copy(artifact, ninfer::ContextLookupPolicy::Off, "off");
        require(off.lookup_rounds == 0 && off.lookup_entry_rounds == 0 &&
                    off.lookup_drafted_tokens == 0,
                "disabled context lookup verified a copied continuation");

        const auto fixed = generate_copy(artifact, ninfer::ContextLookupPolicy::Fixed, "fixed");
        require(fixed.lookup_rounds != 0 && fixed.lookup_entry_rounds == 0,
                "fixed context lookup did not widen the copy at the full width only");

        const auto adaptive =
            generate_copy(artifact, ninfer::ContextLookupPolicy::Adaptive, "adaptive");
        require(adaptive.lookup_entry_rounds != 0 &&
                    adaptive.lookup_rounds >= adaptive.lookup_entry_rounds &&
                    adaptive.lookup_accepted_tokens != 0,
                "adaptive context lookup did not enter the edited copy through the entry tier");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
