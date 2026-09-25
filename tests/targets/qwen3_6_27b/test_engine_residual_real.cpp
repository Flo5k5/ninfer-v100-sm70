// FP32 text residual stream on the artifact named by NINFER_QWEN3_8_27B_NVFP4_WEIGHTS
// (qwen3.8-27b/nvfp4, nvfp4-full-a, nvfp4-full-b or nvfp4-full-c) in the production decode
// configuration: MTP with four drafts and the optimized proposal head, CUDA graphs, INT8 KV and
// 2048-token prefill chunks. The prompt spans two prefill chunks (the wide FP32 residual
// projections and split-D attention), then greedy decoding runs through the captured MTP verify
// and draft graphs with the FP32 stream.
//
// Greedy MTP decoding is lossless: its tokens must equal plain eager decoding with the same FP32
// residual. The drafts must keep being accepted, so the MTP head still reads a valid target hidden
// state. On this near-deterministic continuation a BF16-residual engine produces the same tokens.

#include "ninfer/engine.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kOutputTokens = 96;

ninfer::EngineOptions engine_options(const char* artifact, ninfer::TextResidualStorage residual,
                                     bool speculative) {
    ninfer::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_context          = 4096;
    options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk        = 2048;
    options.kv_cache             = ninfer::KvCacheStorage::Int8Group64;
    options.text_residual        = residual;
    options.max_concurrency      = 1;
    options.max_pending_requests = 1;
    options.use_cuda_graph       = speculative;
    if (speculative) {
        options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens  = 4;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    }
    return options;
}

// A counting sequence long enough for two 2048-token prefill chunks; its continuation is the
// next numbers.
std::vector<ninfer::TokenId> counting_prompt(const ninfer::Engine& engine) {
    std::string text;
    std::vector<ninfer::TokenId> tokens;
    for (int number = 1; tokens.size() < 2600; ++number) {
        text += std::to_string(number) + ", ";
        if (number % 50 == 0) { tokens = engine.tokenize_text(text); }
    }
    return tokens;
}

struct Decoded {
    std::vector<ninfer::TokenId> tokens;
    ninfer::SpeculativeStats speculative;
};

bool decode(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& prompt, Decoded& out,
            const char* label) {
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = kOutputTokens;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    request.stop.include_model_defaults       = false;
    const ninfer::GenerationResult result = engine.generate(engine.prepare_tokens(prompt), request);
    if (result.generated_token_ids.size() != kOutputTokens ||
        result.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << label << ": generated " << result.generated_token_ids.size()
                  << " tokens instead of " << kOutputTokens << '\n';
        return false;
    }
    out.tokens      = result.generated_token_ids;
    out.speculative = result.speculative;
    return true;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_NVFP4_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_QWEN3_8_27B_NVFP4_WEIGHTS is not set\n";
        return 77;
    }

    std::vector<ninfer::TokenId> prompt;
    Decoded speculative;
    {
        ninfer::Engine engine(
            engine_options(artifact, ninfer::TextResidualStorage::Float32, true));
        if (engine.options().text_residual != ninfer::TextResidualStorage::Float32 ||
            !engine.options().use_cuda_graph ||
            engine.options().speculative.backend != ninfer::SpeculativeBackend::Mtp) {
            std::cerr << "the FP32-residual MTP engine did not keep its configuration\n";
            return 1;
        }
        prompt = counting_prompt(engine);
        if (!decode(engine, prompt, speculative, "FP32 MTP graphs")) { return 1; }
    }
    const double acceptance =
        speculative.speculative.drafted_tokens == 0
            ? 0.0
            : static_cast<double>(speculative.speculative.accepted_tokens) /
                  static_cast<double>(speculative.speculative.drafted_tokens);
    if (speculative.speculative.rounds == 0 || acceptance < 0.5) {
        std::cerr << "FP32 MTP drafts were not accepted: "
                  << speculative.speculative.accepted_tokens << " of "
                  << speculative.speculative.drafted_tokens << '\n';
        return 1;
    }

    Decoded plain;
    {
        ninfer::Engine engine(
            engine_options(artifact, ninfer::TextResidualStorage::Float32, false));
        if (!decode(engine, prompt, plain, "FP32 plain eager")) { return 1; }
    }
    if (plain.tokens != speculative.tokens) {
        std::cerr << "greedy FP32 MTP decoding diverged from plain FP32 decoding\n";
        return 1;
    }

    Decoded bf16;
    {
        ninfer::Engine engine(
            engine_options(artifact, ninfer::TextResidualStorage::BFloat16, true));
        if (!decode(engine, prompt, bf16, "BF16 MTP graphs")) { return 1; }
    }
    if (bf16.tokens != speculative.tokens) {
        std::cerr << "the FP32 residual changed the greedy continuation of the BF16 engine\n";
        return 1;
    }

    std::cout << "OK residual_real prompt=" << prompt.size() << " rounds="
              << speculative.speculative.rounds << " acceptance=" << acceptance << '\n';
    return 0;
}
