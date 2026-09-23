#include "ninfer/engine.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

// Records the shape of every logits block and recomputes the target log-probability of each row
// in FP64 from the exported BF16 logits.
class RecordingSink final : public ninfer::ScoreLogitsSink {
public:
    explicit RecordingSink(std::span<const ninfer::TokenId> targets) : targets_(targets) {}

    void consume(const std::uint16_t* bf16_logits, std::uint32_t columns, std::uint32_t row_stride,
                 std::uint32_t valid_rows) override {
        blocks.push_back(columns);
        row_strides.push_back(row_stride);
        valid_row_counts.push_back(valid_rows);
        for (std::uint32_t column = 0; column < columns; ++column) {
            const std::uint16_t* row = bf16_logits + static_cast<std::size_t>(column) * row_stride;
            double maximum           = -INFINITY;
            for (std::uint32_t i = 0; i < valid_rows; ++i) {
                maximum = std::max(maximum, value(row[i]));
            }
            double sum = 0.0;
            for (std::uint32_t i = 0; i < valid_rows; ++i) {
                sum += std::exp(value(row[i]) - maximum);
            }
            const ninfer::TokenId target = targets_[recomputed.size()];
            recomputed.push_back(value(row[target]) - maximum - std::log(sum));
        }
    }

    std::vector<std::uint32_t> blocks;
    std::vector<std::uint32_t> row_strides;
    std::vector<std::uint32_t> valid_row_counts;
    std::vector<double> recomputed;

private:
    static double value(std::uint16_t bits) {
        return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
    }

    std::span<const ninfer::TokenId> targets_;
};

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_QWEN3_6_27B_WEIGHTS is not set\n";
        return 77;
    }

    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.purpose       = ninfer::EnginePurpose::CausalScoring;
    options.max_context   = 2048;
    options.kv_cache      = ninfer::KvCacheStorage::Fp8E4M3Row256;
    ninfer::Engine engine(options);
    const auto& effective = engine.options();
    if (effective.max_concurrency != 1 || effective.prefill_chunk != 1024 ||
        effective.kv_capacity.mode != ninfer::KvCapacityMode::Explicit ||
        effective.kv_capacity.explicit_tokens != effective.max_context ||
        effective.context_cache.enabled ||
        effective.speculative.backend != ninfer::SpeculativeBackend::None ||
        effective.kv_cache != ninfer::KvCacheStorage::Fp8E4M3Row256) {
        std::cerr << "causal scoring options were not normalized correctly\n";
        return 1;
    }

    std::string text;
    const std::string paragraph =
        "NInfer scores each target token from the preceding hidden state. "
        "Every evaluation window owns fresh state and a fresh KV address space.\n";
    std::vector<ninfer::TokenId> tokens;
    while (tokens.size() < 1537) {
        text += paragraph;
        tokens = engine.tokenize_text(text);
    }
    tokens.resize(1537);

    const std::vector<float> all      = engine.score_tokens(tokens, 1);
    const std::vector<float> suffix   = engine.score_tokens(tokens, 513);
    const std::vector<float> repeated = engine.score_tokens(tokens, 513);
    if (all.size() != 1536 || suffix.size() != 1024 || repeated.size() != suffix.size()) {
        std::cerr << "causal scoring returned an invalid result shape\n";
        return 1;
    }
    float maximum_overlap_error = 0.0F;
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        if (!std::isfinite(all[i + 512]) || !std::isfinite(suffix[i])) {
            std::cerr << "causal scoring returned a non-finite logprob\n";
            return 1;
        }
        maximum_overlap_error = std::max(maximum_overlap_error, std::abs(all[i + 512] - suffix[i]));
        if (suffix[i] != repeated[i]) {
            std::cerr << "a repeated score window inherited prior State/KV\n";
            return 1;
        }
    }
    if (maximum_overlap_error > 0.25F) {
        std::cerr << "overlapping target suffix changed by " << maximum_overlap_error << '\n';
        return 1;
    }
    // Exported logits: 1536 scored positions span a full 1024-column tile and a partial one.
    RecordingSink sink(std::span<const ninfer::TokenId>(tokens).subspan(1));
    const std::vector<float> with_sink = engine.score_tokens(tokens, 1, &sink);
    if (with_sink != all) {
        std::cerr << "a logits sink changed the returned logprobs\n";
        return 1;
    }
    if (sink.blocks != std::vector<std::uint32_t>{1024, 512} ||
        std::any_of(sink.row_strides.begin(), sink.row_strides.end(),
                    [](std::uint32_t stride) { return stride != 248320; }) ||
        std::any_of(sink.valid_row_counts.begin(), sink.valid_row_counts.end(),
                    [](std::uint32_t valid) { return valid != 248077; }) ||
        sink.recomputed.size() != all.size()) {
        std::cerr << "the logits sink received an unexpected block shape\n";
        return 1;
    }
    double maximum_sink_error = 0.0;
    for (std::size_t i = 0; i < all.size(); ++i) {
        maximum_sink_error = std::max(maximum_sink_error,
                                      std::abs(sink.recomputed[i] - static_cast<double>(all[i])));
    }
    // Both sides read the same BF16 logits; only the FP32 reduction order differs.
    if (maximum_sink_error > 2e-3) {
        std::cerr << "exported logits disagree with the target logprobs by " << maximum_sink_error
                  << '\n';
        return 1;
    }
    std::cout << "OK causal_score_real max_overlap_error=" << maximum_overlap_error
              << " max_sink_error=" << maximum_sink_error << '\n';
    return 0;
}
