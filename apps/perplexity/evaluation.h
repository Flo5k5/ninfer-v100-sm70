#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::perplexity {

struct WindowPlan {
    std::size_t input_begin    = 0;
    std::size_t input_end      = 0;
    std::size_t target_begin   = 0;
    std::size_t target_end     = 0;
    std::uint32_t first_target = 0;
};

[[nodiscard]] std::vector<WindowPlan> plan_windows(std::size_t tokens, std::uint32_t context,
                                                   std::uint32_t stride);

// The llama.cpp `llama-perplexity` KL-divergence protocol: floor(tokens/context) non-overlapping
// windows of exactly `context` tokens, each scoring only the targets after its first context/2
// predictors, i.e. local targets [context/2+1, context). Tokens after the last full window are not
// evaluated. Requires at least two windows, as llama-perplexity does.
[[nodiscard]] std::vector<WindowPlan> plan_kld_chunks(std::size_t tokens, std::uint32_t context);

struct ScoreAggregate {
    std::uint64_t scored_tokens = 0;
    double total_nll            = 0.0;

    void add(std::span<const float> logprobs);
    void add(const ScoreAggregate& other) noexcept;
    [[nodiscard]] double mean_nll() const;
    [[nodiscard]] double ppl() const;
};

} // namespace ninfer::perplexity
