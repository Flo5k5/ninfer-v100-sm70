#pragma once

#include "ninfer/types.h"
#include "product/logging/logging.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace ninfer::perplexity {

struct Options {
    bool help_requested = false;
    std::filesystem::path artifact;
    std::optional<std::filesystem::path> corpus;
    std::optional<std::filesystem::path> text;
    std::optional<std::filesystem::path> output;
    std::optional<std::filesystem::path> logits_out;
    std::optional<std::filesystem::path> logits_reference;
    std::optional<std::uint32_t> chunks;
    std::uint32_t context       = 4096;
    std::uint32_t stride        = 2048;
    // Prefill chunk of the scoring engine, and the forward width over each window's scored
    // targets (0: prefill chunks; see CausalScoreOptions).
    std::uint32_t prefill_chunk = 1024;
    std::uint32_t scored_chunk  = 0;
    int device                  = 0;
    KvCacheStorage kv           = KvCacheStorage::Fp8E4M3Row256;
    bool quick                  = false;
    product::LogLevel log_level = product::LogLevel::Info;
};

// Throws std::invalid_argument for an invalid command line. With --logits-out the stride is
// context/2 (the llama.cpp KLD window plan): an omitted --stride takes that value.
[[nodiscard]] Options parse_options(int argc, char** argv);
[[nodiscard]] std::string usage_text();
[[nodiscard]] std::string kv_dtype_name(KvCacheStorage value);

} // namespace ninfer::perplexity
