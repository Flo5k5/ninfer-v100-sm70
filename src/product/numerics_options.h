#pragma once

#include "ninfer/types.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

// Command-line spellings of EngineOptions::text_residual (--text-residual).
[[nodiscard]] inline TextResidualStorage parse_text_residual(std::string_view value) {
    if (value == "bf16") { return TextResidualStorage::BFloat16; }
    if (value == "fp32") { return TextResidualStorage::Float32; }
    throw std::invalid_argument("invalid text-residual (expected bf16 or fp32): " +
                                std::string(value));
}

[[nodiscard]] inline const char* text_residual_name(TextResidualStorage storage) noexcept {
    switch (storage) {
    case TextResidualStorage::BFloat16:
        return "bf16";
    case TextResidualStorage::Float32:
        return "fp32";
    }
    return "unknown";
}

// Command-line spellings of EngineOptions::prefill_attention (--prefill-attention).
[[nodiscard]] inline PrefillAttentionKernel parse_prefill_attention(std::string_view value) {
    if (value == "auto") { return PrefillAttentionKernel::Automatic; }
    if (value == "splitd") { return PrefillAttentionKernel::SplitD; }
    if (value == "flash") { return PrefillAttentionKernel::Flash; }
    if (value == "reference") { return PrefillAttentionKernel::Reference; }
    throw std::invalid_argument(
        "invalid prefill-attention (expected auto, splitd, flash or reference): " +
        std::string(value));
}

[[nodiscard]] inline const char* prefill_attention_name(PrefillAttentionKernel kernel) noexcept {
    switch (kernel) {
    case PrefillAttentionKernel::Automatic:
        return "auto";
    case PrefillAttentionKernel::SplitD:
        return "splitd";
    case PrefillAttentionKernel::Flash:
        return "flash";
    case PrefillAttentionKernel::Reference:
        return "reference";
    }
    return "unknown";
}

} // namespace ninfer::product
