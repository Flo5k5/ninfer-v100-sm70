#pragma once

#include "logits_dump.h"
#include "options.h"

#include <filesystem>
#include <optional>

namespace ninfer::perplexity {

// Creates `directory` when it is missing and requires an existing one to be an empty directory.
// Returns its absolute, normalized path.
[[nodiscard]] std::filesystem::path
prepare_report_directory(const std::filesystem::path& directory);

// File-system work that does not need the model, done before the engine is constructed. An
// explicit --output directory is prepared first, so that a --logits-out file inside it can be
// created once the corpus is tokenized; then the dump destination is checked and the
// --logits-reference header is read and matched against --context.
struct Preflight {
    std::optional<std::filesystem::path> report_directory; // set when --output is given
    std::optional<KldBaseHeader> reference;                // set with --logits-reference
};

[[nodiscard]] Preflight run_preflight(const Options& options);

} // namespace ninfer::perplexity
