#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::perplexity {

// Log-probability dump in the file layout of `llama-perplexity --kl-divergence-base`, so one reader
// handles NInfer and llama.cpp runs and llama.cpp can consume an NInfer dump as its base. All
// fields are little-endian:
//
//   char[8] "_logits_" | u32 context | i32 vocab | i32 chunks | i32 tokens[chunks*context]
//   per scored position, in order:
//   f32 scale | f32 min_log_prob | u16 q[vocab] | u16 pad (odd vocab only)
//
// with log p(i) = min_log_prob + scale*q[i]. Each position stores its log-softmax over the row with
// a per-position affine uint16 code spanning [max_logit-16, max_logit]; smaller logits encode as 0.
// Positions follow plan_kld_chunks(): context-1-context/2 per chunk.
[[nodiscard]] std::size_t kld_base_words_per_position(std::uint32_t vocab) noexcept;
[[nodiscard]] std::uint64_t kld_base_header_bytes(std::uint32_t context, std::uint32_t chunks);

// Encodes one logits row exactly like llama.cpp's perplexity log_softmax(): logits[0,valid) define
// the distribution and entries [valid,vocab) (output-head padding) encode as 0. `out` receives
// kld_base_words_per_position(vocab) words.
void encode_kld_base_position(const float* logits, std::uint32_t valid, std::uint32_t vocab,
                              std::uint16_t* out);

struct KldBaseHeader {
    std::uint32_t context = 0;
    std::uint32_t vocab   = 0;
    std::uint32_t chunks  = 0;
    std::vector<TokenId> tokens;
};

[[nodiscard]] KldBaseHeader read_kld_base_header(const std::filesystem::path& path);

// Rejects a reference dump written with another context.
void check_reference_context(const KldBaseHeader& reference, std::uint32_t context);

// Checks a dump destination: neither the dump nor a partial file of an interrupted run may exist,
// and its directory must. KldBaseWriter runs the same check when it creates the partial file.
void check_logits_destination(const std::filesystem::path& path);

// Rejects a run whose evaluated tokens differ from a reference dump, before any scoring time is
// spent. `chunks` windows of `context` tokens are evaluated from the start of `tokens`. Without
// `prefix` the reference must hold exactly `chunks` windows; with it (a --chunks run) it may hold
// more, and only the first chunks*context tokens are compared, as `kld.py compare --chunks` does.
void check_reference_tokens(const KldBaseHeader& reference, std::uint32_t context,
                            std::size_t chunks, std::span<const TokenId> tokens, bool prefix);

// ScoreLogitsSink that writes a dump for the windows of plan_kld_chunks(). The file is written as
// `<path>.partial` and renamed by finish(); an unfinished writer deletes its partial file. The
// constructor creates the partial file, so an unwritable destination fails before scoring; it
// refuses an existing destination or a stale partial file left by an interrupted run.
class KldBaseWriter final : public ScoreLogitsSink {
public:
    // `tokens` is the evaluated stream prefix of exactly chunks*context tokens. A set
    // `expected_vocab` rejects a logits row width that differs from a reference dump.
    KldBaseWriter(std::filesystem::path path, std::uint32_t context, std::uint32_t chunks,
                  std::span<const TokenId> tokens, std::optional<std::uint32_t> expected_vocab);
    ~KldBaseWriter() override;

    KldBaseWriter(const KldBaseWriter&)            = delete;
    KldBaseWriter& operator=(const KldBaseWriter&) = delete;

    void consume(const std::uint16_t* bf16_logits, std::uint32_t columns, std::uint32_t row_stride,
                 std::uint32_t valid_rows) override;

    // Requires every scored position to be written; returns the final file size in bytes.
    std::uint64_t finish();

    [[nodiscard]] std::uint32_t vocab() const noexcept { return vocab_; }

    [[nodiscard]] std::uint32_t valid_rows() const noexcept { return valid_rows_; }

    [[nodiscard]] std::uint64_t expected_positions() const noexcept { return expected_positions_; }

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    void open(std::uint32_t vocab, std::uint32_t valid_rows);
    void require_space(std::uint32_t vocab) const;

    std::filesystem::path path_;
    std::filesystem::path partial_path_;
    std::uint32_t context_ = 0;
    std::uint32_t chunks_  = 0;
    std::vector<TokenId> tokens_;
    std::optional<std::uint32_t> expected_vocab_;
    std::uint64_t expected_positions_ = 0;
    std::uint64_t written_positions_  = 0;
    std::uint32_t vocab_              = 0;
    std::uint32_t valid_rows_         = 0;
    std::ofstream output_;
    std::vector<std::uint16_t> encoded_;
    bool finished_ = false;
};

} // namespace ninfer::perplexity
