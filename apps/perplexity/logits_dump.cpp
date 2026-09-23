#include "logits_dump.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

namespace ninfer::perplexity {
namespace {

constexpr char kMagic[8]       = {'_', 'l', 'o', 'g', 'i', 't', 's', '_'};
constexpr float kLogitWindow   = 16.0F;
constexpr float kCodeMaximum   = 65535.0F;
constexpr unsigned kMaxWorkers = 32;

// llama.cpp's nearest_int(): round-to-nearest-even through the FP32 mantissa.
int nearest_int(float value) noexcept {
    const float shifted = value + 12582912.0F;
    std::int32_t bits   = 0;
    std::memcpy(&bits, &shifted, sizeof(bits));
    return (bits & 0x007fffff) - 0x00400000;
}

float bf16_to_float(std::uint16_t bits) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

template <class Value>
void write_value(std::ofstream& output, const Value& value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <class Value>
Value read_value(std::ifstream& input, const std::filesystem::path& path, const char* label) {
    Value value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) {
        throw std::runtime_error(std::string("cannot read ") + label + " from " + path.string());
    }
    return value;
}

} // namespace

std::size_t kld_base_words_per_position(std::uint32_t vocab) noexcept {
    return 2 * ((static_cast<std::size_t>(vocab) + 1) / 2) + 4;
}

std::uint64_t kld_base_header_bytes(std::uint32_t context, std::uint32_t chunks) {
    return sizeof(kMagic) + 3 * sizeof(std::uint32_t) +
           static_cast<std::uint64_t>(context) * chunks * sizeof(TokenId);
}

void encode_kld_base_position(const float* logits, std::uint32_t valid, std::uint32_t vocab,
                              std::uint16_t* out) {
    if (valid == 0 || valid > vocab) {
        throw std::invalid_argument("KLD base row must have 1<=valid<=vocab logits");
    }
    float max_logit = logits[0];
    float min_logit = logits[0];
    for (std::uint32_t i = 1; i < valid; ++i) {
        max_logit = std::max(max_logit, logits[i]);
        min_logit = std::min(min_logit, logits[i]);
    }
    if (!std::isfinite(max_logit) || !std::isfinite(min_logit)) {
        throw std::runtime_error("KLD base row contains a non-finite logit");
    }
    min_logit      = std::max(min_logit, max_logit - kLogitWindow);
    double sum_exp = 0.0;
    for (std::uint32_t i = 0; i < valid; ++i) { sum_exp += std::exp(logits[i] - max_logit); }
    const auto log_sum_exp   = static_cast<float>(std::log(sum_exp));
    const float min_log_prob = min_logit - max_logit - log_sum_exp;
    const float scale        = (max_logit - min_logit) / kCodeMaximum;
    std::memcpy(out, &scale, sizeof(scale));
    std::memcpy(out + 2, &min_log_prob, sizeof(min_log_prob));
    std::uint16_t* codes = out + 4;
    std::fill(codes, codes + (kld_base_words_per_position(vocab) - 4), std::uint16_t{0});
    if (scale == 0.0F) { return; }
    const float inverse_scale = 1.0F / scale;
    for (std::uint32_t i = 0; i < valid; ++i) {
        if (logits[i] > min_logit) {
            codes[i] =
                static_cast<std::uint16_t>(nearest_int(inverse_scale * (logits[i] - min_logit)));
        }
    }
}

KldBaseHeader read_kld_base_header(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("cannot open KLD base: " + path.string()); }
    char magic[sizeof(kMagic)] = {};
    input.read(magic, sizeof(magic));
    if (!input || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("not a llama-perplexity KLD base file: " + path.string());
    }
    KldBaseHeader header;
    header.context   = read_value<std::uint32_t>(input, path, "context");
    const auto vocab = read_value<std::int32_t>(input, path, "vocabulary size");
    const auto count = read_value<std::int32_t>(input, path, "chunk count");
    if (header.context < 4 || vocab <= 0 || count <= 0) {
        throw std::runtime_error("KLD base header is invalid: " + path.string());
    }
    header.vocab  = static_cast<std::uint32_t>(vocab);
    header.chunks = static_cast<std::uint32_t>(count);
    header.tokens.resize(static_cast<std::size_t>(header.context) * header.chunks);
    input.read(reinterpret_cast<char*>(header.tokens.data()),
               static_cast<std::streamsize>(header.tokens.size() * sizeof(TokenId)));
    if (!input) { throw std::runtime_error("KLD base token block is truncated: " + path.string()); }
    return header;
}

void check_reference_context(const KldBaseHeader& reference, std::uint32_t context) {
    if (reference.context != context) {
        throw std::runtime_error("reference dump context " + std::to_string(reference.context) +
                                 " differs from --context " + std::to_string(context));
    }
}

void check_logits_destination(const std::filesystem::path& path) {
    if (std::filesystem::exists(path)) {
        throw std::runtime_error("logits output already exists: " + path.string());
    }
    std::filesystem::path partial = path;
    partial += ".partial";
    if (std::filesystem::exists(partial)) {
        throw std::runtime_error("a partial logits dump from an interrupted run exists: " +
                                 partial.string() + "; delete it or choose another --logits-out");
    }
    const std::filesystem::path directory = std::filesystem::absolute(path).parent_path();
    if (!std::filesystem::is_directory(directory)) {
        throw std::runtime_error("logits output directory does not exist: " + directory.string());
    }
}

void check_reference_tokens(const KldBaseHeader& reference, std::uint32_t context,
                            std::size_t chunks, std::span<const TokenId> tokens, bool prefix) {
    check_reference_context(reference, context);
    const std::size_t evaluated = chunks * context;
    if (tokens.size() < evaluated) {
        throw std::logic_error("reference check received fewer tokens than the window plan");
    }
    const std::size_t compared = std::min(evaluated, reference.tokens.size());
    for (std::size_t i = 0; i < compared; ++i) {
        if (tokens[i] != reference.tokens[i]) {
            throw std::runtime_error("tokenization differs from the reference dump at token " +
                                     std::to_string(i) + " (ours " + std::to_string(tokens[i]) +
                                     ", reference " + std::to_string(reference.tokens[i]) + ")");
        }
    }
    if (prefix ? reference.chunks < chunks : reference.chunks != chunks) {
        throw std::runtime_error(
            "reference dump has " + std::to_string(reference.chunks) + " chunks, this run " +
            (prefix ? "scores the first " : "yields ") + std::to_string(chunks) +
            (prefix ? "" : " (use --chunks to compare with a prefix of a longer reference)"));
    }
}

KldBaseWriter::KldBaseWriter(std::filesystem::path path, std::uint32_t context,
                             std::uint32_t chunks, std::span<const TokenId> tokens,
                             std::optional<std::uint32_t> expected_vocab)
    : path_(std::move(path)), context_(context), chunks_(chunks),
      tokens_(tokens.begin(), tokens.end()), expected_vocab_(expected_vocab) {
    if (context_ < 4 || chunks_ == 0) {
        throw std::invalid_argument("KLD base requires context>=4 and at least one chunk");
    }
    if (tokens_.size() != static_cast<std::size_t>(context_) * chunks_) {
        throw std::invalid_argument("KLD base token block must hold chunks*context tokens");
    }
    expected_positions_ = static_cast<std::uint64_t>(chunks_) * (context_ - 1 - context_ / 2);
    partial_path_       = path_;
    partial_path_ += ".partial";
    check_logits_destination(path_);
    if (expected_vocab_) { require_space(*expected_vocab_); }
    // Created now so an unwritable destination fails before scoring; the header is written by
    // the first consume(), when the logits row width is known.
    output_.open(partial_path_, std::ios::binary | std::ios::trunc);
    if (!output_) { throw std::runtime_error("cannot create " + partial_path_.string()); }
}

void KldBaseWriter::require_space(std::uint32_t vocab) const {
    const std::uint64_t bytes =
        kld_base_header_bytes(context_, chunks_) +
        expected_positions_ * kld_base_words_per_position(vocab) * sizeof(std::uint16_t);
    const std::filesystem::path directory   = std::filesystem::absolute(path_).parent_path();
    const std::filesystem::space_info space = std::filesystem::space(directory);
    if (space.available < bytes) {
        throw std::runtime_error("logits dump needs " + std::to_string(bytes) + " bytes but " +
                                 directory.string() + " has " + std::to_string(space.available));
    }
}

KldBaseWriter::~KldBaseWriter() {
    if (finished_) { return; }
    if (output_.is_open()) { output_.close(); }
    std::error_code ignored;
    std::filesystem::remove(partial_path_, ignored);
}

void KldBaseWriter::open(std::uint32_t vocab, std::uint32_t valid_rows) {
    if (expected_vocab_ && *expected_vocab_ != vocab) {
        throw std::runtime_error("logits row width " + std::to_string(vocab) +
                                 " differs from the reference dump vocabulary " +
                                 std::to_string(*expected_vocab_));
    }
    vocab_      = vocab;
    valid_rows_ = valid_rows;
    require_space(vocab_);
    output_.write(kMagic, sizeof(kMagic));
    write_value(output_, context_);
    write_value(output_, static_cast<std::int32_t>(vocab_));
    write_value(output_, static_cast<std::int32_t>(chunks_));
    output_.write(reinterpret_cast<const char*>(tokens_.data()),
                  static_cast<std::streamsize>(tokens_.size() * sizeof(TokenId)));
    if (!output_) { throw std::runtime_error("cannot write " + partial_path_.string()); }
}

void KldBaseWriter::consume(const std::uint16_t* bf16_logits, std::uint32_t columns,
                            std::uint32_t row_stride, std::uint32_t valid_rows) {
    if (finished_) { throw std::logic_error("KLD base writer is already finished"); }
    if (columns == 0 || row_stride == 0 || valid_rows == 0 || valid_rows > row_stride) {
        throw std::invalid_argument("logits block has an invalid shape");
    }
    if (vocab_ == 0) {
        open(row_stride, valid_rows);
    } else if (row_stride != vocab_ || valid_rows != valid_rows_) {
        throw std::logic_error("logits row shape changed during a dump");
    }
    if (written_positions_ + columns > expected_positions_) {
        throw std::logic_error("logits dump received more positions than its chunk plan");
    }

    const std::size_t words = kld_base_words_per_position(vocab_);
    encoded_.resize(static_cast<std::size_t>(columns) * words);
    const unsigned workers =
        std::min({std::max(1U, std::thread::hardware_concurrency()), kMaxWorkers, columns});
    std::vector<std::exception_ptr> errors(workers);
    const auto encode_range = [&](unsigned worker) {
        try {
            std::vector<float> row(row_stride);
            for (std::uint32_t column = worker; column < columns; column += workers) {
                const std::uint16_t* source =
                    bf16_logits + static_cast<std::size_t>(column) * row_stride;
                for (std::uint32_t i = 0; i < valid_rows; ++i) {
                    row[i] = bf16_to_float(source[i]);
                }
                encode_kld_base_position(row.data(), valid_rows, vocab_,
                                         encoded_.data() +
                                             static_cast<std::size_t>(column) * words);
            }
        } catch (...) { errors[worker] = std::current_exception(); }
    };
    {
        // jthread joins on destruction: a failed thread creation cannot std::terminate.
        std::vector<std::jthread> threads;
        threads.reserve(workers - 1);
        for (unsigned worker = 1; worker < workers; ++worker) {
            threads.emplace_back(encode_range, worker);
        }
        encode_range(0);
    }
    for (const std::exception_ptr& error : errors) {
        if (error) { std::rethrow_exception(error); }
    }

    output_.write(reinterpret_cast<const char*>(encoded_.data()),
                  static_cast<std::streamsize>(encoded_.size() * sizeof(std::uint16_t)));
    if (!output_) { throw std::runtime_error("cannot write " + partial_path_.string()); }
    written_positions_ += columns;
}

std::uint64_t KldBaseWriter::finish() {
    if (finished_) { throw std::logic_error("KLD base writer is already finished"); }
    if (vocab_ == 0 || written_positions_ != expected_positions_) {
        throw std::logic_error("logits dump holds " + std::to_string(written_positions_) + " of " +
                               std::to_string(expected_positions_) + " scored positions");
    }
    output_.flush();
    output_.close();
    if (!output_) { throw std::runtime_error("cannot finish " + partial_path_.string()); }
    std::filesystem::rename(partial_path_, path_);
    finished_ = true;
    return std::filesystem::file_size(path_);
}

} // namespace ninfer::perplexity
