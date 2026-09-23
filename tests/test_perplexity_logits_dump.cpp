#include "logits_dump.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int require(bool condition, const char* label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

float word_float(const std::uint16_t* words) {
    float value = 0.0F;
    std::memcpy(&value, words, sizeof(value));
    return value;
}

std::uint16_t float_to_bf16(float value) {
    // Test inputs are exactly representable in BF16.
    return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16U);
}

// Independent FP64 log-softmax oracle for the encoded code points.
int check_encoding_against_oracle() {
    constexpr std::uint32_t vocab = 1001;
    constexpr std::uint32_t valid = 990;
    std::mt19937 generator(7);
    std::normal_distribution<float> distribution(0.0F, 6.0F);
    std::vector<float> logits(vocab);
    for (float& logit : logits) { logit = distribution(generator); }
    logits[17] = 30.0F;

    std::vector<std::uint16_t> words(ninfer::perplexity::kld_base_words_per_position(vocab),
                                     0xffff);
    ninfer::perplexity::encode_kld_base_position(logits.data(), valid, vocab, words.data());

    double maximum = logits[0];
    for (std::uint32_t i = 0; i < valid; ++i) { maximum = std::max<double>(maximum, logits[i]); }
    double sum = 0.0;
    for (std::uint32_t i = 0; i < valid; ++i) { sum += std::exp(logits[i] - maximum); }
    const double log_sum = std::log(sum);

    const float scale        = word_float(words.data());
    const float min_log_prob = word_float(words.data() + 2);
    const std::uint16_t* q   = words.data() + 4;
    int failures             = 0;
    failures += require(words.size() == 1001 + 1 + 4, "odd vocabularies carry one pad word");
    failures += require(std::abs(scale - 16.0 / 65535.0) < 1e-9, "scale spans the 16-nat window");
    failures += require(std::abs(min_log_prob - (-16.0 - log_sum)) < 1e-4,
                        "minimum code is max-16 in log-probability space");
    for (std::uint32_t i = 0; i < valid; ++i) {
        const double exact = logits[i] - maximum - log_sum;
        if (logits[i] > maximum - 16.0) {
            const double decoded = min_log_prob + static_cast<double>(scale) * q[i];
            if (std::abs(decoded - exact) > 0.5 * scale + 1e-4) {
                failures += require(false, "decoded log-probability is within half a code step");
                break;
            }
        } else if (q[i] != 0) {
            failures += require(false, "log-probabilities below the window clamp to code 0");
            break;
        }
    }
    failures += require(q[17] == 65535, "the maximum logit takes the top code");
    failures +=
        require(std::all_of(q + valid, q + vocab + 1, [](std::uint16_t v) { return v == 0; }),
                "padding rows and the pad word encode as 0");
    return failures;
}

int check_writer_round_trip(const std::filesystem::path& directory) {
    constexpr std::uint32_t context = 8;
    constexpr std::uint32_t chunks  = 2;
    constexpr std::uint32_t rows    = 6;
    constexpr std::uint32_t valid   = 5;
    std::vector<ninfer::TokenId> tokens(context * chunks);
    for (std::size_t i = 0; i < tokens.size(); ++i) { tokens[i] = static_cast<int>(100 + i); }

    const std::vector<float> row_values{1.0F, 2.5F, -3.0F, 0.5F, 2.5F, 99.0F};
    std::vector<std::uint16_t> block;
    for (int column = 0; column < 4; ++column) {
        for (const float value : row_values) { block.push_back(float_to_bf16(value + column)); }
    }

    int failures                     = 0;
    const std::filesystem::path path = directory / "dump.kld";
    {
        ninfer::perplexity::KldBaseWriter writer(path, context, chunks, tokens, rows);
        failures += require(writer.expected_positions() == 6, "writer expects context-1-context/2 "
                                                              "positions per chunk");
        writer.consume(block.data(), 2, rows, valid);
        writer.consume(block.data(), 4, rows, valid);
        const std::uint64_t bytes = writer.finish();
        failures += require(bytes == ninfer::perplexity::kld_base_header_bytes(context, chunks) +
                                         6 * ninfer::perplexity::kld_base_words_per_position(rows) *
                                             sizeof(std::uint16_t),
                            "file size matches the llama.cpp layout");
    }
    const auto header = ninfer::perplexity::read_kld_base_header(path);
    failures += require(header.context == context && header.vocab == rows &&
                            header.chunks == chunks && header.tokens == tokens,
                        "header round-trips");

    std::vector<float> column_one(row_values.size());
    std::transform(row_values.begin(), row_values.end(), column_one.begin(),
                   [](float value) { return value + 1.0F; });
    std::vector<std::uint16_t> expected(ninfer::perplexity::kld_base_words_per_position(rows));
    ninfer::perplexity::encode_kld_base_position(column_one.data(), valid, rows, expected.data());
    std::ifstream input(path, std::ios::binary);
    input.seekg(
        static_cast<std::streamoff>(ninfer::perplexity::kld_base_header_bytes(context, chunks) +
                                    expected.size() * sizeof(std::uint16_t)));
    std::vector<std::uint16_t> stored(expected.size());
    input.read(reinterpret_cast<char*>(stored.data()),
               static_cast<std::streamsize>(stored.size() * sizeof(std::uint16_t)));
    failures += require(static_cast<bool>(input) && stored == expected,
                        "positions are stored in target order from BF16 rows");
    failures += require(stored[4 + valid] == 0, "the padding row is excluded from the softmax");
    return failures;
}

int check_writer_rejections(const std::filesystem::path& directory) {
    int failures = 0;
    std::vector<ninfer::TokenId> tokens(16, 1);
    std::vector<std::uint16_t> block(6 * 6, float_to_bf16(1.0F));
    const std::filesystem::path path = directory / "rejected.kld";
    try {
        ninfer::perplexity::KldBaseWriter writer(path, 8, 2, tokens, 7);
        writer.consume(block.data(), 2, 6, 5);
        failures += require(false, "writer rejects a row width that differs from the reference");
    } catch (const std::runtime_error&) {}
    try {
        ninfer::perplexity::KldBaseWriter writer(path, 8, 2, tokens, std::nullopt);
        writer.consume(block.data(), 2, 6, 5);
        (void)writer.finish();
        failures += require(false, "writer rejects an incomplete dump");
    } catch (const std::logic_error&) {}
    failures += require(!std::filesystem::exists(path) &&
                            !std::filesystem::exists(directory / "rejected.kld.partial"),
                        "an unfinished dump leaves no file behind");
    return failures;
}

} // namespace

int main() {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() /
        ("ninfer_logits_dump_test_" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(directory);
    int failures = 0;
    try {
        failures += check_encoding_against_oracle();
        failures += check_writer_round_trip(directory);
        failures += check_writer_rejections(directory);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        ++failures;
    }
    std::filesystem::remove_all(directory);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " perplexity_logits_dump\n";
    return failures == 0 ? 0 : 1;
}
