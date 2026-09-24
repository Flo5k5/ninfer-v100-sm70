// Public-contract qualification for apply_token_bitmask().
//
// The masking transform is exact, so every case compares the complete logits tensor bit for bit
// with an independent CPU application of the contract. The composition cases then check the
// property masks exist for: argmax and sample() never select an excluded token or a padding row
// beyond the token domain.
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/token_bitmask.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::uint16_t kNegativeInfinity = 0xff80u;
constexpr int kRealRows                   = 248320;
constexpr int kRealDomain                 = 248077;

struct Geometry {
    int physical_rows = 0;
    int token_domain  = 0;
    int width         = 0;
    int batch         = 0;
    int mask_columns  = 0; // C of the [words,C,R] buffer
    int mask_rows     = 0; // R of the [words,C,R] buffer

    [[nodiscard]] int words() const { return ops::token_bitmask_words(token_domain); }

    [[nodiscard]] std::size_t logits_count() const {
        return static_cast<std::size_t>(physical_rows) * width * batch;
    }

    [[nodiscard]] std::size_t mask_count() const {
        return static_cast<std::size_t>(words()) * mask_columns * mask_rows;
    }
};

std::vector<std::uint16_t> random_logits(const Geometry& g, std::uint32_t seed) {
    std::vector<float> values(g.logits_count());
    fill_uniform(values, seed, -12.0f, 12.0f);
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    // Signed zeros and an existing minus infinity must survive where the mask allows them.
    for (std::size_t i = 0; i < bits.size(); i += 997) { bits[i] = 0x8000u; }
    for (std::size_t i = 13; i < bits.size(); i += 1009) { bits[i] = 0x0000u; }
    for (std::size_t i = 29; i < bits.size(); i += 4099) { bits[i] = kNegativeInfinity; }
    return bits;
}

// Random words with the given share of set bits. Bits past the token domain in the last word,
// and the unused columns and rows of the buffer, are filled too: the Op must ignore them.
std::vector<std::int32_t> random_mask(const Geometry& g, double density, std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::bernoulli_distribution bit(density);
    std::vector<std::int32_t> mask(g.mask_count());
    for (auto& word : mask) {
        std::uint32_t value = 0;
        for (int b = 0; b < 32; ++b) {
            if (bit(generator)) { value |= 1u << b; }
        }
        word = static_cast<std::int32_t>(value);
    }
    const int tail = g.token_domain % 32;
    if (tail != 0) {
        for (int row = 0; row < g.mask_rows; ++row) {
            for (int column = 0; column < g.mask_columns; ++column) {
                const std::size_t last =
                    (static_cast<std::size_t>(row) * g.mask_columns + column) * g.words() +
                    (g.words() - 1);
                mask[last] = static_cast<std::int32_t>(static_cast<std::uint32_t>(mask[last]) |
                                                       (~0u << tail));
            }
        }
    }
    return mask;
}

bool allowed(const std::vector<std::int32_t>& mask, const Geometry& g, int row, int column,
             int token) {
    if (token >= g.token_domain) { return false; }
    const std::size_t word =
        (static_cast<std::size_t>(row) * g.mask_columns + column) * g.words() + (token >> 5);
    return ((static_cast<std::uint32_t>(mask[word]) >> (token & 31)) & 1u) != 0u;
}

std::vector<std::uint16_t> oracle(const Geometry& g, std::vector<std::uint16_t> logits,
                                  const std::vector<std::int32_t>& mask,
                                  const std::vector<std::int32_t>& mask_columns) {
    for (int row = 0; row < g.batch; ++row) {
        const int active = std::clamp(mask_columns[static_cast<std::size_t>(row)], 0, g.width);
        for (int column = 0; column < active; ++column) {
            const std::size_t base =
                (static_cast<std::size_t>(row) * g.width + column) * g.physical_rows;
            for (int token = 0; token < g.physical_rows; ++token) {
                if (!allowed(mask, g, row, column, token)) {
                    logits[base + static_cast<std::size_t>(token)] = kNegativeInfinity;
                }
            }
        }
    }
    return logits;
}

// `misalign` offsets the logits by one BF16 element so that no column starts on a 16-byte
// boundary; the Op must still produce the exact result.
int exact_case(const std::string& label, const Geometry& g, double density,
               const std::vector<std::int32_t>& mask_columns, std::uint32_t seed,
               bool misalign = false) {
    const std::vector<std::uint16_t> input    = random_logits(g, seed);
    const std::vector<std::int32_t> mask      = random_mask(g, density, seed * 7u + 1u);
    const std::vector<std::uint16_t> expected = oracle(g, input, mask, mask_columns);
    const std::size_t offset                  = misalign ? 1 : 0;
    const std::size_t payload                 = (input.size() + offset) * sizeof(std::uint16_t);
    GuardedDeviceBuffer d_logits(payload);
    const std::uint16_t lead = 0x1234u;
    if (misalign) { d_logits.copy_from_host(&lead, sizeof(lead)); }
    d_logits.copy_from_host(input.data(), input.size() * sizeof(std::uint16_t),
                            offset * sizeof(std::uint16_t));
    DeviceBuffer d_mask    = to_device(mask);
    DeviceBuffer d_columns = to_device(mask_columns);

    auto* logits_data = static_cast<std::uint16_t*>(d_logits.data()) + offset;
    Tensor logits(logits_data, DType::BF16, {g.physical_rows, g.width, g.batch});
    Tensor bitmask(d_mask.p, DType::I32, {g.words(), g.mask_columns, g.mask_rows});
    Tensor columns(d_columns.p, DType::I32, {static_cast<std::int32_t>(mask_columns.size())});
    ops::apply_token_bitmask(logits, bitmask, columns, g.token_domain, nullptr);
    cuda_synchronize();

    int failures = verify_exact(label.c_str(),
                                from_device<std::uint16_t>(logits_data, expected.size()), expected);
    failures += verify_exact((label + " bitmask read-only").c_str(),
                             from_device<std::int32_t>(d_mask, mask.size()), mask);
    failures +=
        verify_exact((label + " counts read-only").c_str(),
                     from_device<std::int32_t>(d_columns, mask_columns.size()), mask_columns);
    failures += d_logits.verify_guards(label + " guards");
    if (misalign) {
        failures += verify_exact((label + " leading element").c_str(),
                                 from_device<std::uint16_t>(d_logits.data(), 1), {lead});
    }
    return failures;
}

int exact_contract() {
    int failures = 0;
    // The engine layout: one stable [words,16,8] buffer serving every verification width.
    const Geometry mtp{kRealRows, kRealDomain, 5, 1, 16, 8};
    for (const double density : {0.0, 0.001, 0.5, 1.0}) {
        failures += exact_case("bitmask real W=5 B=1 density=" + std::to_string(density), mtp,
                               density, {5}, 11u);
    }
    failures += exact_case("bitmask real W=5 B=1 prefix", mtp, 0.01, {2}, 12u);
    // Every row count form: none, partial, full, beyond the width, and negative.
    failures += exact_case("bitmask real W=16 B=8 mixed", {kRealRows, kRealDomain, 16, 8, 16, 8},
                           0.01, {0, 1, 5, 16, 17, -3, 16, 2}, 13u);
    failures += exact_case("bitmask real ordinary W=1 B=8", {kRealRows, kRealDomain, 1, 8, 16, 8},
                           0.2, {1, 0, 1, 1, 0, 1, 1, 1}, 14u);
    failures += exact_case("bitmask counts beyond B", {kRealRows, kRealDomain, 2, 3, 16, 8}, 0.3,
                           {2, 1, 2, 0, 2, 2, 2, 2}, 15u);
    // One-token route: rows not a multiple of eight, a domain inside a word, a misaligned base.
    failures += exact_case("bitmask irregular rows", {1003, 999, 3, 2, 4, 3}, 0.4, {3, 2}, 16u);
    failures +=
        exact_case("bitmask misaligned base", {1024, 1000, 2, 2, 2, 2}, 0.4, {2, 1}, 17u, true);
    failures += exact_case("bitmask domain below one group", {8, 5, 1, 1, 1, 1}, 0.6, {1}, 18u);
    failures +=
        exact_case("bitmask domain equals rows", {4096, 4096, 3, 2, 3, 2}, 0.5, {3, 3}, 19u);
    // Columns and rows past the CUDA grid limit of 65535 per launch.
    failures += exact_case("bitmask columns past the grid limit", {8, 8, 65536, 1, 65536, 1}, 0.5,
                           {65536}, 20u);
    std::vector<std::int32_t> alternate(65536);
    for (std::size_t row = 0; row < alternate.size(); ++row) { alternate[row] = row % 3 == 0; }
    failures += exact_case("bitmask rows past the grid limit", {8, 8, 1, 65536, 1, 65536}, 0.5,
                           alternate, 21u);
    return failures;
}

// One captured execution must follow the per-row counts and mask contents present at replay.
int capture_contract() {
    const Geometry g{kRealRows, kRealDomain, 5, 2, 16, 8};
    DeviceContext context;
    DeviceBuffer d_logits(g.logits_count() * sizeof(std::uint16_t));
    DeviceBuffer d_mask(g.mask_count() * sizeof(std::int32_t));
    DeviceBuffer d_columns(8 * sizeof(std::int32_t));
    Tensor logits(d_logits.p, DType::BF16, {g.physical_rows, g.width, g.batch});
    Tensor bitmask(d_mask.p, DType::I32, {g.words(), g.mask_columns, g.mask_rows});
    Tensor columns(d_columns.p, DType::I32, {8});
    const auto launch = [&] {
        ops::apply_token_bitmask(logits, bitmask, columns, g.token_domain, context.stream);
    };
    DecodeGraphDefinition definition;
    definition.capture(context.stream, launch);
    DecodeGraphExecutable graph;
    graph.instantiate(definition);

    int failures = 0;
    const std::vector<std::vector<std::int32_t>> counts{
        {5, 0, 0, 0, 0, 0, 0, 0}, {0, 3, 0, 0, 0, 0, 0, 0}, {2, 5, 0, 0, 0, 0, 0, 0}};
    for (std::size_t replay = 0; replay < counts.size(); ++replay) {
        const auto seed                           = static_cast<std::uint32_t>(40 + replay);
        const std::vector<std::uint16_t> input    = random_logits(g, seed);
        const std::vector<std::int32_t> mask      = random_mask(g, 0.05, seed + 100u);
        const std::vector<std::uint16_t> expected = oracle(g, input, mask, counts[replay]);
        d_logits.copy_from_host(input.data(), input.size() * sizeof(std::uint16_t));
        d_mask.copy_from_host(mask.data(), mask.size() * sizeof(std::int32_t));
        d_columns.copy_from_host(counts[replay].data(),
                                 counts[replay].size() * sizeof(std::int32_t));
        graph.launch(context.stream);
        context.synchronize();
        failures += verify_exact(("bitmask graph replay " + std::to_string(replay)).c_str(),
                                 from_device<std::uint16_t>(d_logits, expected.size()), expected);
    }
    return failures;
}

// Rows keep 1 to 50 allowed tokens with close logits, and the padding rows past the domain hold
// the largest values of the column, so reading them would win any selection.
int selection_contract() {
    constexpr int batch = 8;
    const Geometry g{kRealRows, kRealDomain, 1, batch, 1, batch};
    const int allowed_counts[batch] = {1, 2, 3, 7, 19, 20, 21, 50};
    std::vector<float> values(g.logits_count(), -4.0f);
    std::vector<std::int32_t> mask(g.mask_count(), 0);
    std::vector<std::vector<int>> allowed_sets(batch);
    std::mt19937 generator(77u);
    std::uniform_int_distribution<int> token(0, kRealDomain - 1);
    std::uniform_real_distribution<float> logit(-0.5f, 0.5f);
    for (int row = 0; row < batch; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * kRealRows;
        auto& set              = allowed_sets[static_cast<std::size_t>(row)];
        while (static_cast<int>(set.size()) < allowed_counts[row]) {
            const int candidate = token(generator);
            if (std::find(set.begin(), set.end(), candidate) != set.end()) { continue; }
            set.push_back(candidate);
            values[base + static_cast<std::size_t>(candidate)] = logit(generator);
            mask[static_cast<std::size_t>(row) * g.words() + (candidate >> 5)] |=
                static_cast<std::int32_t>(1u << (candidate & 31));
        }
        // Masked tokens outrank every allowed one before masking.
        for (int i = 0; i < 64; ++i) {
            const int candidate = token(generator);
            if (std::find(set.begin(), set.end(), candidate) == set.end()) {
                values[base + static_cast<std::size_t>(candidate)] = 8.0f + 0.01f * i;
            }
        }
        for (int padding = kRealDomain; padding < kRealRows; ++padding) {
            values[base + static_cast<std::size_t>(padding)] = 100.0f;
        }
    }
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }

    DeviceBuffer d_logits  = to_device(bits);
    DeviceBuffer d_mask    = to_device(mask);
    DeviceBuffer d_columns = to_device(std::vector<std::int32_t>(batch, 1));
    Tensor logits(d_logits.p, DType::BF16, {kRealRows, 1, batch});
    Tensor bitmask(d_mask.p, DType::I32, {g.words(), 1, batch});
    Tensor columns(d_columns.p, DType::I32, {batch});
    ops::apply_token_bitmask(logits, bitmask, columns, kRealDomain, nullptr);

    Tensor matrix = logits.view({kRealRows, batch});
    DeviceBuffer d_greedy(batch * sizeof(std::int32_t));
    Tensor greedy(d_greedy.p, DType::I32, {batch});
    ops::argmax(matrix, greedy, kRealDomain, nullptr);
    cuda_synchronize();

    int failures             = 0;
    const auto greedy_tokens = from_device<std::int32_t>(d_greedy, batch);
    for (int row = 0; row < batch; ++row) {
        const auto& set        = allowed_sets[static_cast<std::size_t>(row)];
        const std::size_t base = static_cast<std::size_t>(row) * kRealRows;
        int best               = set.front();
        for (const int candidate : set) {
            const float value = bf16_to_f32(bits[base + static_cast<std::size_t>(candidate)]);
            const float held  = bf16_to_f32(bits[base + static_cast<std::size_t>(best)]);
            if (value > held || (value == held && candidate < best)) { best = candidate; }
        }
        if (greedy_tokens[static_cast<std::size_t>(row)] != best) {
            std::cerr << "masked argmax row " << row << " selected "
                      << greedy_tokens[static_cast<std::size_t>(row)] << ", expected " << best
                      << '\n';
            ++failures;
        }
    }

    // Unfiltered sampling (full top-20 cap, top_p = 1, min_p = 0) at many counter positions.
    ops::SamplingConfig config;
    config.temperature     = 1.0f;
    config.top_k           = 20;
    config.seed            = 424242;
    DeviceBuffer d_configs = to_device(std::vector<ops::SamplingConfig>(batch, config));
    DeviceBuffer d_positions(batch * sizeof(std::int32_t));
    DeviceBuffer d_sampled(batch * sizeof(std::int32_t));
    Tensor positions(d_positions.p, DType::I32, {batch});
    Tensor sampled(d_sampled.p, DType::I32, {batch});
    WorkspaceArena workspace(std::max<std::size_t>(
        256, ops::sampling_workspace_capacity_bytes(kRealDomain, batch, batch)));
    for (int draw = 0; draw < 256; ++draw) {
        std::vector<std::int32_t> draw_positions(batch);
        for (int row = 0; row < batch; ++row) { draw_positions[row] = 1000 + draw * batch + row; }
        d_positions.copy_from_host(draw_positions.data(), batch * sizeof(std::int32_t));
        ops::sample(matrix, sampled, kRealDomain,
                    static_cast<const ops::SamplingConfig*>(d_configs.p), positions,
                    ops::kSamplePurposeDecode, workspace, nullptr);
        cuda_synchronize();
        const auto tokens = from_device<std::int32_t>(d_sampled, batch);
        for (int row = 0; row < batch; ++row) {
            const auto& set = allowed_sets[static_cast<std::size_t>(row)];
            if (std::find(set.begin(), set.end(), tokens[static_cast<std::size_t>(row)]) ==
                set.end()) {
                std::cerr << "masked sample row " << row << " drew excluded token "
                          << tokens[static_cast<std::size_t>(row)] << '\n';
                return failures + 1;
            }
        }
    }
    return failures;
}

// A row whose mask allows nothing becomes minus infinity throughout, and argmax and sample() return
// token 0 for it, as their contracts state. That token would look like a valid choice, so callers
// keep at least one token allowed per row; this case pins the edge they must avoid.
int all_excluded_contract() {
    constexpr int batch    = 2;
    constexpr int kAllowed = 131071;
    const Geometry g{kRealRows, kRealDomain, 1, batch, 1, batch};
    std::vector<float> values(g.logits_count());
    fill_uniform(values, 91u, -6.0f, 6.0f);
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    // Row 0 allows nothing; row 1 allows one token.
    std::vector<std::int32_t> mask(g.mask_count(), 0);
    mask[static_cast<std::size_t>(g.words()) + (kAllowed >> 5)] =
        static_cast<std::int32_t>(1u << (kAllowed & 31));

    DeviceBuffer d_logits  = to_device(bits);
    DeviceBuffer d_mask    = to_device(mask);
    DeviceBuffer d_columns = to_device(std::vector<std::int32_t>(batch, 1));
    Tensor logits(d_logits.p, DType::BF16, {kRealRows, 1, batch});
    Tensor bitmask(d_mask.p, DType::I32, {g.words(), 1, batch});
    Tensor columns(d_columns.p, DType::I32, {batch});
    ops::apply_token_bitmask(logits, bitmask, columns, kRealDomain, nullptr);
    cuda_synchronize();

    int failures        = 0;
    const auto masked   = from_device<std::uint16_t>(d_logits, g.logits_count());
    const auto excluded = std::count(masked.begin(), masked.begin() + kRealRows, kNegativeInfinity);
    if (excluded != kRealRows) {
        std::cerr << "an all-zero mask left " << kRealRows - excluded << " logits unmasked\n";
        ++failures;
    }

    Tensor matrix = logits.view({kRealRows, batch});
    DeviceBuffer d_greedy(batch * sizeof(std::int32_t));
    Tensor greedy(d_greedy.p, DType::I32, {batch});
    ops::argmax(matrix, greedy, kRealDomain, nullptr);
    cuda_synchronize();
    const auto greedy_tokens = from_device<std::int32_t>(d_greedy, batch);
    failures +=
        verify_exact("all-excluded argmax", greedy_tokens, std::vector<std::int32_t>{0, kAllowed});

    for (const float temperature : {0.0f, 1.0f}) {
        ops::SamplingConfig config;
        config.temperature       = temperature;
        config.top_k             = 20;
        config.seed              = 7;
        DeviceBuffer d_configs   = to_device(std::vector<ops::SamplingConfig>(batch, config));
        DeviceBuffer d_positions = to_device(std::vector<std::int32_t>{11, 12});
        DeviceBuffer d_sampled(batch * sizeof(std::int32_t));
        Tensor positions(d_positions.p, DType::I32, {batch});
        Tensor sampled(d_sampled.p, DType::I32, {batch});
        WorkspaceArena workspace(std::max<std::size_t>(
            256, ops::sampling_workspace_capacity_bytes(kRealDomain, batch, batch)));
        ops::sample(matrix, sampled, kRealDomain,
                    static_cast<const ops::SamplingConfig*>(d_configs.p), positions,
                    ops::kSamplePurposeDecode, workspace, nullptr);
        cuda_synchronize();
        failures += verify_exact(
            temperature == 0.0f ? "all-excluded greedy sample" : "all-excluded stochastic sample",
            from_device<std::int32_t>(d_sampled, batch), std::vector<std::int32_t>{0, kAllowed});
    }
    return failures;
}

int validation_contract() {
    int failures = 0;
    DeviceBuffer d_logits(64 * 2 * sizeof(std::uint16_t));
    DeviceBuffer d_mask(2 * 2 * 2 * sizeof(std::int32_t));
    DeviceBuffer d_columns(2 * sizeof(std::int32_t));
    Tensor logits(d_logits.p, DType::BF16, {64, 1, 2});
    Tensor mask(d_mask.p, DType::I32, {2, 2, 2});
    Tensor columns(d_columns.p, DType::I32, {2});
    const auto rejects = [&](const char* label, Tensor l, Tensor m, Tensor c, std::int32_t domain) {
        try {
            ops::apply_token_bitmask(l, m, c, domain, nullptr);
            std::cerr << "apply_token_bitmask accepted " << label << '\n';
            ++failures;
        } catch (const std::invalid_argument&) {}
    };
    rejects("a short word axis", logits, Tensor(d_mask.p, DType::I32, {1, 2, 2}), columns, 64);
    rejects("a domain past the rows", logits, mask, columns, 65);
    rejects("an empty domain", logits, mask, columns, 0);
    rejects("too few mask columns", Tensor(d_logits.p, DType::BF16, {32, 2, 2}),
            Tensor(d_mask.p, DType::I32, {1, 1, 2}), columns, 32);
    rejects("too few mask rows", logits, Tensor(d_mask.p, DType::I32, {2, 2, 1}), columns, 64);
    rejects("too few counts", logits, mask, Tensor(d_columns.p, DType::I32, {1}), 64);
    rejects("FP32 logits", Tensor(d_logits.p, DType::FP32, {64, 1, 1}), mask, columns, 64);
    rejects("rank-4 logits", Tensor(d_logits.p, DType::BF16, {16, 2, 2, 2}), mask, columns, 16);
    rejects("logits aliasing the mask", logits, Tensor(d_logits.p, DType::I32, {2, 2, 2}), columns,
            64);
    rejects("logits aliasing the counts", logits, mask, Tensor(d_logits.p, DType::I32, {2}), 64);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += validation_contract();
    failures += exact_contract();
    failures += capture_contract();
    failures += selection_contract();
    failures += all_excluded_contract();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " apply_token_bitmask public contract\n";
    return failures == 0 ? 0 : 1;
}
