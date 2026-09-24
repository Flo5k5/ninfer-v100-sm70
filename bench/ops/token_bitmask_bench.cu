// Benchmark of the public apply_token_bitmask at the Qwen3.6 decode logits shapes: 248320 physical
// rows, a 248077-token domain, verification widths W=1 (ordinary), 5 (MTP K=4) and 16 (a
// context-lookup window) for B=1 and B=8. A captured decode graph launches the Op every round, so
// the `free` mode (no constrained row, every block exits at once) is the cost every request pays.
//
//   ./ninfer_token_bitmask_bench
//   ./ninfer_token_bitmask_bench --mode excluding --width 16 --batch 8
//   ./ninfer_token_bitmask_bench --mode one --width 16 --batch 8
//
// Modes:
//   free       no constrained row: every column count is zero;
//   excluding  every column constrained, one token in 1024 allowed, as a JSON grammar mostly
//              excludes the vocabulary: excluded groups are stored without a load;
//   mixed      every column constrained, every other token allowed: every group is loaded,
//              blended and stored, the most work per token;
//   one        one constrained row (excluding) among B, the others free.
#include "ninfer/ops/token_bitmask.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr std::int32_t kPhysicalRows = 248320;
constexpr std::int32_t kTokenDomain  = 248077;

enum class Mode { Free, Excluding, Mixed, One };

struct Options {
    std::string mode;
    int width = 0;
    int batch = 0;
};

void usage(const char* argv0) {
    std::printf("usage: %s [--mode free|excluding|mixed|one --width W --batch B]\n", argv0);
}

int parse_int(std::string_view value, const char* name) {
    try {
        std::size_t parsed = 0;
        const int out      = std::stoi(std::string(value), &parsed);
        if (parsed != value.size()) { throw std::invalid_argument("trailing"); }
        return out;
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name) + " expects an integer");
    }
}

Mode parse_mode(std::string_view name) {
    if (name == "free") { return Mode::Free; }
    if (name == "excluding") { return Mode::Excluding; }
    if (name == "mixed") { return Mode::Mixed; }
    if (name == "one") { return Mode::One; }
    throw std::invalid_argument("--mode must be free, excluding, mixed or one");
}

const char* mode_name(Mode mode) {
    switch (mode) {
    case Mode::Free:
        return "free";
    case Mode::Excluding:
        return "excluding";
    case Mode::Mixed:
        return "mixed";
    case Mode::One:
        return "one";
    }
    return "?";
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto need_value = [&](const char* name) -> std::string_view {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string(name) + " needs a value");
            }
            return argv[++i];
        };
        if (arg == "--mode") {
            options.mode = need_value("--mode");
        } else if (arg == "--width") {
            options.width = parse_int(need_value("--width"), "--width");
        } else if (arg == "--batch") {
            options.batch = parse_int(need_value("--batch"), "--batch");
        } else if (arg == "-h" || arg == "--help") {
            usage(argc > 0 ? argv[0] : "ninfer_token_bitmask_bench");
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (options.mode.empty()) {
        if (options.width != 0 || options.batch != 0) {
            throw std::invalid_argument("--width and --batch require --mode");
        }
        return options;
    }
    (void)parse_mode(options.mode);
    if (options.width == 0) { options.width = 1; }
    if (options.batch == 0) { options.batch = 1; }
    if (options.width < 1 || options.width > 16 || options.batch < 1 || options.batch > 8) {
        throw std::invalid_argument("--width must be in [1,16] and --batch in [1,8]");
    }
    return options;
}

void run_case(Mode mode, int width, int batch) {
    const std::int32_t words = ops::token_bitmask_words(kTokenDomain);
    DeviceBuffer logits      = make_bf16(static_cast<std::size_t>(kPhysicalRows) * width * batch);

    // Word k of every column: one allowed token in 1024 (bit 0 of every 32nd word), or every
    // other token.
    std::vector<std::int32_t> mask(static_cast<std::size_t>(words) * width * batch);
    for (std::size_t word = 0; word < mask.size(); ++word) {
        const std::size_t in_column = word % static_cast<std::size_t>(words);
        mask[word] = mode == Mode::Mixed ? 0x55555555 : (in_column % 32 == 0 ? 1 : 0);
    }
    DeviceBuffer bitmask(mask.size() * sizeof(std::int32_t));
    bitmask.copy_from_host(mask.data(), bitmask.bytes);

    std::vector<std::int32_t> columns(static_cast<std::size_t>(batch),
                                      mode == Mode::Free ? 0 : width);
    if (mode == Mode::One) {
        for (std::size_t row = 1; row < columns.size(); ++row) { columns[row] = 0; }
    }
    DeviceBuffer mask_columns(columns.size() * sizeof(std::int32_t));
    mask_columns.copy_from_host(columns.data(), mask_columns.bytes);

    Tensor tlogits(logits.p, DType::BF16, {kPhysicalRows, width, batch});
    Tensor tmask(bitmask.p, DType::I32, {words, width, batch});
    Tensor tcolumns(mask_columns.p, DType::I32, {batch});

    // Logit bytes the constrained columns move: a store per excluded group, and a load as well
    // for a mixed one. Mask words are negligible next to them.
    const int constrained_rows = mode == Mode::Free ? 0 : mode == Mode::One ? 1 : batch;
    const double per_token     = mode == Mode::Mixed ? 4.0 : 2.0;
    const double bytes  = per_token * kPhysicalRows * static_cast<double>(width) * constrained_rows;
    const Result result = bench_loop(
        [&](cudaStream_t stream) {
            ops::apply_token_bitmask(tlogits, tmask, tcolumns, kTokenDomain, stream);
        },
        bytes);

    const std::string label = std::string("token_bitmask ") + mode_name(mode) +
                              " W=" + std::to_string(width) + " B=" + std::to_string(batch);
    print_result(label.c_str(), result);
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    try {
        const Options options = parse_args(argc, argv);
        if (options.mode.empty()) {
            for (const int batch : {1, 8}) {
                for (const int width : {1, 5, 16}) {
                    for (const Mode mode : {Mode::Free, Mode::Excluding, Mode::Mixed}) {
                        run_case(mode, width, batch);
                    }
                }
            }
            run_case(Mode::One, 16, 8);
        } else {
            run_case(parse_mode(options.mode), options.width, options.batch);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ninfer_token_bitmask_bench: %s\n", e.what());
        return 2;
    }
    return 0;
}
