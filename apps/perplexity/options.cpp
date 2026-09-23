#include "options.h"

#include <charconv>
#include <stdexcept>
#include <string_view>
#include <system_error>

namespace ninfer::perplexity {
namespace {

[[noreturn]] void usage_error(std::string_view message) {
    throw std::invalid_argument(std::string(message));
}

template <class Integer>
Integer parse_integer(std::string_view text, const char* label) {
    Integer value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        usage_error(std::string("invalid ") + label + ": " + std::string(text));
    }
    return value;
}

KvCacheStorage parse_kv_dtype(std::string_view dtype) {
    if (dtype == "bf16") { return KvCacheStorage::BFloat16; }
    if (dtype == "int8") { return KvCacheStorage::Int8Group64; }
    if (dtype == "fp8") { return KvCacheStorage::Fp8E4M3Row256; }
    if (dtype == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (dtype == "k8v4") { return KvCacheStorage::Fp8KeyNvfp4Value; }
    usage_error("--kv-dtype must be bf16, int8, fp8, nvfp4, or k8v4");
}

} // namespace

std::string usage_text() {
    return "usage: ninfer-perplexity <model.ninfer> "
           "(--corpus <manifest.json> [--quick] | --text <utf8-file>)\n"
           "       [--context N] [--stride N] [--device N]\n"
           "       [--kv-dtype bf16|int8|fp8|nvfp4|k8v4] [--output <directory>]\n"
           "       [--logits-out <file> [--logits-reference <file>] [--chunks N]]\n"
           "       [--log-level trace|debug|info|warning|error|critical|off]\n";
}

std::string kv_dtype_name(KvCacheStorage value) {
    switch (value) {
    case KvCacheStorage::BFloat16:
        return "bf16";
    case KvCacheStorage::Int8Group64:
        return "int8-g64";
    case KvCacheStorage::Fp8E4M3Row256:
        return "fp8-e4m3-r256";
    case KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    }
    throw std::logic_error("unknown KV dtype");
}

Options parse_options(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        return Options{.help_requested = true};
    }
    if (argc < 2 || std::string_view(argv[1]).starts_with("--")) {
        usage_error("artifact path is required");
    }
    Options out;
    out.artifact    = argv[1];
    bool stride_set = false;
    for (int i = 2; i < argc; ++i) {
        const std::string_view option = argv[i];
        const auto value              = [&](const char* label) -> std::string_view {
            if (++i >= argc) { usage_error(std::string(label) + " requires a value"); }
            return argv[i];
        };
        if (option == "--corpus") {
            out.corpus = std::filesystem::path(value("--corpus"));
        } else if (option == "--text") {
            out.text = std::filesystem::path(value("--text"));
        } else if (option == "--quick") {
            out.quick = true;
        } else if (option == "--context") {
            out.context = parse_integer<std::uint32_t>(value("--context"), "context");
        } else if (option == "--stride") {
            out.stride = parse_integer<std::uint32_t>(value("--stride"), "stride");
            stride_set = true;
        } else if (option == "--device") {
            out.device = parse_integer<int>(value("--device"), "device");
        } else if (option == "--kv-dtype") {
            out.kv = parse_kv_dtype(value("--kv-dtype"));
        } else if (option == "--output") {
            out.output = std::filesystem::path(value("--output"));
        } else if (option == "--logits-out") {
            out.logits_out = std::filesystem::path(value("--logits-out"));
        } else if (option == "--logits-reference") {
            out.logits_reference = std::filesystem::path(value("--logits-reference"));
        } else if (option == "--chunks") {
            out.chunks = parse_integer<std::uint32_t>(value("--chunks"), "chunks");
        } else if (option == "--log-level") {
            out.log_level = product::parse_log_level(value("--log-level"));
        } else {
            usage_error("unknown option: " + std::string(option));
        }
    }
    if (out.corpus.has_value() == out.text.has_value()) {
        usage_error("exactly one of --corpus and --text is required");
    }
    if (out.quick && !out.corpus) { usage_error("--quick requires --corpus"); }
    if ((out.logits_reference || out.chunks) && !out.logits_out) {
        usage_error("--logits-reference and --chunks require --logits-out");
    }
    if (out.chunks && *out.chunks == 0) { usage_error("--chunks must be positive"); }
    if (out.logits_out) {
        // The KLD window plan fixes the stride before the generic context/stride check.
        if (!out.text) { usage_error("--logits-out requires --text (one token stream)"); }
        if (out.context < 4) { usage_error("--logits-out requires --context >= 4"); }
        if (stride_set && out.stride != out.context / 2) {
            usage_error("--logits-out scores the second half of non-overlapping context windows; "
                        "--stride must be omitted or equal context/2");
        }
        out.stride = out.context / 2;
    }
    if (out.context < 2 || out.stride == 0 || out.stride >= out.context) {
        usage_error("context/stride must satisfy context>=2 and 1<=stride<context");
    }
    return out;
}

} // namespace ninfer::perplexity
