#include "options.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

ninfer::perplexity::Options parse(std::vector<std::string> arguments) {
    arguments.insert(arguments.begin(), "ninfer-perplexity");
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::perplexity::parse_options(static_cast<int>(argv.size()), argv.data());
}

bool rejects(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;

    const auto sliding = parse({"model.ninfer", "--text", "t.txt"});
    failures += check(sliding.context == 4096 && sliding.stride == 2048 && !sliding.logits_out,
                      "sliding windows default to context 4096, stride 2048");
    failures += check(sliding.prefill_chunk == 1024 && sliding.scored_chunk == 0,
                      "scoring defaults to 1024-token prefill chunks for every target");
    const auto narrow = parse({"model.ninfer", "--text", "t.txt", "--prefill-chunk", "2048",
                               "--scored-chunk", "5"});
    failures += check(narrow.prefill_chunk == 2048 && narrow.scored_chunk == 5,
                      "--prefill-chunk and --scored-chunk are parsed");
    failures += check(rejects([] {
                          (void)parse({"model.ninfer", "--text", "t.txt", "--prefill-chunk", "1000"});
                      }),
                      "--prefill-chunk must be a multiple of 128");
    failures += check(rejects([] {
                          (void)parse({"model.ninfer", "--text", "t.txt", "--scored-chunk", "2048"});
                      }),
                      "--scored-chunk must not exceed the prefill chunk");
    failures +=
        check(rejects([] { (void)parse({"model.ninfer", "--text", "t.txt", "--context", "512"}); }),
              "a sliding plan still rejects the default stride above a small context");

    const auto small =
        parse({"model.ninfer", "--text", "t.txt", "--context", "512", "--logits-out", "d.kld"});
    failures += check(small.stride == 256 && small.context == 512,
                      "--logits-out without --stride takes context/2 before the stride check");
    const auto explicit_stride = parse({"model.ninfer", "--text", "t.txt", "--context", "512",
                                        "--stride", "256", "--logits-out", "d.kld"});
    failures += check(explicit_stride.stride == 256, "--logits-out accepts --stride context/2");
    failures += check(rejects([] {
                          (void)parse({"model.ninfer", "--text", "t.txt", "--context", "512",
                                       "--stride", "100", "--logits-out", "d.kld"});
                      }),
                      "--logits-out rejects any other stride");
    failures +=
        check(rejects([] {
                  (void)parse({"model.ninfer", "--corpus", "m.json", "--logits-out", "d.kld"});
              }),
              "--logits-out requires one --text stream");
    failures += check(rejects([] {
                          (void)parse({"model.ninfer", "--text", "t.txt", "--context", "3",
                                       "--logits-out", "d.kld"});
                      }),
                      "--logits-out requires a context of at least 4");

    const auto prefix = parse({"model.ninfer", "--text", "t.txt", "--logits-out", "d.kld",
                               "--logits-reference", "r.kld", "--chunks", "8"});
    failures += check(prefix.chunks == 8u && prefix.logits_reference.has_value(),
                      "--chunks and --logits-reference are parsed with --logits-out");
    failures +=
        check(rejects([] { (void)parse({"model.ninfer", "--text", "t.txt", "--chunks", "8"}); }),
              "--chunks requires --logits-out");
    failures +=
        check(rejects([] {
                  (void)parse({"model.ninfer", "--text", "t.txt", "--logits-reference", "r.kld"});
              }),
              "--logits-reference requires --logits-out");
    failures += check(rejects([] {
                          (void)parse({"model.ninfer", "--text", "t.txt", "--logits-out", "d.kld",
                                       "--chunks", "0"});
                      }),
                      "--chunks must be positive");

    const auto kv = parse({"model.ninfer", "--text", "t.txt", "--kv-dtype", "int8"});
    failures += check(kv.kv == ninfer::KvCacheStorage::Int8Group64 &&
                          ninfer::perplexity::kv_dtype_name(kv.kv) == "int8-g64",
                      "--kv-dtype int8 selects and names the int8 group-64 cache");
    failures += check(
        rejects([] { (void)parse({"model.ninfer", "--text", "t.txt", "--kv-dtype", "fp4"}); }),
        "an unknown KV dtype is rejected");
    failures += check(sliding.text_residual == ninfer::TextResidualStorage::BFloat16 &&
                          sliding.prefill_attention == ninfer::PrefillAttentionKernel::Automatic,
                      "numerics default to a BF16 residual and the automatic prefill kernel");
    const auto numerics = parse({"model.ninfer", "--text", "t.txt", "--text-residual", "fp32",
                                 "--prefill-attention", "splitd"});
    failures += check(numerics.text_residual == ninfer::TextResidualStorage::Float32 &&
                          numerics.prefill_attention == ninfer::PrefillAttentionKernel::SplitD,
                      "--text-residual and --prefill-attention are parsed");
    failures += check(rejects([] {
                          (void)parse(
                              {"model.ninfer", "--text", "t.txt", "--text-residual", "bf32"});
                      }) && rejects([] {
                          (void)parse({"model.ninfer", "--text", "t.txt", "--prefill-attention",
                                       "llama"});
                      }),
                      "unknown numerics values are rejected");
    failures += check(ninfer::perplexity::usage_text().find("--text-residual bf16|fp32") !=
                              std::string::npos &&
                          ninfer::perplexity::usage_text().find(
                              "--prefill-attention auto|splitd|flash|reference") !=
                              std::string::npos,
                      "usage lists the numerics controls");
    failures += check(rejects([] { (void)parse({"model.ninfer"}); }),
                      "exactly one of --corpus and --text is required");
    failures += check(parse({"--help"}).help_requested, "--help alone requests help");

    std::cout << (failures == 0 ? "OK" : "FAIL") << " perplexity_options\n";
    return failures == 0 ? 0 : 1;
}
