#include "options.h"

#include <functional>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

ninfer::cli::Options parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

bool rejects(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

// The rejection message of a command line, or empty when it parses.
std::string rejection(std::vector<std::string> arguments) {
    try {
        (void)parse(std::move(arguments));
    } catch (const std::invalid_argument& error) { return error.what(); }
    return {};
}

std::vector<std::string> mtp_prompt(std::initializer_list<std::string> extra) {
    std::vector<std::string> arguments{"ninfer-cli", "model.ninfer", "--prompt", "hello"};
    arguments.insert(arguments.end(), extra.begin(), extra.end());
    return arguments;
}

int check_context_lookup_flags() {
    int failures = 0;
    const ninfer::cli::Options defaults =
        parse(mtp_prompt({"--spec", "mtp", "--draft-tokens", "4"}));
    failures +=
        check(defaults.speculative.context_lookup.policy == ninfer::ContextLookupPolicy::Fixed &&
                  defaults.speculative.context_lookup.min_suffix == 16 &&
                  defaults.speculative.context_lookup.max_proposal == 15,
              "MTP context lookup defaults do not preserve the fixed 16-token rule");
    const ninfer::cli::Options adaptive =
        parse(mtp_prompt({"--spec", "mtp", "--draft-tokens", "4", "--lookup-policy", "adaptive",
                          "--lookup-min-suffix", "4", "--lookup-max-proposal", "8"}));
    failures +=
        check(adaptive.speculative.context_lookup.policy == ninfer::ContextLookupPolicy::Adaptive &&
                  adaptive.speculative.context_lookup.min_suffix == 4 &&
                  adaptive.speculative.context_lookup.max_proposal == 8,
              "CLI did not preserve the context lookup flags");
    const ninfer::cli::Options off =
        parse(mtp_prompt({"--spec", "mtp", "--draft-tokens", "7", "--lookup-policy", "off"}));
    failures += check(off.speculative.context_lookup.policy == ninfer::ContextLookupPolicy::Off,
                      "--lookup-policy off was rejected with a seven-token draft window");
    const std::string help = ninfer::cli::usage_text("ninfer-cli");
    failures += check(help.find("--lookup-policy off|fixed|adaptive") != std::string::npos &&
                          help.find("--lookup-min-suffix") != std::string::npos &&
                          help.find("--lookup-max-proposal") != std::string::npos,
                      "CLI help omits a context lookup flag");

    // Explicit flags are validated even when their value equals the default.
    for (const auto& flag : {std::vector<std::string>{"--lookup-policy", "fixed"},
                             std::vector<std::string>{"--lookup-min-suffix", "16"},
                             std::vector<std::string>{"--lookup-max-proposal", "15"}}) {
        std::vector<std::string> without_mtp = mtp_prompt({});
        without_mtp.insert(without_mtp.end(), flag.begin(), flag.end());
        failures += check(rejection(without_mtp).find("require --spec mtp") != std::string::npos,
                          "CLI accepted an explicit context lookup flag without MTP");
        std::vector<std::string> with_dflash =
            mtp_prompt({"--spec", "dflash", "--draft-tokens", "7"});
        with_dflash.insert(with_dflash.end(), flag.begin(), flag.end());
        failures += check(rejection(with_dflash).find("require --spec mtp") != std::string::npos,
                          "CLI accepted an explicit context lookup flag with DFlash");
    }
    // With lookup off, sizes are rejected as unused rather than checked against the draft window.
    failures +=
        check(rejection(mtp_prompt({"--spec", "mtp", "--draft-tokens", "7", "--lookup-policy",
                                    "off", "--lookup-max-proposal", "6"}))
                      .find("--lookup-policy fixed or adaptive") != std::string::npos,
              "CLI did not reject a lookup proposal with --lookup-policy off as unused");
    failures += check(rejection(mtp_prompt({"--spec", "mtp", "--draft-tokens", "4",
                                            "--lookup-policy", "off", "--lookup-min-suffix", "4"}))
                              .find("--lookup-policy fixed or adaptive") != std::string::npos,
                      "CLI did not reject a lookup suffix with --lookup-policy off as unused");
    for (const auto& invalid : {std::vector<std::string>{"--lookup-policy", "greedy"},
                                std::vector<std::string>{"--lookup-min-suffix", "0"},
                                std::vector<std::string>{"--lookup-min-suffix", "1"},
                                std::vector<std::string>{"--lookup-min-suffix", "65"},
                                std::vector<std::string>{"--lookup-max-proposal", "4"},
                                std::vector<std::string>{"--lookup-max-proposal", "16"},
                                std::vector<std::string>{"--lookup-max-proposal"}}) {
        std::vector<std::string> arguments = mtp_prompt({"--spec", "mtp", "--draft-tokens", "4"});
        arguments.insert(arguments.end(), invalid.begin(), invalid.end());
        failures += check(!rejection(std::move(arguments)).empty(),
                          "CLI accepted an invalid context lookup flag");
    }
    return failures;
}


} // namespace

int main() {
    int failures = 0;
    const ninfer::cli::Options configured =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "37"});
    failures += check(configured.thinking_budget == 37,
                      "--thinking-budget did not preserve its positive value");
    failures +=
        check(ninfer::cli::usage_text("ninfer-cli").find("--thinking-budget") != std::string::npos,
              "CLI help omits --thinking-budget");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "0"});
                      }),
                      "zero --thinking-budget was accepted");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "8", "--no-thinking"});
                      }),
                      "--thinking-budget was accepted with --no-thinking");
    const ninfer::cli::Options with_effort =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "8",
               "--reasoning-effort", "medium"});
    failures += check(with_effort.thinking_budget == 8 && with_effort.reasoning_effort,
                      "thinking budget did not coexist with reasoning effort");
    const ninfer::cli::Options dflash_vision =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--vision", "--spec", "dflash",
               "--draft-tokens", "7"});
    failures += check(dflash_vision.enable_vision &&
                          dflash_vision.speculative.backend == ninfer::SpeculativeBackend::DFlash &&
                          dflash_vision.speculative.draft_tokens == 7,
                      "CLI did not preserve the combined DFlash and Vision startup features");
    for (const auto k : {1U, 2U, 7U, 15U}) {
        const auto dflash2 = parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                    "dflash2", "--draft-tokens", std::to_string(k)});
        failures += check(dflash2.speculative.backend == ninfer::SpeculativeBackend::DFlash2 &&
                              dflash2.speculative.draft_tokens == k,
                          "CLI did not preserve the DFlash2 draft count");
    }
    for (const auto k : {0U, 16U}) {
        failures +=
            check(rejects([&] {
                      (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                   "dflash2", "--draft-tokens", std::to_string(k)});
                  }),
                  "CLI accepted an unsupported DFlash2 draft count");
    }
    const ninfer::cli::Options nvfp4 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "nvfp4"});
    failures += check(nvfp4.kv_cache == ninfer::KvCacheStorage::Nvfp4Group16,
                      "--kv-dtype nvfp4 did not select group-16 NVFP4 KV");
    const ninfer::cli::Options k8v4 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "k8v4"});
    failures += check(k8v4.kv_cache == ninfer::KvCacheStorage::Fp8KeyNvfp4Value,
                      "--kv-dtype k8v4 did not select asymmetric K8V4 KV");
    const std::string help = ninfer::cli::usage_text("ninfer-cli");
    failures +=
        check(help.find("nvfp4") != std::string::npos && help.find("k8v4") != std::string::npos,
              "CLI help omits a production KV storage mode");
    const ninfer::cli::Options logging =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--log-level", "debug"});
    failures += check(logging.log_level == ninfer::product::LogLevel::Debug,
                      "CLI log level was not parsed");
    failures += check(help.find("--log-level") != std::string::npos,
                      "CLI help omits the log-level control");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--log-level", "verbose"});
                      }),
                      "CLI accepted an unknown log level");
    failures +=
        check(rejects([] {
                  (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--top-k", "21"});
              }),
              "CLI accepted top_k beyond the executable candidate domain");
    failures += check_context_lookup_flags();
    const ninfer::cli::Options numerics_default =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello"});
    failures += check(numerics_default.text_residual == ninfer::TextResidualStorage::BFloat16 &&
                          numerics_default.prefill_attention ==
                              ninfer::PrefillAttentionKernel::Automatic,
                      "CLI numerics defaults are not bf16/auto");
    const ninfer::cli::Options numerics =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--text-residual", "fp32",
               "--prefill-attention", "reference"});
    failures += check(numerics.text_residual == ninfer::TextResidualStorage::Float32 &&
                          numerics.prefill_attention == ninfer::PrefillAttentionKernel::Reference,
                      "--text-residual/--prefill-attention were not parsed");
    for (const char* kernel : {"auto", "splitd", "flash", "reference"}) {
        failures += check(!rejects([kernel] {
                              (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                           "--prefill-attention", kernel});
                          }),
                          "CLI rejected a prefill attention kernel");
    }
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--text-residual", "fp16"});
                      }),
                      "CLI accepted an unknown text residual storage");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--prefill-attention", "SplitD"});
                      }),
                      "CLI accepted an unknown prefill attention kernel");
    failures += check(help.find("--text-residual bf16|fp32") != std::string::npos &&
                          help.find("--prefill-attention auto|splitd|flash|reference") !=
                              std::string::npos,
                      "CLI help omits the numerics controls");
    return failures == 0 ? 0 : 1;
}
