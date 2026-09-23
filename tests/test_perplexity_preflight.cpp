#include "logits_dump.h"
#include "options.h"
#include "preflight.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
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

ninfer::perplexity::Options parse(std::vector<std::string> arguments) {
    arguments.insert(arguments.begin(), "ninfer-perplexity");
    std::vector<char*> argv;
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::perplexity::parse_options(static_cast<int>(argv.size()), argv.data());
}

std::string rejection(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::exception& error) { return error.what(); }
    return {};
}

// run() in main.cpp calls run_preflight() before loading the model and constructs the dump
// writer after tokenization; this reproduces that order without a model.
std::string preflight_then_writer(const std::vector<std::string>& arguments) {
    return rejection([&] {
        const auto options   = parse(arguments);
        const auto preflight = ninfer::perplexity::run_preflight(options);
        const std::vector<ninfer::TokenId> tokens(16, 1);
        ninfer::perplexity::KldBaseWriter writer(*options.logits_out, 8, 2, tokens, std::nullopt);
    });
}

void write_reference(const std::filesystem::path& path, std::uint32_t context) {
    std::ofstream out(path, std::ios::binary);
    out.write("_logits_", 8);
    const std::uint32_t chunks = 1;
    const std::int32_t vocab   = 6;
    out.write(reinterpret_cast<const char*>(&context), sizeof(context));
    out.write(reinterpret_cast<const char*>(&vocab), sizeof(vocab));
    out.write(reinterpret_cast<const char*>(&chunks), sizeof(chunks));
    const std::vector<std::int32_t> tokens(context, 3);
    out.write(reinterpret_cast<const char*>(tokens.data()),
              static_cast<std::streamsize>(tokens.size() * sizeof(std::int32_t)));
}

} // namespace

int main() {
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        ("ninfer_perplexity_preflight_" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(root);
    int failures = 0;

    // --output X --logits-out X/d.kld: the report directory is prepared before the dump exists.
    const std::filesystem::path missing = root / "missing";
    failures +=
        require(preflight_then_writer({"m.ninfer", "--text", "t.txt", "--output", missing.string(),
                                       "--logits-out", (missing / "d.kld").string()})
                    .empty(),
                "a missing --output directory is created before the dump inside it");
    const std::filesystem::path empty = root / "empty";
    std::filesystem::create_directories(empty);
    failures +=
        require(preflight_then_writer({"m.ninfer", "--text", "t.txt", "--output", empty.string(),
                                       "--logits-out", (empty / "d.kld").string()})
                    .empty(),
                "an empty --output directory can hold the dump");

    const std::filesystem::path used = root / "used";
    std::filesystem::create_directories(used);
    std::ofstream(used / "report.json") << "{}";
    failures += require(rejection([&] {
                            (void)ninfer::perplexity::run_preflight(
                                parse({"m.ninfer", "--text", "t.txt", "--output", used.string()}));
                        }).find("exists and is not empty") != std::string::npos,
                        "a non-empty --output directory is refused before loading the model");

    std::ofstream(root / "done.kld") << "x";
    failures += require(
        rejection([&] {
            (void)ninfer::perplexity::run_preflight(parse(
                {"m.ninfer", "--text", "t.txt", "--logits-out", (root / "done.kld").string()}));
        }).find("already exists") != std::string::npos,
        "an existing dump is refused before loading the model");
    std::ofstream(root / "stale.kld.partial") << "x";
    failures += require(
        rejection([&] {
            (void)ninfer::perplexity::run_preflight(parse(
                {"m.ninfer", "--text", "t.txt", "--logits-out", (root / "stale.kld").string()}));
        }).find("stale.kld.partial") != std::string::npos,
        "a stale partial dump is refused before loading the model");
    failures += require(rejection([&] {
                            (void)ninfer::perplexity::run_preflight(
                                parse({"m.ninfer", "--text", "t.txt", "--logits-out",
                                       (root / "absent" / "d.kld").string()}));
                        }).find("directory does not exist") != std::string::npos,
                        "a dump without a directory is refused before loading the model");

    write_reference(root / "ref.kld", 8);
    const auto preflight = ninfer::perplexity::run_preflight(
        parse({"m.ninfer", "--text", "t.txt", "--context", "8", "--logits-out",
               (root / "new.kld").string(), "--logits-reference", (root / "ref.kld").string()}));
    failures += require(preflight.reference && preflight.reference->context == 8 &&
                            preflight.reference->tokens.size() == 8 && !preflight.report_directory,
                        "the reference header is read before loading the model");
    failures += require(rejection([&] {
                            (void)ninfer::perplexity::run_preflight(
                                parse({"m.ninfer", "--text", "t.txt", "--context", "16",
                                       "--logits-out", (root / "new.kld").string(),
                                       "--logits-reference", (root / "ref.kld").string()}));
                        }).find("context 8 differs from --context 16") != std::string::npos,
                        "a reference with another context is refused before loading the model");
    failures += require(
        !rejection([&] {
             (void)ninfer::perplexity::run_preflight(
                 parse({"m.ninfer", "--text", "t.txt", "--logits-out", (root / "new.kld").string(),
                        "--logits-reference", (root / "absent.kld").string()}));
         }).empty(),
        "a missing reference dump is refused before loading the model");
    failures += require(!std::filesystem::exists(root / "new.kld.partial"),
                        "the preflight creates no dump file");

    std::filesystem::remove_all(root);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " perplexity_preflight\n";
    return failures == 0 ? 0 : 1;
}
