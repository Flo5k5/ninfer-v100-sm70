#include "artifact/reader.h"
#include "artifact_fixture.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using ninfer::artifact::ArtifactError;
using ninfer::artifact::Reader;
using ninfer::test::artifact_fixture::Json;
using ninfer::test::artifact_fixture::TemporaryArtifact;

using Magic = std::array<std::uint8_t, 8>;

constexpr std::uint64_t kPayloadBytes = 10240;

Json v2_directory() {
    return {{"identity", {{"model_id", "qwen3.8-27b"}, {"weights_id", "nvfp4"}}},
            {"objects",
             Json::array({{{"name", "text/final_norm"},
                           {"kind", "tensor"},
                           {"shape", {5120}},
                           {"format", "BF16"},
                           {"layout", "contiguous-le-v1"},
                           {"offset", 0},
                           {"bytes", kPayloadBytes}}})}};
}

// A complete v2 entry, framed and laid out as the v2 writer produced it: the magic, the directory
// byte count, the directory, and the payload at the next 4096-byte boundary. The shared fixture
// writes v3 entries.
TemporaryArtifact write_v2_entry(const Magic& magic, std::string_view suffix) {
    const std::string json    = v2_directory().dump();
    const auto payload_offset = ninfer::test::artifact_fixture::align_up(16 + json.size(), 4096);
    std::vector<std::byte> file(payload_offset + kPayloadBytes, std::byte{0});
    for (std::size_t i = 0; i < magic.size(); ++i) { file[i] = std::byte{magic[i]}; }
    ninfer::test::artifact_fixture::write_u64_le(file.data() + 8, json.size());
    std::memcpy(file.data() + 16, json.data(), json.size());

    auto path = std::filesystem::temp_directory_path() /
                ("ninfer_artifact_" + std::string(suffix) + ".ninfer");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(file.data()),
                 static_cast<std::streamsize>(file.size()));
    if (!output) { throw std::runtime_error("failed to write the v2 entry"); }
    return {std::move(path)};
}

std::string open_error(const Magic& magic, std::string_view suffix) {
    const auto fixture = write_v2_entry(magic, suffix);
    try {
        Reader reader(fixture.path);
    } catch (const ArtifactError& error) { return error.what(); }
    throw std::runtime_error(std::string(suffix) + " entry was accepted");
}

void require_contains(const std::string& text, std::string_view expected, std::string_view label) {
    if (text.find(expected) == std::string::npos) {
        throw std::runtime_error(std::string(label) + " error lacks '" + std::string(expected) +
                                 "': " + text);
    }
}

} // namespace

int main() {
    try {
        // A v2 entry is named as such and sent to the published v3 artifacts.
        const std::string v2 = open_error(Magic{'N', 'I', 'N', 'F', 'E', 'R', 0, 2}, "v2_entry");
        require_contains(v2, "NInfer v2 artifact is not supported", "v2 entry");
        require_contains(v2, "README.md", "v2 entry");

        // Only version 2 has a replacement to point to; other versions keep the magic error.
        const std::string v1 = open_error(Magic{'N', 'I', 'N', 'F', 'E', 'R', 0, 1}, "v1_entry");
        require_contains(v1, "expected NInfer v3 entry magic", "v1 entry");
        std::cout << "artifact v2 rejection checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
