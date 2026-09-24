#include "artifact/reader.h"
#include "artifact_fixture.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace {

using ninfer::artifact::ArtifactError;
using ninfer::artifact::Reader;
using Json = nlohmann::json;
using ninfer::test::artifact_fixture::kArtifactId;
using ninfer::test::artifact_fixture::kPayloadAlignment;
using ninfer::test::artifact_fixture::write_fixture;

Json normative_directory() {
    return {
        {"components", {{"text", {{"config", Json::object()}}}}},
        {"objects", Json::array({
                        {{"id", "resource"},
                         {"kind", "resource"},
                         {"encoding", "raw_bytes_v1"},
                         {"offset", 0},
                         {"bytes", 3}},
                        {{"id", "bf16"},
                         {"kind", "tensor"},
                         {"shape", {2, 3}},
                         {"format", "bf16"},
                         {"layout", "contiguous_le_v1"},
                         {"offset", 256},
                         {"bytes", 12}},
                        {{"id", "fp32_scalar"},
                         {"kind", "tensor"},
                         {"shape", Json::array()},
                         {"format", "fp32"},
                         {"layout", "contiguous_le_v1"},
                         {"offset", 512},
                         {"bytes", 4}},
                        {{"id", "i32"},
                         {"kind", "tensor"},
                         {"shape", {2}},
                         {"format", "int32"},
                         {"layout", "contiguous_le_v1"},
                         {"offset", 768},
                         {"bytes", 8}},
                        {{"id", "q4"},
                         {"kind", "tensor"},
                         {"shape", {1, 1}},
                         {"format", "q4_g64_fp16"},
                         {"layout", "row_split_k128_v1"},
                         {"offset", 1024},
                         {"bytes", 260}},
                        {{"id", "q5"},
                         {"kind", "tensor"},
                         {"shape", {2, 130}},
                         {"format", "q5_g64_fp16"},
                         {"layout", "row_split_k128_v1"},
                         {"offset", 1536},
                         {"bytes", 528}},
                        {{"id", "q6"},
                         {"kind", "tensor"},
                         {"shape", {1, 64}},
                         {"format", "q6_g64_fp16"},
                         {"layout", "row_split_k128_v1"},
                         {"offset", 2304},
                         {"bytes", 516}},
                        {{"id", "q8"},
                         {"kind", "tensor"},
                         {"shape", {1, 33}},
                         {"format", "q8_g32_fp16"},
                         {"layout", "row_split_k128_v1"},
                         {"offset", 3072},
                         {"bytes", 264}},
                        {{"id", "fp8_row"},
                         {"kind", "tensor"},
                         {"shape", {2, 4}},
                         {"format", "fp8_e4m3fn_row_bf16"},
                         {"layout", "row_scale_v1"},
                         {"offset", 3584},
                         {"bytes", 260}},
                    })},
        {"bindings", Json::object()},
        {"uses", Json::array()},
    };
}

template <typename Function>
void expect_artifact_error(Function&& function, std::string_view label) {
    try {
        function();
    } catch (const ArtifactError&) { return; }
    throw std::runtime_error(std::string(label) + " was accepted");
}

void test_normative_fixture() {
    const Json directory = normative_directory();
    auto fixture         = write_fixture(directory, "valid");
    Reader reader(fixture.path);
    // The directory fits in the first alignment unit, so the payload starts at 4096.
    const auto& objects = reader.directory().objects;
    if (objects.size() != 9 || reader.file_bytes() != kPayloadAlignment + 3844 ||
        !std::equal(kArtifactId.begin(), kArtifactId.end(), reader.artifact_id().begin(),
                    [](std::uint8_t expected, std::byte actual) {
                        return std::byte{expected} == actual;
                    })) {
        throw std::runtime_error("fixture framing mismatch");
    }

    for (std::size_t i = 0; i < objects.size(); ++i) {
        const auto handle = reader.find(directory.at("objects").at(i).at("id").get<std::string>());
        const auto offset = ninfer::artifact::object_offset(objects[i]);
        const auto bytes  = ninfer::artifact::object_bytes(objects[i]);
        if (handle.index != i) { throw std::runtime_error("fixture id index mismatch"); }
        // Validation checks the registered encoded size and alignment of each object.
        reader.validate_object(handle);
        const auto segments = reader.segments(offset, bytes);
        if (segments.size() != 1 || segments[0].file_index != 0 ||
            segments[0].file_offset != kPayloadAlignment + offset || segments[0].bytes != bytes) {
            throw std::runtime_error("fixture payload span mismatch");
        }
        const auto payload = reader.read_object(handle);
        if (payload.size() != bytes || !std::ranges::all_of(payload, [&](std::byte value) {
                return value == std::byte(i + 1);
            })) {
            throw std::runtime_error("fixture payload mismatch");
        }
    }
    expect_artifact_error([&] { (void)reader.find("missing"); }, "missing object");

    const auto& q5  = reader.geometry(reader.find("q5"));
    const auto& fp8 = reader.geometry(reader.find("fp8_row"));
    if (!std::holds_alternative<ninfer::artifact::ResourceObject>(objects.front()) ||
        q5.shape != std::vector<std::uint64_t>({2, 130}) ||
        q5.format != ninfer::QType::Q5_G64_FP16 || q5.layout != ninfer::QuantLayout::RowSplit ||
        fp8.shape != std::vector<std::uint64_t>({2, 4}) ||
        fp8.format != ninfer::QType::FP8_E4M3FN_ROW_BF16 ||
        fp8.layout != ninfer::QuantLayout::RowScale) {
        throw std::runtime_error("fixture object signature mismatch");
    }
}

// Encodings are interpreted only for requested objects: the entry opens, and validating the
// altered object fails.
void expect_invalid_object(const Json& directory, std::string_view suffix, std::string_view id) {
    auto fixture = write_fixture(directory, suffix);
    Reader reader(fixture.path);
    expect_artifact_error([&] { reader.validate_object(reader.find(id)); }, suffix);
}

void test_common_validation() {
    {
        auto directory                   = normative_directory();
        directory["objects"][5]["bytes"] = 527;
        expect_invalid_object(directory, "wrong_encoded_size", "q5");
    }
    {
        auto directory                    = normative_directory();
        directory["objects"][1]["offset"] = 257;
        expect_invalid_object(directory, "misaligned_offset", "bf16");
    }
    {
        auto directory                    = normative_directory();
        directory["objects"][8]["format"] = "nvfp4";
        expect_invalid_object(directory, "row_scale_format_mismatch", "fp8_row");
    }
    {
        auto directory                   = normative_directory();
        directory["objects"][8]["shape"] = Json::array({8});
        expect_invalid_object(directory, "row_scale_rank_mismatch", "fp8_row");
    }
    {
        constexpr std::array<std::uint8_t, 8> invalid_magic = {
            'I', 'N', 'V', 'A', 'L', 'I', 'D', '!',
        };
        auto fixture = write_fixture(normative_directory(), "invalid_magic", invalid_magic);
        expect_artifact_error([&] { Reader reader(fixture.path); }, "invalid magic");
    }
}

} // namespace

int main() {
    try {
        test_normative_fixture();
        test_common_validation();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
