#include "artifact/binder.h"
#include "artifact/framing.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact/typed_binding.h"
#include "artifact_fixture.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {

using ninfer::artifact::NumericFormat;
using ninfer::artifact::TensorPlacement;

constexpr std::array<std::byte, 3> kResource = {
    std::byte{1},
    std::byte{1},
    std::byte{1},
};
constexpr std::array<std::byte, 4> kTensor = {
    std::byte{2},
    std::byte{2},
    std::byte{2},
    std::byte{2},
};
constexpr std::array<std::byte, 8> kSecondTensor = {
    std::byte{3}, std::byte{3}, std::byte{3}, std::byte{3},
    std::byte{3}, std::byte{3}, std::byte{3}, std::byte{3},
};
constexpr std::size_t kFp8TensorBytes = 260;
// The last direct block starts with the second tensor and ends the file after the FP8 tensor.
constexpr std::size_t kTailReadBytes = 256 + kFp8TensorBytes;

ninfer::test::artifact_fixture::TemporaryArtifact write_fixture() {
    using Json = ninfer::test::artifact_fixture::Json;
    return ninfer::test::artifact_fixture::write_fixture(
        {
            {"components",
             {{"text",
               {{"config", Json::object()},
                {"resources", {{"test.json", "resource/text/test.json"}}}}}}},
            {"objects", Json::array({
                            {{"id", "resource/text/test.json"},
                             {"kind", "resource"},
                             {"encoding", "raw_bytes_v1"},
                             {"offset", 0},
                             {"bytes", 3}},
                            {{"id", "weight/test"},
                             {"kind", "tensor"},
                             {"shape", {2}},
                             {"format", "bf16"},
                             {"layout", "contiguous_le_v1"},
                             {"offset", 256},
                             {"bytes", 4}},
                            {{"id", "weight/second"},
                             {"kind", "tensor"},
                             {"shape", {4}},
                             {"format", "bf16"},
                             {"layout", "contiguous_le_v1"},
                             {"offset", 8192},
                             {"bytes", 8}},
                            {{"id", "weight/fp8"},
                             {"kind", "tensor"},
                             {"shape", {2, 4}},
                             {"format", "fp8_e4m3fn_row_bf16"},
                             {"layout", "row_scale_v1"},
                             {"offset", 8448},
                             {"bytes", kFp8TensorBytes}},
                        })},
            {"bindings",
             {{"text/test", {{"object", "weight/test"}}},
              {"text/second", {{"object", "weight/second"}}},
              {"text/fp8", {{"object", "weight/fp8"}}}}},
            {"uses", Json::array()},
        },
        "materialization");
}

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

template <typename Function>
void expect_artifact_error(Function&& function, const char* message) {
    try {
        function();
    } catch (const ninfer::artifact::ArtifactError&) { return; }
    throw std::runtime_error(message);
}

} // namespace

int main() {
    try {
        auto fixture = write_fixture();
        ninfer::artifact::Reader reader(fixture.path);
        {
            ninfer::artifact::Binder binder(reader);
            expect_artifact_error(
                [&] {
                    (void)ninfer::artifact::bind_tensor(binder, "text/fp8",
                                                        NumericFormat::FP8_E4M3FN_ROW_BF16S,
                                                        {4, 2}, TensorPlacement::ValidateOnly);
                },
                "validate-only binding accepted a shape that differs from the artifact");
        }
        ninfer::artifact::Binder validation_binder(reader);
        const auto validated_resource = validation_binder.resource("text", "test.json");
        (void)ninfer::artifact::bind_tensor(validation_binder, "text/test", NumericFormat::BF16,
                                            {2}, TensorPlacement::ValidateOnly);
        const auto retained_tensor = ninfer::artifact::bind_tensor(
            validation_binder, "text/second", NumericFormat::BF16, {4}, TensorPlacement::Device);
        (void)ninfer::artifact::bind_tensor(validation_binder, "text/fp8",
                                            NumericFormat::FP8_E4M3FN_ROW_BF16S, {2, 4},
                                            TensorPlacement::ValidateOnly);
        const auto validation_plan = std::move(validation_binder).finish();
        require(validation_plan.object_count == 4 && validation_plan.host_objects.size() == 1 &&
                    validation_plan.host_objects[0].object == validated_resource &&
                    validation_plan.device_objects.size() == 1 &&
                    validation_plan.device_objects[0].object == retained_tensor &&
                    validation_plan.device_capacity_bytes == kSecondTensor.size(),
                "validate-only tensor was included in the materialization plan");

        int device_count              = 0;
        const cudaError_t count_error = cudaGetDeviceCount(&device_count);
        if (cuda_unavailable(count_error)) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        CUDA_CHECK(count_error);
        if (device_count == 0) {
            std::cout << "SKIP: no CUDA devices\n";
            return 77;
        }

        ninfer::artifact::Binder binder(reader);

        const auto resource = binder.resource("text", "test.json");
        const auto second   = ninfer::artifact::bind_tensor(binder, "text/second",
                                                            NumericFormat::BF16, {4},
                                                            TensorPlacement::Device);

        // Bind in the opposite order from the artifact. Placements follow the artifact order, and
        // the second and FP8 tensors share one direct-I/O block that is scattered to both parents.
        const auto tensor = ninfer::artifact::bind_tensor(binder, "text/test", NumericFormat::BF16,
                                                          {2}, TensorPlacement::Device);
        const auto fp8    = ninfer::artifact::bind_tensor(binder, "text/fp8",
                                                          NumericFormat::FP8_E4M3FN_ROW_BF16S,
                                                          {2, 4}, TensorPlacement::Device);

        ninfer::artifact::MaterializationPlan plan = std::move(binder).finish();
        require(plan.object_count == 4 && plan.host_objects.size() == 1 &&
                    plan.device_objects.size() == 3 && plan.device_capacity_bytes == 772,
                "binder produced the wrong materialization plan");
        const std::uint64_t device_capacity = plan.device_capacity_bytes;

        ninfer::DeviceContext device(0);
        auto materialized = ninfer::artifact::materialize(reader, std::move(plan), device);

        std::array<std::byte, kTensor.size()> copied{};
        CUDA_CHECK(cudaMemcpy(copied.data(), materialized.device_parent(tensor).data,
                              copied.size(), cudaMemcpyDeviceToHost));
        require(copied == kTensor, "device tensor payload differs from the artifact");
        std::array<std::byte, kSecondTensor.size()> second_copied{};
        CUDA_CHECK(cudaMemcpy(second_copied.data(), materialized.device_parent(second).data,
                              second_copied.size(), cudaMemcpyDeviceToHost));
        require(second_copied == kSecondTensor,
                "second device tensor payload differs from the artifact");
        std::array<std::byte, kFp8TensorBytes> fp8_copied{};
        CUDA_CHECK(cudaMemcpy(fp8_copied.data(), materialized.device_parent(fp8).data,
                              fp8_copied.size(), cudaMemcpyDeviceToHost));
        require(std::all_of(fp8_copied.begin(), fp8_copied.end(),
                            [](std::byte value) { return value == std::byte{4}; }),
                "FP8 device tensor payload differs from the artifact");

        const ninfer::Weight fp8_weight = ninfer::artifact::materialized_weight(
            materialized, fp8, NumericFormat::FP8_E4M3FN_ROW_BF16S, 2, 4);
        require(fp8_weight.qtype == ninfer::QType::FP8_E4M3FN_ROW_BF16S &&
                    fp8_weight.layout == ninfer::QuantLayout::RowScale &&
                    fp8_weight.scale_dtype == ninfer::DType::BF16 && fp8_weight.n == 2 &&
                    fp8_weight.k == 4 && fp8_weight.group == 4 && fp8_weight.group_size == 4 &&
                    fp8_weight.payload == materialized.device_parent(fp8).data &&
                    fp8_weight.qdata == fp8_weight.payload && fp8_weight.qhigh == nullptr &&
                    fp8_weight.scales == static_cast<const std::byte*>(fp8_weight.payload) + 256 &&
                    fp8_weight.payload_bytes == kFp8TensorBytes,
                "materialized FP8 Weight metadata is incomplete");

        const auto retained = materialized.host_bytes(resource);
        require(std::equal(retained.begin(), retained.end(), kResource.begin(), kResource.end()),
                "retained resource payload differs from the artifact");

        // The Binder read the resource. Direct reads cover whole 4 KiB blocks, except the last
        // one, which ends with the file.
        const auto& stats = materialized.stats();
        require(stats.device_object_count == 3 && stats.host_object_count == 1 &&
                    stats.h2d_bytes == kTensor.size() + kSecondTensor.size() + kFp8TensorBytes &&
                    stats.device_capacity_bytes == device_capacity &&
                    stats.retained_host_bytes == kResource.size() &&
                    stats.file_bytes == std::filesystem::file_size(fixture.path) &&
                    stats.read_bytes == kResource.size() + ninfer::artifact::kPayloadAlignment +
                                            kTailReadBytes,
                "materialization statistics are incomplete");
        require(materialized.device_arena().capacity() == device_capacity &&
                    materialized.device_arena().used() == device_capacity,
                "materialized tensors do not own the planned device backing");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
