#include "ninfer/ops/linear_add.h"
#include "core/device.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"
#include "ops/linear/nvfp4/nvfp4_prepack_sm70.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

constexpr double kBf16UnitRoundoff = 1.0 / 256.0;
constexpr ReductionCriterion kA16Tolerance{
    kBf16UnitRoundoff,
    kBf16UnitRoundoff,
    2.0 * kBf16UnitRoundoff,
};
constexpr ReductionCriterion kA4Tolerance{0.16, kBf16UnitRoundoff, 0.16};

struct Invocation {
    std::int32_t tokens;
    ops::LinearPolicy policy;
};

std::vector<std::int32_t> sampled_indices(std::int32_t extent) {
    std::vector<std::int32_t> result;
    for (const std::int32_t index :
         {0, 1, extent / 4, extent / 2, (3 * extent) / 4, extent - 2, extent - 1}) {
        if (index >= 0 && index < extent &&
            std::find(result.begin(), result.end(), index) == result.end()) {
            result.push_back(index);
        }
    }
    return result;
}

std::vector<std::uint16_t> make_activation(std::int32_t rows, std::int32_t tokens,
                                           std::uint32_t seed) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t row = 0; row < rows; ++row) {
            std::uint32_t value = seed ^ (static_cast<std::uint32_t>(row) * 0x9e3779b9U) ^
                                  (static_cast<std::uint32_t>(token) * 0x85ebca6bU);
            value ^= value >> 16;
            value *= 0x7feb352dU;
            value ^= value >> 15;
            const float represented =
                static_cast<float>(static_cast<int>(value & 0xffU) - 128) * (1.0F / 256.0F);
            result[static_cast<std::size_t>(token) * rows + row] = f32_to_bf16(represented);
        }
    }
    return result;
}

std::vector<std::uint16_t> make_residual(std::int32_t rows, std::int32_t tokens,
                                         std::uint32_t seed) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t row = 0; row < rows; ++row) {
            const std::uint32_t coordinate = static_cast<std::uint32_t>(row) * 23U +
                                             static_cast<std::uint32_t>(token) * 41U + seed * 7U;
            const float represented =
                static_cast<float>(static_cast<int>(coordinate & 0xffU) - 128) * (1.0F / 128.0F);
            result[static_cast<std::size_t>(token) * rows + row] = f32_to_bf16(represented);
        }
    }
    return result;
}

int verify_preserved(const GuardedDeviceBuffer& device, std::span<const std::uint8_t> expected,
                     std::string_view label) {
    std::vector<std::uint8_t> actual(expected.size());
    device.copy_to_host(actual.data(), actual.size());
    if (std::equal(actual.begin(), actual.end(), expected.begin(), expected.end())) { return 0; }
    std::cerr << label << ": payload was modified\n";
    return 1;
}

int run_shape(std::int32_t n, std::int32_t k, std::uint32_t seed, bool prepack) {
#ifdef NINFER_VOLTA_BUILD
    // 5 is the K=4 verify width; 9..32 cover the two- and four-tile QPN2 buckets.
    const std::array invocations{
        Invocation{1, ops::LinearPolicy::A16Only},  Invocation{4, ops::LinearPolicy::A16Only},
        Invocation{5, ops::LinearPolicy::A16Only},  Invocation{8, ops::LinearPolicy::A16Only},
        Invocation{9, ops::LinearPolicy::A16Only},  Invocation{16, ops::LinearPolicy::A16Only},
        Invocation{32, ops::LinearPolicy::A16Only},
    };
#else
    const std::int32_t first_a4 = k == 6144 ? 7 : 8;
    const std::array invocations{
        Invocation{1, ops::LinearPolicy::A16Only},
        Invocation{4, ops::LinearPolicy::A16Only},
        Invocation{first_a4, ops::LinearPolicy::AllowA4},
        Invocation{17, ops::LinearPolicy::AllowA4},
        Invocation{1024, ops::LinearPolicy::AllowA4},
        Invocation{8, ops::LinearPolicy::AllowA4},
        Invocation{16, ops::LinearPolicy::AllowA4},
        Invocation{32, ops::LinearPolicy::AllowA4},
        Invocation{64, ops::LinearPolicy::AllowA4},
        Invocation{96, ops::LinearPolicy::AllowA4},
        Invocation{128, ops::LinearPolicy::AllowA4},
        Invocation{129, ops::LinearPolicy::AllowA4},
    };
#endif
    constexpr std::int32_t kMaximumTokens = 1024;
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    quantized_weight::PackedWeight host_weight =
        quantized_weight::make_patterned_weight(QType::NVFP4, n, k, seed, options);
    const std::vector<std::int32_t> rows = sampled_indices(n);
    const std::vector<float> materialized_weight =
        quantized_weight::materialize_rows_fp32(host_weight, rows);
    const std::vector<std::uint16_t> activation = make_activation(k, kMaximumTokens, seed + 1U);
    const std::vector<std::uint16_t> initial_residual = make_residual(n, kMaximumTokens, seed + 2U);

    GuardedDeviceBuffer device_activation(activation.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(activation.data(), device_activation.bytes());
    GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    Weight weight = host_weight.device_weight(device_weight.data());
    // Production loads prepack NVFP4 weights for the Volta QPN kernel (bindings.cpp), which then
    // runs the prepacked kernel; the prepack rewrites the payload in place.
    std::vector<std::uint8_t> expected_weight = host_weight.payload;
#ifdef NINFER_VOLTA_BUILD
    if (prepack) {
        ops::detail::nvfp4_prepack_qpn_sm70(weight);
        device_weight.copy_to_host(expected_weight.data(), expected_weight.size());
    }
#endif

    int failures = 0;
    for (const Invocation invocation : invocations) {
        const std::size_t output_words = static_cast<std::size_t>(n) * invocation.tokens;
        GuardedDeviceBuffer output(output_words * sizeof(std::uint16_t));
        output.copy_from_host(initial_residual.data(), output.bytes());
        Tensor x(device_activation.data(), DType::BF16, {k, invocation.tokens});
        Tensor residual(output.data(), DType::BF16, {n, invocation.tokens});
        const std::size_t capacity = ops::linear_add_workspace_capacity_bytes(
            QType::NVFP4, n, k, invocation.policy, invocation.tokens, invocation.tokens);
        WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
        ops::linear_add(x, weight, residual, invocation.policy, workspace, nullptr);
        cuda_check(cudaDeviceSynchronize(), "synchronize NVFP4 linear_add");

        if (invocation.tokens == 128) {
            cudaStream_t stream;
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            ops::linear_add(x, weight, residual, invocation.policy, workspace, stream);
            CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            for (int replay = 0; replay < 2; ++replay) {
                CUDA_CHECK(cudaMemcpyAsync(output.data(), initial_residual.data(), output.bytes(),
                    cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaGraphLaunch(executable, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            CUDA_CHECK(cudaGraphExecDestroy(executable));
            CUDA_CHECK(cudaGraphDestroy(graph));
            CUDA_CHECK(cudaStreamDestroy(stream));
        }

        const bool a4           = invocation.policy == ops::LinearPolicy::AllowA4;
        const std::string label = "NVFP4 linear_add [" + std::to_string(n) + "," +
                                  std::to_string(k) + "] " + (a4 ? "A4" : "A16") +
                                  " T=" + std::to_string(invocation.tokens);
        if (workspace.peak_used() != capacity) {
            std::cerr << label << ": workspace query/execution high-water mismatch\n";
            ++failures;
        }
        failures += output.verify_guards(label);

        std::vector<std::uint16_t> actual_bits(output_words);
        output.copy_to_host(actual_bits.data(), output.bytes());
        const std::vector<std::int32_t> tokens = sampled_indices(invocation.tokens);
        std::vector<double> actual;
        std::vector<double> expected;
        actual.reserve(rows.size() * tokens.size());
        expected.reserve(rows.size() * tokens.size());
        for (std::size_t sampled_row = 0; sampled_row < rows.size(); ++sampled_row) {
            const std::int32_t row = rows[sampled_row];
            const float* weight_row =
                materialized_weight.data() + sampled_row * static_cast<std::size_t>(k);
            for (const std::int32_t token : tokens) {
                double sum = 0.0;
                const std::uint16_t* activation_row =
                    activation.data() + static_cast<std::size_t>(token) * k;
                for (std::int32_t column = 0; column < k; ++column) {
                    sum += static_cast<double>(weight_row[column]) *
                           static_cast<double>(bf16_to_f32(activation_row[column]));
                }
                const std::size_t index = static_cast<std::size_t>(token) * n + row;
                actual.push_back(static_cast<double>(bf16_to_f32(actual_bits[index])));
                expected.push_back(sum + static_cast<double>(bf16_to_f32(initial_residual[index])));
            }
        }
        failures += verify_reduction(label, actual, expected, a4 ? kA4Tolerance : kA16Tolerance);
    }

    failures += device_activation.verify_guards("NVFP4 linear_add activation");
    failures += device_weight.verify_guards("NVFP4 linear_add weight");
    failures += verify_preserved(
        device_activation,
        std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(activation.data()),
                                      activation.size() * sizeof(std::uint16_t)),
        "NVFP4 linear_add activation");
    failures += verify_preserved(device_weight, expected_weight, "NVFP4 linear_add weight");
    return failures;
}


#ifdef NINFER_VOLTA_BUILD
// FP32 residual stream: both routes accumulate in FP32 and round the sum once, to FP32 (measured
// relative L2 below 4e-6 on these shapes), so the criterion sits far below the BF16 unit roundoff:
// a BF16-rounded residual or a BF16-materialized projection (about 1e-3 and 1.6e-3 relative L2
// here) fails it.
constexpr ReductionCriterion kFp32ResidualTolerance{1.0e-4, 1.0 / 16384.0, 1.0 / 16384.0};

// FP32 residual stream: the same weight and activation update an FP32 residual. The fused QPN
// epilogue (T <= 32) and the wide GEMM route (T > 32, accumulating onto the residual from its FP32
// accumulators) both round the sum once, to FP32.
int run_fp32_residual_shape(std::int32_t n, std::int32_t k, std::int32_t first_w4a4,
                            std::uint32_t seed) {
    struct Call {
        std::int32_t tokens;
        ops::LinearPolicy policy;
    };
    // The permissive policy stays on the FP32-accumulating route below the W4A4 width.
    const std::array<Call, 5> calls{{{1, ops::LinearPolicy::A16Only},
                                     {5, ops::LinearPolicy::A16Only},
                                     {first_w4a4 - 1, ops::LinearPolicy::AllowA4},
                                     {33, ops::LinearPolicy::A16Only},
                                     {64, ops::LinearPolicy::A16Only}}};
    constexpr std::int32_t kMaximumTokens = 64;
    quantized_weight::PackedWeight host_weight = [&] {
        quantized_weight::PatternedWeightOptions options;
        options.weight_scale_divisor = 0.125F;
        options.input_scale_divisor  = 3.5F;
        return quantized_weight::make_patterned_weight(QType::NVFP4, n, k, seed, options);
    }();
    const std::vector<std::int32_t> rows = sampled_indices(n);
    const std::vector<float> materialized_weight =
        quantized_weight::materialize_rows_fp32(host_weight, rows);
    std::vector<std::uint16_t> activation = make_activation(k, kMaximumTokens, seed + 1U);
    // Token 2 projects to exact zero, so its residual column must come back bit-identical.
    constexpr std::int32_t kZeroToken = 2;
    std::fill_n(activation.begin() + static_cast<std::ptrdiff_t>(kZeroToken) * k, k,
                f32_to_bf16(0.0F));
    const std::vector<std::uint16_t> residual_bits = make_residual(n, kMaximumTokens, seed + 2U);
    std::vector<float> initial_residual(residual_bits.size());
    for (std::size_t i = 0; i < residual_bits.size(); ++i) {
        // Values off the BF16 grid, so an FP32 residual that were rounded to BF16 would show.
        initial_residual[i] = bf16_to_f32(residual_bits[i]) * 1.0009765625F;
    }

    GuardedDeviceBuffer device_activation(activation.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(activation.data(), device_activation.bytes());
    GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    Weight weight = host_weight.device_weight(device_weight.data());
    ops::detail::nvfp4_prepack_qpn_sm70(weight);

    int failures = 0;
    {
        // From the W4A4 width the permissive policy takes the W4A4 route, which writes BF16:
        // an FP32 residual is rejected before any launch and keeps its values.
        const std::int32_t tokens = first_w4a4;
        const std::size_t words   = static_cast<std::size_t>(n) * tokens;
        GuardedDeviceBuffer output(words * sizeof(float));
        output.copy_from_host(initial_residual.data(), output.bytes());
        Tensor x(device_activation.data(), DType::BF16, {k, tokens});
        Tensor residual(output.data(), DType::FP32, {n, tokens});
        const std::size_t capacity = ops::linear_add_workspace_capacity_bytes(
            QType::NVFP4, n, k, ops::LinearPolicy::AllowA4, tokens, tokens);
        WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
        const std::string label = std::string("NVFP4 linear_add FP32 residual W4A4 route [") +
                                  std::to_string(n) + "," + std::to_string(k) + "]";
        try {
            ops::linear_add(x, weight, residual, ops::LinearPolicy::AllowA4, workspace, nullptr);
            std::cerr << label << ": an FP32 residual was accepted\n";
            ++failures;
        } catch (const std::invalid_argument&) {
        }
        cuda_check(cudaDeviceSynchronize(), "synchronize rejected FP32-residual linear_add");
        std::vector<float> after(words);
        output.copy_to_host(after.data(), output.bytes());
        failures += verify_exact((label + " residual unchanged").c_str(), after,
                                 std::vector<float>(initial_residual.begin(),
                                                    initial_residual.begin() +
                                                        static_cast<std::ptrdiff_t>(words)));
    }
    for (const Call& call : calls) {
        const std::int32_t tokens       = call.tokens;
        const std::size_t output_words = static_cast<std::size_t>(n) * tokens;
        GuardedDeviceBuffer output(output_words * sizeof(float));
        output.copy_from_host(initial_residual.data(), output.bytes());
        Tensor x(device_activation.data(), DType::BF16, {k, tokens});
        Tensor residual(output.data(), DType::FP32, {n, tokens});
        const std::size_t capacity = ops::linear_add_workspace_capacity_bytes(
            QType::NVFP4, n, k, call.policy, tokens, tokens);
        WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
        ops::linear_add(x, weight, residual, call.policy, workspace, nullptr);
        cuda_check(cudaDeviceSynchronize(), "synchronize FP32-residual linear_add");

        const std::string label =
            std::string("NVFP4") + " linear_add FP32 residual [" + std::to_string(n) + "," +
            std::to_string(k) + "] T=" + std::to_string(tokens) +
            (call.policy == ops::LinearPolicy::A16Only ? "" : " AllowA4");
        if (workspace.peak_used() > std::max<std::size_t>(capacity, 256)) {
            std::cerr << label << ": workspace overrun\n";
            ++failures;
        }
        failures += output.verify_guards(label);
        std::vector<float> actual_values(output_words);
        output.copy_to_host(actual_values.data(), output.bytes());
        std::vector<double> actual;
        std::vector<double> expected;
        for (std::size_t sampled_row = 0; sampled_row < rows.size(); ++sampled_row) {
            const std::int32_t row = rows[sampled_row];
            const float* weight_row =
                materialized_weight.data() + sampled_row * static_cast<std::size_t>(k);
            for (const std::int32_t token : sampled_indices(tokens)) {
                double sum = 0.0;
                const std::uint16_t* activation_row =
                    activation.data() + static_cast<std::size_t>(token) * k;
                for (std::int32_t column = 0; column < k; ++column) {
                    sum += static_cast<double>(weight_row[column]) *
                           static_cast<double>(bf16_to_f32(activation_row[column]));
                }
                const std::size_t index = static_cast<std::size_t>(token) * n + row;
                actual.push_back(static_cast<double>(actual_values[index]));
                expected.push_back(sum + static_cast<double>(initial_residual[index]));
            }
        }
        failures += verify_reduction(label, actual, expected, kFp32ResidualTolerance);
        if (tokens > kZeroToken) {
            const auto column = [&](const std::vector<float>& values) {
                const auto begin = values.begin() + static_cast<std::ptrdiff_t>(kZeroToken) * n;
                return std::vector<float>(begin, begin + n);
            };
            failures += verify_exact((label + " zero-projection column").c_str(),
                                     column(actual_values), column(initial_residual));
        }
    }
    failures += device_activation.verify_guards("FP32-residual linear_add activation");
    failures += device_weight.verify_guards("FP32-residual linear_add weight");
    return failures;
}

// The W4A4 route reads the checkpoint-native planes only. A QPN-prepacked weight, the layout of
// every NVFP4 down and mixer output projection loaded on Volta, must be refused before any launch
// and leave the residual as it was.
int run_prepacked_w4a4_refusal(std::int32_t n, std::int32_t k, std::int32_t first_w4a4,
                               std::uint32_t seed) {
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    const quantized_weight::PackedWeight host_weight =
        quantized_weight::make_patterned_weight(QType::NVFP4, n, k, seed, options);
    const std::vector<std::uint16_t> activation = make_activation(k, first_w4a4, seed + 1U);
    const std::vector<std::uint16_t> initial    = make_residual(n, first_w4a4, seed + 2U);

    GuardedDeviceBuffer device_activation(activation.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(activation.data(), device_activation.bytes());
    GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    Weight weight = host_weight.device_weight(device_weight.data());
    ops::detail::nvfp4_prepack_qpn_sm70(weight);
    GuardedDeviceBuffer output(initial.size() * sizeof(std::uint16_t));
    output.copy_from_host(initial.data(), output.bytes());

    Tensor x(device_activation.data(), DType::BF16, {k, first_w4a4});
    Tensor residual(output.data(), DType::BF16, {n, first_w4a4});
    const std::size_t capacity = ops::linear_add_workspace_capacity_bytes(
        QType::NVFP4, n, k, ops::LinearPolicy::AllowA4, first_w4a4, first_w4a4);
    WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
    const std::string label = "NVFP4 linear_add W4A4 route, prepacked [" + std::to_string(n) + "," +
                              std::to_string(k) + "]";
    bool refused            = false;
    try {
        ops::linear_add(x, weight, residual, ops::LinearPolicy::AllowA4, workspace, nullptr);
    } catch (const std::invalid_argument& error) {
        refused = true;
        std::cout << label << ": refused (" << error.what() << ")\n";
    }
    int failures = 0;
    if (!refused) {
        std::cerr << label << ": a prepacked weight was accepted\n";
        ++failures;
    }
    cuda_check(cudaDeviceSynchronize(), "synchronize refused NVFP4 linear_add");
    std::vector<std::uint16_t> after(initial.size());
    output.copy_to_host(after.data(), output.bytes());
    failures += verify_exact((label + " residual unchanged").c_str(), after, initial);
    failures += output.verify_guards(label);
    return failures;
}
#endif
} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run_shape(5120, 6144, 811U, false);
    failures += run_shape(5120, 17408, 821U, false);
#ifdef NINFER_VOLTA_BUILD
    failures += run_shape(5120, 6144, 812U, true);
    failures += run_shape(5120, 17408, 822U, true);
    failures += run_fp32_residual_shape(5120, 6144, 7, 815U);
    failures += run_fp32_residual_shape(5120, 17408, 8, 825U);
    failures += run_prepacked_w4a4_refusal(5120, 6144, 7, 817U);
#endif
    std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4 linear_add\n";
    return failures == 0 ? 0 : 1;
}
