// Volta wide-T row-scaled FP8 projections (the CUTLASS routes of attn_input_proj, gdn_input_proj,
// linear_add and linear_swiglu) against checkpoint-like weights.
//
// A row-scaled checkpoint stores scale = amax/448 per row, so code * scale spans each row's own
// range and small rows put it below the smallest normal FP16 value. The routes stage the codes
// unscaled in FP16 and apply the scale to the FP32 accumulator, so a BF16 output is one rounding
// of the exact result and an FP32 residual update carries only FP32 accumulation error. Against
// the FP64 oracle over the represented inputs:
//
//   BF16 output:   |y - ideal| <= ulp_bf16(ideal) / 2 + A
//   FP32 residual: |y - (r + ideal)| <= A + 2^-23 |r + ideal|
//
// where A = kAccumulationAllowance * sum_k |x_k * w_k| bounds the FP32 tensor-core accumulation.
// The same criteria are evaluated on exact emulations of the two alternatives, and the test
// requires both to fail them: scaling a BF16 GEMM output afterwards (two roundings) and folding the
// scale into the staged FP16 weights (FP16 rounding and subnormal loss of code * scale).

#include "ops/linear/fp8/fp8_cutlass_sm70.h"
#include "ops/linear/fp8/fp8_prepack_sm70.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_cutlass_sm70.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include "core/arena.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;
namespace qw = ninfer::test::quantized_weight;

// FP32 accumulation allowance relative to sum_k |x_k * w_k|: 16 units of 2^-24. Volta HMMA.884
// truncates its alignment and normalization; the FP32 residual cases print the measured worst case
// (1.6 units on these shapes).
constexpr double kAccumulationAllowance = 0x1p-20;
constexpr std::int32_t kTinyRowPeriod   = 16;

bool cuda_device_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

double ulp_bf16(double value) {
    const double magnitude = std::fabs(value);
    if (magnitude < 0x1p-126) { return 0x1p-133; }
    return std::ldexp(1.0, std::ilogb(magnitude) - 7);
}

double round_bf16(double value) {
    return static_cast<double>(qw::detail::bf16_to_f32(
        qw::detail::f32_to_bf16_rne(static_cast<float>(value))));
}

double round_fp16(double value) {
    return static_cast<double>(
        qw::detail::f16_to_f32(qw::detail::f32_to_f16(static_cast<float>(value))));
}

// BF16 activations kept inside the normal FP16 range, so the routes' FP16 staging is exact.
std::vector<std::uint16_t> make_activation(std::int32_t k, std::int32_t tokens,
                                           std::uint32_t seed) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(k) * tokens);
    std::uint64_t state = 0x5eedULL ^ (static_cast<std::uint64_t>(seed) << 20);
    for (std::uint16_t& value : bits) {
        double sample = qw::detail::standard_normal(state);
        sample        = std::copysign(std::clamp(std::fabs(sample), 0x1p-6, 4.0), sample);
        value         = qw::detail::f32_to_bf16_rne(static_cast<float>(sample));
    }
    return bits;
}

std::vector<std::int32_t> sampled_rows(std::int32_t n) {
    std::vector<std::int32_t> rows;
    for (std::int32_t i = 0; i < 48; ++i) {
        std::int32_t row = static_cast<std::int32_t>((static_cast<std::int64_t>(i) * 997 + 5) % n);
        if (row % kTinyRowPeriod == 0) { ++row; }
        rows.push_back(row);
    }
    for (std::int32_t i = 0; i < 16; ++i) {
        rows.push_back(kTinyRowPeriod * static_cast<std::int32_t>(
                                            (static_cast<std::int64_t>(i) * 131 + 7) %
                                            (n / kTinyRowPeriod)));
    }
    return rows;
}

std::vector<std::int32_t> sampled_tokens(std::int32_t tokens) {
    std::vector<std::int32_t> result;
    for (std::int32_t i = 0; i < 24; ++i) {
        result.push_back(
            static_cast<std::int32_t>(static_cast<std::int64_t>(i) * (tokens - 1) / 23));
    }
    return result;
}

struct Sample {
    std::int32_t row;
    std::int32_t token;
    bool tiny;
    double ideal;       // scale * sum x * code
    double magnitude;   // scale * sum |x * code|
    double two_roundings;
    double folded;      // sum x * FP16(code * scale), FP64 accumulation
};

std::vector<Sample> oracle(const qw::PackedWeight& packed, const std::vector<std::uint16_t>& x,
                           std::int32_t k, std::int32_t tokens) {
    std::vector<Sample> samples;
    std::vector<double> code(static_cast<std::size_t>(k));
    std::vector<double> folded(static_cast<std::size_t>(k));
    for (const std::int32_t row : sampled_rows(packed.weight.n)) {
        const double scale = static_cast<double>(qw::detail::bf16_to_f32(qw::detail::load_u16_le(
            packed.payload, packed.scale_plane_offset + static_cast<std::size_t>(row) * 2)));
        for (std::int32_t c = 0; c < k; ++c) {
            code[static_cast<std::size_t>(c)] = qw::detail::decode_e4m3fn(
                packed.payload[static_cast<std::size_t>(row) * k + c]);
            folded[static_cast<std::size_t>(c)] =
                round_fp16(static_cast<double>(static_cast<float>(code[c] * scale)));
        }
        for (const std::int32_t token : sampled_tokens(tokens)) {
            const std::uint16_t* activation = x.data() + static_cast<std::size_t>(token) * k;
            double sum = 0.0, magnitude = 0.0, folded_sum = 0.0;
            for (std::int32_t c = 0; c < k; ++c) {
                const double value = static_cast<double>(qw::detail::bf16_to_f32(activation[c]));
                sum += value * code[static_cast<std::size_t>(c)];
                magnitude += std::fabs(value * code[static_cast<std::size_t>(c)]);
                folded_sum += value * folded[static_cast<std::size_t>(c)];
            }
            Sample sample;
            sample.row       = row;
            sample.token     = token;
            sample.tiny      = row % kTinyRowPeriod == 0;
            sample.ideal     = sum * scale;
            sample.magnitude = magnitude * scale;
            // BF16 GEMM output of the unscaled accumulator, then the row scale in FP32 and a
            // second BF16 rounding (the pre-epilogue route).
            sample.two_roundings = round_bf16(static_cast<double>(
                static_cast<float>(round_bf16(sum)) * static_cast<float>(scale)));
            sample.folded = folded_sum;
            samples.push_back(sample);
        }
    }
    return samples;
}

struct Tally {
    std::int64_t samples    = 0;
    std::int64_t violations = 0;
    std::int64_t tiny_violations = 0;
    double worst_excess = 0.0; // largest |error| minus its bound, in units of the bound's ulp term
    double squared_error = 0.0;
    double squared_ideal = 0.0;

    void add(bool tiny, double error, double bound, double unit, double ideal) {
        ++samples;
        if (std::fabs(error) > bound) {
            ++violations;
            if (tiny) { ++tiny_violations; }
        }
        worst_excess = std::max(worst_excess, (std::fabs(error) - bound) / unit);
        squared_error += error * error;
        squared_ideal += ideal * ideal;
    }

    double relative_l2() const {
        return std::sqrt(squared_error / std::max(squared_ideal, 1e-300));
    }
};

void print(const std::string& label, const char* what, const Tally& tally) {
    std::printf("%s %-18s violations %5lld/%lld (tiny rows %lld), worst excess %+.3f, "
                "relative L2 %.3e\n",
                label.c_str(), what, static_cast<long long>(tally.violations),
                static_cast<long long>(tally.samples),
                static_cast<long long>(tally.tiny_violations), tally.worst_excess,
                tally.relative_l2());
}

struct DeviceWeight {
    GuardedDeviceBuffer payload;
    Weight weight;
};

DeviceWeight upload(const qw::PackedWeight& packed, bool swiglu) {
    DeviceWeight device{GuardedDeviceBuffer(packed.payload.size()), {}};
    device.payload.copy_from_host(packed.payload.data(), packed.payload.size());
    device.weight = packed.device_weight(device.payload.data());
    // Production weights are permuted into the Volta QPN stream at load.
    ops::detail::fp8_prepack_qpn_sm70(device.weight, nullptr, swiglu);
    cuda_synchronize();
    return device;
}

// BF16 projection route (attn/gdn input projections and the BF16 linear_add). `swiglu` runs the
// linear_swiglu gate/up GEMM on the interleaved layout instead.
int run_bf16_route(std::int32_t n, std::int32_t k, std::int32_t tokens, std::uint32_t seed,
                   bool swiglu) {
    const std::string label = std::string(swiglu ? "swiglu gate/up" : "projection") + " [" +
                              std::to_string(n) + "," + std::to_string(k) +
                              "] T=" + std::to_string(tokens);
    const qw::PackedWeight packed = qw::make_checkpoint_like_fp8_weight(n, k, seed, kTinyRowPeriod);
    const std::vector<std::uint16_t> x = make_activation(k, tokens, seed + 1U);
    DeviceWeight device                = upload(packed, swiglu);

    GuardedDeviceBuffer activation(x.size() * sizeof(std::uint16_t));
    activation.copy_from_host(x.data(), activation.bytes());
    GuardedDeviceBuffer output(static_cast<std::size_t>(n) * tokens * sizeof(std::uint16_t));
    output.fill(0xff);
    Tensor tx(activation.data(), DType::BF16, {k, tokens});
    Tensor tout(output.data(), DType::BF16, {n, tokens});
    const std::size_t capacity =
        swiglu ? ops::detail::fp8_linear_swiglu_cutlass_workspace_bytes(n, k, tokens)
               : ops::detail::fp8_cutlass_sm70_workspace_bytes(n, k, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
    if (swiglu) {
        ops::detail::fp8_linear_swiglu_cutlass_sm70_launch(tx, device.weight, tout, workspace,
                                                           nullptr);
    } else {
        ops::detail::fp8_cutlass_sm70_launch(tx, device.weight, tout, workspace, nullptr);
    }
    cuda_synchronize();
    std::vector<std::uint16_t> actual(static_cast<std::size_t>(n) * tokens);
    output.copy_to_host(actual.data(), output.bytes());

    Tally route, two_roundings, folded;
    for (const Sample& sample : oracle(packed, x, k, tokens)) {
        const double half_ulp = ulp_bf16(sample.ideal) / 2.0;
        const double bound    = half_ulp + kAccumulationAllowance * sample.magnitude;
        const double value    = static_cast<double>(qw::detail::bf16_to_f32(
            actual[static_cast<std::size_t>(sample.token) * n + sample.row]));
        route.add(sample.tiny, value - sample.ideal, bound, 2.0 * half_ulp, sample.ideal);
        two_roundings.add(sample.tiny, sample.two_roundings - sample.ideal, bound, 2.0 * half_ulp,
                          sample.ideal);
        folded.add(sample.tiny, round_bf16(sample.folded) - sample.ideal, bound, 2.0 * half_ulp,
                   sample.ideal);
    }
    print(label, "route", route);
    print(label, "two roundings", two_roundings);
    print(label, "FP16-folded scale", folded);

    int failures = 0;
    if (route.violations != 0) {
        std::cerr << label << ": outputs exceed half a BF16 ulp plus accumulation\n";
        ++failures;
    }
    // The criterion must separate one rounding from two and from FP16-folded weights.
    if (two_roundings.violations * 50 < two_roundings.samples) {
        std::cerr << label << ": the criterion does not separate two roundings\n";
        ++failures;
    }
    if (folded.tiny_violations == 0) {
        std::cerr << label << ": the criterion does not separate FP16-folded scales\n";
        ++failures;
    }
    failures += output.verify_guards(label + " output");
    failures += device.payload.verify_guards(label + " weight");
    return failures;
}

// FP32 residual update route (FP32-residual linear_add at T > 32).
int run_fp32_residual_route(std::int32_t n, std::int32_t k, std::int32_t tokens,
                            std::uint32_t seed) {
    const std::string label = "FP32 residual [" + std::to_string(n) + "," + std::to_string(k) +
                              "] T=" + std::to_string(tokens);
    const qw::PackedWeight packed = qw::make_checkpoint_like_fp8_weight(n, k, seed, kTinyRowPeriod);
    const std::vector<std::uint16_t> x = make_activation(k, tokens, seed + 1U);
    DeviceWeight device                = upload(packed, false);

    // A residual comparable to each row's projection, off the BF16 grid.
    std::vector<float> residual(static_cast<std::size_t>(n) * tokens);
    std::uint64_t state = 0xabcdULL ^ seed;
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t row = 0; row < n; ++row) {
            const double row_size = row % kTinyRowPeriod == 0 ? 1e-3 : 1.0;
            residual[static_cast<std::size_t>(token) * n + row] =
                static_cast<float>(row_size * qw::detail::standard_normal(state));
        }
    }
    GuardedDeviceBuffer activation(x.size() * sizeof(std::uint16_t));
    activation.copy_from_host(x.data(), activation.bytes());
    GuardedDeviceBuffer output(residual.size() * sizeof(float));
    output.copy_from_host(residual.data(), output.bytes());
    Tensor tx(activation.data(), DType::BF16, {k, tokens});
    Tensor tresidual(output.data(), DType::FP32, {n, tokens});
    WorkspaceArena workspace(
        std::max<std::size_t>(ops::detail::fp8_cutlass_sm70_workspace_bytes(n, k, tokens), 256));
    ops::detail::fp8_cutlass_sm70_residual_launch(tx, device.weight, tresidual, workspace, nullptr);
    cuda_synchronize();
    std::vector<float> actual(residual.size());
    output.copy_to_host(actual.data(), output.bytes());

    Tally route, folded;
    double worst_accumulation = 0.0; // |y - expected| / (2^-24 * sum |x * w|)
    for (const Sample& sample : oracle(packed, x, k, tokens)) {
        const std::size_t index = static_cast<std::size_t>(sample.token) * n + sample.row;
        const double initial    = static_cast<double>(residual[index]);
        const double expected   = initial + sample.ideal;
        const double unit       = kAccumulationAllowance * sample.magnitude;
        const double bound      = unit + 0x1p-23 * std::fabs(expected);
        const double error      = static_cast<double>(actual[index]) - expected;
        route.add(sample.tiny, error, bound, unit, expected);
        folded.add(sample.tiny, static_cast<double>(static_cast<float>(initial + sample.folded)) -
                                    expected,
                   bound, unit, expected);
        worst_accumulation = std::max(
            worst_accumulation,
            std::max(0.0, std::fabs(error) - 0x1p-24 * std::fabs(expected)) /
                (0x1p-24 * sample.magnitude));
    }
    print(label, "route", route);
    print(label, "FP16-folded scale", folded);
    std::printf("%s measured accumulation error up to %.1f x 2^-24 sum|x*w| (allowance %.0f)\n",
                label.c_str(), worst_accumulation, kAccumulationAllowance / 0x1p-24);

    int failures = 0;
    if (route.violations != 0) {
        std::cerr << label << ": updates exceed the FP32 accumulation allowance\n";
        ++failures;
    }
    if (folded.tiny_violations == 0) {
        std::cerr << label << ": the criterion does not separate FP16-folded scales\n";
        ++failures;
    }
    failures += output.verify_guards(label + " residual");
    failures += device.payload.verify_guards(label + " weight");
    return failures;
}

} // namespace

int main() {
    if (!cuda_device_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = 0;
        // Attention input and residual projections of the 27B at wide prompt widths: two M tiles,
        // the second partial.
        failures += run_bf16_route(14336, 5120, 200, 901U, false);
        failures += run_bf16_route(5120, 17408, 200, 903U, false);
        failures += run_bf16_route(4096, 5120, 72, 905U, true);
        failures += run_fp32_residual_route(5120, 6144, 200, 907U);
        failures += run_fp32_residual_route(5120, 17408, 72, 909U);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " FP8 row-scaled Volta routes\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FP8 row-scaled Volta routes: " << error.what() << '\n';
        return 1;
    }
}
