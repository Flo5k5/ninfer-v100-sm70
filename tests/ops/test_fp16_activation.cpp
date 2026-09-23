// The Volta fp16 activation domain: wherever linear_swiglu / linear_add report FP16 support, an
// FP16 activation (the BF16 value converted to fp16) must give results bit-identical to the BF16
// tensor, and an FP16 SwiGLU output must equal the BF16 output converted to fp16.
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"

#include "core/arena.h"
#include "core/device.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/fp8/fp8_prepack_sm70.h"
#include "ops/linear/nvfp4/nvfp4_prepack_sm70.h"
#endif

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kHidden       = 5120;
constexpr std::int32_t kIntermediate = 17408;

enum class Layout { Fp8, Fp8SwiGlu, Nvfp4Plain, Nvfp4SwiGlu };

std::uint16_t fp16_of_bf16(std::uint16_t bf16_bits) {
    const std::uint32_t widened = static_cast<std::uint32_t>(bf16_bits) << 16;
    float value                 = 0.0F;
    std::memcpy(&value, &widened, sizeof(value));
    const __half converted = __float2half_rn(value);
    std::uint16_t bits     = 0;
    std::memcpy(&bits, &converted, sizeof(bits));
    return bits;
}

std::vector<std::uint16_t> random_bf16(std::size_t count, std::uint32_t seed, float scale) {
    std::vector<std::uint16_t> bits(count);
    for (auto& value : bits) {
        seed             = seed * 1664525U + 1013904223U;
        const float real = (static_cast<float>(seed >> 8) * (2.0F / 16777216.0F) - 1.0F) * scale;
        std::uint32_t raw = 0;
        std::memcpy(&raw, &real, sizeof(raw));
        value = static_cast<std::uint16_t>((raw + 0x7FFFU + ((raw >> 16) & 1U)) >> 16);
    }
    return bits;
}

std::vector<std::uint16_t> to_fp16(const std::vector<std::uint16_t>& bf16) {
    std::vector<std::uint16_t> out(bf16.size());
    std::transform(bf16.begin(), bf16.end(), out.begin(), fp16_of_bf16);
    return out;
}

struct DeviceWords {
    DeviceBuffer buffer;
    explicit DeviceWords(const std::vector<std::uint16_t>& host)
        : buffer(host.size() * sizeof(std::uint16_t)) {
        CUDA_CHECK(cudaMemcpy(buffer.p, host.data(), host.size() * 2, cudaMemcpyHostToDevice));
    }
    explicit DeviceWords(std::size_t count) : buffer(count * sizeof(std::uint16_t)) {}
    std::vector<std::uint16_t> read(std::size_t count) const {
        std::vector<std::uint16_t> host(count);
        CUDA_CHECK(cudaMemcpy(host.data(), buffer.p, count * 2, cudaMemcpyDeviceToHost));
        return host;
    }
};

int compare(const std::string& label, const std::vector<std::uint16_t>& actual,
            const std::vector<std::uint16_t>& expected) {
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::cerr << label << ": element " << i << " is 0x" << std::hex << actual[i]
                      << ", expected 0x" << expected[i] << std::dec << '\n';
            return 1;
        }
    }
    return 0;
}

struct DeviceWeight {
    DeviceBuffer payload;
    Weight weight;
};

DeviceWeight make_weight(QType qtype, std::int32_t n, std::int32_t k, std::uint32_t seed,
                         Layout layout) {
    test::quantized_weight::PatternedWeightOptions options;
    if (qtype == QType::NVFP4) {
        options.weight_scale_divisor = 2688.0F;
        options.input_scale_divisor  = 3.5F;
    }
    const auto host = test::quantized_weight::make_patterned_weight(qtype, n, k, seed, options);
    DeviceWeight result{DeviceBuffer(host.payload.size()), {}};
    CUDA_CHECK(cudaMemcpy(result.payload.p, host.payload.data(), host.payload.size(),
                          cudaMemcpyHostToDevice));
    result.weight = host.device_weight(result.payload.p);
#ifdef NINFER_VOLTA_BUILD
    if (layout == Layout::Fp8 || layout == Layout::Fp8SwiGlu) {
        ops::detail::fp8_prepack_qpn_sm70(result.weight, nullptr, layout == Layout::Fp8SwiGlu);
    } else {
        ops::detail::nvfp4_prepack_qpn_sm70(result.weight, nullptr,
                                            layout == Layout::Nvfp4SwiGlu);
    }
#else
    (void)layout;
#endif
    return result;
}

int run_swiglu(const std::string& label, QType qtype, Layout layout, std::int32_t tokens) {
    const ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
    DeviceWeight weight = make_weight(qtype, 2 * kIntermediate, kHidden, 3301U, layout);
    if (!ops::linear_swiglu_fp16_activation_supported(weight.weight, policy, tokens)) {
        std::cerr << label << ": fp16 activation domain unexpectedly unsupported\n";
        return 1;
    }
    const std::size_t in_count  = static_cast<std::size_t>(kHidden) * tokens;
    const std::size_t out_count = static_cast<std::size_t>(kIntermediate) * tokens;
    const auto x_bf16           = random_bf16(in_count, 3303U + tokens, 3.0F);
    DeviceWords x_bf16_device(x_bf16);
    DeviceWords x_fp16_device(to_fp16(x_bf16));
    WorkspaceArena workspace(std::max<std::size_t>(
        ops::linear_swiglu_workspace_capacity_bytes(qtype, 2 * kIntermediate, kHidden, policy,
                                                    tokens, tokens),
        256));

    const auto run = [&](const DeviceWords& input, DType in_dtype, DType out_dtype) {
        DeviceWords output(out_count);
        Tensor x(input.buffer.p, in_dtype, {kHidden, tokens});
        Tensor out(output.buffer.p, out_dtype, {kIntermediate, tokens});
        workspace.reset();
        ops::linear_swiglu(x, weight.weight, out, policy, workspace, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        return output.read(out_count);
    };
    const auto reference = run(x_bf16_device, DType::BF16, DType::BF16);
    const auto reference_fp16 = to_fp16(reference);
    int failures = 0;
    failures += compare(label + " fp16 x", run(x_fp16_device, DType::FP16, DType::BF16), reference);
    failures +=
        compare(label + " fp16 out", run(x_bf16_device, DType::BF16, DType::FP16), reference_fp16);
    failures += compare(label + " fp16 x+out", run(x_fp16_device, DType::FP16, DType::FP16),
                        reference_fp16);
    return failures;
}

int run_linear_add(const std::string& label, QType qtype, std::int32_t k, std::int32_t tokens) {
    const ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
    DeviceWeight weight =
        make_weight(qtype, kHidden, k, 3311U + k, qtype == QType::NVFP4 ? Layout::Nvfp4Plain
                                                                         : Layout::Fp8);
    if (!ops::linear_add_fp16_activation_supported(weight.weight, policy, tokens)) {
        std::cerr << label << ": fp16 activation domain unexpectedly unsupported\n";
        return 1;
    }
    const std::size_t in_count  = static_cast<std::size_t>(k) * tokens;
    const std::size_t out_count = static_cast<std::size_t>(kHidden) * tokens;
    const auto x_bf16           = random_bf16(in_count, 3313U + tokens, 2.0F);
    const auto residual         = random_bf16(out_count, 3317U + tokens, 8.0F);
    DeviceWords x_bf16_device(x_bf16);
    DeviceWords x_fp16_device(to_fp16(x_bf16));
    WorkspaceArena workspace(std::max<std::size_t>(
        ops::linear_add_workspace_capacity_bytes(qtype, kHidden, k, policy, tokens, tokens), 256));

    const auto run = [&](const DeviceWords& input, DType in_dtype) {
        DeviceWords output(residual);
        Tensor x(input.buffer.p, in_dtype, {k, tokens});
        Tensor out(output.buffer.p, DType::BF16, {kHidden, tokens});
        workspace.reset();
        ops::linear_add(x, weight.weight, out, policy, workspace, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        return output.read(out_count);
    };
    return compare(label, run(x_fp16_device, DType::FP16), run(x_bf16_device, DType::BF16));
}

int run_attn_input(std::int32_t tokens) {
    constexpr std::int32_t kParentRows = 14336;
    constexpr std::int32_t kQRows      = 6144;
    constexpr std::int32_t kKvRows     = 1024;
    const ops::LinearPolicy policy     = ops::LinearPolicy::A16Only;
    const std::string label            = "attn_input_proj FP8 T=" + std::to_string(tokens);
    DeviceWeight weight =
        make_weight(QType::FP8_E4M3FN_ROW_BF16S, kParentRows, kHidden, 3341U, Layout::Fp8);
    if (!ops::attn_input_proj_fp16_activation_supported(weight.weight, policy, tokens)) {
        std::cerr << label << ": fp16 activation domain unexpectedly unsupported\n";
        return 1;
    }
    const auto x_bf16 = random_bf16(static_cast<std::size_t>(kHidden) * tokens, 3343U, 3.0F);
    DeviceWords x_bf16_device(x_bf16);
    DeviceWords x_fp16_device(to_fp16(x_bf16));
    WorkspaceArena workspace(std::max<std::size_t>(
        ops::attn_input_proj_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16S, kParentRows,
                                                      kHidden, policy, tokens, tokens),
        256));
    const std::size_t q_count  = static_cast<std::size_t>(kQRows) * tokens;
    const std::size_t kv_count = static_cast<std::size_t>(kKvRows) * tokens;
    const auto run = [&](const DeviceWords& input, DType dtype) {
        DeviceWords q(q_count), gate(q_count), k(kv_count), v(kv_count);
        Tensor x(input.buffer.p, dtype, {kHidden, tokens});
        Tensor q_tensor(q.buffer.p, DType::BF16, {kQRows, tokens});
        Tensor gate_tensor(gate.buffer.p, DType::BF16, {kQRows, tokens});
        Tensor k_tensor(k.buffer.p, DType::BF16, {kKvRows, tokens});
        Tensor v_tensor(v.buffer.p, DType::BF16, {kKvRows, tokens});
        workspace.reset();
        ops::attn_input_proj(x, weight.weight, q_tensor, gate_tensor, k_tensor, v_tensor, policy,
                             workspace, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<std::uint16_t> all = q.read(q_count);
        for (const auto* part : {&gate, &k, &v}) {
            const auto bits = part->read(part == &gate ? q_count : kv_count);
            all.insert(all.end(), bits.begin(), bits.end());
        }
        return all;
    };
    return compare(label, run(x_fp16_device, DType::FP16), run(x_bf16_device, DType::BF16));
}

// The fused 27B GDN norm/control kernel writes h itself: an FP16 h must be the BF16 h converted.
int run_norm_gating(std::int32_t tokens) {
    constexpr std::int32_t kHeads = 48;
    const std::string label = "gdn_norm_gating_proj h T=" + std::to_string(tokens);
    if (!ops::gdn_norm_gating_proj_fp16_hidden_supported(kHeads, kHidden, tokens)) {
        std::cerr << label << ": fp16 hidden unexpectedly unsupported\n";
        return 1;
    }
    const std::size_t count = static_cast<std::size_t>(kHidden) * tokens;
    DeviceWords x(random_bf16(count, 3331U + tokens, 4.0F));
    DeviceWords norm(random_bf16(kHidden, 3333U, 0.5F));
    DeviceWords ab(random_bf16(static_cast<std::size_t>(2 * kHeads) * kHidden, 3335U, 0.05F));
    std::vector<float> host_log(kHeads, -0.5F), host_bias(kHeads, 0.25F);
    DeviceBuffer a_log(kHeads * sizeof(float)), dt_bias(kHeads * sizeof(float));
    CUDA_CHECK(cudaMemcpy(a_log.p, host_log.data(), kHeads * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(dt_bias.p, host_bias.data(), kHeads * sizeof(float), cudaMemcpyHostToDevice));
    DeviceBuffer g(kHeads * tokens * sizeof(float)), beta(kHeads * tokens * sizeof(float));
    Weight ab_weight{};
    ab_weight.payload         = ab.buffer.p;
    ab_weight.payload_bytes   = static_cast<std::uint64_t>(2 * kHeads) * kHidden * 2;
    ab_weight.qtype           = QType::BF16_CTRL;
    ab_weight.layout          = QuantLayout::Contiguous;
    ab_weight.ndim            = 2;
    ab_weight.shape[0]        = 2 * kHeads;
    ab_weight.shape[1]        = kHidden;
    ab_weight.padded_shape[0] = 2 * kHeads;
    ab_weight.padded_shape[1] = kHidden;
    ab_weight.qdata           = ab.buffer.p;
    ab_weight.n               = 2 * kHeads;
    ab_weight.k               = kHidden;
    WorkspaceArena workspace(std::max<std::size_t>(
        ops::gdn_norm_gating_proj_workspace_capacity_bytes(kHeads, kHidden, tokens, tokens), 256));
    DeviceContext device;
    const auto run = [&](DType dtype) {
        DeviceWords h(count);
        Tensor x_tensor(x.buffer.p, DType::BF16, {kHidden, tokens});
        Tensor norm_tensor(norm.buffer.p, DType::BF16, {kHidden});
        Tensor log_tensor(a_log.p, DType::FP32, {kHeads});
        Tensor bias_tensor(dt_bias.p, DType::FP32, {kHeads});
        Tensor h_tensor(h.buffer.p, dtype, {kHidden, tokens});
        Tensor g_tensor(g.p, DType::FP32, {kHeads, tokens});
        Tensor beta_tensor(beta.p, DType::FP32, {kHeads, tokens});
        workspace.reset();
        ops::gdn_norm_gating_proj(x_tensor, norm_tensor, 1.0e-6F, ab_weight, log_tensor,
                                  bias_tensor, workspace, h_tensor, g_tensor, beta_tensor,
                                  device.execution_view());
        CUDA_CHECK(cudaDeviceSynchronize());
        return h.read(count);
    };
    return compare(label, run(DType::FP16), to_fp16(run(DType::BF16)));
}

} // namespace

int main() {
    if (test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = 0;
#ifdef NINFER_VOLTA_BUILD
        for (const std::int32_t t : {1, 5, 32}) {
            const std::string suffix = " T=" + std::to_string(t);
            failures += run_swiglu("swiglu NVFP4 interleaved" + suffix, QType::NVFP4,
                                   Layout::Nvfp4SwiGlu, t);
            failures +=
                run_swiglu("swiglu NVFP4 plain" + suffix, QType::NVFP4, Layout::Nvfp4Plain, t);
            failures += run_swiglu("swiglu FP8" + suffix, QType::FP8_E4M3FN_ROW_BF16S,
                                   Layout::Fp8, t);
            failures += run_swiglu("swiglu FP8 interleaved" + suffix, QType::FP8_E4M3FN_ROW_BF16S,
                                   Layout::Fp8SwiGlu, t);
            for (const std::int32_t k : {6144, 17408}) {
                const std::string shape = " K=" + std::to_string(k) + suffix;
                failures += run_linear_add("linear_add NVFP4" + shape, QType::NVFP4, k, t);
                failures +=
                    run_linear_add("linear_add FP8" + shape, QType::FP8_E4M3FN_ROW_BF16S, k, t);
            }
        }
        for (const std::int32_t t : {1, 5, 32}) {
            failures += run_norm_gating(t);
            failures += run_attn_input(t);
        }
        // Past the QPN width the domain is closed: callers must keep BF16 there.
        DeviceWeight wide = make_weight(QType::NVFP4, 2 * kIntermediate, kHidden, 3399U,
                                        Layout::Nvfp4SwiGlu);
        if (ops::linear_swiglu_fp16_activation_supported(wide.weight, ops::LinearPolicy::A16Only,
                                                         33)) {
            std::cerr << "swiglu NVFP4 T=33: fp16 activation domain must be closed\n";
            ++failures;
        }
#else
        const Weight none{};
        if (ops::linear_add_fp16_activation_supported(none, ops::LinearPolicy::A16Only, 5) ||
            ops::linear_swiglu_fp16_activation_supported(none, ops::LinearPolicy::A16Only, 5)) {
            std::cerr << "fp16 activation domain must be Volta-only\n";
            ++failures;
        }
#endif
        std::cout << (failures == 0 ? "OK" : "FAIL") << " fp16 activation domain\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "fp16 activation test failed: " << error.what() << '\n';
        return 1;
    }
}
