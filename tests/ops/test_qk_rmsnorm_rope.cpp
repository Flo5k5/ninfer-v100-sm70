// qk_rmsnorm_rope must reproduce rmsnorm(q) + rmsnorm(k) + rope(q, k) bit for bit, on its fused
// route (fixed text geometries, 1-D and 3-D positions) and on the composed fallback.
#include "ninfer/ops/qk_rmsnorm_rope.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"

#include "core/arena.h"
#include "core/device.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

using namespace ninfer;

namespace {

std::vector<std::uint16_t> random_bf16(std::size_t count, std::uint32_t seed, float scale) {
    std::vector<std::uint16_t> bits(count);
    for (auto& value : bits) {
        seed              = seed * 1664525U + 1013904223U;
        const float real  = (static_cast<float>(seed >> 8) * (2.0F / 16777216.0F) - 1.0F) * scale;
        std::uint32_t raw = 0;
        std::memcpy(&raw, &real, sizeof(raw));
        value = static_cast<std::uint16_t>((raw + 0x7FFFU + ((raw >> 16) & 1U)) >> 16);
    }
    return bits;
}

template <class T>
DeviceBuffer upload(const std::vector<T>& host) {
    DeviceBuffer buffer(host.size() * sizeof(T));
    CUDA_CHECK(cudaMemcpy(buffer.p, host.data(), buffer.bytes, cudaMemcpyHostToDevice));
    return buffer;
}

std::vector<std::uint16_t> download(const DeviceBuffer& buffer) {
    std::vector<std::uint16_t> host(buffer.bytes / sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpy(host.data(), buffer.p, buffer.bytes, cudaMemcpyDeviceToHost));
    return host;
}

int run_case(std::int32_t q_heads, std::int32_t k_heads, std::int32_t tokens, int axes,
             std::uint32_t seed) {
    constexpr std::int32_t kDim = 256;
    const std::string label = "qk_rmsnorm_rope " + std::to_string(q_heads) + "Q/" +
                              std::to_string(k_heads) + "K T=" + std::to_string(tokens) +
                              " axes=" + std::to_string(axes);
    const std::size_t q_count = static_cast<std::size_t>(kDim) * q_heads * tokens;
    const std::size_t k_count = static_cast<std::size_t>(kDim) * k_heads * tokens;
    DeviceBuffer q     = upload(random_bf16(q_count, seed, 6.0F));
    DeviceBuffer k     = upload(random_bf16(k_count, seed + 1U, 6.0F));
    DeviceBuffer q_norm = upload(random_bf16(kDim, seed + 2U, 0.8F));
    DeviceBuffer k_norm = upload(random_bf16(kDim, seed + 3U, 0.8F));
    std::vector<std::int32_t> host_positions(static_cast<std::size_t>(tokens) * axes);
    for (std::size_t i = 0; i < host_positions.size(); ++i) {
        host_positions[i] = static_cast<std::int32_t>((i * 7919U + seed) % 32768U);
    }
    DeviceBuffer positions = upload(host_positions);

    Tensor q_tensor(q.p, DType::BF16, {kDim, q_heads, tokens});
    Tensor k_tensor(k.p, DType::BF16, {kDim, k_heads, tokens});
    Tensor q_norm_tensor(q_norm.p, DType::BF16, {kDim});
    Tensor k_norm_tensor(k_norm.p, DType::BF16, {kDim});
    Tensor position_tensor = axes == 1 ? Tensor(positions.p, DType::I32, {tokens})
                                       : Tensor(positions.p, DType::I32, {tokens, axes});

    DeviceBuffer q_fused(q_count * 2), k_fused(k_count * 2), q_ref(q_count * 2),
        k_ref(k_count * 2);
    Tensor q_fused_tensor(q_fused.p, DType::BF16, {kDim, q_heads, tokens});
    Tensor k_fused_tensor(k_fused.p, DType::BF16, {kDim, k_heads, tokens});
    Tensor q_ref_tensor(q_ref.p, DType::BF16, {kDim, q_heads, tokens});
    Tensor k_ref_tensor(k_ref.p, DType::BF16, {kDim, k_heads, tokens});

    ops::qk_rmsnorm_rope(q_tensor, k_tensor, q_norm_tensor, k_norm_tensor, 1.0e-6F,
                         position_tensor, 64, 1.0e7F, q_fused_tensor, k_fused_tensor, nullptr);
    ops::rmsnorm(q_tensor, q_norm_tensor, 1.0e-6F, true, q_ref_tensor, nullptr);
    ops::rmsnorm(k_tensor, k_norm_tensor, 1.0e-6F, true, k_ref_tensor, nullptr);
    ops::rope(position_tensor, 64, 1.0e7F, q_ref_tensor, k_ref_tensor, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    int failures = 0;
    for (const auto& [actual, expected, name] :
         {std::tuple{&q_fused, &q_ref, "q"}, std::tuple{&k_fused, &k_ref, "k"}}) {
        const auto a = download(*actual);
        const auto e = download(*expected);
        for (std::size_t i = 0; i < e.size(); ++i) {
            if (a[i] != e[i]) {
                std::cerr << label << " " << name << ": element " << i << " is 0x" << std::hex
                          << a[i] << ", composed gives 0x" << e[i] << std::dec << '\n';
                ++failures;
                break;
            }
        }
    }
    return failures;
}

} // namespace

int main() {
    if (test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = 0;
        std::uint32_t seed = 4101U;
        for (const std::int32_t tokens : {1, 5, 17, 128}) {
            for (const int axes : {1, 3}) {
                failures += run_case(24, 4, tokens, axes, seed++);
                failures += run_case(16, 2, tokens, axes, seed++);
            }
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " qk_rmsnorm_rope\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "qk_rmsnorm_rope test failed: " << error.what() << '\n';
        return 1;
    }
}
