// Volta (sm_70) regression for the shared-memory carveout requests of the q5 row-split GEMVs and
// the w8 SIMT routes. Every kernel these launchers run must carry its own
// cudaFuncAttributePreferredSharedMemoryCarveout = 100. When the request was guarded by a flag
// keyed by the kernel pointer type, all kernels sharing a signature shared one flag, and only the
// first one launched in the process received the hint.
//
// This test deliberately names private launchers and kernel symbols: it checks a launch attribute
// that no Op contract exposes, not Op semantics, so the weight contents are irrelevant.
#include "core/arena.h"
#include "core/device.h"
#include "ops/linear/q5/q5_launch.h"
#include "ops/linear/q5/q5_rowsplit_gemv.cuh"
#include "ops/linear/w8/w8_launch.h"
#include "ops/linear/w8/w8_rowsplit_gemm_simt.cuh"
#include "ops/linear_add/q5/q5_linear_add_kernels.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>

namespace {

using namespace ninfer;
using namespace ninfer::ops::detail;
namespace fixture = ninfer::test::quantized_weight;

constexpr std::uint32_t kSeed = 0x5eedU;

struct DeviceWeight {
    fixture::PackedWeight packed;
    DeviceBuffer payload;
    Weight weight;

    DeviceWeight(QType qtype, std::int32_t n, std::int32_t k)
        : packed(fixture::make_patterned_weight(qtype, n, k, kSeed)),
          payload(packed.payload.size()) {
        payload.copy_from_host(packed.payload.data(), packed.payload.size());
        weight = packed.device_weight(payload.p);
    }
};

DeviceBuffer zeroed_bf16(std::int32_t rows, std::int32_t columns) {
    DeviceBuffer buffer(static_cast<std::size_t>(rows) * columns * sizeof(std::uint16_t));
    buffer.fill(0);
    return buffer;
}

void launch_q5_gemv(std::int32_t n) {
    constexpr std::int32_t kK = 5120;
    const DeviceWeight weight(QType::Q5G64_F16S, n, kK);
    DeviceBuffer x   = zeroed_bf16(kK, 1);
    DeviceBuffer out = zeroed_bf16(n, 1);
    const Tensor x_view(x.p, DType::BF16, {kK, 1});
    Tensor out_view(out.p, DType::BF16, {n, 1});
    launch_q5_gemv_r16_s2_x(x_view, weight.weight, out_view, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_q5_residual(std::int32_t k) {
    constexpr std::int32_t kN = 5120;
    const DeviceWeight weight(QType::Q5G64_F16S, kN, k);
    DeviceBuffer x        = zeroed_bf16(k, 1);
    DeviceBuffer residual = zeroed_bf16(kN, 1);
    const Tensor x_view(x.p, DType::BF16, {k, 1});
    Tensor residual_view(residual.p, DType::BF16, {kN, 1});
    q5_linear_add_gemv_residual_launch(x_view, weight.weight, residual_view, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_w8_simt(W8Launch launch, std::int32_t tokens) {
    constexpr std::int32_t kN = 512;
    constexpr std::int32_t kK = 2048;
    const DeviceWeight weight(QType::W8G32_F16S, kN, kK);
    DeviceBuffer x   = zeroed_bf16(kK, tokens);
    DeviceBuffer out = zeroed_bf16(kN, tokens);
    const Tensor x_view(x.p, DType::BF16, {kK, tokens});
    Tensor out_view(out.p, DType::BF16, {kN, tokens});
    launch(x_view, weight.weight, out_view, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename Kernel>
int expect_carveout(Kernel kernel, const char* label) {
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    if (attributes.preferredShmemCarveout == 100) { return 0; }
    std::cerr << label << ": preferredShmemCarveout " << attributes.preferredShmemCarveout
              << ", expected 100\n";
    return 1;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (count_err == cudaErrorNoDevice || count_err == cudaErrorInsufficientDriver ||
        count_err == cudaErrorStubLibrary || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_err);
    CUDA_CHECK(cudaSetDevice(0));

    launch_q5_gemv(6144);
    launch_q5_gemv(7168);
    launch_q5_residual(6144);
    launch_q5_residual(17408);
    launch_w8_simt(launch_w8_simt_r8_c4, 4);
    launch_w8_simt(launch_w8_simt_r8_c8, 8);
    launch_w8_simt(launch_w8_simt_r4_c16, 16);

    int failures = 0;
    failures +=
        expect_carveout(q5_rowsplit_gemv_kernel<6144, 5120, 16, 2, true, false>, "q5 GEMV N=6144");
    failures +=
        expect_carveout(q5_rowsplit_gemv_kernel<7168, 5120, 16, 2, true, false>, "q5 GEMV N=7168");
    failures += expect_carveout(q5_rowsplit_gemv_kernel<5120, 6144, 16, 2, true, true>,
                                "q5 residual GEMV K=6144");
    failures += expect_carveout(q5_rowsplit_gemv_kernel<5120, 17408, 16, 2, false, true>,
                                "q5 residual GEMV K=17408");
    // The routes run 8 warps per CTA, 2 pipeline stages, and on Volta only the predicated body.
    failures +=
        expect_carveout(w8_rowsplit_gemm_simt_kernel<W8RowSplitSimtSchedule, 4, 8, 2, false,
                                                     W8Epilogue::Store, W8ContiguousOutput, 1>,
                        "w8 SIMT r8_c4");
    failures +=
        expect_carveout(w8_rowsplit_gemm_simt_kernel<W8RowSplitSimtSchedule, 8, 8, 2, false,
                                                     W8Epilogue::Store, W8ContiguousOutput, 1>,
                        "w8 SIMT r8_c8");
    failures +=
        expect_carveout(w8_rowsplit_gemm_simt_kernel<W8RowSplitSimtSchedule, 8, 8, 2, false,
                                                     W8Epilogue::Store, W8ContiguousOutput, 2>,
                        "w8 SIMT r4_c16");
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
