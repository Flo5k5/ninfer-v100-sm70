// Q4G64 x BF16 GEMV for T=1 on Volta (sm_70), register-streamed.
//
// The Q4 sibling of w8_volta_gemv.cu, for the MTP proposal head (Q4 [131072,5120], read four
// times per speculative step). A warp owns kRows output rows; per iteration every lane issues
// kUnroll x kRows coalesced 16-byte code loads (512 contiguous bytes per warp instruction,
// streaming) and its activation words before consuming any, so the resident warps keep enough
// bytes in flight without a shared-memory pipeline, which Volta cannot run asynchronously.
//
// A lane's 16 code bytes are 32 consecutive k inside one 64-value group, so one scale applies.
// Nibbles (4-bit two's complement, value i of a word at bits 4i) decode through the 2^23
// magic-number identity: biased by XOR 8, the low and high nibbles of the word's four bytes are
// masked into bytes, then each byte is placed under 0x4B00_00xx by one PRMT and recentred by one
// FADD -- full-rate ops only, where the SIMT decoders use the quarter-rate I2F or F2F pipes.
//
// V100-PCIE, cold cache, [131072,5120] at T=1: 581.7 -> 452.6 us (614 -> 794 GB/s). Configurations
// of 1-4 rows per warp, 1-4 steps per iteration and 4-8 warps per CTA land within 1% of each
// other except those that spill or keep too few bytes in flight.

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {

constexpr int kQ4GemvWarps     = 8;
constexpr int kQ4GemvRows      = 2; // output rows per warp
constexpr int kQ4GemvUnroll    = 2; // 1024-value K steps per iteration
constexpr int kQ4GemvMinBlocks = 2; // 80 registers, no spill (64-register caps spill here)
constexpr int kQ4GemvKStep     = 1024; // k per 512-byte code step

__device__ __forceinline__ float q4_byte_to_float(std::uint32_t nibble_bytes, int byte) {
    // Each byte holds one biased nibble u = c + 8; 0x4B0000uu is 2^23 + u, so this returns c.
    const std::uint32_t bits = __byte_perm(nibble_bytes, 0x4B000000u, 0x7540u + byte);
    return __int_as_float(static_cast<int>(bits)) - 8388616.0f; // 2^23 + 8
}

template <int kRows, int kUnroll>
__device__ __forceinline__ void q4_gemv_step(const std::uint8_t* const (&crow)[kRows],
                                             const std::uint16_t* const (&srow)[kRows],
                                             const uint4* __restrict__ xv, int kb, int lane,
                                             float (&acc)[kRows]) {
    uint4 cw[kUnroll][kRows];
    std::uint16_t sc[kUnroll][kRows];
    uint4 xa[kUnroll][4];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
        const int step = kb / kQ4GemvKStep + u;
#pragma unroll
        for (int j = 0; j < kRows; ++j) {
            cw[u][j] = __ldcs(reinterpret_cast<const uint4*>(crow[j] + 512 * step));
            sc[u][j] = __ldg(srow[j] + 16 * step);
        }
        // 32 activations: k = kb + 1024 u + 32 lane .. +32, four uint4 of bf16.
        const int xi = (kb + kQ4GemvKStep * u) / 8 + 4 * lane;
#pragma unroll
        for (int v = 0; v < 4; ++v) { xa[u][v] = __ldg(xv + xi + v); }
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
        const std::uint32_t xw[16] = {xa[u][0].x, xa[u][0].y, xa[u][0].z, xa[u][0].w,
                                      xa[u][1].x, xa[u][1].y, xa[u][1].z, xa[u][1].w,
                                      xa[u][2].x, xa[u][2].y, xa[u][2].z, xa[u][2].w,
                                      xa[u][3].x, xa[u][3].y, xa[u][3].z, xa[u][3].w};
#pragma unroll
        for (int j = 0; j < kRows; ++j) {
            const std::uint32_t words[4] = {cw[u][j].x, cw[u][j].y, cw[u][j].z, cw[u][j].w};
            float part = 0.0f;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                // Word q covers k = 8q .. 8q+7; byte b holds k = 8q+2b (low) and 8q+2b+1 (high).
                // Codes are 4-bit two's complement: u ^ 8 is the offset-binary form.
                const std::uint32_t biased = words[q] ^ 0x88888888u;
                const std::uint32_t lo     = biased & 0x0F0F0F0Fu;
                const std::uint32_t hi     = (biased >> 4) & 0x0F0F0F0Fu;
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const std::uint32_t xpair = xw[4 * q + b];
                    part = fmaf(q4_byte_to_float(lo, b), __uint_as_float(xpair << 16), part);
                    part = fmaf(q4_byte_to_float(hi, b), __uint_as_float(xpair & 0xffff0000u),
                                part);
                }
            }
            acc[j] = fmaf(part, __half2float(__ushort_as_half(sc[u][j])), acc[j]);
        }
    }
}

template <int kWarps, int kRows, int kUnroll, int kMinBlocks>
__global__ __launch_bounds__(kWarps * 32, kMinBlocks) void q4_volta_gemv_t1_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, int n, int k) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row0 = (static_cast<int>(blockIdx.x) * kWarps + warp) * kRows;
    if (row0 >= n) { return; }

    const int groups             = k / Q4RowSplitStorage::kGroupK;
    const std::int64_t row_bytes = static_cast<std::int64_t>(groups) *
                                   Q4RowSplitStorage::kCodeBytesPerGroup;
    const std::uint8_t* crow[kRows];
    const std::uint16_t* srow[kRows];
#pragma unroll
    for (int j = 0; j < kRows; ++j) {
        const int row = min(row0 + j, n - 1);
        crow[j]       = codes + row * row_bytes + 16 * lane;
        srow[j]       = reinterpret_cast<const std::uint16_t*>(scales) +
                  static_cast<std::int64_t>(row) * groups + lane / 2;
    }
    const uint4* xv = reinterpret_cast<const uint4*>(x);

    float acc[kRows];
#pragma unroll
    for (int j = 0; j < kRows; ++j) { acc[j] = 0.0f; }

    int kb = 0;
    for (; kb + kQ4GemvKStep * kUnroll <= k; kb += kQ4GemvKStep * kUnroll) {
        q4_gemv_step<kRows, kUnroll>(crow, srow, xv, kb, lane, acc);
    }
    for (; kb < k; kb += kQ4GemvKStep) { q4_gemv_step<kRows, 1>(crow, srow, xv, kb, lane, acc); }

#pragma unroll
    for (int j = 0; j < kRows; ++j) {
        float v = acc[j];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xffffffffu, v, offset);
        }
        if (lane == 0 && row0 + j < n) { out[row0 + j] = __float2bfloat16_rn(v); }
    }
}

} // namespace

bool q4_volta_gemv_t1_supported(const Tensor& x, const Weight& w) noexcept {
    const std::int32_t k = x.ne[0];
    return x.ne[1] == 1 && w.layout == QuantLayout::RowSplit && k % kQ4GemvKStep == 0 &&
           w.padded_shape[1] == k && (reinterpret_cast<std::uintptr_t>(x.data) & 0xfu) == 0 &&
           (reinterpret_cast<std::uintptr_t>(w.qdata) & 0xfu) == 0;
}

void launch_q4_volta_gemv_t1(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (!q4_volta_gemv_t1_supported(x, w)) {
        launch_q4_gemv_r4_w1_direct(x, w, out, stream);
        return;
    }
    const std::int32_t n      = out.ne[0];
    const std::int32_t k      = x.ne[0];
    constexpr int kRowsPerCta = kQ4GemvWarps * kQ4GemvRows;
    const dim3 grid(static_cast<unsigned>(div_up(n, kRowsPerCta)));
    q4_volta_gemv_t1_kernel<kQ4GemvWarps, kQ4GemvRows, kQ4GemvUnroll, kQ4GemvMinBlocks>
        <<<grid, kQ4GemvWarps * 32, 0, stream>>>(
            static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.scales),
            static_cast<const __nv_bfloat16*>(x.data), static_cast<__nv_bfloat16*>(out.data), n,
            k);
    CUDA_CHECK(cudaGetLastError());
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
