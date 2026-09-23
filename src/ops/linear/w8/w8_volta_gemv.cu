// W8G32 x BF16 GEMV for T=1 on Volta (sm_70), register-streamed.
//
// The MTP draft layer runs its five W8 projections at T=1 once per drafted token, and the
// sliced SIMT route (w8_rowsplit_gemm_simt.cuh) measured 430-620 GB/s on them, long-scoreboard
// bound. Two things hold it there on Volta:
//
//   - Its weight "pipeline" stages 1 KB slabs through shared memory with cp.async, which sm_70
//     does not have: pipe_copy is a synchronous load+store, so every slab exposes a full DRAM
//     latency before it can be consumed, and a warp never has more than one slab in flight.
//   - Each 256-value phase reloads its activation from global memory, one more exposed latency
//     per phase, and the int8 -> fp32 conversion runs on the quarter-rate I2F pipe.
//
// Here a warp owns kRows output rows and streams them straight into registers: per iteration
// every lane issues kUnroll x kRows coalesced 16-byte code loads (512 contiguous bytes per warp
// instruction, marked streaming since nothing in them is reused) plus its activation words before
// consuming any, so a warp keeps kUnroll * kRows * 512 bytes of weight in flight and the SM's
// 32 resident warps cover DRAM latency by themselves. The activation (at most 34 KB here) is read
// through L1, where every warp on the SM shares it; staging it in shared memory measured no
// better and capped occupancy on the k=17408 shape. bf16 -> fp32 is a shift. Codes decode with
// the 2^23 magic-number identity (one PRMT and one FADD per value, full rate), and the per-group
// scale is applied once per 16 values since a lane's 16 codes always sit in one 32-value group.
//
// V100-PCIE, cold cache, us (sliced SIMT -> this kernel), all within +-1% across configurations
// of 1-4 rows per warp, 1-4 steps per iteration and 4-16 warps per CTA; splitting each row pair's
// K over four warps of the CTA (shared-memory reduce) then takes another ~2% (685.7 -> 672):
//
//   (n, k)           before   after
//   5120 x 10240      116.8    93.2
//   14336 x 5120      144.4   124.0
//   5120 x 6144        77.8    60.4
//   34816 x 5120      306.2   257.6
//   5120 x 17408      184.3   150.5
//   sum (one draft)   829.5   685.7
//
// Requires k % 512 == 0, 16-byte aligned activation, and a W8 row stride that keeps 16-byte code
// alignment.

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/w8/w8_launch.h"
#include "ops/linear/w8/w8_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {

constexpr int kGemvWarps     = 8;
constexpr int kGemvRows      = 2; // output rows per warp
constexpr int kGemvUnroll    = 2; // 512-value K steps per iteration
constexpr int kGemvMinBlocks = 4; // 32 resident warps, 64 registers
constexpr int kGemvSplit     = 4; // warps splitting K per row pair: ~2% on every shape
constexpr int kGemvKStep     = 512; // single-step tail covers k % (512 * kGemvUnroll)
constexpr int kGemvMaxTokens = 7;   // multi-token form; loses its edge over QPN at T=8

__device__ __forceinline__ float w8_code_to_float(std::uint32_t biased_word, int byte) {
    // biased_word = codes ^ 0x80808080, so each byte is u = b + 128. 0x4B0000uu is 2^23 + u.
    const std::uint32_t bits = __byte_perm(biased_word, 0x4B000000u, 0x7540u + byte);
    return __int_as_float(static_cast<int>(bits)) - 8388736.0f; // 2^23 + 128
}

// One iteration: kUnroll 512-value K steps of kRows rows, every load issued before any use.
template <int kRows, int kUnroll>
__device__ __forceinline__ void w8_gemv_step(const std::uint8_t* const (&crow)[kRows],
                                             const std::uint16_t* const (&srow)[kRows],
                                             const uint4* __restrict__ xs, int kb, int lane,
                                             float (&acc)[kRows]) {
    uint4 cw[kUnroll][kRows];
    std::uint16_t sc[kUnroll][kRows];
    uint4 xa[kUnroll];
    uint4 xb[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
        const int xi = (kb + 512 * u) / 8 + 2 * lane;
        xa[u]        = __ldg(xs + xi);
        xb[u]        = __ldg(xs + xi + 1);
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
#pragma unroll
        for (int j = 0; j < kRows; ++j) {
            cw[u][j] = __ldcs(reinterpret_cast<const uint4*>(crow[j] + kb + 512 * u));
            sc[u][j] = __ldg(srow[j] + (kb + 512 * u) / 32);
        }
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
        // This lane's 16 activations, k = kb + 512u + 16 lane .. +16, as fp32.
        const std::uint32_t xw[8] = {xa[u].x, xa[u].y, xa[u].z, xa[u].w,
                                     xb[u].x, xb[u].y, xb[u].z, xb[u].w};
        float xf[16];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            xf[2 * i]     = __uint_as_float(xw[i] << 16);
            xf[2 * i + 1] = __uint_as_float(xw[i] & 0xffff0000u);
        }
#pragma unroll
        for (int j = 0; j < kRows; ++j) {
            const std::uint32_t words[4] = {cw[u][j].x ^ 0x80808080u, cw[u][j].y ^ 0x80808080u,
                                            cw[u][j].z ^ 0x80808080u, cw[u][j].w ^ 0x80808080u};
            float part = 0.0f;
#pragma unroll
            for (int q = 0; q < 16; ++q) {
                part = fmaf(w8_code_to_float(words[q / 4], q % 4), xf[q], part);
            }
            acc[j] = fmaf(part, __half2float(__ushort_as_half(sc[u][j])), acc[j]);
        }
    }
}

template <int kWarps, int kRows, int kUnroll, int kMinBlocks, int kSplit>
__global__ __launch_bounds__(kWarps * 32, kMinBlocks) void w8_volta_gemv_t1_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, int n, int k,
    int padded_groups) {
    static_assert(kWarps % kSplit == 0);
    __shared__ float partials[kSplit > 1 ? kWarps : 1][kRows];
    const uint4* xs = reinterpret_cast<const uint4*>(x);
    const int tid   = static_cast<int>(threadIdx.x);

    const int lane  = tid & 31;
    const int warp  = tid >> 5;
    const int group = warp / kSplit; // row group within the CTA
    const int split = warp % kSplit; // K slice
    const int row0  = (static_cast<int>(blockIdx.x) * (kWarps / kSplit) + group) * kRows;

    const std::int64_t row_bytes = static_cast<std::int64_t>(padded_groups) *
                                   W8RowSplitStorage::kCodeBytesPerGroup;
    const std::uint8_t* crow[kRows];
    const std::uint16_t* srow[kRows];
#pragma unroll
    for (int j = 0; j < kRows; ++j) {
        const int row = min(row0 + j, n - 1);
        crow[j]       = codes + row * row_bytes + 16 * lane;
        srow[j]       = reinterpret_cast<const std::uint16_t*>(scales) +
                  static_cast<std::int64_t>(row) * padded_groups + lane / 2;
    }

    float acc[kRows];
#pragma unroll
    for (int j = 0; j < kRows; ++j) { acc[j] = 0.0f; }

    const int steps = k / 512;
    int kb          = (split * steps / kSplit) * 512;
    const int kend  = ((split + 1) * steps / kSplit) * 512;
    for (; kb + 512 * kUnroll <= kend; kb += 512 * kUnroll) {
        w8_gemv_step<kRows, kUnroll>(crow, srow, xs, kb, lane, acc);
    }
    for (; kb < kend; kb += 512) { w8_gemv_step<kRows, 1>(crow, srow, xs, kb, lane, acc); }

#pragma unroll
    for (int j = 0; j < kRows; ++j) {
        float v = acc[j];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xffffffffu, v, offset);
        }
        acc[j] = v;
    }
    if constexpr (kSplit > 1) {
        if (lane == 0) {
#pragma unroll
            for (int j = 0; j < kRows; ++j) { partials[warp][j] = acc[j]; }
        }
        __syncthreads();
        if (split != 0 || lane != 0) { return; }
#pragma unroll
        for (int j = 0; j < kRows; ++j) {
            float v = 0.0f;
#pragma unroll
            for (int s = 0; s < kSplit; ++s) { v += partials[warp + s][j]; }
            acc[j] = v;
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < kRows; ++j) {
            if (row0 + j < n) { out[row0 + j] = __float2bfloat16_rn(acc[j]); }
        }
    }
}

// T = 2..7 (the MTP catch-up pass, T = K + 1). Same row ownership and decode as the T=1 kernel,
// one 512-value step per iteration with the next step's codes prefetched into registers, and the
// decoded fp32 weights reused across the kT activation columns, which are loaded per column just
// in time (L1 hits: every warp on the SM walks the same activation). The FP32 pipe, not DRAM,
// sets the pace from T~4 on, which is why this form stops paying at T=8.
template <int kT, int kRows, int kWarps, int kMinBlocks>
__global__ __launch_bounds__(kWarps * 32, kMinBlocks) void w8_volta_gemv_tn_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, int n, int k,
    int padded_groups) {
    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row0 = (static_cast<int>(blockIdx.x) * kWarps + warp) * kRows;
    if (row0 >= n) { return; }
    const std::int64_t row_bytes = static_cast<std::int64_t>(padded_groups) *
                                   W8RowSplitStorage::kCodeBytesPerGroup;
    const std::uint8_t* crow[kRows];
    const std::uint16_t* srow[kRows];
#pragma unroll
    for (int j = 0; j < kRows; ++j) {
        const int row = min(row0 + j, n - 1);
        crow[j]       = codes + row * row_bytes + 16 * lane;
        srow[j]       = reinterpret_cast<const std::uint16_t*>(scales) +
                  static_cast<std::int64_t>(row) * padded_groups + lane / 2;
    }
    const uint4* xs = reinterpret_cast<const uint4*>(x);
    float acc[kRows][kT];
#pragma unroll
    for (int j = 0; j < kRows; ++j) {
#pragma unroll
        for (int c = 0; c < kT; ++c) { acc[j][c] = 0.0f; }
    }
    uint4 cn[kRows];
    std::uint16_t sn[kRows];
#pragma unroll
    for (int j = 0; j < kRows; ++j) {
        cn[j] = __ldcs(reinterpret_cast<const uint4*>(crow[j]));
        sn[j] = __ldg(srow[j]);
    }
    for (int kb = 0; kb < k; kb += 512) {
        uint4 cw[kRows];
        float sc[kRows];
#pragma unroll
        for (int j = 0; j < kRows; ++j) {
            cw[j] = cn[j];
            sc[j] = __half2float(__ushort_as_half(sn[j]));
        }
        if (kb + 512 < k) {
#pragma unroll
            for (int j = 0; j < kRows; ++j) {
                cn[j] = __ldcs(reinterpret_cast<const uint4*>(crow[j] + kb + 512));
                sn[j] = __ldg(srow[j] + (kb + 512) / 32);
            }
        }
        float wf[kRows][16];
#pragma unroll
        for (int j = 0; j < kRows; ++j) {
            const std::uint32_t words[4] = {cw[j].x ^ 0x80808080u, cw[j].y ^ 0x80808080u,
                                            cw[j].z ^ 0x80808080u, cw[j].w ^ 0x80808080u};
#pragma unroll
            for (int q = 0; q < 16; ++q) { wf[j][q] = w8_code_to_float(words[q / 4], q % 4); }
        }
#pragma unroll
        for (int c = 0; c < kT; ++c) {
            {
                const int xi  = (c * k + kb) / 8 + 2 * lane;
                const uint4 a = __ldg(xs + xi);
                const uint4 b = __ldg(xs + xi + 1);
                const std::uint32_t xw[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
                float xf[16];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    xf[2 * i]     = __uint_as_float(xw[i] << 16);
                    xf[2 * i + 1] = __uint_as_float(xw[i] & 0xffff0000u);
                }
#pragma unroll
                for (int j = 0; j < kRows; ++j) {
                    float part = 0.0f;
#pragma unroll
                    for (int q = 0; q < 16; ++q) { part = fmaf(wf[j][q], xf[q], part); }
                    acc[j][c] = fmaf(part, sc[j], acc[j][c]);
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < kRows; ++j) {
#pragma unroll
        for (int c = 0; c < kT; ++c) {
            float v = acc[j][c];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                v += __shfl_xor_sync(0xffffffffu, v, offset);
            }
            if (lane == 0 && row0 + j < n) {
                out[static_cast<std::int64_t>(c) * n + row0 + j] = __float2bfloat16_rn(v);
            }
        }
    }
}

template <int kT, int kRows, int kWarps, int kMinBlocks>
void launch_tn_t(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t n             = out.ne[0];
    const std::int32_t k             = x.ne[0];
    const std::int32_t padded_groups = w.padded_shape[1] / W8RowSplitStorage::kGroupK;
    const dim3 grid(static_cast<unsigned>(div_up(n, kWarps * kRows)));
    w8_volta_gemv_tn_kernel<kT, kRows, kWarps, kMinBlocks><<<grid, kWarps * 32, 0, stream>>>(
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.scales),
        static_cast<const __nv_bfloat16*>(x.data), static_cast<__nv_bfloat16*>(out.data), n, k,
        padded_groups);
    CUDA_CHECK(cudaGetLastError());
}

template <int kRows, int kWarps, int kMinBlocks>
void launch_tn(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    switch (x.ne[1]) {
    case 2: launch_tn_t<2, kRows, kWarps, kMinBlocks>(x, w, out, stream); break;
    case 3: launch_tn_t<3, kRows, kWarps, kMinBlocks>(x, w, out, stream); break;
    case 4: launch_tn_t<4, kRows, kWarps, kMinBlocks>(x, w, out, stream); break;
    case 5: launch_tn_t<5, kRows, kWarps, kMinBlocks>(x, w, out, stream); break;
    case 6: launch_tn_t<6, kRows, kWarps, kMinBlocks>(x, w, out, stream); break;
    default: launch_tn_t<7, kRows, kWarps, kMinBlocks>(x, w, out, stream); break;
    }
}

template <int kWarps, int kRows, int kUnroll, int kMinBlocks, int kSplit>
void launch_t1(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t n             = out.ne[0];
    const std::int32_t k             = x.ne[0];
    const std::int32_t padded_groups = w.padded_shape[1] / W8RowSplitStorage::kGroupK;
    constexpr int kRowsPerCta        = kWarps / kSplit * kRows;
    const dim3 grid(static_cast<unsigned>(div_up(n, kRowsPerCta)));
    w8_volta_gemv_t1_kernel<kWarps, kRows, kUnroll, kMinBlocks, kSplit>
        <<<grid, kWarps * 32, 0, stream>>>(
            static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.scales),
            static_cast<const __nv_bfloat16*>(x.data), static_cast<__nv_bfloat16*>(out.data), n,
            k, padded_groups);
    CUDA_CHECK(cudaGetLastError());
}

bool w8_volta_gemv_aligned(const Tensor& x, const Weight& w) noexcept {
    return (reinterpret_cast<std::uintptr_t>(x.data) & 0xfu) == 0 &&
           (reinterpret_cast<std::uintptr_t>(w.qdata) & 0xfu) == 0;
}

} // namespace

bool w8_volta_gemv_t1_supported(std::int32_t n, std::int32_t k, std::int32_t padded_k) noexcept {
    return n > 0 && k > 0 && k % kGemvKStep == 0 && padded_k % 16 == 0;
}

void launch_w8_volta_gemv_t1(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t n = out.ne[0];
    const std::int32_t k = x.ne[0];
    if (x.ne[1] != 1 || !w8_volta_gemv_t1_supported(n, k, w.padded_shape[1]) ||
        !w8_volta_gemv_aligned(x, w)) {
        launch_w8_small_t(x, w, out, stream);
        return;
    }
    launch_t1<kGemvWarps, kGemvRows, kGemvUnroll, kGemvMinBlocks, kGemvSplit>(x, w, out, stream);
}

void launch_w8_volta_gemv_tn(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t n = out.ne[0];
    const std::int32_t t = x.ne[1];
    if (t < 2 || t > kGemvMaxTokens || !w8_volta_gemv_t1_supported(n, x.ne[0], w.padded_shape[1]) ||
        !w8_volta_gemv_aligned(x, w)) {
        launch_w8_small_t(x, w, out, stream);
        return;
    }
    // The n=5120 shapes have too few rows to fill the machine two per warp.
    if (n <= 8192) {
        launch_tn<1, 8, 4>(x, w, out, stream);
    } else {
        launch_tn<2, 4, 4>(x, w, out, stream);
    }
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
