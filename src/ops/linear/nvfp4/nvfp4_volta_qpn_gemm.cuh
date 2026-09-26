#pragma once

// Fused-dequant NVFP4 (E2M1 + E4M3 K16 block scale) x BF16 GEMM on Volta tensor cores,
// quadpair-split-N form (sm_70 only). Fourth sibling of q4/w8/fp8_volta_qpn_gemm.cuh: same
// geometry, same fragment maps, same one-barrier cross-warp K reduce. What changes is the decoder
// and the scale cadence.
//
// This is v100-skinny's skinny_nvfp4_qpn2 (kernels/skinny_kernels.cu), adapted to ninfer's
// checkpoint-native (non-prepacked) weight layout instead of their load-time fragment-order
// prepack -- see the pairing note below for why that adaptation is exact rather than approximate.
// Copy the decoder and scale-cadence choices; do not re-derive them.
//
//   - e2m1 -> half2, TurboMind's shift+rebias decode (Apache-2.0, 1Cat-vLLM). Shifting the sign
//     and three EM bits into fp16 lane positions reproduces the value scaled by 2^-14; v100-skinny
//     folds the correction into the group scale (gscale * 16384) rather than paying an extra
//     multiply per value, and their note says this fold is also the fix for FP16 overflow on real
//     activation outliers -- code * raw fp8 scale reaches ~2.7e3 without it.
//   - e4m3 -> half2, the same shift decode QPN8 already ships (fp8_volta_qpn_gemm.cuh,
//     fp8_decode_quad). Mask 0x3F80, not 0x7F80: with the wider mask bit 14 catches the sign,
//     which lands in the fp16 exponent, and every negative scale overflows to inf.
//   - Pairing is load-bearing, and it is where this kernel earns its keep over a straight port.
//     dequant8_tm decodes one 32-bit word (8 e2m1 codes = half an NVFP4 group) by shifting each
//     nibble into a fixed fp16 lane; the geometry of that shift makes out[j] hold codes (j, j+4)
//     of the word, not (j, j+1). v100-skinny's checkpoint prepacks codes into an order where that
//     structural (j, j+4) grouping already IS adjacent k, so their consumer reads activations
//     sequentially. Ours does not prepack -- the code plane is plain row-major, adjacent-k nibble
//     packing (validated against nvfp4_codec.cuh's pack routine) -- so decoding it directly with
//     dequant8_tm's shift trick yields (k, k+4) pairs. The fix is not to prepack the weights but to
//     apply the *same* permutation to the activations: two __byte_perm calls turn eight
//     sequential-k halves into the identical (k, k+4) grouping the decoder emits, entirely in
//     registers, no shared memory, no extra loads (nvfp4_stage_pairs, ported from
//     skinny_kernels.cu's stage_pairs, the non-prepacked SIMT kernel's version of this same trick).
//     Because both operands go through the same permutation, every mma slice still contracts the
//     matching k on both sides -- the permutation is a relabeling of the reduction, not a
//     reordering of which values multiply which.
//   - Scale cadence is 4x tighter than Q4's: a new group scale every 4 mma slices (one 16-k NVFP4
//     group) instead of every 16 (one 64-k Q4 group). The saving grace is the swizzled scale plane
//     (BlockScaleK16M128x4): for one output row, scale bytes for four *consecutive* groups sit
//     contiguously (the "scale_lane" term in nvfp4_scale_offset has unit stride), so one aligned
//     4-byte read covers a whole 64-k chunk's worth of group scales -- the same physical chunk
//     size as Q4's group, just decoded into four scales instead of held as one.
//
// No shared memory or barrier in the main loop for the same reason as the other three siblings:
// activations are read redundantly by quadpair siblings (L1/L2-resident, KB-scale) and weights
// stream global -> register. The single __syncthreads() is the cross-warp K reduction at the
// output.

#include "core/device.h"
#include "ops/common/volta_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_output.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

struct Nvfp4VoltaQpnSchedule {
    static constexpr int kColsPerCta  = 32; // output rows per CTA (mma's N axis)
    static constexpr int kRowsPerTile = 8;  // tokens per A tile (mma's M axis)
    static constexpr int kGroupK       = 16; // k-values per NVFP4 block-scale group
    static constexpr int kChunkGroups  = 4;  // groups per scale-plane contiguous run
    static constexpr int kChunkK       = kGroupK * kChunkGroups; // 64: one Q4-sized K unit
    static constexpr int kCodeBytesPerChunk = kChunkK / 2;       // 32
};

// One 32-bit word = 8 e2m1 codes = half of one 16-k group. Output pairing is (j, j+4) -- see the
// file header. The shift decode gives 2^-14 * true magnitude; `rebias16384` (the constant 2^14,
// hoisted by the caller) corrects it back to the true 0..6 range. That correction has to land here
// rather than folded into the group scale: the group scale can be arbitrarily large (weight's
// global divisor composed with an e4m3 byte up to 448), and gscale * 2^14 overflows fp16 long
// before the *actual* weight value -- shift_decoded(<=6 * 2^-14) * 2^14 -- ever could. Splitting
// the multiply costs one extra hmul2 per value; both factors stay within fp16 range throughout.
__device__ __forceinline__ void nvfp4_decode_e2m1_quad(std::uint32_t q, half2 rebias16384,
                                                        half2 (&out)[4]) {
    constexpr std::uint32_t kSign = 0x80008000u;
    constexpr std::uint32_t kExpM = 0x0E000E00u;
    std::uint32_t v0 = ((q << 12) & kSign) | ((q << 9) & kExpM);
    std::uint32_t v1 = ((q << 8) & kSign) | ((q << 5) & kExpM);
    std::uint32_t v2 = ((q << 4) & kSign) | ((q << 1) & kExpM);
    std::uint32_t v3 = (q & kSign) | ((q >> 3) & kExpM);
    out[0]           = __hmul2(*reinterpret_cast<half2*>(&v0), rebias16384);
    out[1]           = __hmul2(*reinterpret_cast<half2*>(&v1), rebias16384);
    out[2]           = __hmul2(*reinterpret_cast<half2*>(&v2), rebias16384);
    out[3]           = __hmul2(*reinterpret_cast<half2*>(&v3), rebias16384);
}

// e4m3 group-scale decode (mirrors fp8_volta_qpn_gemm.cuh's fp8_decode_quad exactly): sign to fp16
// bit 15, seven exponent+mantissa bits up by 7. fp16's exponent bias (15) sits 8 above e4m3's (7),
// so this reproduces the value scaled by 2^-8; the caller folds that 256x correction into the
// divisor constant instead of paying a second multiply here (safe: unlike the e2m1 rebias, 256 is
// small enough that dividing it into the weight's global divisor never approaches fp16's range).
__device__ __forceinline__ half2 nvfp4_decode_e4m3_scale(std::uint8_t b) {
    const std::uint16_t hb = static_cast<std::uint16_t>((b & 0x80u) << 8) |
                             static_cast<std::uint16_t>((b & 0x7Fu) << 7);
    return __half2half2(__ushort_as_half(hb));
}

// Reorders eight sequential-k activations into the decoder's (j, j+4) grouping -- two PRMTs,
// register-only. `seq` and `out` do not alias.
__device__ __forceinline__ void nvfp4_stage_pairs(const half* seq, half2* out) {
    const auto* r          = reinterpret_cast<const std::uint32_t*>(seq);
    std::uint32_t o0 = __byte_perm(r[0], r[2], 0x5410);
    std::uint32_t o1 = __byte_perm(r[0], r[2], 0x7632);
    std::uint32_t o2 = __byte_perm(r[1], r[3], 0x5410);
    std::uint32_t o3 = __byte_perm(r[1], r[3], 0x7632);
    out[0]            = *reinterpret_cast<half2*>(&o0);
    out[1]            = *reinterpret_cast<half2*>(&o1);
    out[2]            = *reinterpret_cast<half2*>(&o2);
    out[3]            = *reinterpret_cast<half2*>(&o3);
}

// `kTiles` is the number of 8-row A tiles, so T <= 8 * kTiles. `SPLITK` is the number of warps per
// CTA splitting K (generalizing the fixed 4 the first three siblings still use); `NACC` is the
// number of independent accumulator fragments the four MMA slices of one group round-robin into,
// breaking the RAW dependency chain between consecutive mma.sync instructions into the same
// registers. Both are v100-skinny's own generation-2 knobs (the NVFP4 decode sweep):
// SPLITK in {4,8,16}, NACC in {1,2}, measured 441 -> 637 GB/s at M=8 on their kernel. The shared
// reduce buffer is SPLITK * kTiles * 256 floats, so the caller must keep that under Volta's 48 KB
// static limit -- SPLITK=16 at kTiles=4 does not fit and is not instantiated.
// Per-thread register need doesn't grow with SPLITK (each thread still does the same per-k work,
// just a shorter K loop), so the occupancy target below holds the *total* resident-block-warp
// count roughly fixed at what the SPLITK=4 design measured well -- 32 for kTiles=1, 16 for
// kTiles=2, 4 for kTiles=4 -- rather than holding minBlocksPerSm itself fixed while the block
// quadruples in size underneath it. Getting this wrong doesn't corrupt results, but it silently
// invalidates a sweep: an over-tight target at large SPLITK forces spills that have nothing to do
// with the SPLITK/NACC question being asked (measured 75-141 GB/s "SPLITK8" candidates before this
// fix, against SPLITK4's 230 -- a regression from the register budget, not from the schedule).
template <int kTiles, int SPLITK, int NACC, class OutputPolicy, class Activation>
__global__ __launch_bounds__(
    SPLITK * 32, (kTiles == 1 ? 32 : kTiles == 2 ? 16 : 4) / SPLITK < 1
        ? 1
        : (kTiles == 1 ? 32 : kTiles == 2 ? 16 : 4) / SPLITK) void nvfp4_volta_qpn_gemm_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    const Activation* __restrict__ x, int n, int k, int t, float inverse_weight_divisor,
    OutputPolicy output) {
    using S = Nvfp4VoltaQpnSchedule;

    __shared__ float cs[SPLITK][kTiles * S::kRowsPerTile * S::kColsPerCta];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int qp   = (lane >> 2) & 3;
    const int r    = (lane & 3) + ((lane & 16) != 0 ? 4 : 0);

    const int col  = static_cast<int>(blockIdx.x) * S::kColsPerCta + qp * 8 + r;
    const int good = col < n;
    const int use_col = good ? col : 0;

    const int scale_tiles = k / S::kChunkK;
    const int stq          = scale_tiles / SPLITK;
    const int st0           = warp * stq;
    const int stend         = (warp == SPLITK - 1) ? scale_tiles : st0 + stq;

    // BlockScaleK16M128x4 addressing (nvfp4_gemv.cuh's nvfp4_scale_offset, group = st * 4): scale
    // bytes for one row's four-group chunk are four contiguous bytes at this base, then a 512-byte
    // stride to the next chunk.
    const int m_tile      = use_col / 128;
    const int row_inner    = use_col - m_tile * 128;
    const int row_mod32    = row_inner & 31;
    const int row_quartile = row_inner >> 5;
    const std::int64_t scale_row_base = static_cast<std::int64_t>(m_tile) * scale_tiles * 512 +
                                        row_mod32 * 16 + row_quartile * 4;

    const std::uint8_t* crow = codes + static_cast<std::int64_t>(use_col) * (k / 2);

    const half2 rebias  = __float2half2_rn(16384.0f);
    const half2 divisor2 = __float2half2_rn(inverse_weight_divisor * 256.0f);

    float c[kTiles][NACC][8];
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int a = 0; a < NACC; ++a) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { c[tile][a][i] = 0.0f; }
        }
    }

    for (int st = st0; st < stend; ++st) {
        const std::uint8_t* cp = crow + static_cast<std::int64_t>(st) * S::kCodeBytesPerChunk;
        const uint4 code01 = __ldg(reinterpret_cast<const uint4*>(cp));
        const uint4 code23 = __ldg(reinterpret_cast<const uint4*>(cp + 16));
        const std::uint32_t scale_word = *reinterpret_cast<const std::uint32_t*>(
            scales + scale_row_base + static_cast<std::int64_t>(st) * 512);

#pragma unroll
        for (int g = 0; g < 4; ++g) {
            const std::uint32_t word_lo = g == 0 ? code01.x : g == 1 ? code01.z
                                          : g == 2 ? code23.x         : code23.z;
            const std::uint32_t word_hi = g == 0 ? code01.y : g == 1 ? code01.w
                                          : g == 2 ? code23.y         : code23.w;
            const std::uint8_t scale_byte =
                static_cast<std::uint8_t>(scale_word >> (8 * g));
            const half2 sc2 = __hmul2(nvfp4_decode_e4m3_scale(scale_byte), divisor2);
            half2 b[8];
            nvfp4_decode_e2m1_quad(word_lo, rebias, *reinterpret_cast<half2(*)[4]>(&b[0]));
            nvfp4_decode_e2m1_quad(word_hi, rebias, *reinterpret_cast<half2(*)[4]>(&b[4]));
#pragma unroll
            for (int j = 0; j < 8; ++j) { b[j] = __hmul2(b[j], sc2); }
            const unsigned* B = reinterpret_cast<const unsigned*>(b);
            const int kbase   = st * S::kChunkK + g * S::kGroupK;

#pragma unroll
            for (int tile = 0; tile < kTiles; ++tile) {
                const int row = tile * S::kRowsPerTile + r;
                half2 a[8];
                if (row < t) {
                    const Activation* xrow = x + static_cast<std::int64_t>(row) * k + kbase;
                    const uint4 raw0 = *reinterpret_cast<const uint4*>(xrow);
                    const uint4 raw1 = *reinterpret_cast<const uint4*>(xrow + 8);
                    const auto* src0 = reinterpret_cast<const Activation*>(&raw0);
                    const auto* src1 = reinterpret_cast<const Activation*>(&raw1);
                    half seq0[8];
                    half seq1[8];
                    if constexpr (std::is_same_v<Activation, half>) {
#pragma unroll
                        for (int j = 0; j < 8; ++j) { seq0[j] = src0[j]; }
#pragma unroll
                        for (int j = 0; j < 8; ++j) { seq1[j] = src1[j]; }
                    } else {
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            seq0[j] = __float2half(__bfloat162float(src0[j]));
                        }
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            seq1[j] = __float2half(__bfloat162float(src1[j]));
                        }
                    }
                    nvfp4_stage_pairs(seq0, &a[0]);
                    nvfp4_stage_pairs(seq1, &a[4]);
                } else {
#pragma unroll
                    for (int j = 0; j < 8; ++j) { a[j] = __half2half2(__ushort_as_half(0)); }
                }
                const unsigned* A = reinterpret_cast<const unsigned*>(a);
                volta_mma_qp_n(c[tile][0 % NACC], A[0], A[1], B[0], B[1]);
                volta_mma_qp_n(c[tile][1 % NACC], A[2], A[3], B[2], B[3]);
                volta_mma_qp_n(c[tile][2 % NACC], A[4], A[5], B[4], B[5]);
                volta_mma_qp_n(c[tile][3 % NACC], A[6], A[7], B[6], B[7]);
            }
        }
    }

    // Fold the independent accumulator chains together before the cross-warp reduce.
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int a = 1; a < NACC; ++a) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { c[tile][0][i] += c[tile][a][i]; }
        }
    }

    // C map (v100-skinny mma8_probe.cu, roles swapped): identical to the other three siblings.
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = (i & 2) | ((lane & 16) != 0 ? 4 : 0) | (lane & 1);
            const int cl  = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
            cs[warp][(tile * S::kRowsPerTile + row) * S::kColsPerCta + qp * 8 + cl] = c[tile][0][i];
        }
    }
    __syncthreads(); // the only barrier: cross-warp K reduce

    constexpr int kOut = kTiles * S::kRowsPerTile * S::kColsPerCta;
    for (int e = static_cast<int>(threadIdx.x); e < kOut; e += SPLITK * 32) {
        const int row  = e / S::kColsPerCta;
        const int cl   = e % S::kColsPerCta;
        const int ocol = static_cast<int>(blockIdx.x) * S::kColsPerCta + cl;
        if (row < t && ocol < n) {
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < SPLITK; ++w) { v += cs[w][e]; }
            // The group scale (and the 2^-14 rebias correction) was already applied per value
            // during decode, so unlike the FP8 sibling the epilogue needs no further multiply --
            // matching Q4, whose per-group scale is likewise folded in before the mma.
            output.store(ocol, row, v);
        }
    }
}

// e2m1 decode without the 2^14 rebias: out[j] = 2^-14 * value. The prepacked kernel folds the
// rebias into its per-group scale multiply instead (see nvfp4_prepacked_scale).
__device__ __forceinline__ void nvfp4_decode_e2m1_quad_unscaled(std::uint32_t q, half2 (&out)[4]) {
    constexpr std::uint32_t kSign = 0x80008000u;
    constexpr std::uint32_t kExpM = 0x0E000E00u;
    std::uint32_t v0 = ((q << 12) & kSign) | ((q << 9) & kExpM);
    std::uint32_t v1 = ((q << 8) & kSign) | ((q << 5) & kExpM);
    std::uint32_t v2 = ((q << 4) & kSign) | ((q << 1) & kExpM);
    std::uint32_t v3 = (q & kSign) | ((q >> 3) & kExpM);
    out[0]           = *reinterpret_cast<half2*>(&v0);
    out[1]           = *reinterpret_cast<half2*>(&v1);
    out[2]           = *reinterpret_cast<half2*>(&v2);
    out[3]           = *reinterpret_cast<half2*>(&v3);
}

// The prepacked kernel's dequant constant, split on the host. The unprepacked kernel multiplies
// every decoded e2m1 value by 2^14 and then by the group scale: two HMUL2 per half2. Folding the
// 2^14 into the group scale leaves one, but the file header's warning stands -- the group scale
// times 2^14 can leave fp16. The fold is only exact while it stays finite, so the host bounds it
// for the *worst* e4m3 byte (448 * 2^-8 = 1.75 after the shift decode) and hands any excess to a
// power-of-two `output_scale` the epilogue applies in fp32. Real checkpoints never need it
// (inverse divisors ~1e-4 keep the product near 1e3), and while `output_scale` is 1 every weight
// is bit-identical to the two-multiply form: scaling by a power of two commutes with rounding.
struct Nvfp4PrepackedScale {
    float divisor;
    float output_scale;
};

inline Nvfp4PrepackedScale nvfp4_prepacked_scale(float inverse_weight_divisor) {
    Nvfp4PrepackedScale scale{inverse_weight_divisor * 256.0F * 16384.0F, 1.0F};
    while (scale.divisor * 1.75F > 32768.0F) {
        scale.divisor *= 0.5F;
        scale.output_scale *= 2.0F;
    }
    return scale;
}

// Load-time-prepacked companion. Codes are [N/32 tile][K/16 group][lane32][8B], with
// the nibble order chosen so the shift decoder's structural (i,i+4) pairs become adjacent K.
// This is v100-skinny's QPN2 main loop: no activation PRMTs and one scale byte per group.
//
// The loop is issue-bound as much as it is DRAM-bound (ncu: long_scoreboard 33%, not_selected
// 19%, math_pipe_throttle 15% at ~81% of DRAM), so it is written to issue as little as possible
// per 16-k group besides the 16 HMMA steps it cannot shed:
//   - one HMUL2 per decoded half2 (the rebias folded into the group scale, above);
//   - activation rows past `t` read row t-1 instead of predicating the load and zeroing the
//     fragment: a garbage A row only lands in its own C row, which the epilogue never stores;
//   - running pointers, so every load in a batch is a constant offset from one base.
template <int kTiles, int SPLITK, int NACC, class OutputPolicy, class Activation>
__global__ __launch_bounds__(
    SPLITK * 32, (kTiles == 1 ? 32 : kTiles == 2 ? 16 : 4) / SPLITK < 1
        ? 1
        : (kTiles == 1 ? 32 : kTiles == 2 ? 16 : 4) / SPLITK)
void nvfp4_volta_qpn_prepacked_kernel(const std::uint8_t* __restrict__ codes,
                                      const std::uint8_t* __restrict__ scales,
                                      const Activation* __restrict__ x, int n, int k, int t,
                                      float folded_divisor, float output_scale,
                                      OutputPolicy output) {
    using S = Nvfp4VoltaQpnSchedule;
    __shared__ float cs[SPLITK][kTiles * S::kRowsPerTile * S::kColsPerCta];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int qp   = (lane >> 2) & 3;
    const int r    = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups = k / S::kGroupK;
    const int quotient = groups / SPLITK;
    const int g0       = warp * quotient;
    const int gend     = warp == SPLITK - 1 ? groups : g0 + quotient;
    const std::int64_t first_tuple =
        (static_cast<std::int64_t>(blockIdx.x) * groups + g0) * 32 + lane;
    const half2 divisor2 = __float2half2_rn(folded_divisor);

    const uint2* code_ptr          = reinterpret_cast<const uint2*>(codes) + first_tuple;
    const std::uint8_t* scale_ptr  = scales + first_tuple;
    const Activation* xrow[kTiles];
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
        const int row = min(tile * S::kRowsPerTile + r, t - 1);
        xrow[tile]    = x + static_cast<std::int64_t>(row) * k + g0 * S::kGroupK;
    }

    float c[kTiles][NACC][8];
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int a = 0; a < NACC; ++a) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { c[tile][a][i] = 0.0f; }
        }
    }

    // One group is only 8 bytes of codes per lane, and each is consumed by a dependent
    // decode + four mma before the next load issues, so a warp keeps a single 256-byte request
    // in flight -- far below what HBM needs at 2 CTAs/SM (the 5120x17408 down projection ran at
    // 37% of bandwidth). Issue kLoadBatch groups' code and scale loads together first.
    constexpr int kLoadBatch = 4;
    const auto consume_group = [&](uint2 q2, std::uint8_t scale_byte, int k_offset) {
        const half2 sc2 = __hmul2(nvfp4_decode_e4m3_scale(scale_byte), divisor2);
        half2 b[8];
        nvfp4_decode_e2m1_quad_unscaled(q2.x, *reinterpret_cast<half2(*)[4]>(&b[0]));
        nvfp4_decode_e2m1_quad_unscaled(q2.y, *reinterpret_cast<half2(*)[4]>(&b[4]));
#pragma unroll
        for (int j = 0; j < 8; ++j) { b[j] = __hmul2(b[j], sc2); }
        const unsigned* B = reinterpret_cast<const unsigned*>(b);

#pragma unroll
        for (int tile = 0; tile < kTiles; ++tile) {
            half values[16];
            const Activation* source = xrow[tile] + k_offset;
            const uint4 raw0 = *reinterpret_cast<const uint4*>(source);
            const uint4 raw1 = *reinterpret_cast<const uint4*>(source + 8);
            const auto* src0 = reinterpret_cast<const Activation*>(&raw0);
            const auto* src1 = reinterpret_cast<const Activation*>(&raw1);
            if constexpr (std::is_same_v<Activation, half>) {
#pragma unroll
                for (int j = 0; j < 8; ++j) { values[j] = src0[j]; }
#pragma unroll
                for (int j = 0; j < 8; ++j) { values[j + 8] = src1[j]; }
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) { values[j] = __float2half(__bfloat162float(src0[j])); }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    values[j + 8] = __float2half(__bfloat162float(src1[j]));
                }
            }
            const unsigned* A = reinterpret_cast<const unsigned*>(values);
            volta_mma_qp_n(c[tile][0 % NACC], A[0], A[1], B[0], B[1]);
            volta_mma_qp_n(c[tile][1 % NACC], A[2], A[3], B[2], B[3]);
            volta_mma_qp_n(c[tile][2 % NACC], A[4], A[5], B[4], B[5]);
            volta_mma_qp_n(c[tile][3 % NACC], A[6], A[7], B[6], B[7]);
        }
    };

    const int count = gend - g0;
    int done        = 0;
    for (; done + kLoadBatch <= count; done += kLoadBatch) {
        uint2 q2[kLoadBatch];
        std::uint8_t scale_bytes[kLoadBatch];
#pragma unroll
        for (int u = 0; u < kLoadBatch; ++u) {
            q2[u]          = __ldg(code_ptr + u * 32);
            scale_bytes[u] = __ldg(scale_ptr + u * 32);
        }
#pragma unroll
        for (int u = 0; u < kLoadBatch; ++u) {
            consume_group(q2[u], scale_bytes[u], u * S::kGroupK);
        }
        code_ptr += kLoadBatch * 32;
        scale_ptr += kLoadBatch * 32;
#pragma unroll
        for (int tile = 0; tile < kTiles; ++tile) { xrow[tile] += kLoadBatch * S::kGroupK; }
    }
    for (; done < count; ++done) {
        consume_group(__ldg(code_ptr), __ldg(scale_ptr), 0);
        code_ptr += 32;
        scale_ptr += 32;
#pragma unroll
        for (int tile = 0; tile < kTiles; ++tile) { xrow[tile] += S::kGroupK; }
    }

#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int a = 1; a < NACC; ++a) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { c[tile][0][i] += c[tile][a][i]; }
        }
    }
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
            const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
            cs[warp][(tile * S::kRowsPerTile + row) * S::kColsPerCta + qp * 8 + col] =
                c[tile][0][i];
        }
    }
    __syncthreads();

    if constexpr (nvfp4_swiglu_pairs_v<OutputPolicy>) {
        // VoltaQpnPrepackedSwiGlu: columns [0, 16) of this CTA are gate features, [16, 32) the
        // same up features.
        constexpr int kHalf  = S::kColsPerCta / 2;
        constexpr int kPairs = kTiles * S::kRowsPerTile * kHalf;
        for (int e = static_cast<int>(threadIdx.x); e < kPairs; e += SPLITK * 32) {
            const int row     = e / kHalf;
            const int local   = e % kHalf;
            const int feature = static_cast<int>(blockIdx.x) * kHalf + local;
            if (row < t && feature < n / 2) {
                const int gate_index = row * S::kColsPerCta + local;
                float gate           = 0.0f;
                float up             = 0.0f;
#pragma unroll
                for (int w = 0; w < SPLITK; ++w) {
                    gate += cs[w][gate_index];
                    up += cs[w][gate_index + kHalf];
                }
                output.store_pair(feature, row, gate * output_scale, up * output_scale);
            }
        }
    } else {
        constexpr int kOut = kTiles * S::kRowsPerTile * S::kColsPerCta;
        for (int e = static_cast<int>(threadIdx.x); e < kOut; e += SPLITK * 32) {
            const int row  = e / S::kColsPerCta;
            const int col  = e % S::kColsPerCta;
            const int ocol = static_cast<int>(blockIdx.x) * S::kColsPerCta + col;
            if (row < t && ocol < n) {
                float value = 0.0f;
#pragma unroll
                for (int w = 0; w < SPLITK; ++w) { value += cs[w][e]; }
                output.store(ocol, row, value * output_scale);
            }
        }
    }
}

// Shared launcher. Every NVFP4 consumer that wants the plain contiguous output takes this
// directly; a fused consumer supplies its own OutputPolicy.
//
// Generation-2 SPLITK/NACC winners from a private sweep (bench/ops/nvfp4_qpn2_splitk_sweep.cu,
// the NVFP4 decode sweep for the measured envelope). Per-shape, not
// universal -- CTA count (n/32) is what actually decides it, and the three measured shapes land in
// three different places:
//   - gate_up (34816 rows, 1088 CTAs): SPLITK=16 at kTiles=1, SPLITK=8 at kTiles=2 (16 is *worse*
//     there -- 217.1 vs 236.5 GB/s at T=16).
//   - the split SwiGLU kernel's per-half shape (17408 rows, 544 CTAs): SPLITK=16 at *both*
//     kTiles=1 and kTiles=2 (344.2 vs 279.8 GB/s at T=4; 214.4 vs 209.3 at T=16) --
//     bench/ops/nvfp4_qpn2_split_shape_sweep.cu, deleted.
//   - down (5120 rows, 160 CTAs): SPLITK=8 tops out, 16 is worse -- for the unbatched loop. The
//     prepacked kernel's batched group loads flip that: SPLITK=16 there (see `down` below).
// NACC barely moves any of them once SPLITK is right, so it only varies where it measured a real
// (if small) edge. Geometries this dispatch doesn't name fall to the SPLITK=8 default, which was
// never worse than production's SPLITK=4 baseline on any measured shape: the attn/gdn inputs, and
// the 6144-residual, the attention and GDN output projections (prepacked for Qwen3.8 nvfp4-full-c,
// checkpoint-native for Qwen3.6 nvfp4). SPLITK=16 on the prepacked 6144-residual, as on `down`,
// has not been measured.
template <int kTiles, int SPLITK, int NACC, class Activation, class OutputPolicy>
void launch_nvfp4_qpn_schedule(bool prepacked, dim3 grid, const std::uint8_t* codes,
                               const std::uint8_t* scales, const Activation* x, int n, int k,
                               int t, float inverse_weight_divisor, OutputPolicy output,
                               cudaStream_t stream) {
    if (prepacked) {
        const Nvfp4PrepackedScale scale = nvfp4_prepacked_scale(inverse_weight_divisor);
        nvfp4_volta_qpn_prepacked_kernel<kTiles, SPLITK, NACC>
            <<<grid, SPLITK * 32, 0, stream>>>(codes, scales, x, n, k, t, scale.divisor,
                                               scale.output_scale, output);
    } else if constexpr (!nvfp4_swiglu_pairs_v<OutputPolicy>) {
        nvfp4_volta_qpn_gemm_kernel<kTiles, SPLITK, NACC>
            <<<grid, SPLITK * 32, 0, stream>>>(codes, scales, x, n, k, t,
                                               inverse_weight_divisor, output);
    } else {
        throw std::logic_error("NVFP4 QPN2 SwiGLU pairs need the interleaved prepacked layout");
    }
}

template <class Activation, class OutputPolicy>
void launch_nvfp4_volta_qpn_with_activation(const Tensor& x, const Weight& w,
                                            const Activation* xd, OutputPolicy output,
                                            std::int32_t n, float inverse_weight_divisor,
                                            cudaStream_t stream) {
    using S              = Nvfp4VoltaQpnSchedule;
    const std::int32_t k = x.ne[0];
    const std::int32_t t = x.ne[1];

    const dim3 grid(static_cast<unsigned>((n + S::kColsPerCta - 1) / S::kColsPerCta));
    const auto* codes  = static_cast<const std::uint8_t*>(w.qdata);
    const auto* scales = static_cast<const std::uint8_t*>(w.scales);
    const bool prepacked  = w.layout == QuantLayout::VoltaQpnPrepacked ||
                           w.layout == QuantLayout::VoltaQpnPrepackedSwiGlu;
    const bool gate_up    = (n == 34816 && k == 5120);
    const bool split_half = (n == 17408 && k == 5120);
    // The down projection's 160 CTAs leave SPLITK=8 at 2 CTAs/SM (16 warps) short of the
    // requests in flight HBM needs; with the batched group loads SPLITK=16 measured 157 -> 114 us
    // at T=6 (322 -> 444 GB/s), where the unbatched kernel had SPLITK=8 on top.
    // Only the prepacked kernel has the batched loads that make SPLITK=16 pay on this shape.
    const bool down       = prepacked && n == 5120 && k == 17408;
    if (t <= S::kRowsPerTile) {
        if (gate_up || split_half || down) {
            launch_nvfp4_qpn_schedule<1, 16, 2>(prepacked, grid, codes, scales, xd, n, k, t,
                                                inverse_weight_divisor, output, stream);
        } else {
            launch_nvfp4_qpn_schedule<1, 8, 2>(prepacked, grid, codes, scales, xd, n, k, t,
                                               inverse_weight_divisor, output, stream);
        }
    } else if (t <= 2 * S::kRowsPerTile) {
        if (split_half) {
            // NACC1 and NACC2 tied exactly at T=16 (233.8us both); NACC1 for the smaller register
            // footprint.
            launch_nvfp4_qpn_schedule<2, 16, 1>(prepacked, grid, codes, scales, xd, n, k, t,
                                                inverse_weight_divisor, output, stream);
        } else {
            launch_nvfp4_qpn_schedule<2, 8, 1>(prepacked, grid, codes, scales, xd, n, k, t,
                                               inverse_weight_divisor, output, stream);
        }
    } else {
        launch_nvfp4_qpn_schedule<4, 8, 1>(prepacked, grid, codes, scales, xd, n, k, t,
                                           inverse_weight_divisor, output, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class OutputPolicy>
void launch_nvfp4_volta_qpn_with_output(const Tensor& x, const Weight& w, OutputPolicy output,
                                        std::int32_t n, float inverse_weight_divisor,
                                        cudaStream_t stream) {
    launch_nvfp4_volta_qpn_with_activation(
        x, w, static_cast<const __nv_bfloat16*>(x.data), output, n, inverse_weight_divisor,
        stream);
}

template <class OutputPolicy>
void launch_nvfp4_volta_qpn_with_fp16_activation(const Tensor& x, const Weight& w,
                                                 const half* x_fp16, OutputPolicy output,
                                                 std::int32_t n, float inverse_weight_divisor,
                                                 cudaStream_t stream) {
    launch_nvfp4_volta_qpn_with_activation(x, w, x_fp16, output, n, inverse_weight_divisor,
                                           stream);
}

#endif // sm_70

} // namespace ninfer::ops::detail
