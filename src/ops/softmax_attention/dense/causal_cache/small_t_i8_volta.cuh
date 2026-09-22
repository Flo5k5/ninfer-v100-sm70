#pragma once

// ninfer::ops - Volta (sm_70) tensor-core GQA small-T attention partial kernel, INT8-G64 cache.
//
// This is gqa_attention_small_t_tc_volta_partial_kernel (gqa_attention_prefill_volta.cuh) with
// the KV cache dtype swapped from bf16 to the int8 + per-64-group-scale codec. The tile
// topology, QK^T/PV compute core, online-softmax recurrence and partial_* output format are
// character-for-character the same, and the file comment below still governs them; keep the two
// files in step when either changes.
//
// Why it exists: the INT8 decode path was SIMT-only, and SIMT is what caps it. Measured at 82k,
// width 4, the INT8 SIMT kernel runs at 1.86 TFLOP/s and the bf16 tensor-core kernel at 2.71 --
// so bf16 beat int8 by 1.46x *while reading twice the bytes*, purely because it had tensor cores.
// The two properties are separable: this kernel takes the tensor-core math and keeps int8's
// halved traffic. See docs/v100.md.
//
// The only structural difference is where the fp16 that feeds mma.sync comes from. The bf16
// kernel copies raw bf16 bytes into k_s/v_s and converts them in place, which works because bf16
// and fp16 are both two bytes. int8 codes are one byte, so an in-place expansion would clobber
// its own neighbours; instead the staging loop dequantizes through registers on the way in. That
// costs nothing here, because cp_async below sm_80 is already a synchronous load+store through
// registers (ops/common/memory.cuh) -- there is no async pipeline to break.
//
// Structurally mirrors gqa_attention_small_t_tc_partial_bf16_kernel's Ampere+ branch
// (gqa_attention_decode_bf16.cuh) -- same paged-KV addressing, GQA row mapping, and
// cache-write handling, reused via the shared helpers in gqa_attention_decode.cuh -- but
// the tile/warp topology and QK^T/PV compute core are rebuilt from scratch on Volta's
// mma.sync.m8n8k4 primitives (ops/common/volta_mma.cuh), since Volta has neither
// ldmatrix nor m16n8k16. Writes the same partial_acc/partial_m/partial_l format the
// existing reduce kernel already consumes, so no changes are needed anywhere else in the
// online-softmax merge pipeline.
//
// Why this kernel's tile topology differs from the Ampere+ kernel's (see docs/v100.md
// for the full derivation -- this is the load-bearing design note, read it before changing
// any of the constants below):
//
//   1. Register wall. Ampere's ldmatrix-based PV tile packs 2 output rows x 2 columns per
//      thread per mma call; Volta's mma.m8n8k4 packs only 1 row x 8 columns per thread per
//      call. Holding the *entire* D=256 head-dim output resident across the whole key loop
//      (as the Ampere kernel does, ~128 registers) would cost Volta 32 chunks x 8 floats =
//      256 registers/thread for a *single* output row -- already at Volta's hard 255/thread
//      cap with nothing left for Q/K fragments or loop state. Fix: split the head dimension
//      across warps (DimSplit=4 below) instead of giving each warp the full D range, so each
//      warp only holds D/4=64 columns resident (8 chunks x 8 floats = 64 registers). Warps
//      that share a row-tile also split the QK^T contraction along D (split-K): each
//      contracts its own D/4 slice and the four partial score tiles are summed through shared
//      memory in a fixed order, so every warp runs the (cheap) online-softmax step on
//      bit-identical scores and they only diverge for the PV accumulate. Recomputing the
//      full-D QK^T in every warp made this kernel issue-bound at long context: QK^T is four
//      times the PV tensor work per warp.
//
//   2. Shared-memory wall. This kernel reloads Q from shared memory every key-tile
//      iteration rather than keeping Q fragments resident in registers (that trade-off is
//      what keeps (1)'s register budget low -- persisting Q *and* a full-width accumulator
//      would both want ~128 registers, right back at the wall). That means Q must stay
//      resident in shared memory for the whole key loop, so -- unlike the Ampere kernel,
//      which stages Q into the same shared buffer it later reuses for K/V once Q has been
//      consumed into registers -- this kernel needs separate, simultaneously-live q_s/k_s/
//      v_s buffers. To fit that under the 48KB static-shared-memory default, Br is fixed at
//      one 32-row Volta tile (not scaled by warp count) and Bc is halved to 16 keys/tile;
//      q_s+k_s+v_s then costs 16KB+8KB+8KB = 32KB, comfortably under budget.
//
//   3. Row capacity. Fixing Br=32 means a single tile can't cover every (TokenTile x
//      GroupSize) row count this op needs (up to 6*8=48 for the widest GQA geometry this
//      codebase instantiates). The general route therefore loops over 32-row passes. The hot
//      27B width-six shape is exactly 36 rows, so its long-context route instead gives the
//      four-row tail to a fifth warp. That warp uses the four independent mma quadpairs as
//      D slices, for PV as well as for its QK^T contraction (reduced across quadpairs by
//      shuffle), sharing the first tile's K/V walk rather than starting a second pass.

#include "ops/common/volta_mma.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

__device__ __forceinline__ int4 causal_kv_dequant_i8x8_f16_from(
    const std::int8_t* codes8, float scale) {
    const int2 raw       = load_vec<int2>(codes8);
    const std::int8_t* c = reinterpret_cast<const std::int8_t*>(&raw);
    __half2 packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        packed[i] = __floats2half2_rn(static_cast<float>(c[2 * i]) * scale,
                                      static_cast<float>(c[2 * i + 1]) * scale);
    }
    return *reinterpret_cast<const int4*>(packed);
}

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput>
__launch_bounds__(WarpsPerCta * 32, 2) __global__
    void causal_attention_small_t_tc_volta_partial_i8_kernel(
    const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, std::int8_t* cache_k_i8,
    std::int8_t* cache_v_i8, __half* cache_k_scale, __half* cache_v_scale,
    const std::int32_t* block_tables, const std::int32_t* valid_columns,
    const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t logical_capacity, float scale,
    float* partial_acc, float* partial_m, float* partial_l) {
    static_assert(TokenTile >= 1 && TokenTile * Geometry::GroupSize <= 48);
    static_assert(WarpsPerCta == 4 || WarpsPerCta == 5);

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
    constexpr int DimSplit      = 4;
    constexpr bool CompactTail  = WarpsPerCta == 5;
    constexpr int Br            = 32;          // one Volta tile's worth of rows per pass
    constexpr int Bc            = 16;          // keys per shared-memory tile
    constexpr int D             = kCausalHeadDim;
    constexpr int Threads       = WarpsPerCta * 32;
    constexpr int PVChunks      = Bc / 8;          // key sub-groups per Bc tile
    constexpr int DSlice        = D / DimSplit;    // this warp's PV output width
    constexpr int DChunksLocal  = DSlice / 8;       // this warp's resident accumulator chunks
    constexpr int PageIds       = 64;
    constexpr int Groups        = D / kKVCacheInt8Group; // quant groups per key row
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(D % kKVCacheInt8Group == 0, "head dim must divide into whole quant groups");
    static_assert(kKVCacheInt8Group % 8 == 0,
                  "an 8-wide staging chunk must sit inside one quant group, so it needs one scale");

    // Declared fp16 (not bf16): Volta's mma.sync.m8n8k4 only accepts fp16 operands. Unlike the
    // bf16 sibling kernel, which lands raw cache bytes here and reinterprets them in place, K/V
    // arrive already dequantized to fp16 -- int8 codes are half the width of the destination, so
    // an in-place expansion would overwrite the neighbouring chunk. Q is converted inline at the
    // point of load, since it's read scalar-wise.
    // Row stride is padded, not D. The mma feed reads *down* a column of these tiles -- one
    // half2 per row at a fixed d -- so with an unpadded 512-byte row stride (D=256 halves) every
    // row lands on the same shared-memory bank, since banks wrap every 128 bytes. ncu measured
    // 724M bank conflicts against 891M wavefronts for 42.2M shared loads: ~21 replays per
    // instruction, near the 32-way worst case, and L1/TEX throughput pinned at 85% while the
    // tensor pipe idled. SmemPad shifts each row by a whole 16-byte vector so consecutive rows
    // start 4 banks apart, which keeps every 16-byte store in the staging loop aligned.
    constexpr int SmemPad    = 8;
    constexpr int SmemStride = D + SmemPad;
    static_assert(SmemPad % 8 == 0, "pad must preserve 16-byte alignment of the staging stores");
    __shared__ __align__(16) half q_s[Br * SmemStride];
    __shared__ __align__(16) half q_tail_s[4 * SmemStride];
    __shared__ __align__(16) half p_tail_s[8 * 8];
    __shared__ __align__(16) half k_s[Bc * SmemStride];
    __shared__ __align__(16) half v_s[Bc * SmemStride];
    __shared__ std::int32_t physical_pages_s[PageIds];
    // Split-K QK^T exchange. Each dim-split warp contracts only its own DSlice of the head dim
    // and publishes that partial 32x8 score tile per 8-key sub-group; every dim-split warp then
    // sums the DimSplit partials in the same fixed order, so all of them hold bit-identical
    // scores and therefore identical online-softmax state (dim_warp 0 writes the m/l the others'
    // PV slices are normalized by). Layout [warp][sub][half][lane][4] keeps the 16-byte
    // accesses of a warp on distinct banks.
    __shared__ __align__(16) float qk_part_s[DimSplit * PVChunks * 2 * 32 * 4];
    const auto qk_part_index = [](int part_warp, int sub, int half_index, int part_lane) {
        return (((part_warp * PVChunks + sub) * 2 + half_index) * 32 + part_lane) * 4;
    };
    // Quadpair-split QK^T for the compact tail warp: at step s quadpair qp contracts head dims
    // [d0, d0+4). The interleave spreads one step's K-row loads over all 32 banks
    // (bank = 4*row + 4*(s&3) + 2*(qp&1) + 16*(qp>>1) mod 32, for both the Q and the K row)
    // while the 16 steps of the four quadpairs still tile D=256 exactly once.
    const auto tail_qk_d0 = [](int step, int quadpair) {
        return (step >> 2) * 64 + (step & 3) * 8 + (quadpair & 1) * 4 + (quadpair >> 1) * 32;
    };
    constexpr int TailQkSteps = D / (4 * 4);

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    const int dim_warp    = warp < DimSplit ? warp : 0;
    int valid_tokens       = tokens;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens         = remaining <= 0 ? 0 : (remaining < tokens ? remaining : tokens);
    }
    const int row_count = tokens * Geometry::GroupSize;

    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(kCausalHeadDim) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(kCausalHeadDim) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(kCausalHeadDim) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc += static_cast<std::int64_t>(batch) * kCausalHeadDim * Geometry::QHeads * tokens *
                       split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < row_count; row += Threads) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] =
                    -CUDART_INF_F;
                partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] = 0.0f;
            }
        }
        for (int idx = tid; idx < row_count * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[causal_partial_acc_index<Geometry>(q_head, d, token, split, tokens)] =
                    0.0f;
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || tokens < 1 || tokens > TokenTile ||
        split_count <= 0) {
        return;
    }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[tokens - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int window = last_pos + 1;
    const int active_split_count =
        causal_small_t_active_splits<Geometry, true>(window, split_count, TokenTile);
    if (split >= active_split_count) { return; }

    const int logical_tiles = div_up(window, Bc);
    const bool tile_split   = logical_tiles >= active_split_count;
    const int units_per_split =
        tile_split ? div_up(logical_tiles, active_split_count) : div_up(window, active_split_count);
    const int split_start = split * units_per_split * (tile_split ? Bc : 1);
    const int split_limit = split_start + units_per_split * (tile_split ? Bc : 1);
    const int split_end   = (split_limit < window) ? split_limit : window;
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    const int first_tile = (split_start / Bc) * Bc;
    const int key_blocks = div_up(split_end - first_tile, Bc);
    const int first_page = first_tile >> kPagedKVPageShift;
    const int page_count = ((split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    for (int page = tid; page < page_count; page += Threads) {
        physical_pages_s[page] = block_table[first_page + page];
    }

    if constexpr (CacheInput::writes_cache) {
        // One warp owns a D256 row, applies the registered normalized transform to K, and then
        // emits all four G64 groups. V remains in its native coordinates.
        for (int token = warp; token < valid_tokens; token += WarpsPerCta) {
            const int position = pos[token];
            if (position < split_start || position >= split_end || position < 0 ||
                position >= logical_capacity) {
                continue;
            }
            float k_values[8];
            float v_values[8];
#pragma unroll
            for (int part = 0; part < 8; ++part) {
                const int d = lane + 32 * part;
                const std::int64_t source = kv_cache_int8_new_index<Geometry>(kv_head, d, token);
                k_values[part] = __bfloat162float(input.k[source]);
                v_values[part] = __bfloat162float(input.v[source]);
            }
            normalized_hadamard_d256_inplace(k_values, lane);

            int physical_page = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
            physical_page     = __shfl_sync(FullMask, physical_page, 0);
            const int page_offset = position & kPagedKVPageMask;
#pragma unroll
            for (int grp = 0; grp < Groups; ++grp) {
                float kamax = fmaxf(fabsf(k_values[2 * grp]), fabsf(k_values[2 * grp + 1]));
                float vamax = fmaxf(fabsf(v_values[2 * grp]), fabsf(v_values[2 * grp + 1]));
                kamax = warp_max(kamax, FullMask);
                vamax = warp_max(vamax, FullMask);
                const KVCacheInt8QuantParams kp = kv_cache_int8_quant_params(kamax);
                const KVCacheInt8QuantParams vp = kv_cache_int8_quant_params(vamax);
                const int d0 = grp * kKVCacheInt8Group + lane;
                const int d1 = d0 + 32;
                cache_k_i8[kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d0,
                                                                     page_offset)] =
                    kv_cache_int8_quant_code(k_values[2 * grp], kp.inverse_scale);
                cache_k_i8[kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d1,
                                                                     page_offset)] =
                    kv_cache_int8_quant_code(k_values[2 * grp + 1], kp.inverse_scale);
                cache_v_i8[kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d0,
                                                                     page_offset)] =
                    kv_cache_int8_quant_code(v_values[2 * grp], vp.inverse_scale);
                cache_v_i8[kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d1,
                                                                     page_offset)] =
                    kv_cache_int8_quant_code(v_values[2 * grp + 1], vp.inverse_scale);
                if (lane == 0) {
                const std::int64_t so =
                    kv_cache_int8_quant_scale_index<Geometry>(physical_page, kv_head, grp, page_offset);
                    cache_k_scale[so] = kp.scale;
                    cache_v_scale[so] = vp.scale;
                }
            }
        }
        __syncthreads();
    }

    // The width-six 27B specialization keeps the first 32 rows on the established four-warp
    // tensor-core mapping and assigns the four-row tail to one compact quadpair-split-D warp.
    // Both consume each staged K/V tile before it is replaced.
    const int main_row_count = CompactTail ? Br : row_count;
    for (int row_base = 0; row_base < main_row_count; row_base += Br) {
        const int rows_here =
            (main_row_count - row_base < Br) ? main_row_count - row_base : Br;

        if (!CompactTail || warp < DimSplit) {
            for (int row = dim_warp; row < Br; row += DimSplit) {
                int q_head      = 0;
                int token       = 0;
                float values[8] = {};
                if (row < rows_here) {
                    causal_small_t_tc_row_to_qt<Geometry>(row_base + row, tokens, kv_head, q_head,
                                                          token);
                    if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
#pragma unroll
                        for (int part = 0; part < 8; ++part) {
                            const int d = lane + 32 * part;
                            values[part] =
                                __bfloat162float(q[causal_q_index<Geometry>(q_head, d, token)]);
                        }
                        normalized_hadamard_d256_inplace(values, lane);
                    }
                }
#pragma unroll
                for (int part = 0; part < 8; ++part) {
                    q_s[row * SmemStride + lane + 32 * part] = __float2half(values[part]);
                }
            }
        }
        if constexpr (CompactTail) {
            if (warp == DimSplit) {
                for (int row = 0; row < 4; ++row) {
                    int q_head = 0;
                    int token  = 0;
                    causal_small_t_tc_row_to_qt<Geometry>(Br + row, tokens, kv_head, q_head,
                                                          token);
                    float values[8];
#pragma unroll
                    for (int part = 0; part < 8; ++part) {
                        const int d = lane + 32 * part;
                        values[part] =
                            __bfloat162float(q[causal_q_index<Geometry>(q_head, d, token)]);
                    }
                    normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
                    for (int part = 0; part < 8; ++part) {
                        q_tail_s[row * SmemStride + lane + 32 * part] =
                            __float2half(values[part]);
                    }
                }
            }
        }
        __syncthreads();

        int physical_page = physical_pages_s[0];

        // QK^T operands are read from shared memory as whole 16-byte fragments (the four half2
        // of a Volta Q/K fragment row are contiguous). With the 528-byte padded row stride the
        // eight lanes of each LDS.128 phase land on distinct 4-bank groups, so these reads are
        // conflict-free, unlike four scalar half2 loads. Keeping Q in registers instead would
        // cost 32 registers the five-warp route cannot spare under its 2-CTA/SM bound.
        const auto load_frag16 = [](half2 (&dst)[4], const half* src) {
            const int4 raw = *reinterpret_cast<const int4*>(src);
            const half2* h = reinterpret_cast<const half2*>(&raw);
#pragma unroll
            for (int l = 0; l < 4; ++l) { dst[l] = h[l]; }
        };
        const int q_frag_row  = volta_qp_get_i();
        const int k_frag_row  = volta_k_get_i();
        const int tail_quadpair = (lane >> 2) & 3;
        const int tail_row      = (lane & 3) + ((lane & 16) != 0 ? 4 : 0);

        // acc_f[c] holds this warp's DSlice-wide PV output for head-dim chunk c (columns
        // [dim_warp*DSlice + c*8, +8)), folded across key-tiles by the online-softmax
        // alpha rescale -- see file comment #1 for why this is DChunksLocal (not DChunks)
        // wide.
        float acc_f[DChunksLocal][8];
#pragma unroll
        for (int c = 0; c < DChunksLocal; ++c) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { acc_f[c][i] = 0.0f; }
        }
        // Every thread's 8 D-tile registers straddle TWO distinct rows (r_lo, r_hi -- see
        // volta_mma.cuh), so their softmax state must be tracked as two independent series,
        // not collapsed into one: thread `lane` and thread `lane^2` both compute IDENTICAL
        // bm_lo/bm_hi (each is already a full 2-lane reduction), and both update m_lo/l_lo
        // and m_hi/l_hi identically from those identical inputs -- so every thread has a
        // fully correct, non-mixed view of both rows' state, and can normalize d_score's
        // row_lo group against m_lo and its row_hi group against m_hi (previously this used
        // a single shared max/alpha for both groups, silently corrupting whichever group
        // wasn't "this thread's own row" -- masked by the very first softmax update, since
        // then m_lo==m_hi==-inf makes alpha 0 regardless, but wrong from the second update
        // onward whenever the two rows' true maxima differ).
        float m_lo = -CUDART_INF_F, m_hi = -CUDART_INF_F;
        float l_lo = 0.0f, l_hi = 0.0f;

        for (int kb = 0; kb < key_blocks; ++kb) {
            const int k0 = first_tile + kb * Bc;
            if (kb != 0 && (k0 & kPagedKVPageMask) == 0) {
                physical_page = physical_pages_s[(k0 >> kPagedKVPageShift) - first_page];
            }

            // Stage K/V for this key tile, dequantizing int8 -> fp16 on the way in. Unlike the
            // bf16 sibling there is no second in-place conversion pass: the codes are half the
            // width of the fp16 destination, so they are widened in registers between the load
            // and the store. Every key in range is read from the cache, including this tile's
            // own new tokens -- the append block above has already written and __syncthreads()'d
            // them, so there is no need for the bf16 kernel's separate "from_new" source.
            for (int chunk = tid; chunk < Bc * (D / 8); chunk += Threads) {
                const int key_l = chunk / (D / 8);
                const int d     = (chunk - key_l * (D / 8)) * 8;
                const int key   = k0 + key_l;
                half* k_dst     = &k_s[key_l * SmemStride + d];
                half* v_dst     = &v_s[key_l * SmemStride + d];
                if (key >= split_start && key < split_end) {
                    const int page_offset = key & kPagedKVPageMask;
                    const std::int64_t code_off =
                        kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d, page_offset);
                    const std::int64_t scale_off = kv_cache_int8_quant_scale_index<Geometry>(
                        physical_page, kv_head, d / kKVCacheInt8Group, page_offset);
                    const float ks = __half2float(cache_k_scale[scale_off]);
                    const float vs = __half2float(cache_v_scale[scale_off]);
                    store_vec(k_dst, causal_kv_dequant_i8x8_f16_from(&cache_k_i8[code_off], ks));
                    store_vec(v_dst, causal_kv_dequant_i8x8_f16_from(&cache_v_i8[code_off], vs));
                } else {
                    store_vec(k_dst, make_int4(0, 0, 0, 0));
                    store_vec(v_dst, make_int4(0, 0, 0, 0));
                }
            }
            __syncthreads();

            // --- QK^T, split-K over the head dim. A dim-split warp contracts only its own
            // DSlice (DChunksLocal of the D/8 chunks) for both 8-key sub-groups and publishes the
            // partial tiles; the tail warp contracts its four rows with the quadpair-split
            // mapping and reduces across quadpairs by shuffle. This replaces DimSplit+1
            // identical full-D contractions per sub-group (file comment #1) with one. ---
            float tail_scores[CompactTail ? PVChunks : 1][8];
            if (!CompactTail || warp < DimSplit) {
                // The sub-groups' accumulator chains are independent: interleaving them gives
                // the fixed-latency mma pipe two chains to alternate between.
                float partial[PVChunks][8] = {};
#pragma unroll
                for (int cl = 0; cl < DChunksLocal; ++cl) {
                    half2 qf[4];
                    load_frag16(qf, &q_s[q_frag_row * SmemStride + (dim_warp * DChunksLocal + cl) * 8]);
#pragma unroll
                    for (int sub = 0; sub < PVChunks; ++sub) {
                        half2 kf[4];
                        load_frag16(kf, &k_s[(sub * 8 + k_frag_row) * SmemStride +
                                             (dim_warp * DChunksLocal + cl) * 8]);
                        volta_mma_qk(partial[sub], qf, kf);
                    }
                }
#pragma unroll
                for (int sub = 0; sub < PVChunks; ++sub) {
                    *reinterpret_cast<float4*>(&qk_part_s[qk_part_index(dim_warp, sub, 0, lane)]) =
                        make_float4(partial[sub][0], partial[sub][1], partial[sub][2],
                                    partial[sub][3]);
                    *reinterpret_cast<float4*>(&qk_part_s[qk_part_index(dim_warp, sub, 1, lane)]) =
                        make_float4(partial[sub][4], partial[sub][5], partial[sub][6],
                                    partial[sub][7]);
                }
            }
            if constexpr (CompactTail) {
                if (warp == DimSplit) {
#pragma unroll
                    for (int sub = 0; sub < PVChunks; ++sub) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) { tail_scores[sub][i] = 0.0f; }
                    }
#pragma unroll
                    for (int step = 0; step < TailQkSteps; ++step) {
                        const int d0 = tail_qk_d0(step, tail_quadpair);
                        // A operand: row tail_row of the 8x4 slice, rows 4..7 being zero padding.
                        uint2 q_codes = make_uint2(0u, 0u);
                        if (tail_row < 4) {
                            q_codes = *reinterpret_cast<const uint2*>(
                                &q_tail_s[tail_row * SmemStride + d0]);
                        }
#pragma unroll
                        for (int sub = 0; sub < PVChunks; ++sub) {
                            // B operand: column tail_row of the 8x4 slice is key tail_row's four
                            // head dims [d0, d0+4), two packed half2 registers.
                            const uint2 key_codes = *reinterpret_cast<const uint2*>(
                                &k_s[(sub * 8 + tail_row) * SmemStride + d0]);
                            volta_mma_qp_n(tail_scores[sub], q_codes.x, q_codes.y, key_codes.x,
                                           key_codes.y);
                        }
                    }
                    // Quadpairs hold disjoint head-dim partials of the same 8x8 tile. The xor-4
                    // then xor-8 butterfly adds commutative pairs, so every quadpair ends with
                    // bit-identical sums.
#pragma unroll
                    for (int sub = 0; sub < PVChunks; ++sub) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            tail_scores[sub][i] +=
                                __shfl_xor_sync(FullMask, tail_scores[sub][i], 4, 32);
                            tail_scores[sub][i] +=
                                __shfl_xor_sync(FullMask, tail_scores[sub][i], 8, 32);
                        }
                    }
                }
            }
            __syncthreads();

            // Bc=16 keys are staged together, but softmax/PV operate on 8-key sub-groups
            // within that stage: each mma.sync.m8n8k4 call only covers an 8-key "N"
            // dimension, so each sub-group gets its own complete online-softmax rescale step
            // (same recurrence as the outer kb loop, one level finer). Every dim-split warp
            // sharing this row-tile runs this step on identical scores.
#pragma unroll
            for (int sub = 0; sub < PVChunks; ++sub) {
                const int sub_k0 = k0 + sub * 8;
                const bool tail_worker = CompactTail && warp == DimSplit;

                float d_score[8];
                if (tail_worker) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) { d_score[i] = tail_scores[CompactTail ? sub : 0][i]; }
                } else {
                    // Fixed summation order 0..DimSplit-1: identical in every dim-split warp.
                    float4 lo4 = *reinterpret_cast<const float4*>(
                        &qk_part_s[qk_part_index(0, sub, 0, lane)]);
                    float4 hi4 = *reinterpret_cast<const float4*>(
                        &qk_part_s[qk_part_index(0, sub, 1, lane)]);
#pragma unroll
                    for (int part_warp = 1; part_warp < DimSplit; ++part_warp) {
                        const float4 plo = *reinterpret_cast<const float4*>(
                            &qk_part_s[qk_part_index(part_warp, sub, 0, lane)]);
                        const float4 phi = *reinterpret_cast<const float4*>(
                            &qk_part_s[qk_part_index(part_warp, sub, 1, lane)]);
                        lo4.x += plo.x; lo4.y += plo.y; lo4.z += plo.z; lo4.w += plo.w;
                        hi4.x += phi.x; hi4.y += phi.y; hi4.z += phi.z; hi4.w += phi.w;
                    }
                    d_score[0] = lo4.x; d_score[1] = lo4.y; d_score[2] = lo4.z; d_score[3] = lo4.w;
                    d_score[4] = hi4.x; d_score[5] = hi4.y; d_score[6] = hi4.z; d_score[7] = hi4.w;
                }

                // --- Causal mask + scale. Each thread's 8 registers span exactly two distinct
                // rows (l&2==0 vs l&2==2, see volta_mma.cuh), so two qabs lookups (not eight)
                // cover this thread's own rows. The column mapping of the warp-wide D tile and
                // of the quadpair-split tail tile is the same; only the row mapping differs:
                // the tail tile is 8 rows per quadpair, rows 4..7 being zero padding. ---
                const int r_lo = tail_worker ? (((lane & 16) != 0 ? 4 : 0) | (lane & 1))
                                             : (volta_d_get_i(0) & ~2);
                const int r_hi = r_lo | 2;
                const int worker_row_base = tail_worker ? Br : row_base;
                const int worker_rows     = tail_worker ? row_count - Br : rows_here;
                int q_head_lo = 0, tok_lo = 0, q_head_hi = 0, tok_hi = 0;
                causal_small_t_tc_row_to_qt<Geometry>(worker_row_base + r_lo, tokens, kv_head,
                                                      q_head_lo, tok_lo);
                causal_small_t_tc_row_to_qt<Geometry>(worker_row_base + r_hi, tokens, kv_head,
                                                      q_head_hi, tok_hi);
                const int qabs_lo = (r_lo < worker_rows) ? pos[tok_lo] : -1;
                const int qabs_hi = (r_hi < worker_rows) ? pos[tok_hi] : -1;
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    const bool lo   = (l & 2) == 0;
                    const int row   = lo ? r_lo : r_hi;
                    const int col   = volta_d_get_j(l);
                    const int key   = sub_k0 + col;
                    const int qabs  = lo ? qabs_lo : qabs_hi;
                    const bool ok =
                        row < worker_rows && key >= split_start && key < split_end && key <= qabs;
                    d_score[l] = ok ? d_score[l] * scale : -CUDART_INF_F;
                }

                // --- Per-row online-softmax update. Threads tid and tid^2 together hold
                // the complete 8-key row for both rows this thread touches, so a single
                // offset-2 shfl_xor gives an exact per-row reduction. ---
                float bm_lo = fmaxf(fmaxf(d_score[0], d_score[1]), fmaxf(d_score[4], d_score[5]));
                float bm_hi = fmaxf(fmaxf(d_score[2], d_score[3]), fmaxf(d_score[6], d_score[7]));
                bm_lo        = fmaxf(bm_lo, __shfl_xor_sync(FullMask, bm_lo, 2, 32));
                bm_hi        = fmaxf(bm_hi, __shfl_xor_sync(FullMask, bm_hi, 2, 32));

                const float new_m_lo = fmaxf(m_lo, bm_lo);
                const float new_m_hi = fmaxf(m_hi, bm_hi);
                const float alpha_lo =
                    (m_lo == -CUDART_INF_F) ? 0.0f : exp2_approx((m_lo - new_m_lo) * Log2E);
                const float alpha_hi =
                    (m_hi == -CUDART_INF_F) ? 0.0f : exp2_approx((m_hi - new_m_hi) * Log2E);

#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    const float new_m = ((l & 2) == 0) ? new_m_lo : new_m_hi;
                    d_score[l] = (new_m > -CUDART_INF_F && d_score[l] > -CUDART_INF_F)
                                     ? exp2_approx((d_score[l] - new_m) * Log2E)
                                     : 0.0f;
                }
                float bl_lo = d_score[0] + d_score[1] + d_score[4] + d_score[5];
                float bl_hi = d_score[2] + d_score[3] + d_score[6] + d_score[7];
                bl_lo        = bl_lo + __shfl_xor_sync(FullMask, bl_lo, 2, 32);
                bl_hi        = bl_hi + __shfl_xor_sync(FullMask, bl_hi, 2, 32);

                l_lo = l_lo * alpha_lo + bl_lo;
                l_hi = l_hi * alpha_hi + bl_hi;
                m_lo = new_m_lo;
                m_hi = new_m_hi;
                // Own-row selection: this thread's PV accumulator (acc_f) is tied to output
                // row=lane specifically, which is r_lo when lane&2==0 and r_hi otherwise --
                // see volta_mma.cuh's I_MAJOR addressing for the Q/P/PV-output tile.
                const float alpha = ((lane & 2) == 0) ? alpha_lo : alpha_hi;

                if (!CompactTail || warp < DimSplit) {
                    half2 p[4];
                    volta_softmax_to_half2(p, d_score);
#pragma unroll
                    for (int c = 0; c < DChunksLocal; ++c) {
                        half2 vf[4];
                        volta_load_v(vf,
                                     reinterpret_cast<const half2*>(
                                         &v_s[sub * 8 * SmemStride + dim_warp * DSlice + c * 8]),
                                     SmemStride / 2);
                        half2 pv[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
                        volta_mma_pv(pv, p, vf);
#pragma unroll
                        for (int n = 0; n < 4; ++n) {
                            const float2 contrib = __half22float2(pv[n]);
                            acc_f[c][2 * n + 0]   = acc_f[c][2 * n + 0] * alpha + contrib.x;
                            acc_f[c][2 * n + 1]   = acc_f[c][2 * n + 1] * alpha + contrib.y;
                        }
                    }
                }
                if constexpr (CompactTail) {
                    if (warp == DimSplit) {
                        // Every quadpair holds the same reduced tile; quadpair 0 publishes it.
                        if (tail_quadpair == 0) {
#pragma unroll
                            for (int l = 0; l < 8; ++l) {
                                const int prow = (l & 2) == 0 ? r_lo : r_hi;
                                if (prow < 4) {
                                    p_tail_s[prow * 8 + volta_d_get_j(l)] =
                                        __float2half(d_score[l]);
                                }
                            }
                        }
                        __syncwarp();

                        // The fifth warp flips the usual attention MMA mapping: its four
                        // independent quadpairs own four 8-column D slices. Rows 4..7 are
                        // zero padding, leaving one warp to cover all four real tail rows and
                        // the full D=256 output in eight iterations.
                        const int qp        = (lane >> 2) & 3;
                        const int input_row = (lane & 3) + ((lane & 16) != 0 ? 4 : 0);
                        half2 p[4];
#pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            p[l] = input_row < 4
                                       ? __halves2half2(p_tail_s[input_row * 8 + 2 * l],
                                                       p_tail_s[input_row * 8 + 2 * l + 1])
                                       : __half2half2(__ushort_as_half(0));
                        }
                        // p_tail_s is rewritten by the next sub-group; keep every lane's read
                        // ordered before that write.
                        __syncwarp();
                        const unsigned* P = reinterpret_cast<const unsigned*>(p);
                        // Tail row r lives in lane r&1 of the quadpair-split tile, as its lo
                        // series for r<2 and its hi series for r>=2.
                        const float alpha_rows[4] = {
                            __shfl_sync(FullMask, alpha_lo, 0, 32),
                            __shfl_sync(FullMask, alpha_lo, 1, 32),
                            __shfl_sync(FullMask, alpha_hi, 0, 32),
                            __shfl_sync(FullMask, alpha_hi, 1, 32)};

#pragma unroll
                        for (int c = 0; c < DChunksLocal; ++c) {
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                const int accumulator_row =
                                    (i & 2) | ((lane & 16) != 0 ? 4 : 0) | (lane & 1);
                                acc_f[c][i] *= accumulator_row < 4
                                                   ? alpha_rows[accumulator_row]
                                                   : 0.0f;
                            }
                            const int d = c * 32 + qp * 8 + input_row;
                            half2 vf[4];
#pragma unroll
                            for (int l = 0; l < 4; ++l) {
                                vf[l] = __halves2half2(
                                    v_s[(sub * 8 + 2 * l) * SmemStride + d],
                                    v_s[(sub * 8 + 2 * l + 1) * SmemStride + d]);
                            }
                            const unsigned* V = reinterpret_cast<const unsigned*>(vf);
                            volta_mma_qp_n(acc_f[c], P[0], P[1], V[0], V[1]);
                            volta_mma_qp_n(acc_f[c], P[2], P[3], V[2], V[3]);
                        }
                    }
                }
            }
            __syncthreads();
        }

        // --- Write-out: partial_m/partial_l once per row-tile (dim_warp==0 only -- every
        // dim-split warp computed the same m/l redundantly), partial_acc per dim-split
        // warp's own DSlice-wide column range. ---
        const int row      = lane;
        const float own_m = ((lane & 2) == 0) ? m_lo : m_hi;
        const float own_l = ((lane & 2) == 0) ? l_lo : l_hi;
        if (!CompactTail || warp < DimSplit) {
            if (dim_warp == 0 && row < rows_here) {
                int q_head = 0;
                int token  = 0;
                causal_small_t_tc_row_to_qt<Geometry>(row_base + row, tokens, kv_head, q_head,
                                                      token);
                if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                    partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] =
                        own_m;
                    partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] =
                        own_l;
                }
            }
            if (row < rows_here) {
                int q_head = 0;
                int token  = 0;
                causal_small_t_tc_row_to_qt<Geometry>(row_base + row, tokens, kv_head, q_head,
                                                      token);
                if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
#pragma unroll
                    for (int c = 0; c < DChunksLocal; ++c) {
                        const int d = dim_warp * DSlice + c * 8;
                        const std::int64_t dst = causal_partial_acc_index<Geometry>(
                            q_head, d, token, split, tokens);
                        store_vec(&partial_acc[dst],
                                  *reinterpret_cast<const int4*>(&acc_f[c][0]));
                        store_vec(&partial_acc[dst + 4],
                                  *reinterpret_cast<const int4*>(&acc_f[c][4]));
                    }
                }
            }
        }
        if constexpr (CompactTail) {
            if (warp == DimSplit) {
                // Tail row r's softmax state lives in lane r&1 (lo series for r<2, hi for r>=2)
                // of the quadpair-split tile; gather it so lane r writes row r.
                const float tail_m_lo = __shfl_sync(FullMask, m_lo, lane & 1, 32);
                const float tail_m_hi = __shfl_sync(FullMask, m_hi, lane & 1, 32);
                const float tail_l_lo = __shfl_sync(FullMask, l_lo, lane & 1, 32);
                const float tail_l_hi = __shfl_sync(FullMask, l_hi, lane & 1, 32);
                if (lane < 4) {
                    int q_head = 0;
                    int token  = 0;
                    causal_small_t_tc_row_to_qt<Geometry>(Br + lane, tokens, kv_head, q_head,
                                                          token);
                    partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] =
                        (lane & 2) == 0 ? tail_m_lo : tail_m_hi;
                    partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] =
                        (lane & 2) == 0 ? tail_l_lo : tail_l_hi;
                }
#pragma unroll
                for (int c = 0; c < DChunksLocal; ++c) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int accumulator_row =
                            (i & 2) | ((lane & 16) != 0 ? 4 : 0) | (lane & 1);
                        if (accumulator_row < 4) {
                            const int qp = (lane >> 2) & 3;
                            const int cl = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
                            const int d  = c * 32 + qp * 8 + cl;
                            int q_head   = 0;
                            int token    = 0;
                            causal_small_t_tc_row_to_qt<Geometry>(Br + accumulator_row, tokens,
                                                                  kv_head, q_head, token);
                            const std::int64_t dst = causal_partial_acc_index<Geometry>(
                                q_head, d, token, split, tokens);
                            partial_acc[dst] = acc_f[c][i];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
}

} // namespace ninfer::ops
