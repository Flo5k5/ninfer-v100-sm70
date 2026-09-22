#pragma once

// ninfer::ops - Volta (sm_70) INT8-G64 small-T causal attention, key-major tensor-core kernel.
//
// Same partial_acc/partial_m/partial_l contract and split policy as
// causal_attention_small_t_tc_volta_partial_i8_kernel (small_t_i8_volta.cuh); only the inside of
// one (kv_head, split) CTA differs. The query-major kernel puts query rows on the mma M axis,
// which forces 32-row tiles, a CTA-wide dequantizing stage of every K/V tile into fp16 shared
// memory and three barriers per 16 keys. At 26K it measured ~1370 issued instructions per warp
// per 16-key tile at IPC 0.28, i.e. latency bound, with the width-one draft barely faster than
// the width-five verify. This kernel transposes both products so that keys (QK^T) and head dims
// (PV) sit on the 32-wide M axis and query rows on the 8-wide N axis:
//
//   S^T[key][row]  = K[key][d] . Q[row][d]      mma.m8n8k4.row.col, A = K codes, B = Q
//   O^T[d][row]   += V^T[d][key] . P^T[key][row] mma.m8n8k4.col.row, A = V codes, B = P
//
// Consequences, each load-bearing:
//
//   1. Row tiles are 8 wide. Width one (6 rows) runs one N tile instead of a 32-row tile, width
//      four runs three; ceil(rows / 8) N tiles cover every width without a tail special case.
//   2. Each of the four warps owns one 64-wide head-dim slice, which is exactly one INT8 quant
//      group. A warp therefore reads only its own slice of every key row, straight from global
//      memory into mma fragment registers: the K A-fragment is one key row per lane (row-major
//      A), the V A-fragment is four consecutive head dims of one key per lane (column-major A).
//      There is no shared-memory K/V stage and no staging barrier. The raw codes of the next
//      step are loaded right after the current step consumes its registers, so each load has a
//      whole step to land.
//   3. Codes enter the tensor core as exact fp16 integers. The per-key K scale of the warp's
//      group multiplies the fp32 partial score; the per-key V scale multiplies the fp16 V
//      fragment. Each code is converted once per CTA.
//   4. QK^T is split-K over the four slices, so partial score tiles are summed through shared
//      memory in a fixed order (bit-identical scores in every consumer). With one N tile every
//      warp sums and runs the softmax itself behind one barrier (double-buffered exchange);
//      with more, the warp owning an N tile runs its softmax and publishes P^T and the
//      rescale factors behind a second barrier.
//   5. Online softmax with a stale reference maximum: a row keeps its reference m until a score
//      exceeds it by more than kKtHeadroom (log2 units), so P is bounded by 2^kKtHeadroom and
//      the max reduction plus accumulator rescale run only on the rare steps that raise it. l
//      is accumulated per lane and reduced once at the end. The result is the same online
//      softmax identity with a different (valid) reference, written in the same natural-log m.

#include "ops/common/volta_mma.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kKtStepKeys   = 32;
inline constexpr int kKtWarps      = 4;
inline constexpr float kKtHeadroom = 8.0f;

template <typename Geometry, int TokenTile>
inline constexpr int kKtRowTiles = (TokenTile * Geometry::GroupSize + 7) / 8;

// Widths this kernel is instantiated for: up to five 8-row N tiles.
template <typename Geometry, int TokenTile>
inline constexpr bool kKtSupported = kKtRowTiles<Geometry, TokenTile> <= 5;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
// D(32x8 f32) += A(32x4 f16, column-major) * B(4x8 f16, row-major). Lane L holds
// A[(L & ~3) + e][L & 3] and B[L & 3][4 * (L >> 4) + e] for e = 0..3 (measured on sm_70); the
// accumulator is the volta_d_get_i/j layout shared by every f32 m8n8k4 form.
__device__ __forceinline__ void volta_kt_mma_col_row(float (&d)[8], std::uint32_t a0,
                                                     std::uint32_t a1, std::uint32_t b0,
                                                     std::uint32_t b1) {
    asm volatile("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
                 "{%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
                   "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
#endif

// Four int8 codes -> two half2 holding the exact integers. 0x64 over (code ^ 0x80) is the fp16
// value 1152 + code, so one byte permute and one subtraction per pair give the code exactly.
__device__ __forceinline__ void volta_kt_codes_to_half2(std::uint32_t word, half2& lo, half2& hi) {
    constexpr std::uint32_t kExponent1024 = 0x64646464u;
    const half2 bias1152                  = __half2half2(__ushort_as_half(0x6480));
    const std::uint32_t biased            = word ^ 0x80808080u;
    const std::uint32_t low               = __byte_perm(biased, kExponent1024, 0x4140);
    const std::uint32_t high              = __byte_perm(biased, kExponent1024, 0x4342);
    lo = __hsub2(*reinterpret_cast<const half2*>(&low), bias1152);
    hi = __hsub2(*reinterpret_cast<const half2*>(&high), bias1152);
}

__device__ __forceinline__ std::uint32_t volta_kt_bits(half2 value) {
    return *reinterpret_cast<const std::uint32_t*>(&value);
}

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput>
__launch_bounds__(kKtWarps * 32, 2) __global__
    void causal_attention_small_t_volta_kt_partial_i8_kernel(
        const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, std::int8_t* cache_k_i8,
        std::int8_t* cache_v_i8, __half* cache_k_scale, __half* cache_v_scale,
        const std::int32_t* block_tables, const std::int32_t* valid_columns,
        const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t tokens,
        std::int32_t full_width, std::int32_t column_begin, std::int32_t logical_capacity,
        float scale, float* partial_acc, float* partial_m, float* partial_l) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
    constexpr int NT = kKtRowTiles<Geometry, TokenTile>;
    static_assert(NT >= 1 && NT <= 5);
    constexpr int Rows       = NT * 8;
    constexpr int Warps      = kKtWarps;
    constexpr int Threads    = Warps * 32;
    constexpr int D          = kCausalHeadDim;
    constexpr int Slice      = D / Warps;
    constexpr int StepKeys   = kKtStepKeys;
    constexpr int PageIds    = 64;
    constexpr int Groups     = D / kKVCacheInt8Group;
    constexpr bool Redundant = NT == 1; // every warp sums the partials and runs the softmax
    constexpr bool QResident = NT == 1; // Q B-fragments stay in registers for the whole split
    constexpr int XBuffers   = Redundant ? 2 : 1;
    constexpr int OwnedMax   = Redundant ? NT : (NT + Warps - 1) / Warps;
    constexpr int QStride    = D + 8; // 16-byte pad: the 8 fragment rows hit distinct banks
    constexpr float Log2E    = 1.4426950408889634074f;
    constexpr unsigned Full  = 0xffffffffu;
    static_assert(Slice == kKVCacheInt8Group, "one warp slice must be exactly one quant group");
    static_assert(StepKeys * 2 == kPagedKVPageSize, "a step must sit inside one KV page");

    __shared__ __align__(16) half q_s[Rows * QStride];
    // Partial score tiles [buffer][source warp][N tile][half][lane][4]: a warp's 16-byte
    // accesses land on distinct banks.
    __shared__ __align__(16) float x_s[XBuffers * Warps * NT * 256];
    // P^T tiles [N tile][key][row] in fp16; private per warp when every warp computes its own.
    __shared__ __align__(16) half p_s[(Redundant ? Warps : 1) * NT * StepKeys * 8];
    __shared__ float alpha_s[Redundant ? 1 : NT * 8];
    __shared__ int flag_s[Redundant ? 1 : NT];
    __shared__ std::int32_t physical_pages_s[PageIds];

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    int valid_tokens      = tokens;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < tokens ? remaining : tokens);
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
        partial_acc += static_cast<std::int64_t>(batch) * kCausalHeadDim * Geometry::QHeads *
                       tokens * split_count;
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
    // Earliest query position: below it no key of this step can be causally masked.
    std::int32_t min_pos = first_pos;
    for (int token = 1; token < tokens; ++token) { min_pos = min(min_pos, pos[token]); }

    const int window = last_pos + 1;
    const int active_split_count =
        causal_small_t_active_splits<Geometry, true>(window, split_count, TokenTile);
    if (split >= active_split_count) { return; }

    // Same split partition as the query-major kernel (16-key tile units), so both kernels
    // produce interchangeable partials for a given split count.
    constexpr int PartitionTile = 16;
    const int logical_tiles     = div_up(window, PartitionTile);
    const bool tile_split       = logical_tiles >= active_split_count;
    const int units_per_split =
        tile_split ? div_up(logical_tiles, active_split_count) : div_up(window, active_split_count);
    const int split_start = split * units_per_split * (tile_split ? PartitionTile : 1);
    const int split_limit = split_start + units_per_split * (tile_split ? PartitionTile : 1);
    const int split_end   = (split_limit < window) ? split_limit : window;
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    const int first_key  = split_start & ~(StepKeys - 1);
    const int steps      = div_up(split_end - first_key, StepKeys);
    const int first_page = first_key >> kPagedKVPageShift;
    const int page_count = ((split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    for (int page = tid; page < page_count; page += Threads) {
        physical_pages_s[page] = block_table[first_page + page];
    }

    if constexpr (CacheInput::writes_cache) {
        // One warp owns a D256 row, applies the registered normalized transform to K, and then
        // emits all four G64 groups. V remains in its native coordinates.
        for (int token = warp; token < valid_tokens; token += Warps) {
            const int position = pos[token];
            if (position < split_start || position >= split_end || position < 0 ||
                position >= logical_capacity) {
                continue;
            }
            float k_values[8];
            float v_values[8];
#    pragma unroll
            for (int part = 0; part < 8; ++part) {
                const int d               = lane + 32 * part;
                const std::int64_t source = kv_cache_int8_new_index<Geometry>(kv_head, d, token);
                k_values[part]            = __bfloat162float(input.k[source]);
                v_values[part]            = __bfloat162float(input.v[source]);
            }
            normalized_hadamard_d256_inplace(k_values, lane);

            int physical_page     = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
            physical_page         = __shfl_sync(Full, physical_page, 0);
            const int page_offset = position & kPagedKVPageMask;
#    pragma unroll
            for (int grp = 0; grp < Groups; ++grp) {
                float kamax = fmaxf(fabsf(k_values[2 * grp]), fabsf(k_values[2 * grp + 1]));
                float vamax = fmaxf(fabsf(v_values[2 * grp]), fabsf(v_values[2 * grp + 1]));
                kamax       = warp_max(kamax, Full);
                vamax       = warp_max(vamax, Full);
                const KVCacheInt8QuantParams kp = kv_cache_int8_quant_params(kamax);
                const KVCacheInt8QuantParams vp = kv_cache_int8_quant_params(vamax);
                const int d0                    = grp * kKVCacheInt8Group + lane;
                const int d1                    = d0 + 32;
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
                    const std::int64_t so = kv_cache_int8_quant_scale_index<Geometry>(
                        physical_page, kv_head, grp, page_offset);
                    cache_k_scale[so] = kp.scale;
                    cache_v_scale[so] = vp.scale;
                }
            }
        }
    }

    // Q rows in the rotated basis, fp16, zero past the real rows.
    for (int row = warp; row < Rows; row += Warps) {
        float values[8] = {};
        if (row < row_count) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
#    pragma unroll
                for (int part = 0; part < 8; ++part) {
                    values[part] = __bfloat162float(
                        q[causal_q_index<Geometry>(q_head, lane + 32 * part, token)]);
                }
                normalized_hadamard_d256_inplace(values, lane);
            }
        }
#    pragma unroll
        for (int part = 0; part < 8; ++part) {
            q_s[row * QStride + lane + 32 * part] = __float2half(values[part]);
        }
    }
    __syncthreads();

    const int slice0     = warp * Slice;    // this warp's head-dim slice == quant group
    const int q_frag_n   = volta_k_get_i(); // B-fragment row (N index) held by this lane
    const float qk_scale = scale * Log2E;

    const auto load_q_frag = [&](half2(&dst)[4], int row_tile, int pair) {
        const int4 raw = *reinterpret_cast<const int4*>(
            &q_s[(row_tile * 8 + q_frag_n) * QStride + slice0 + pair * 8]);
        const half2* h = reinterpret_cast<const half2*>(&raw);
#    pragma unroll
        for (int i = 0; i < 4; ++i) { dst[i] = h[i]; }
    };
    half2 q_resident[QResident ? 8 : 1][4];
    if constexpr (QResident) {
#    pragma unroll
        for (int pair = 0; pair < 8; ++pair) { load_q_frag(q_resident[pair], 0, pair); }
    }

    // Accumulators: O^T slice [M tile][N tile] in the f32 D layout. M index m of tile mt is
    // head dim slice0 + 8 * (m >> 2) + 4 * mt + (m & 3) (see the V fragment below).
    float acc[2][NT][8];
#    pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#    pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#    pragma unroll
            for (int i = 0; i < 8; ++i) { acc[mt][nt][i] = 0.0f; }
        }
    }
    // Softmax state for the N tiles this warp owns, per D-layout row slot. A lane's eight
    // D values cover rows (lane & 2) + {0, 1, 4, 5}: slot(l) = (l & 1) | ((l >> 1) & 2).
    float m_ref[OwnedMax][4];
    float m_sub[OwnedMax][4]; // m_ref, or 0 while the row has no finite score
    float m_thr[OwnedMax][4]; // headroom, or -inf while the row has no finite score
    float l_run[OwnedMax][4]; // per-lane partial row sums, reduced once at the end
#    pragma unroll
    for (int o = 0; o < OwnedMax; ++o) {
#    pragma unroll
        for (int s = 0; s < 4; ++s) {
            m_ref[o][s] = -CUDART_INF_F;
            m_sub[o][s] = 0.0f;
            m_thr[o][s] = -CUDART_INF_F;
            l_run[o][s] = 0.0f;
        }
    }
    const auto slot_of = [](int l) { return (l & 1) | ((l >> 1) & 2); };

    // Raw codes of one step. K: this lane's key (M row = lane), 64 bytes of the slice.
    // V: key 4u + (lane & 3) for u = 0..7, bytes slice0 + 8 * (lane >> 2) + 0..7 (M tile 0 takes
    // the first word, M tile 1 the second).
    // Masked keys of an edge step still load real codes: they are clamped into the split, whose
    // cache slots this CTA owns (neighbouring slots may be mid-append in another CTA, and a stale
    // fp16 scale there could be NaN, which a zero probability would not cancel).
    const auto clamp_key = [&](int key) { return min(max(key, split_start), split_end - 1); };
    uint4 k_raw[4];
    uint2 v_raw[8];
    // Scales stay raw until their step consumes them: converting at load time would make the
    // prefetch wait for its own data.
    __half k_scale_lane = __ushort_as_half(0);
    __half v_scale_lane = __ushort_as_half(0);
    const auto load_k   = [&](int step) {
        const int k0     = first_key + step * StepKeys;
        const int page   = physical_pages_s[(k0 >> kPagedKVPageShift) - first_page];
        const int key    = clamp_key(k0 + lane);
        const int offset = key & kPagedKVPageMask;
        const uint4* src = reinterpret_cast<const uint4*>(
            &cache_k_i8[kv_cache_int8_quant_code_index<Geometry>(page, kv_head, slice0, offset)]);
#    pragma unroll
        for (int j = 0; j < 4; ++j) { k_raw[j] = src[j]; }
        k_scale_lane =
            cache_k_scale[kv_cache_int8_quant_scale_index<Geometry>(page, kv_head, warp, offset)];
    };
    const auto load_v = [&](int step) {
        const int k0            = first_key + step * StepKeys;
        const int page          = physical_pages_s[(k0 >> kPagedKVPageShift) - first_page];
        const std::int8_t* base = &cache_v_i8[kv_cache_int8_quant_code_index<Geometry>(
            page, kv_head, slice0 + 8 * (lane >> 2), 0)];
#    pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int offset = clamp_key(k0 + 4 * u + (lane & 3)) & kPagedKVPageMask;
            v_raw[u]         = *reinterpret_cast<const uint2*>(base + offset * D);
        }
        const int offset = clamp_key(k0 + lane) & kPagedKVPageMask;
        v_scale_lane =
            cache_v_scale[kv_cache_int8_quant_scale_index<Geometry>(page, kv_head, warp, offset)];
    };

    load_k(0);
    load_v(0);

    for (int step = 0; step < steps; ++step) {
        const int k0 = first_key + step * StepKeys;
        const bool edge =
            k0 < split_start || k0 + StepKeys > split_end || k0 + StepKeys - 1 > min_pos;
        float* x_buf = &x_s[(Redundant ? (step & 1) : 0) * Warps * NT * 256];

        // --- S^T partial over this warp's slice: A = K codes (M = key), B = Q (N = row). ---
        {
            constexpr int Chains = NT == 1 ? 2 : 1;
            float s_part[NT][Chains][8];
#    pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
#    pragma unroll
                for (int c = 0; c < Chains; ++c) {
#    pragma unroll
                    for (int i = 0; i < 8; ++i) { s_part[nt][c][i] = 0.0f; }
                }
            }
#    pragma unroll
            for (int pair = 0; pair < 8; ++pair) {
                const std::uint32_t w0 = (pair & 1) == 0 ? k_raw[pair >> 1].x : k_raw[pair >> 1].z;
                const std::uint32_t w1 = (pair & 1) == 0 ? k_raw[pair >> 1].y : k_raw[pair >> 1].w;
                half2 a[4];
                volta_kt_codes_to_half2(w0, a[0], a[1]);
                volta_kt_codes_to_half2(w1, a[2], a[3]);
#    pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    half2 qf[4];
                    if constexpr (QResident) {
#    pragma unroll
                        for (int i = 0; i < 4; ++i) { qf[i] = q_resident[pair][i]; }
                    } else {
                        load_q_frag(qf, nt, pair);
                    }
                    float (&dst)[8] = s_part[nt][pair % Chains];
                    volta_mma_qp_n(dst, volta_kt_bits(a[0]), volta_kt_bits(a[1]),
                                   volta_kt_bits(qf[0]), volta_kt_bits(qf[1]));
                    volta_mma_qp_n(dst, volta_kt_bits(a[2]), volta_kt_bits(a[3]),
                                   volta_kt_bits(qf[2]), volta_kt_bits(qf[3]));
                }
            }
            // The D rows (keys) this lane holds are (lane & ~2) and (lane & ~2) + 2. Read this
            // step's scales before the next step's load replaces them.
            const float ks_own = __half2float(k_scale_lane);
            const float ks_lo  = __shfl_sync(Full, ks_own, lane & ~2);
            const float ks_hi  = __shfl_sync(Full, ks_own, (lane & ~2) + 2);
            if (step + 1 < steps) { load_k(step + 1); }
#    pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                float v[8];
#    pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float sum = s_part[nt][0][i];
                    if constexpr (Chains == 2) { sum += s_part[nt][1][i]; }
                    v[i] = sum * ((i & 2) == 0 ? ks_lo : ks_hi);
                }
                float* dst                      = &x_buf[((warp * NT + nt) * 2) * 128 + lane * 4];
                *reinterpret_cast<float4*>(dst) = make_float4(v[0], v[1], v[2], v[3]);
                *reinterpret_cast<float4*>(dst + 128) = make_float4(v[4], v[5], v[6], v[7]);
            }
        }
        __syncthreads();

        // --- Softmax on the summed scores of the N tiles this warp handles. ---
        float alpha_own[OwnedMax][4];
        bool rescaled_own[OwnedMax];
#    pragma unroll
        for (int o = 0; o < OwnedMax; ++o) {
            const int nt    = Redundant ? o : warp + Warps * o;
            rescaled_own[o] = false;
#    pragma unroll
            for (int s = 0; s < 4; ++s) { alpha_own[o][s] = 1.0f; }
            if (nt >= NT) { continue; }

            float x[8];
            {
                float4 lo = *reinterpret_cast<const float4*>(&x_buf[(nt * 2) * 128 + lane * 4]);
                float4 hi = *reinterpret_cast<const float4*>(&x_buf[(nt * 2 + 1) * 128 + lane * 4]);
#    pragma unroll
                for (int src = 1; src < Warps; ++src) {
                    const float4 plo = *reinterpret_cast<const float4*>(
                        &x_buf[((src * NT + nt) * 2) * 128 + lane * 4]);
                    const float4 phi = *reinterpret_cast<const float4*>(
                        &x_buf[((src * NT + nt) * 2 + 1) * 128 + lane * 4]);
                    lo.x += plo.x;
                    lo.y += plo.y;
                    lo.z += plo.z;
                    lo.w += plo.w;
                    hi.x += phi.x;
                    hi.y += phi.y;
                    hi.z += phi.z;
                    hi.w += phi.w;
                }
                x[0] = lo.x;
                x[1] = lo.y;
                x[2] = lo.z;
                x[3] = lo.w;
                x[4] = hi.x;
                x[5] = hi.y;
                x[6] = hi.z;
                x[7] = hi.w;
            }
#    pragma unroll
            for (int i = 0; i < 8; ++i) { x[i] *= qk_scale; }
            if (edge) {
#    pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int key   = k0 + (i & 2) + (lane & ~2);
                    const int row   = nt * 8 + (lane & 2) + (i & 5);
                    const int token = min(row / Geometry::GroupSize, tokens - 1);
                    const bool ok   = key >= split_start && key < split_end && key <= pos[token];
                    x[i]            = ok ? x[i] : -CUDART_INF_F;
                }
            }
            bool over = false;
#    pragma unroll
            for (int i = 0; i < 8; ++i) {
                over |= x[i] - m_sub[o][slot_of(i)] > m_thr[o][slot_of(i)];
            }
            if (__any_sync(Full, over)) {
                // Raise the reference to this step's row maximum (xor over the lane bits that
                // index keys: 0, 2, 3, 4) and rescale the running sums.
#    pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const int l0      = (s & 1) | ((s & 2) << 1);
                    float tile_max    = fmaxf(x[l0], x[l0 | 2]);
                    tile_max          = fmaxf(tile_max, __shfl_xor_sync(Full, tile_max, 1));
                    tile_max          = fmaxf(tile_max, __shfl_xor_sync(Full, tile_max, 4));
                    tile_max          = fmaxf(tile_max, __shfl_xor_sync(Full, tile_max, 8));
                    tile_max          = fmaxf(tile_max, __shfl_xor_sync(Full, tile_max, 16));
                    const float new_m = fmaxf(m_ref[o][s], tile_max);
                    const float alpha =
                        new_m == m_ref[o][s] ? 1.0f : exp2_approx(m_ref[o][s] - new_m);
                    alpha_own[o][s] = alpha;
                    l_run[o][s] *= alpha;
                    m_ref[o][s] = new_m;
                    m_sub[o][s] = new_m == -CUDART_INF_F ? 0.0f : new_m;
                    m_thr[o][s] = new_m == -CUDART_INF_F ? -CUDART_INF_F : kKtHeadroom;
                }
                rescaled_own[o] = true;
            }
            float p[8];
#    pragma unroll
            for (int i = 0; i < 8; ++i) {
                p[i] = exp2_approx(x[i] - m_sub[o][slot_of(i)]);
                l_run[o][slot_of(i)] += p[i];
            }
            // P^T[key][row]: rows j and j + 1 of one key are adjacent halves.
            half* p_tile   = &p_s[((Redundant ? warp * NT : 0) + nt) * StepKeys * 8];
            const int key0 = lane & ~2;
            const int row0 = lane & 2;
            if constexpr (Redundant) { __syncwarp(); }
            *reinterpret_cast<half2*>(&p_tile[key0 * 8 + row0])     = __floats2half2_rn(p[0], p[1]);
            *reinterpret_cast<half2*>(&p_tile[key0 * 8 + row0 + 4]) = __floats2half2_rn(p[4], p[5]);
            *reinterpret_cast<half2*>(&p_tile[(key0 + 2) * 8 + row0]) =
                __floats2half2_rn(p[2], p[3]);
            *reinterpret_cast<half2*>(&p_tile[(key0 + 2) * 8 + row0 + 4]) =
                __floats2half2_rn(p[6], p[7]);
            if constexpr (!Redundant) {
                if (rescaled_own[o] && (lane & ~2) == 0) {
#    pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        alpha_s[nt * 8 + row0 + (s & 1) + ((s & 2) << 1)] = alpha_own[o][s];
                    }
                }
                if (lane == 0) { flag_s[nt] = rescaled_own[o] ? 1 : 0; }
            }
        }
        if constexpr (Redundant) {
            __syncwarp();
        } else {
            __syncthreads();
        }

        // --- Rescale, then O^T += V^T P^T over this warp's slice. ---
#    pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            if constexpr (Redundant) {
                if (rescaled_own[nt]) {
#    pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
#    pragma unroll
                        for (int i = 0; i < 8; ++i) { acc[mt][nt][i] *= alpha_own[nt][slot_of(i)]; }
                    }
                }
            } else {
                if (flag_s[nt] != 0) {
                    float alpha[4];
#    pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        alpha[s] = alpha_s[nt * 8 + (lane & 2) + (s & 1) + ((s & 2) << 1)];
                    }
#    pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
#    pragma unroll
                        for (int i = 0; i < 8; ++i) { acc[mt][nt][i] *= alpha[slot_of(i)]; }
                    }
                }
            }
        }
        const half* p_base         = &p_s[(Redundant ? warp * NT : 0) * StepKeys * 8];
        const std::uint32_t vs_own = volta_kt_bits(__half2half2(v_scale_lane));
#    pragma unroll
        for (int u = 0; u < 8; ++u) {
            const std::uint32_t vs_bits = __shfl_sync(Full, vs_own, 4 * u + (lane & 3));
            const half2 vs              = *reinterpret_cast<const half2*>(&vs_bits);
            half2 a0[2];
            half2 a1[2];
            volta_kt_codes_to_half2(v_raw[u].x, a0[0], a0[1]);
            volta_kt_codes_to_half2(v_raw[u].y, a1[0], a1[1]);
            a0[0] = __hmul2(a0[0], vs);
            a0[1] = __hmul2(a0[1], vs);
            a1[0] = __hmul2(a1[0], vs);
            a1[1] = __hmul2(a1[1], vs);
#    pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const uint2 b = *reinterpret_cast<const uint2*>(
                    &p_base[nt * StepKeys * 8 + (4 * u + (lane & 3)) * 8 + 4 * (lane >> 4)]);
                volta_kt_mma_col_row(acc[0][nt], volta_kt_bits(a0[0]), volta_kt_bits(a0[1]), b.x,
                                     b.y);
                volta_kt_mma_col_row(acc[1][nt], volta_kt_bits(a1[0]), volta_kt_bits(a1[1]), b.x,
                                     b.y);
            }
        }
        if (step + 1 < steps) { load_v(step + 1); }
    }

    // --- Write-out. Row sums: reduce the per-lane partials over the key lanes. ---
#    pragma unroll
    for (int o = 0; o < OwnedMax; ++o) {
        const int nt = Redundant ? o : warp + Warps * o;
        if (nt >= NT) { continue; }
#    pragma unroll
        for (int s = 0; s < 4; ++s) {
            float l = l_run[o][s];
            l += __shfl_xor_sync(Full, l, 1);
            l += __shfl_xor_sync(Full, l, 4);
            l += __shfl_xor_sync(Full, l, 8);
            l += __shfl_xor_sync(Full, l, 16);
            l_run[o][s] = l;
        }
        if ((Redundant && warp != 0) || (lane & ~2) != 0) { continue; }
#    pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int row = nt * 8 + (lane & 2) + (s & 1) + ((s & 2) << 1);
            if (row >= row_count) { continue; }
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (!causal_valid_q_head<Geometry>(kv_head, q_head)) { continue; }
            const auto stat = causal_partial_stat_index<Geometry>(q_head, token, split, tokens);
            partial_m[stat] = m_ref[o][s] / Log2E;
            partial_l[stat] = l_run[o][s];
        }
    }
#    pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#    pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = nt * 8 + (lane & 2) + (i & 5);
            if (row >= row_count) { continue; }
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (!causal_valid_q_head<Geometry>(kv_head, q_head)) { continue; }
            const int m = (i & 2) + (lane & ~2);
#    pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int d = slice0 + 8 * (m >> 2) + 4 * mt + (m & 3);
                partial_acc[causal_partial_acc_index<Geometry>(q_head, d, token, split, tokens)] =
                    acc[mt][nt][i];
            }
        }
    }
#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
}

} // namespace ninfer::ops
