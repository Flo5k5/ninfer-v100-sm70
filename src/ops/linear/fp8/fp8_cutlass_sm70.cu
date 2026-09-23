#include "ops/linear/fp8/fp8_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/fp8/fp8_gemv.cuh"
#include "ops/linear/fp8/fp8_prepack_sm70.cuh"

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/half.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Reconstructs the represented FP16 weight code * row_scale (one rounding of an exact FP32 product),
// so the GEMM output is rounded to BF16 once instead of once before and once after the row scale.
__global__ void dequant_fp8_row_to_fp16(const std::uint8_t* __restrict__ codes,
                                        const __nv_bfloat16* __restrict__ scales, int n, int k,
                                        bool prepacked,
                                        cutlass::half_t* __restrict__ out) {
    const int row      = static_cast<int>(blockIdx.y);
    const int pair_idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= n || pair_idx >= k / 2) { return; }
    const int k0 = pair_idx * 2;
    const std::uint8_t* source = nullptr;
    if (prepacked) {
        source = codes + fp8_qpn_prepacked_offset(row, k0, k);
    } else {
        source = codes + static_cast<std::int64_t>(row) * k + k0;
    }
    const std::uint16_t packed = *reinterpret_cast<const std::uint16_t*>(source);
    const float2 weight        = decode_fp8_e4m3x2(packed);
    const float scale          = __bfloat162float(scales[row]);
    cutlass::half_t* out_row   = out + static_cast<std::int64_t>(row) * k;
    out_row[pair_idx * 2]      = cutlass::half_t(weight.x * scale);
    out_row[pair_idx * 2 + 1]  = cutlass::half_t(weight.y * scale);
}

__global__ void bf16_to_fp16_kernel(const __nv_bfloat16* __restrict__ in,
                                    cutlass::half_t* __restrict__ out, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { out[i] = cutlass::half_t(__bfloat162float(in[i])); }
}

using ElementAccumulator     = float;
using ElementComputeEpilogue = float;
using ElementInput           = cutlass::half_t;
// BF16 output for the plain projection; FP32 output accumulated onto an FP32 residual stream
// (D = acc + C with C = D) for the residual-update projections.
template <class ElementOutput>
using GemmFor = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::RowMajor, ElementInput, cutlass::layout::ColumnMajor,
    ElementOutput, cutlass::layout::RowMajor, ElementAccumulator, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70, cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator,
        ElementComputeEpilogue>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;
using Gemm    = GemmFor<cutlass::bfloat16_t>;
using GemmF32 = GemmFor<float>;

template <class Allocator>
struct Scratch {
    Tensor weight;
    Tensor input;
    DeviceSpan gemm;
};

template <class Allocator>
Scratch<Allocator> allocate_scratch(Allocator& allocator, int n, int k, int cols,
                                    std::size_t gemm_bytes) {
    Scratch<Allocator> out;
    out.weight = allocator.alloc(DType::FP16, {k, n});
    out.input  = allocator.alloc(DType::FP16, {k, cols});
    if (gemm_bytes != 0) { out.gemm = allocator.alloc_bytes(gemm_bytes); }
    return out;
}

template <class G>
std::size_t gemm_workspace_bytes_for(int n, int k, int cols) {
    const cutlass::gemm::GemmCoord shape(cols, n, k);
    typename G::Arguments args{shape, {nullptr, k}, {nullptr, k}, {nullptr, n}, {nullptr, n},
                               {1.0F, 0.0F}, 1};
    return G::get_workspace_size(args);
}

std::size_t gemm_workspace_bytes(int n, int k, int cols) {
    return std::max(gemm_workspace_bytes_for<Gemm>(n, k, cols),
                    gemm_workspace_bytes_for<GemmF32>(n, k, cols));
}

// Stages the FP16 weights (code * row scale) and activations, then runs D = acc + beta * C on
// `out` (C = D).
template <class G>
void run(const Tensor& x, const Weight& w, typename G::ElementC* out, float beta,
         WorkspaceArena& ws, cudaStream_t stream) {
    const int n = w.n;
    const int k = w.k;
    const int t = x.ne[1];
    const std::size_t gemm_bytes = gemm_workspace_bytes(n, k, t);
    auto scope = ws.scope();
    Scratch<WorkspaceArena> scratch = allocate_scratch(ws, n, k, t, gemm_bytes);
    auto* weight = static_cast<cutlass::half_t*>(scratch.weight.data);
    auto* input  = static_cast<cutlass::half_t*>(scratch.input.data);

    const dim3 block(256);
    const dim3 grid(static_cast<unsigned>((k / 2 + 255) / 256), static_cast<unsigned>(n), 1u);
    dequant_fp8_row_to_fp16<<<grid, block, 0, stream>>>(
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const __nv_bfloat16*>(w.scales), n,
        k, w.layout == QuantLayout::VoltaQpnPrepacked, weight);
    CUDA_CHECK(cudaGetLastError());
    const std::int64_t input_count = static_cast<std::int64_t>(t) * k;
    bf16_to_fp16_kernel<<<static_cast<int>((input_count + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), input, input_count);
    CUDA_CHECK(cudaGetLastError());

    const cutlass::gemm::GemmCoord shape(t, n, k);
    typename G::Arguments args{shape, {input, k}, {weight, k}, {out, n}, {out, n}, {1.0F, beta}, 1};
    G op;
    cutlass::Status status = op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("fp8_cutlass_sm70: CUTLASS can_implement failed");
    }
    status = op.initialize(args, scratch.gemm.data, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("fp8_cutlass_sm70: CUTLASS initialize failed");
    }
    status = op(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("fp8_cutlass_sm70: CUTLASS gemm failed");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

std::size_t fp8_cutlass_sm70_workspace_bytes(std::int32_t n, std::int32_t k, std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_scratch(layout, n, k, cols, gemm_workspace_bytes(n, k, cols));
    return layout.peak_bytes(1);
}

void fp8_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                             cudaStream_t stream) {
    run<Gemm>(x, w, static_cast<cutlass::bfloat16_t*>(out.data), 0.0F, ws, stream);
}

void fp8_cutlass_sm70_residual_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                      WorkspaceArena& ws, cudaStream_t stream) {
    run<GemmF32>(x, w, static_cast<float*>(residual.data), 1.0F, ws, stream);
}

} // namespace ninfer::ops::detail
