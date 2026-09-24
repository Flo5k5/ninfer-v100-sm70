#include "ops/linear/fp8/fp8_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/fp8/fp8_gemv.cuh"
#include "ops/linear/fp8/fp8_prepack_sm70.cuh"
#include "ops/linear/fp8/fp8_row_scale_gemm_sm70.cuh"

#include "cutlass/bfloat16.h"
#include "cutlass/half.h"

#include <cuda_bf16.h>

#include <algorithm>

namespace ninfer::ops::detail {
namespace {

// Stages the E4M3 codes unscaled: FP16 represents every E4M3 value exactly, and the row scale is
// applied to the FP32 accumulator in the GEMM epilogue (fp8_row_scale_gemm_sm70.cuh).
__global__ void dequant_fp8_row_to_fp16(const std::uint8_t* __restrict__ codes, int n, int k,
                                        bool prepacked, cutlass::half_t* __restrict__ out) {
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
    cutlass::half_t* out_row   = out + static_cast<std::int64_t>(row) * k;
    out_row[pair_idx * 2]      = cutlass::half_t(weight.x);
    out_row[pair_idx * 2 + 1]  = cutlass::half_t(weight.y);
}

__global__ void bf16_to_fp16_kernel(const __nv_bfloat16* __restrict__ in,
                                    cutlass::half_t* __restrict__ out, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { out[i] = cutlass::half_t(__bfloat162float(in[i])); }
}

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

std::size_t gemm_workspace_bytes(int n, int k, int cols) {
    return std::max(fp8_row_scale_gemm_workspace_bytes<cutlass::bfloat16_t>(n, k, cols),
                    fp8_row_scale_gemm_workspace_bytes<float>(n, k, cols));
}

// Stages the FP16 codes and activations, then runs D = (acc * scale) + beta * D.
template <class ElementOut>
void run(const Tensor& x, const Weight& w, ElementOut* out, float beta, WorkspaceArena& ws,
         cudaStream_t stream) {
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
        static_cast<const std::uint8_t*>(w.qdata), n, k,
        w.layout == QuantLayout::VoltaQpnPrepacked, weight);
    CUDA_CHECK(cudaGetLastError());
    const std::int64_t input_count = static_cast<std::int64_t>(t) * k;
    bf16_to_fp16_kernel<<<static_cast<int>((input_count + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), input, input_count);
    CUDA_CHECK(cudaGetLastError());

    run_fp8_row_scale_gemm<ElementOut>(input, weight, static_cast<const __nv_bfloat16*>(w.scales),
                                       out, beta, n, k, t, scratch.gemm.data, stream,
                                       "fp8_cutlass_sm70");
}

} // namespace

std::size_t fp8_cutlass_sm70_workspace_bytes(std::int32_t n, std::int32_t k, std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_scratch(layout, n, k, cols, gemm_workspace_bytes(n, k, cols));
    return layout.peak_bytes(1);
}

void fp8_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                             cudaStream_t stream) {
    run<cutlass::bfloat16_t>(x, w, static_cast<cutlass::bfloat16_t*>(out.data), 0.0F, ws, stream);
}

void fp8_cutlass_sm70_residual_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                      WorkspaceArena& ws, cudaStream_t stream) {
    run<float>(x, w, static_cast<float*>(residual.data), 1.0F, ws, stream);
}

} // namespace ninfer::ops::detail
