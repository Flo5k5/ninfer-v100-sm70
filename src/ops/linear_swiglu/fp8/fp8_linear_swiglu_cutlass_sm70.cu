#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/fp8/fp8_gemv.cuh"
#include "ops/linear/fp8/fp8_prepack_sm70.cuh"
#include "ops/linear/fp8/fp8_row_scale_gemm_sm70.cuh"

#include "cutlass/bfloat16.h"
#include "cutlass/half.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ninfer::ops::detail {
namespace {

// Row-scaled E4M3: one byte per element, one BF16 scale per output row (weight.scales, n
// entries), decoded two elements at a time with the same decode_fp8_e4m3x2 the GEMV/small-T
// kernels use (fp8_gemv.cuh). The codes are staged UNSCALED: FP16 represents every E4M3 value
// exactly, and the row scale multiplies the FP32 accumulator in the GEMM epilogue
// (fp8_row_scale_gemm_sm70.cuh), so each gate/up output is rounded once, to BF16. Folding the
// scale into the staged weights would round each code * scale to FP16 and lose rows whose
// products fall into FP16 subnormals; scaling the BF16 GEMM output afterwards would round every
// output twice.
__global__ void dequant_fp8_row_to_fp16(const std::uint8_t* __restrict__ codes, int n, int k,
                                        bool prepacked, bool swiglu_interleave,
                                        cutlass::half_t* __restrict__ out) {
    const int row      = blockIdx.y;
    const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pairs_per_row = k / 2;
    if (row >= n || pair_idx >= pairs_per_row) { return; }

    const int k0 = pair_idx * 2;
    // `row` is logical; the SwiGLU layout stores gate row f and up row f in the same 32-row tile.
    const int half_n       = n / 2;
    const int physical_row = !swiglu_interleave ? row
                             : row < half_n     ? (row / 16) * 32 + (row & 15)
                                                : ((row - half_n) / 16) * 32 + 16 +
                                                      ((row - half_n) & 15);
    const std::int64_t source_offset = prepacked
                                           ? fp8_qpn_prepacked_offset(physical_row, k0, k)
                                           : static_cast<std::int64_t>(row) * k + k0;
    const std::uint16_t packed =
        *reinterpret_cast<const std::uint16_t*>(codes + source_offset);
    const float2 weight = decode_fp8_e4m3x2(packed);

    cutlass::half_t* out_row = out + static_cast<std::int64_t>(row) * k;
    out_row[pair_idx * 2]     = cutlass::half_t(weight.x);
    out_row[pair_idx * 2 + 1] = cutlass::half_t(weight.y);
}

__global__ void bf16_to_fp16_kernel(const __nv_bfloat16* __restrict__ in,
                                    cutlass::half_t* __restrict__ out, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { out[i] = cutlass::half_t(__bfloat162float(in[i])); }
}

int div_up_i(int a, int b) { return (a + b - 1) / b; }

template <class Allocator>
struct CutlassWorkspace {
    Tensor w_fp16;
    Tensor x_fp16;
    DeviceSpan gemm_workspace;
};

template <class Allocator>
CutlassWorkspace<Allocator> allocate_cutlass_workspace(Allocator& allocator, std::int32_t n,
                                                       std::int32_t k, std::int32_t cols,
                                                       std::size_t gemm_workspace_bytes) {
    CutlassWorkspace<Allocator> out;
    out.w_fp16 = allocator.alloc(DType::FP16, {k, n});
    out.x_fp16 = allocator.alloc(DType::FP16, {k, cols});
    if (gemm_workspace_bytes > 0) { out.gemm_workspace = allocator.alloc_bytes(gemm_workspace_bytes); }
    return out;
}

} // namespace

std::size_t fp8_linear_swiglu_cutlass_workspace_bytes(std::int32_t gate_up_rows, std::int32_t k,
                                                       std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_cutlass_workspace(
        layout, gate_up_rows, k, cols,
        fp8_row_scale_gemm_workspace_bytes<cutlass::bfloat16_t>(gate_up_rows, k, cols));
    return layout.peak_bytes(1);
}

void fp8_linear_swiglu_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& gate_up_out,
                                           WorkspaceArena& ws, cudaStream_t stream) {
    const std::int32_t k    = x.ne[0];
    const std::int32_t cols = x.ne[1];
    const std::int32_t n    = w.n;

    auto scratch_scope = ws.scope();
    CutlassWorkspace<WorkspaceArena> scratch = allocate_cutlass_workspace(
        ws, n, k, cols, fp8_row_scale_gemm_workspace_bytes<cutlass::bfloat16_t>(n, k, cols));

    auto* w_fp16 = static_cast<cutlass::half_t*>(scratch.w_fp16.data);
    auto* x_fp16 = static_cast<cutlass::half_t*>(scratch.x_fp16.data);

    {
        const dim3 block(256);
        const dim3 grid(static_cast<unsigned>(div_up_i(k / 2, 256)), static_cast<unsigned>(n), 1u);
        dequant_fp8_row_to_fp16<<<grid, block, 0, stream>>>(static_cast<const std::uint8_t*>(w.qdata),
                                                            n, k,
                                                            w.layout == QuantLayout::VoltaQpnPrepacked ||
                                                                w.layout == QuantLayout::VoltaQpnPrepackedSwiGlu,
                                                            w.layout == QuantLayout::VoltaQpnPrepackedSwiGlu,
                                                            w_fp16);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        const std::int64_t count = static_cast<std::int64_t>(cols) * k;
        const int threads        = 256;
        const int blocks         = static_cast<int>((count + threads - 1) / threads);
        bf16_to_fp16_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), x_fp16, count);
        CUDA_CHECK(cudaGetLastError());
    }

    run_fp8_row_scale_gemm<cutlass::bfloat16_t>(
        x_fp16, w_fp16, static_cast<const __nv_bfloat16*>(w.scales),
        static_cast<cutlass::bfloat16_t*>(gate_up_out.data), 0.0F, n, k, cols,
        scratch.gemm_workspace.data, stream, "fp8_linear_swiglu_cutlass_sm70");
}

} // namespace ninfer::ops::detail
