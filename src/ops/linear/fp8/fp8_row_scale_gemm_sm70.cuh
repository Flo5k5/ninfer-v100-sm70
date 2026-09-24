#pragma once

// Volta (sm_70) wide-T GEMM for row-scaled FP8 weights on FP16 tensor cores:
//
//   D[t, n] = (sum_k X[t, k] * code[n, k]) * scale[n] + beta * C[t, n]
//
// The caller stages the E4M3 codes unscaled in FP16, which represents every E4M3 value exactly,
// and the activations in FP16. The BF16 row scale multiplies the FP32 accumulator in the
// epilogue, so each output is rounded once, to ElementOut. Folding the scale into the staged FP16
// weights instead rounds every code * scale product to FP16, and a row whose products fall below
// the smallest normal FP16 value (6.1e-5, e.g. a row with amax near 2.7e-2 and codes below 1)
// loses their precision to FP16 subnormals; scaling a BF16 GEMM output afterwards rounds every
// output twice.

#include "core/device.h"

#include "cutlass/array.h"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_with_broadcast.h"
#include "cutlass/half.h"
#include "cutlass/numeric_conversion.h"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

// EpilogueWithBroadcast output operator. The broadcast vector is the per-output-row BF16 scale
// (one entry per GEMM column n); T is not stored.
template <class ElementOut_, int Count>
class Fp8RowScaleOutputOp {
public:
    using ElementOutput      = ElementOut_;
    using ElementD           = ElementOut_;
    using ElementC           = ElementOut_;
    using ElementZ           = ElementOut_;
    using ElementT           = ElementOut_;
    using ElementAccumulator = float;
    using ElementCompute     = float;
    using ElementScalar      = float;
    using ElementVector      = cutlass::bfloat16_t;

    static int const kElementsPerAccess = Count;
    static int const kCount             = Count;
    static bool const kIsSingleSource   = true;
    static bool const kStoreZ           = true;
    static bool const kStoreT           = false;

    using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
    using FragmentCompute     = cutlass::Array<ElementCompute, kCount>;
    using FragmentC           = cutlass::Array<ElementC, kCount>;
    using FragmentZ           = cutlass::Array<ElementZ, kCount>;
    using FragmentT           = cutlass::Array<ElementT, kCount>;
    using FragmentSource      = FragmentC;
    using FragmentOutput      = FragmentZ;

    struct Params {
        // 0 writes the scaled projection; 1 accumulates it onto D (C aliases D).
        ElementCompute beta = ElementCompute(0);
    };

    CUTLASS_HOST_DEVICE explicit Fp8RowScaleOutputOp(Params const& params) : beta_(params.beta) {}

    CUTLASS_HOST_DEVICE bool is_source_needed() const { return beta_ != ElementCompute(0); }

    CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}

    CUTLASS_HOST_DEVICE void operator()(FragmentZ& z, FragmentT&, FragmentAccumulator const& ab,
                                        FragmentC const& c, FragmentCompute const& scale) const {
        const cutlass::NumericArrayConverter<ElementCompute, ElementC, kCount> widen;
        const cutlass::NumericArrayConverter<ElementZ, ElementCompute, kCount> narrow;
        const FragmentCompute source = widen(c);
        FragmentCompute result;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kCount; ++i) { result[i] = ab[i] * scale[i] + beta_ * source[i]; }
        z = narrow(result);
    }

    CUTLASS_HOST_DEVICE void operator()(FragmentZ& z, FragmentT&, FragmentAccumulator const& ab,
                                        FragmentCompute const& scale) const {
        const cutlass::NumericArrayConverter<ElementZ, ElementCompute, kCount> narrow;
        FragmentCompute result;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kCount; ++i) { result[i] = ab[i] * scale[i]; }
        z = narrow(result);
    }

private:
    ElementCompute beta_;
};

// Same tile configuration as the other Volta CUTLASS routes: 128x128x32 threadblocks, 64x64x32
// warps, HMMA.884, two stages.
template <class ElementOut>
using Fp8RowScaleGemm = cutlass::gemm::device::GemmUniversalWithBroadcast<
    cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t, cutlass::layout::ColumnMajor,
    ElementOut, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70, cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<8, 8, 4>,
    Fp8RowScaleOutputOp<ElementOut, 128 / cutlass::sizeof_bits<ElementOut>::value>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

template <class ElementOut>
typename Fp8RowScaleGemm<ElementOut>::Arguments
fp8_row_scale_gemm_arguments(const cutlass::half_t* x, const cutlass::half_t* codes,
                             const __nv_bfloat16* scales, ElementOut* out, float beta,
                             std::int32_t n, std::int32_t k, std::int32_t cols) {
    using Gemm = Fp8RowScaleGemm<ElementOut>;
    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, cutlass::gemm::GemmCoord(cols, n, k), 1,
        typename Gemm::EpilogueOutputOp::Params{beta}, x, codes, out, out,
        const_cast<void*>(static_cast<const void*>(scales)), nullptr, 0, 0, 0, 0, 0, 0, k, k, n,
        n, 0, 0);
}

template <class ElementOut>
std::size_t fp8_row_scale_gemm_workspace_bytes(std::int32_t n, std::int32_t k,
                                               std::int32_t cols) {
    return Fp8RowScaleGemm<ElementOut>::get_workspace_size(
        fp8_row_scale_gemm_arguments<ElementOut>(nullptr, nullptr, nullptr, nullptr, 0.0F, n, k,
                                                 cols));
}

// x: FP16 [cols, k] row-major; codes: unscaled FP16 [n, k] row-major; scales: BF16 [n];
// out: ElementOut [cols, n] row-major, read as C when beta is nonzero.
template <class ElementOut>
void run_fp8_row_scale_gemm(const cutlass::half_t* x, const cutlass::half_t* codes,
                            const __nv_bfloat16* scales, ElementOut* out, float beta,
                            std::int32_t n, std::int32_t k, std::int32_t cols, void* workspace,
                            cudaStream_t stream, const char* op) {
    using Gemm      = Fp8RowScaleGemm<ElementOut>;
    const auto args = fp8_row_scale_gemm_arguments<ElementOut>(x, codes, scales, out, beta, n, k,
                                                               cols);
    Gemm gemm;
    cutlass::Status status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(std::string(op) + ": CUTLASS can_implement failed");
    }
    status = gemm.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(std::string(op) + ": CUTLASS initialize failed");
    }
    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(std::string(op) + ": CUTLASS gemm failed");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
