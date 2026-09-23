# SM70 D256 Split-D flash attention (vendored)

Device code of the Volta (`sm_70`) head-dim-256 FlashAttention "Split-D N32" kernel, used by the
FP32-accumulating Volta prefill attention route (`src/ops/launcher/gqa_attention_volta_splitd.cu`).

## Provenance

| file | origin | license |
|---|---|---|
| `fattn-sm70-d256-kernel.cuh` | 1CatAI D256 Split-D kernel (1Cat-vLLM v1.3.0, commit `6ada86ed64`, `csrc/flash_attn/src/flash_fwd_d256_splitd_sm70.cu` with its `sm70_flash_attn_d256_pipeline` and `splitkv3` patches), as carried by the [sm70-attn](https://github.com/Flo5k5/sm70-attn) llama.cpp plugin | BSD-3-Clause |
| `flash/*` | FlashAttention 2 base layer of the Volta fork `zhinianqin/flash-attention-v100` at `c2eda5e6` (online softmax, masking, kernel traits) | BSD-3-Clause |

`LICENSE` is the FlashAttention BSD 3-Clause text; `NOTICE` lists the copyright holders. CuTe and
CUTLASS come from the CUTLASS release the Volta build already fetches (see the top-level
`CMakeLists.txt`); they are not vendored here.

## Deviations from the source

Code is byte-identical to the plugin copy. Only these lines differ:

- Includes: the kernel's CuTe/CUTLASS includes use the build's CUTLASS include directory
  (`<cute/tensor.hpp>`, `<cutlass/numeric_types.h>`), and the FlashAttention headers are included as
  `"flash/..."` instead of the plugin's `sm70-vendor/` prefix.
- Comments: the Chinese comments of `flash/utils.h` are translated to English, and the header of
  `fattn-sm70-d256-kernel.cuh` points to the NInfer launcher and to `LICENSE`/`NOTICE` instead of
  plugin files that are not vendored.
- The plugin's `flash_namespace_config.h` stub is not vendored. The build defines
  `FLASH_NAMESPACE=ninfer_sm70_flash` for the one translation unit that includes these headers, so
  the code never shares the default `flash` namespace with another FlashAttention copy.

The `HACK` markers in `flash/utils.h` are upstream FlashAttention comments.

## Numerics

Q.K^T and P.V both run on `SM70_8x8x4_F32F16F16F32` tensor-core atoms: FP16 operands, FP32
accumulators, FP32 online softmax. The SplitKV3 variant writes FP32 partial outputs, maxima and
sums that a separate kernel merges in FP32.
