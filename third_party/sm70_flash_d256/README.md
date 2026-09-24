# SM70 D256 Split-D flash attention (vendored)

Device code of the Volta (`sm_70`) head-dim-256 FlashAttention "Split-D N32" kernel, used by the
FP32-accumulating Volta prefill attention route (`src/ops/launcher/gqa_attention_volta_splitd.cu`).

## Provenance and licenses

Every source file was copied from [sm70-attn](https://github.com/fishlikeX/sm70-attn) at commit
`707cf247c1beb5e7c701f47fe6f828c5b84b729d`, which adapts the kernel of 1Cat-vLLM and carries the
FlashAttention base layer of a Volta fork. `NOTICE` lists every origin with its copyright line.

| file | origin | license |
|---|---|---|
| `fattn-sm70-d256-kernel.cuh` | [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) v1.3.0 (`6ada86ed64af6d1a7b3cb0f34df237fd86f06d48`): `csrc/flash_attn/src/flash_fwd_d256_splitd_sm70.cu`, created by `cmake/patches/sm70_flash_attn_d256_pipeline.patch` and extended by `cmake/patches/sm70_flash_attn_d256_splitkv3.patch`; adapted by sm70-attn in `ggml/src/ggml-cuda/fattn-sm70-d256-kernel.cuh` | Apache-2.0 AND MIT |
| `flash/kernel_traits.h`, `flash/namespace_config.h`, `flash/philox.cuh`, `flash/softmax.h` | FlashAttention base layer of the Volta fork [flash-attention-v100](https://github.com/zhinianqin/flash-attention-v100) at `c2eda5e6115b98c3ba4bfd181570668742eece22` (`csrc/flash_attn/src/`), unchanged by sm70-attn (`ggml/src/ggml-cuda/sm70-vendor/flash/`) | BSD-3-Clause |
| `flash/mask.h`, `flash/utils.h` | same FlashAttention base layer, modified by sm70-attn | BSD-3-Clause AND MIT |

License texts: `LICENSE.Apache-2.0` (1Cat-vLLM), `LICENSE.MIT` (sm70-attn), `LICENSE.BSD-3-Clause`
(FlashAttention) with the `AUTHORS` file it refers to. CuTe and CUTLASS come from the CUTLASS
release the Volta build already fetches (see the top-level `CMakeLists.txt`); they are not vendored
here.

## Changes

sm70-attn's changes to the 1Cat-vLLM kernel, as used here: the device code is extracted from the
torch extension (host wrappers dropped), the causal offset becomes the `kv_offset` argument, the
output element type becomes a template parameter so the attention output can be staged in FP32,
the three-way KV split of the `splitkv3` patch is ported with an empty-segment guard and a finite
row-maximum initialization, and an opt-in q4_0 direct-load path is added (NInfer does not
instantiate it). In the FlashAttention layer, sm70-attn replaced `std::min`/`std::max` with
conditional expressions in `flash/mask.h` and removed the paged-KV helper
`resolve_thread_kv_page_slice_offset` from `flash/utils.h`.

Changes made in NInfer, provided under the license of the file they modify:

- `fattn-sm70-d256-kernel.cuh`: new file header; the CuTe/CUTLASS headers are included from the
  build's CUTLASS include directory (`<cute/tensor.hpp>`, `<cutlass/numeric_types.h>`) and the
  FlashAttention headers as `"flash/..."` instead of the sm70-attn `sm70-vendor/` prefix.
- `flash/utils.h`: the Chinese comments are translated to English. Its `HACK` markers are upstream
  FlashAttention comments.

The other files are byte-identical to their sm70-attn copies. The sm70-attn
`flash_namespace_config.h` stub is not vendored: the build defines
`FLASH_NAMESPACE=ninfer_sm70_flash` for the one translation unit that includes these headers, so the
code never shares the default `flash` namespace with another FlashAttention copy.

## Numerics

Q.K^T and P.V both run on `SM70_8x8x4_F32F16F16F32` tensor-core atoms: FP16 operands, FP32
accumulators, FP32 online softmax. The SplitKV3 variant writes FP32 partial outputs, maxima and
sums that a separate kernel merges in FP32.

## Updating

Re-vendor from a single sm70-attn commit, check the provenance of any new code against its
upstreams, update `NOTICE` and this file, and re-run the attention test suite
(`ninfer_softmax_attention_test`) before trusting the result.
