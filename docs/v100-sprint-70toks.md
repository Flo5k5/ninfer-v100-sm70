# V100 decode sprint: 70 tok/s at 26k context (22-23/09/2026)

Target: Qwen3.8-27B on one Tesla V100-PCIE-32GB (sm_70), int8 KV cache, MTP speculative decoding,
decode throughput at a 26k-token context. Measured with the serve API: 3 probes at 26k (summary of a
25.5k-token document, 500-token budget, temperature 1.0) + 2 short probes per server run.

## Metric
tok/s = tok/step / ms/step. ms/step = predicted_ms / (predicted_n - draft_n_accepted) is stable to ~1%
between runs; tok/step moves by ~8% per probe with sampling, so it is pooled (sum tokens / sum steps)
over >= 9 probes. Draft-quality changes are compared on a paired greedy benchmark (identical tokens).

## Result (GPU at 1380 MHz, --spec mtp --draft-tokens 4 --lm-head-draft)

| Build | ms/step @26k | tok/step | tok/s @26k |
|---|---|---|---|
| stock v3 port, K=5, no --lm-head-draft | 70.9 | ~3.0 | ~45 |
| stock + --lm-head-draft | 66.1 | ~3.0 | 46.5 |
| + attention split-K, NVFP4 batching, fp16 staging, Q4 pipeline, K=4 | 43.5 | 2.96 | 68 |
| + draft-path GEMVs (W8 T=1, Q4 head, W8 catch-up) | 41.9 | 2.96 | 70.6 |
| + verify-path fusions (gemv track) | **38.9** | 3.01 | **77.4** |
| same build, second checkpoint of the same architecture | 37.8 | 2.96 | 78.5 |
| + key-major int8 attention kernel (second checkpoint) | **35.0** | 2.83 | **80.8** |
| same, DFlash2 K=7 (second checkpoint; best at short context: 96-122 tok/s) | 47.4 | 3.74 | 78.8 |

## Changes (branch `integration`)
- Attention: key-major Volta int8 kernel (keys/head dims on the 32-wide mma axis, one int8 quant group per
  warp loaded straight to registers, overlapping prefetch, lazy per-row max). 26k: W1 264 -> 93 us, W5 290 -> 128, W8 554 -> 207.
- Attention (small_t_i8_volta): split-K QK^T across the dim-split warps, quadpair-split tail QK^T,
  16-byte fragment loads, device-side wave alignment, magic-bias int8 dequant, fp32 PV accumulation
  with lazy rescale. 26k: W=6 935 -> 390 us, W=1 543 -> 266 us.
- NVFP4 QPN2: batched group loads, SPLITK=16 for the down projection, single gate/up launch, fewer
  instructions per group, residual add and SwiGLU in the epilogue (runtime layout VoltaQpnPrepackedSwiGlu).
- fp16 activations end to end on the QPN routes (Volta mma takes fp16 only; in-loop bf16->fp16
  conversions on the quarter-rate F2F pipe were the limiter): staging, then producers writing fp16.
- FP8 SwiGLU: T was chunked by 4 on the QPN route, re-streaming the 178 MB gate/up weight at T=5.
- MTP draft path: register-streamed W8 GEMV (T=1), Q4 proposal head at 795 GB/s, multi-token W8 catch-up.
- Small kernels: one-pass RMSNorm, fused GDN norm+gating restored, qk RMSNorm+RoPE fused (1255 -> 726 launches/step).

## Rejected (measured)
Register prefetch of K/V in attention (register cap), GDN next-token prefetch (occupancy), W8 QPN at T=1,
truncated proposal head (acceptance loss > time), MTP attention window (acceptance loss > time),
K=5 / K=3 (K=4 best), DFlash2 K=7 (67 tok/s at 26k, best at short context).

## Hardware notes
Application clocks 877/1380 (`nvidia-smi -ac` + `-lgc`) give -4% ms/step over the default boost clocks.
