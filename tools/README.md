# NInfer maintainer tools

`tools/` contains the project-owner workflows for artifact conversion and inspection, benchmark
orchestration, and serving smoke checks. These tools are not part of the public download-and-run
path; normal users should start with the [project README](../README.md). To build your own weights,
use the [weight conversion guide](../docs/weight-conversion.md).

Run commands from the repository root with a Python 3.11 environment containing the dependencies
for the selected tool.

## Task index

| Task | Location |
|---|---|
| Convert weights with an official or custom recipe | [`convert/`](convert/); [user guide](../docs/weight-conversion.md) |
| Rebuild the Qwen3.8-27B NVFP4 artifact from one single-source checkpoint | [`convert/qwen3_8_27b/graft_single_source.py`](convert/qwen3_8_27b/graft_single_source.py) |
| Re-encode its NVFP4 MLP objects from a calibrated checkpoint | [`convert/qwen3_8_27b/reencode_nvfp4.py`](convert/qwen3_8_27b/reencode_nvfp4.py) |
| Inspect artifact metadata and objects | [`artifact/inspect.py`](artifact/inspect.py) |
| Run benchmark matrices | [`bench/`](bench/README.md) |
| Measure external Serve TTFT | [`bench/ttft/`](bench/ttft/README.md) |
| Compare logits dumps (KLD, top-1, PPL) and gate a quantization | [`kld/kld.py`](kld/kld.py), [runbook](../docs/perplexity.md#runbook) |
| Exercise a resident HTTP server | [`smoke/serve_contract.py`](smoke/serve_contract.py) |
| Exercise thinking preservation through a managed server | [`smoke/serve_thinking_preservation.py`](smoke/serve_thinking_preservation.py) |

## Artifact workflow

The common converter reads selected local sources and writes a `.ninfer` artifact plus its
`.conversion.json` report. This example includes the optional weights of the official Qwen3.8-27B
NVFP4 artifact; the input paths are placeholders for local checkpoint checkouts:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.8-27B \
  --recipe qwen3_8_27b_nvfp4 --components text,vision,mtp,dflash2 --proposal \
  --source quantized=/path/to/Qwen3.8-27B-NVFP4 \
  --source dflash2=/path/to/Qwen3.8-27B-DFlash2 \
  --name qwen3.8-27b \
  --out out/qwen3_8_27b_nvfp4.ninfer
```

Inspect a result:

```bash
python3 -m tools.artifact.inspect out/qwen3_8_27b_nvfp4.ninfer --objects
```

The five official recipes, mixed sources, custom methods, resources and sharding are described in
the [conversion guide](../docs/weight-conversion.md), with the artifacts the Volta Engine runs.
Published users download the completed artifacts from Hugging Face instead of running these
workflows.

Two tools start from a published Qwen3.8-27B NVFP4 v3 artifact instead of a whole checkpoint set,
and keep its directory: `graft_single_source` replaces its payloads with those of one
compressed-tensors checkpoint of the same architecture, and `reencode_nvfp4` rewrites its NVFP4 MLP
objects from a calibrated checkpoint. Their module docstrings give the commands.

## Benchmark orchestration

`tools/bench/run_ninfer_bench_matrix.py` builds and runs the public-Engine benchmark matrix and
writes ignored local reports below `profiles/bench/`:

```bash
python3 tools/bench/run_ninfer_bench_matrix.py --preset core --dry-run
python3 tools/bench/run_ninfer_bench_matrix.py --preset core
```

See [`tools/bench/README.md`](bench/README.md) and [`bench/README.md`](../bench/README.md) for the
orchestrator and executable contracts.

For request-arrival latency, use the managed Qwen3.8-27B NVFP4/FP8 TTFT campaign. Its measurement
runner remains an external-only HTTP client; the separate controller owns Serve lifecycle and
artifacts. See [`tools/bench/ttft/README.md`](bench/ttft/README.md).

## Serving smoke

After starting `ninfer-serve --vision` in another terminal:

```bash
python3 -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 \
  --model qwen3.6-27b
```

The client exercises OpenAI, Anthropic, streaming, usage, multimodal, and tool-call response
surfaces against the resident process. Pass `--no-response-store` when the server runs with that
option: the client then requires Responses to be served but never stored, continued, retrieved,
listed, cancelled, or deleted. Pass `--no-vision` for a server started without `--vision`; media
requests must then fail with `vision_disabled`. For a server with an API key, export the key as
`NINFER_API_KEY`; the client sends it as a Bearer token.

For typed rewrite-checkpoint and thinking-history behavior, the managed smoke script launches a
real server and consumes the repository fixture:

```bash
python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_27b.ninfer --backend mtp
```
