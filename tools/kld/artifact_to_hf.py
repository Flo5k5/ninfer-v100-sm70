#!/usr/bin/env python3
"""Write the effective text weights of a Qwen3.8-27B .ninfer artifact as an HF checkpoint.

Fake quantization for KL-divergence runs on any engine: every `text/` parameter of the artifact is
decoded from its stored words (NVFP4, row-scaled FP8, grouped integer, or direct BF16/FP32) and
written in the Hugging Face layout of the source checkpoint, which supplies every other tensor
(vision tower, MTP head), the config and the tokenizer. Converting the result with llama.cpp's
convert_hf_to_gguf.py (`--outtype f16`) yields a GGUF whose text model computes with the same
weights as the artifact, so `llama-perplexity` measures the weight-quantization effect alone,
without NInfer kernels or KV formats.

The artifact-to-HF mapping is the one of tools.convert.qwen3_8_27b.graft_single_source: fused
objects are split by their binding ranges, attention query/gate rows are interleaved back per head,
GDN q/k/v rows are concatenated into in_proj_qkv, and the GDN convolution is transposed back.

    python3 -m tools.kld.artifact_to_hf --artifact model.ninfer \\
      --source /path/to/Qwen3.8-27B-BF16 --out /path/to/fake-quant-hf

Needs torch and safetensors. A JSON report (`artifact_to_hf.json` in the output directory) records
the format and the relative RMS difference to the source of every written tensor.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tools.artifact.codecs.direct import decode_direct
from tools.artifact.codecs.fp8_row import dequantize_fp8_row_scaled
from tools.artifact.codecs.nvfp4 import dequantize_nvfp4
from tools.artifact.codecs.row_split import dequantize_row_split
from tools.artifact.formats import DIRECT_FORMATS, QUANT_FORMATS
from tools.artifact.reader import Artifact
from tools.convert.qwen3_8_27b.graft_single_source import logical_matrix

OUTPUT_DTYPES = {"f16": torch.float16, "f32": torch.float32, "bf16": torch.bfloat16}
def decode_object(artifact: Artifact, object_id: str) -> torch.Tensor:
    obj = artifact.by_id[object_id]
    payload = artifact.read_object(object_id)
    if obj.format in DIRECT_FORMATS:
        return decode_direct(payload, obj.format, obj.shape).float()
    if obj.format == "fp8_e4m3fn_row_bf16":
        return dequantize_fp8_row_scaled(payload, obj.shape)
    if obj.format == "nvfp4":
        return dequantize_nvfp4(payload, obj.shape)
    if obj.format in QUANT_FORMATS:
        return dequantize_row_split(payload, obj.format, obj.shape, dtype=torch.float32)
    raise ValueError(f"{object_id}: unsupported format {obj.format}")


class Decoder:
    """Decodes logical text parameters, keeping recent fused objects for their sibling parts."""

    CACHED_OBJECTS = 4

    def __init__(self, artifact: Artifact) -> None:
        self.artifact = artifact
        self._cache: dict[str, torch.Tensor] = {}

    def _object(self, object_id: str) -> torch.Tensor:
        if object_id not in self._cache:
            if len(self._cache) == self.CACHED_OBJECTS:
                del self._cache[next(iter(self._cache))]
            self._cache[object_id] = decode_object(self.artifact, object_id)
        return self._cache[object_id]

    def logical(self, name: str) -> tuple[torch.Tensor, list[str]]:
        binding = self.artifact.directory.bindings[name]
        if "object" in binding:
            object_id = binding["object"]
            return self._object(object_id), [self.artifact.by_id[object_id].format]
        pieces, formats = [], []
        for part in binding["parts"]:
            obj = self.artifact.by_id[part["object"]]
            begin, end = part["range"]
            flat = self._object(obj.id).reshape(-1)[begin:end]
            pieces.append(flat.reshape(-1, obj.shape[-1]) if len(obj.shape) == 2 else flat)
            formats.append(obj.format)
        return (pieces[0] if len(pieces) == 1 else torch.cat(pieces)), formats


def text_routes(artifact: Artifact) -> dict[str, list[tuple[str, Any]]]:
    """HF tensor name -> [(logical parameter, selected source rows or None)] in row order."""

    routes: dict[str, list[tuple[str, Any]]] = {}
    for logical in artifact.directory.bindings:
        if not logical.startswith("text/"):
            continue
        matrix = logical_matrix(logical)
        routes.setdefault(matrix.source, []).append((logical, matrix.rows))
    return routes


def assemble(source: torch.Tensor, parts: list[tuple[str, Any]], decoder: Decoder
             ) -> tuple[torch.Tensor, list[str]]:
    out = source.float().clone()
    formats: list[str] = []
    for logical, rows in parts:
        values, used = decoder.logical(logical)
        formats.extend(used)
        if logical.endswith("gdn/convolution"):
            # The artifact stores the depthwise kernel as [taps, channels].
            values = values.reshape(values.shape[0], -1).transpose(0, 1)
        if rows is None:
            if values.numel() != out.numel():
                raise ValueError(f"{logical}: {values.numel()} values for {tuple(out.shape)}")
            out = values.reshape(out.shape).clone()
            continue
        cursor = 0
        values = values.reshape(-1, out.shape[-1])
        for begin, end in rows:
            out[begin:end] = values[cursor:cursor + end - begin]
            cursor += end - begin
        if cursor != values.shape[0]:
            raise ValueError(f"{logical}: {values.shape[0]} rows for {cursor} selected rows")
    return out, sorted(set(formats))


def convert_artifact(artifact: Artifact, source_dir: Path, out_dir: Path,
                     dtype: torch.dtype) -> dict:
    """Write the checkpoint for an open artifact (or any object with its reader interface)."""

    decoder = Decoder(artifact)
    routes = text_routes(artifact)
    shards = sorted(source_dir.glob("*.safetensors"))
    available: set[str] = set()
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            available.update(handle.keys())
    missing = sorted(set(routes) - available)
    if missing:
        raise ValueError(f"source checkpoint lacks {len(missing)} routed tensors, e.g. {missing[:3]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        if item.is_file() and item.suffix != ".safetensors":
            shutil.copy2(item, out_dir / item.name)

    report: dict[str, Any] = {"artifact": str(getattr(artifact, "path", "")),
                              "source": str(source_dir), "dtype": str(dtype), "tensors": {}}
    written: set[str] = set()
    for shard in shards:
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            for name in handle.keys():
                source = handle.get_tensor(name)
                if name not in routes:
                    tensors[name] = source
                    continue
                values, formats = assemble(source, routes[name], decoder)
                reference = source.float()
                difference = (values - reference).pow(2).mean().sqrt()
                scale = reference.pow(2).mean().sqrt()
                report["tensors"][name] = {
                    "formats": formats,
                    "relative_rms_difference": float(difference / scale) if scale > 0 else 0.0,
                    "max_abs_difference": float((values - reference).abs().max()),
                }
                # Direct-format parameters keep their exact source dtype.
                direct = all(item in DIRECT_FORMATS for item in formats)
                tensors[name] = source if direct and torch.equal(values, reference) \
                    else values.to(dtype)
                written.add(name)
        save_file(tensors, str(out_dir / shard.name), metadata=metadata or {"format": "pt"})
        print(f"{shard.name}: {sum(1 for name in tensors if name in written)} text tensors",
              flush=True)
    return report


def convert(artifact_path: Path, source_dir: Path, out_dir: Path, dtype: torch.dtype) -> dict:
    with Artifact(artifact_path) as artifact:
        return convert_artifact(artifact, source_dir, out_dir, dtype)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path,
                        help="BF16 HF checkpoint of the same architecture")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dtype", choices=sorted(OUTPUT_DTYPES), default="f16",
                        help="storage dtype of the decoded tensors (default f16)")
    args = parser.parse_args(argv)
    started = time.monotonic()
    report = convert(args.artifact, args.source, args.out, OUTPUT_DTYPES[args.dtype])
    report["seconds"] = time.monotonic() - started
    (args.out / "artifact_to_hf.json").write_text(json.dumps(report, indent=2) + "\n")
    by_format: dict[str, list[float]] = {}
    for item in report["tensors"].values():
        by_format.setdefault("+".join(item["formats"]), []).append(item["relative_rms_difference"])
    for key, values in sorted(by_format.items()):
        print(f"{key:40s} tensors {len(values):4d}  relative RMS difference "
              f"mean {sum(values) / len(values):.4g} max {max(values):.4g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
