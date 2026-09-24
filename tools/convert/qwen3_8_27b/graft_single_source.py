"""Rebuild a Qwen3.8-27B NVFP4 artifact from one compressed-tensors checkpoint.

The published NInfer v3 artifact (``--template``) fixes the complete directory:
components, object geometry and formats, logical bindings, activation uses and
auxiliaries. This tool keeps that directory byte-for-byte in meaning and only
replaces object payloads with values derived from a different checkpoint that
shares the Qwen3.8-27B architecture and the NVFP4/FP8 quantization scheme, for
example an abliterated fine-tune published as one compressed-tensors source.

Per object, the payload comes from one of three routes:

* ``copy``: bytes are copied from the template (frontend resources, the vision
  tower, the DFlash2 draft model and the proposal token ids), because they do
  not depend on the language-model weights, or were checked identical.
* pre-quantized source words: FP8 row-scaled and NVFP4 matrices are assembled
  from the exact source codes and scales, without numerical conversion.
* BF16 source values: norms, GDN parameters and the MTP head are encoded
  directly or with the registered grouped quantizers; BF16 FP8 targets (token
  embedding, output head when the source keeps it in BF16) use the canonical
  row-maxabs FP8 encoder.

Example (a single-source NVFP4 checkpoint of the same architecture)::

    python3 -m tools.convert.qwen3_8_27b.graft_single_source \\
      --template <published>/qwen3_8_27b_nvfp4.ninfer \\
      --source <checkpoint>/Qwen3.8-27B-NVFP4 --source-label <org>/Qwen3.8-27B-NVFP4 \\
      --out <output>/qwen3_8_27b_nvfp4.ninfer

The artifact's provenance names the source by its required label (a repository id, for example)
and the template by file name and artifact id, never by path; the template's own provenance is
kept without its path members, which are listed under ``graft.removed_template_paths``.

``--verify-against-template`` regenerates the selected objects and compares
them with the template payloads instead of writing an artifact. Run it with the
template's own quantized source (unsloth/Qwen3.8-27B-NVFP4) to prove the object
mapping reproduces the published artifact bit-exactly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fnmatch
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import time
from typing import Callable, Iterator, Sequence

import torch
from safetensors import safe_open

from tools.artifact.codecs.direct import decode_direct, encode_direct
from tools.artifact.codecs.fp8_row import encode_fp8_row_scaled
from tools.artifact.codecs.nvfp4 import encode_nvfp4
from tools.artifact.formats import QUANT_FORMATS
from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorObject, TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.convert.common.provenance import input_label, local_name, strip_local_paths
from tools.convert.common.quantize import quantize_and_encode
from tools.convert.qwen3_8_27b.fp8_embedding import (
    ENCODER_PROFILE,
    quantize_bf16_rows,
)


COPY_PREFIXES = ("resource/", "vision/", "dflash2/", "proposal/token_ids")
ATTENTION_HEADS = 24
ATTENTION_HEAD_ROWS = 512
ATTENTION_QUERY_ROWS = 256
GDN_KEY_ROWS = 2048
GDN_QKV_ROWS = 10240
FP8_ROW_CHUNK = 4096


class GraftError(ValueError):
    pass


class SourceCheckpoint:
    """Name-indexed view over every safetensors file of one checkpoint dir."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        self.files: dict[str, Path] = {}
        self.meta: dict[str, tuple[str, tuple[int, ...]]] = {}
        for path in sorted(self.model_dir.glob("*.safetensors")):
            with path.open("rb") as handle:
                (length,) = struct.unpack("<Q", handle.read(8))
                header = json.loads(handle.read(length))
            for name, value in header.items():
                if name == "__metadata__":
                    continue
                if name in self.files:
                    raise GraftError(f"{name} is stored in more than one shard")
                self.files[name] = path
                self.meta[name] = (value["dtype"], tuple(value["shape"]))
        if not self.files:
            raise GraftError(f"{self.model_dir}: no safetensors shards")
        self._path: Path | None = None
        self._handle = None

    def _open(self, name: str):
        try:
            path = self.files[name]
        except KeyError:
            raise GraftError(f"source is missing {name}") from None
        if path != self._path:
            self.close()
            self._handle = safe_open(str(path), framework="pt", device="cpu")
            self._path = path
        return self._handle

    def dtype(self, name: str) -> str:
        return self.meta[name][0] if name in self.meta else ""

    def get(self, name: str) -> torch.Tensor:
        return self._open(name).get_tensor(name)

    def rows(self, name: str, begin: int, end: int) -> torch.Tensor:
        return self._open(name).get_slice(name)[begin:end]

    def close(self) -> None:
        if self._handle is not None:
            self._handle.__exit__(None, None, None)
        self._handle = None
        self._path = None


@dataclass(frozen=True, slots=True)
class Matrix:
    """One logical parameter as rows selected from one source matrix."""

    source: str
    rows: tuple[tuple[int, int], ...] | None = None


@dataclass(frozen=True, slots=True)
class Part:
    logical: str
    begin: int
    end: int


def _attention_rows(gate: bool) -> tuple[tuple[int, int], ...]:
    offset = ATTENTION_QUERY_ROWS if gate else 0
    return tuple(
        (
            head * ATTENTION_HEAD_ROWS + offset,
            head * ATTENTION_HEAD_ROWS + offset + ATTENTION_QUERY_ROWS,
        )
        for head in range(ATTENTION_HEADS)
    )


def _block_matrix(prefix: str, leaf: str) -> Matrix | None:
    """Map a text/MTP decoder-block leaf name to its HF source matrix."""

    table = {
        "input_norm": Matrix(prefix + "input_layernorm.weight"),
        "post_attention_norm": Matrix(prefix + "post_attention_layernorm.weight"),
        "attention/query": Matrix(prefix + "self_attn.q_proj.weight", _attention_rows(False)),
        "attention/gate": Matrix(prefix + "self_attn.q_proj.weight", _attention_rows(True)),
        "attention/key": Matrix(prefix + "self_attn.k_proj.weight"),
        "attention/value": Matrix(prefix + "self_attn.v_proj.weight"),
        "attention/query_norm": Matrix(prefix + "self_attn.q_norm.weight"),
        "attention/key_norm": Matrix(prefix + "self_attn.k_norm.weight"),
        "attention/output": Matrix(prefix + "self_attn.o_proj.weight"),
        "gdn/query": Matrix(prefix + "linear_attn.in_proj_qkv.weight", ((0, GDN_KEY_ROWS),)),
        "gdn/key": Matrix(
            prefix + "linear_attn.in_proj_qkv.weight", ((GDN_KEY_ROWS, 2 * GDN_KEY_ROWS),)
        ),
        "gdn/value": Matrix(
            prefix + "linear_attn.in_proj_qkv.weight", ((2 * GDN_KEY_ROWS, GDN_QKV_ROWS),)
        ),
        "gdn/z": Matrix(prefix + "linear_attn.in_proj_z.weight"),
        "gdn/a_projection": Matrix(prefix + "linear_attn.in_proj_a.weight"),
        "gdn/b_projection": Matrix(prefix + "linear_attn.in_proj_b.weight"),
        "gdn/a_log": Matrix(prefix + "linear_attn.A_log"),
        "gdn/dt_bias": Matrix(prefix + "linear_attn.dt_bias"),
        "gdn/convolution": Matrix(prefix + "linear_attn.conv1d.weight"),
        "gdn/norm": Matrix(prefix + "linear_attn.norm.weight"),
        "gdn/output": Matrix(prefix + "linear_attn.out_proj.weight"),
        "mlp/gate": Matrix(prefix + "mlp.gate_proj.weight"),
        "mlp/up": Matrix(prefix + "mlp.up_proj.weight"),
        "mlp/down": Matrix(prefix + "mlp.down_proj.weight"),
    }
    return table.get(leaf)


_TEXT_LAYER = re.compile(r"^text/layers/(\d+)/(.+)$")
_MTP_LAYER = re.compile(r"^mtp/layers/(\d+)/(.+)$")
_ROOT_MATRICES = {
    "text/token_embedding": Matrix("model.language_model.embed_tokens.weight"),
    "text/output_head": Matrix("lm_head.weight"),
    "text/final_norm": Matrix("model.language_model.norm.weight"),
    "mtp/input_projection": Matrix("mtp.fc.weight"),
    "mtp/embedding_norm": Matrix("mtp.pre_fc_norm_embedding.weight"),
    "mtp/hidden_norm": Matrix("mtp.pre_fc_norm_hidden.weight"),
    "mtp/final_norm": Matrix("mtp.norm.weight"),
}


def logical_matrix(logical: str) -> Matrix:
    if logical in _ROOT_MATRICES:
        return _ROOT_MATRICES[logical]
    match = _TEXT_LAYER.match(logical)
    if match:
        prefix = f"model.language_model.layers.{int(match.group(1))}."
        leaf = match.group(2)
    else:
        match = _MTP_LAYER.match(logical)
        if not match:
            raise GraftError(f"no source route for logical parameter {logical}")
        prefix = f"mtp.layers.{int(match.group(1))}."
        leaf = match.group(2)
    matrix = _block_matrix(prefix, leaf)
    if matrix is None:
        raise GraftError(f"no source route for logical parameter {logical}")
    return matrix


def _weight_base(source: str) -> str:
    if not source.endswith(".weight"):
        raise GraftError(f"{source}: quantized routes need a .weight source")
    return source[: -len(".weight")]


def _select_rows(tensor: torch.Tensor, rows: tuple[tuple[int, int], ...] | None) -> torch.Tensor:
    if rows is None:
        return tensor
    pieces = [tensor[begin:end] for begin, end in rows]
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)


class ObjectBuilder:
    def __init__(
        self,
        template: Artifact,
        source: SourceCheckpoint,
        *,
        device: str,
    ) -> None:
        self.template = template
        self.source = source
        self.device = device
        directory = template.directory
        self.parts: dict[str, list[Part]] = {}
        for logical, binding in directory.bindings.items():
            if "object" in binding:
                self.parts.setdefault(binding["object"], []).append(
                    Part(logical, -1, -1)
                )
                continue
            for part in binding["parts"]:
                begin, end = part["range"]
                self.parts.setdefault(part["object"], []).append(
                    Part(logical, int(begin), int(end))
                )
        self.auxiliary: dict[str, tuple[str, str]] = {}
        for use in directory.uses:
            for role, reference in use.get("auxiliaries", {}).items():
                self.auxiliary[reference["object"]] = (use["parameter"], role)
        self._proposal_ids: torch.Tensor | None = None

    # -- routing -----------------------------------------------------------

    def is_copied(self, obj) -> bool:
        if obj.id.startswith("resource/"):
            return True
        names = [part.logical for part in self.parts.get(obj.id, ())]
        return bool(names) and all(name.startswith(COPY_PREFIXES) for name in names)

    def ordered_parts(self, obj: TensorObject) -> list[Part]:
        parts = self.parts.get(obj.id)
        if not parts:
            raise GraftError(f"{obj.id}: no binding references this object")
        if len(parts) == 1 and parts[0].begin < 0:
            return parts
        elements = 1
        for dim in obj.shape:
            elements *= dim
        ordered = sorted(parts, key=lambda part: part.begin)
        cursor = 0
        for part in ordered:
            if part.begin != cursor or part.end <= part.begin:
                raise GraftError(f"{obj.id}: bindings do not tile the object")
            cursor = part.end
        if cursor != elements:
            raise GraftError(f"{obj.id}: bindings cover {cursor} of {elements} elements")
        return ordered

    # -- value materialization --------------------------------------------

    def _float_rows(self, logical: str) -> torch.Tensor:
        matrix = logical_matrix(logical)
        dtype = self.source.dtype(matrix.source)
        if dtype != "BF16":
            raise GraftError(f"{matrix.source}: expected BF16 source, got {dtype or 'missing'}")
        return _select_rows(self.source.get(matrix.source), matrix.rows)

    def _values(self, obj: TensorObject, parts: list[Part]) -> torch.Tensor:
        if obj.id.startswith("weight/") and parts[0].logical == "proposal/head":
            return self._proposal_head_values()
        tensors = []
        for part in parts:
            tensor = self._float_rows(part.logical)
            if part.logical.endswith("gdn/convolution"):
                tensor = tensor.reshape(tensor.shape[0], tensor.shape[-1]).transpose(0, 1)
            tensors.append(tensor.reshape(-1) if len(obj.shape) == 1 else tensor)
        value = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
        return value.reshape(obj.shape).contiguous()

    def _proposal_ids_tensor(self) -> torch.Tensor:
        if self._proposal_ids is None:
            for obj in self.template.objects:
                names = [part.logical for part in self.parts.get(obj.id, ())]
                if names == ["proposal/token_ids"]:
                    raw = self.template.read_object(obj.id)
                    self._proposal_ids = decode_direct(raw, "int32", obj.shape).long()
                    break
            else:
                raise GraftError("template has no proposal/token_ids object")
        return self._proposal_ids

    def _proposal_head_values(self) -> torch.Tensor:
        lm_head = "lm_head.weight"
        if self.source.dtype(lm_head) != "BF16":
            raise GraftError("proposal/head needs a BF16 lm_head.weight source")
        return self.source.get(lm_head).index_select(0, self._proposal_ids_tensor())

    # -- encoders ------------------------------------------------------------

    def _fp8_words(self, logical: str) -> tuple[torch.Tensor, torch.Tensor]:
        matrix = logical_matrix(logical)
        dtype = self.source.dtype(matrix.source)
        if dtype == "F8_E4M3":
            codes = self.source.get(matrix.source).view(torch.uint8)
            scales = self.source.get(_weight_base(matrix.source) + ".weight_scale")
            if scales.dtype != torch.bfloat16 or scales.numel() != codes.shape[0]:
                raise GraftError(f"{matrix.source}: FP8 channel scale signature mismatch")
            scales = scales.reshape(-1)
            return _select_rows(codes, matrix.rows), _select_rows(scales, matrix.rows)
        if dtype == "BF16":
            quantized = quantize_bf16_rows(_select_rows(self.source.get(matrix.source), matrix.rows))
            return quantized.codes, quantized.scales
        raise GraftError(f"{matrix.source}: no FP8 route for source dtype {dtype or 'missing'}")

    def _iter_fp8_streamed(self, obj: TensorObject, logical: str) -> Iterator[bytes]:
        """Encode a large BF16 matrix (vocab rows) chunk by chunk."""

        matrix = logical_matrix(logical)
        n, k = obj.shape
        scale_words = []
        for begin in range(0, n, FP8_ROW_CHUNK):
            end = min(begin + FP8_ROW_CHUNK, n)
            quantized = quantize_bf16_rows(self.source.rows(matrix.source, begin, end))
            scale_words.append(quantized.scales)
            yield quantized.codes.numpy().tobytes()
        from tools.artifact.layouts import row_scale_geometry

        geometry = row_scale_geometry("fp8_e4m3fn_row_bf16", (n, k))
        padding = geometry.scale_plane_offset - geometry.code_plane_bytes
        if padding:
            yield bytes(padding)
        scales = encode_direct(torch.cat(scale_words), "bf16")
        yield scales
        tail = geometry.payload_bytes - geometry.scale_plane_offset - len(scales)
        if tail:
            yield bytes(tail)

    def _encode_fp8(self, obj: TensorObject, parts: list[Part]) -> bytes | Iterator[bytes]:
        if len(parts) == 1 and parts[0].begin < 0:
            matrix = logical_matrix(parts[0].logical)
            if (
                self.source.dtype(matrix.source) == "BF16"
                and matrix.rows is None
                and obj.shape[0] >= 65536
            ):
                return self._iter_fp8_streamed(obj, parts[0].logical)
        codes, scales = [], []
        for part in parts:
            part_codes, part_scales = self._fp8_words(part.logical)
            codes.append(part_codes)
            scales.append(part_scales)
        code_matrix = codes[0].contiguous() if len(codes) == 1 else torch.cat(codes, dim=0)
        scale_vector = scales[0].contiguous() if len(scales) == 1 else torch.cat(scales)
        return encode_fp8_row_scaled(code_matrix, scale_vector, obj.shape)

    def _encode_nvfp4(self, obj: TensorObject, parts: list[Part]) -> bytes:
        packed, scales, divisors = [], [], set()
        for part in parts:
            matrix = logical_matrix(part.logical)
            base = _weight_base(matrix.source)
            if self.source.dtype(base + ".weight_packed") != "U8":
                raise GraftError(f"{base}: expected an NVFP4 source (weight_packed U8)")
            packed.append(_select_rows(self.source.get(base + ".weight_packed"), matrix.rows))
            scales.append(
                _select_rows(self.source.get(base + ".weight_scale").view(torch.uint8), matrix.rows)
            )
            divisor = self.source.get(base + ".weight_global_scale")
            divisors.add(divisor.float().reshape(()).view(torch.int32).item() & 0xFFFFFFFF)
        if len(divisors) != 1:
            raise GraftError(f"{obj.id}: fused NVFP4 parts carry different global scales")
        word = struct.pack("<I", divisors.pop())
        code_matrix = packed[0].contiguous() if len(packed) == 1 else torch.cat(packed, dim=0)
        scale_matrix = scales[0].contiguous() if len(scales) == 1 else torch.cat(scales, dim=0)
        return encode_nvfp4(code_matrix, scale_matrix, word, obj.shape)

    def _encode_auxiliary(self, obj: TensorObject) -> bytes:
        try:
            parameter, role = self.auxiliary[obj.id]
        except KeyError:
            raise GraftError(f"{obj.id}: unreferenced auxiliary object") from None
        if role != "activation_input_divisor":
            raise GraftError(f"{obj.id}: unsupported auxiliary role {role}")
        base = _weight_base(logical_matrix(parameter).source)
        scale = self.source.get(base + ".input_global_scale")
        if scale.dtype != torch.float32 or scale.numel() != 1:
            raise GraftError(f"{base}.input_global_scale must be FP32[1]")
        return encode_direct(scale.reshape(obj.shape), "fp32")

    def payload(self, obj) -> bytes | Iterator[bytes]:
        if self.is_copied(obj):
            return self.template.read_object(obj.id)
        if obj.id.startswith("auxiliary/"):
            return self._encode_auxiliary(obj)
        parts = self.ordered_parts(obj)
        if obj.format == "fp8_e4m3fn_row_bf16":
            return self._encode_fp8(obj, parts)
        if obj.format == "nvfp4":
            return self._encode_nvfp4(obj, parts)
        if obj.format in QUANT_FORMATS:
            values = self._values(obj, parts)
            return quantize_and_encode(values, obj.format, device=self.device)
        if obj.format in ("bf16", "fp32", "int32"):
            values = self._values(obj, parts)
            target = {"bf16": torch.bfloat16, "fp32": torch.float32, "int32": torch.int32}
            return encode_direct(values.to(target[obj.format]), obj.format)
        raise GraftError(f"{obj.id}: unsupported object format {obj.format}")


def _describe(builder: ObjectBuilder, obj) -> str:
    names = [part.logical for part in builder.parts.get(obj.id, ())]
    if obj.id in builder.auxiliary:
        names = [f"{builder.auxiliary[obj.id][0]}#{builder.auxiliary[obj.id][1]}"]
    return ",".join(names[:2]) + ("..." if len(names) > 2 else "")


def _payload_bytes(payload: bytes | Iterator[bytes]) -> bytes:
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return bytes(payload)
    return b"".join(payload)


def _selected(builder: ObjectBuilder, patterns: Sequence[str]) -> Callable[[object], bool]:
    def keep(obj) -> bool:
        if not patterns:
            return True
        names = [part.logical for part in builder.parts.get(obj.id, ())]
        if obj.id in builder.auxiliary:
            names.append(builder.auxiliary[obj.id][0])
        return any(
            fnmatch.fnmatch(name, pattern) for name in names + [obj.id] for pattern in patterns
        )

    return keep


def verify(builder: ObjectBuilder, patterns: Sequence[str]) -> int:
    keep = _selected(builder, patterns)
    mismatches = 0
    checked = 0
    for obj in builder.template.objects:
        if not keep(obj) or builder.is_copied(obj):
            continue
        try:
            generated = _payload_bytes(builder.payload(obj))
        except GraftError as error:
            print(f"SKIP {obj.id} {_describe(builder, obj)}: {error}", flush=True)
            continue
        reference = builder.template.read_object(obj.id)
        checked += 1
        same = generated == reference
        mismatches += 0 if same else 1
        print(
            f"{'OK  ' if same else 'DIFF'} {obj.id} {obj.format} {_describe(builder, obj)}",
            flush=True,
        )
    print(f"verified {checked} objects, {mismatches} mismatches", flush=True)
    return 1 if mismatches else 0


def convert(builder: ObjectBuilder, out_path: Path, provenance: dict) -> None:
    template = builder.template
    directory = template.directory
    specs = []
    for obj in template.objects:
        if isinstance(obj, TensorObject):
            specs.append(TensorSpec(obj.id, tuple(obj.shape), obj.format, obj.layout))
        else:
            specs.append(ResourceSpec(obj.id, obj.bytes, obj.encoding))
    started = time.perf_counter()
    writer = ArtifactWriter(
        out_path,
        specs,
        components=directory.components,
        bindings=directory.bindings,
        uses=directory.uses,
        metadata=directory.metadata,
        provenance=provenance,
    )
    counts = {"copied": 0, "generated": 0}
    try:
        total = len(template.objects)
        for index, obj in enumerate(template.objects, start=1):
            copied = builder.is_copied(obj)
            writer.write_object(obj.id, builder.payload(obj))
            counts["copied" if copied else "generated"] += 1
            print(
                f"[{index}/{total}] {'copy' if copied else 'gen '} {obj.id} "
                f"{getattr(obj, 'format', 'resource')} {_describe(builder, obj)}",
                flush=True,
            )
        writer.finish()
    except BaseException:
        writer.abort()
        raise
    finally:
        builder.source.close()
    elapsed = time.perf_counter() - started
    report = {
        "artifact": local_name(out_path),
        "bytes": out_path.stat().st_size,
        "template": local_name(template.path),
        "source": local_name(builder.source.model_dir),
        "objects": counts,
        "copy_prefixes": list(COPY_PREFIXES),
        "fp8_bf16_encoder": ENCODER_PROFILE,
        "provenance": provenance,
        "elapsed_seconds": round(elapsed, 1),
    }
    report_path = Path(str(out_path) + ".graft.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"complete: {report['bytes']} bytes in {elapsed:.1f}s; report={report_path}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--source-label",
        type=input_label,
        help="repository id or name of --source recorded in the artifact (required with --out)",
    )
    parser.add_argument("--verify-against-template", action="store_true")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="fnmatch pattern on logical names or object ids (verify mode)",
    )
    arguments = parser.parse_args(argv)
    with Artifact(arguments.template) as template:
        builder = ObjectBuilder(
            template, SourceCheckpoint(arguments.source), device=arguments.device
        )
        if arguments.verify_against_template:
            return verify(builder, arguments.only)
        if arguments.out is None:
            parser.error("--out is required unless --verify-against-template is set")
        if arguments.source_label is None:
            parser.error("--source-label is required with --out")
        # The template's sources no longer describe the language-model weights: keep them
        # under the graft record and name the new source. No record carries a local path.
        provenance, removed = strip_local_paths(template.directory.provenance)
        graft = {
            "tool": "tools.convert.qwen3_8_27b.graft_single_source",
            "template": local_name(arguments.template),
            "template_artifact_id": template.artifact_id.hex(),
            "copied": list(COPY_PREFIXES),
        }
        if "sources" in provenance:
            graft["template_sources"] = provenance["sources"]
        if removed:
            graft["removed_template_paths"] = removed
        provenance["sources"] = {"single": {"label": arguments.source_label}}
        provenance["graft"] = graft
        convert(builder, arguments.out, provenance)
    return 0


if __name__ == "__main__":
    sys.exit(main())
