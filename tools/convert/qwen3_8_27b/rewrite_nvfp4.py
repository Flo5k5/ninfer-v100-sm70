"""Move FP8 roles of a Qwen3.8-27B NVFP4 artifact to NVFP4, one stage at a time.

The single-source Qwen3.8-27B NVFP4 artifacts keep a mixed layout: MLP 0-55 in
NVFP4, and attention, GDN, MLP 56-63 and the output head in row-scaled FP8. This
tool starts from such an artifact (``--base``) and rewrites the roles of one
stage to NVFP4 with the round-to-nearest quantizer of
:mod:`tools.convert.common.nvfp4_quantize`. Every other object is copied byte for
byte, bindings are unchanged, and the Uses of the rewritten parameters gain the
``AllowA4`` policy and an ``activation_input_divisor`` auxiliary, like the MLP
layers that were NVFP4 already.

Stage ``a``: MLP 56-63 (gate/up and down) and the output head. The token
embedding, attention, GDN, MTP and the proposal head are untouched.

Values come from safetensors checkpoints. A matrix published as FP8 row-scaled
codes (``F8_E4M3`` weight + ``BF16 [N,1]`` weight_scale, as the MLP 56-63 of a
compressed-tensors NVFP4 checkpoint) is dequantized to FP32 before NVFP4
quantization, which is a double quantization and is recorded as such in the
provenance; a BF16 matrix is quantized directly. ``--mlp-source`` lets the MLP
come from a different (for example BF16) checkpoint than ``--source``.

The runtime selects the new profile from ``provenance.recipe``
(``qwen3_8_27b_nvfp4-full-a`` -> weights id ``nvfp4-full-a``).

Example (single quantization from the BF16 release of the same weights)::

    python3 -m tools.convert.qwen3_8_27b.rewrite_nvfp4 \\
      --base <dir>/qwen3_8_27b_nvfp4.ninfer \\
      --source <checkpoint>/Qwen3.8-27B-BF16 \\
      --out <dir>/qwen3_8_27b_nvfp4_stageA.ninfer --verify

Passing the compressed-tensors NVFP4 checkpoint as ``--mlp-source`` takes the
MLP from its FP8 codes instead (double quantization).

``--check-idempotence`` re-quantizes NVFP4 source matrices with their own global
divisor and requires bit-identical codes and scales. ``--check-reproduction``
quantizes BF16 matrices from scratch and requires the divisor, code and scale
words of a published NVFP4 checkpoint of the same weights, bit for bit.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import time
from typing import Iterator, Sequence

import torch

from tools.artifact.codecs.nvfp4 import decode_nvfp4_words
from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorObject, TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.convert.common import nvfp4_quantize
from tools.convert.official_recipes import QWEN3_8_27B_NVFP4_FULL_A
from tools.convert.qwen3_8_27b.graft_single_source import SourceCheckpoint


NVFP4_FORMAT = "nvfp4"
NVFP4_LAYOUT = "block_scale_k16_m128x4_v1"
FP8_FORMAT = "fp8_e4m3fn_row_bf16"
AUX_FORMAT = "fp32"
AUX_LAYOUT = "contiguous_le_v1"
DIVISOR_ROLE = "activation_input_divisor"
OUTPUT_HEAD = "text/output_head"
OUTPUT_HEAD_SOURCE = "lm_head.weight"
UNIT_INPUT_DIVISOR = 1.0


class RewriteError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    recipe: str
    mlp_layers: tuple[int, ...]
    output_head: bool


STAGES = {
    "a": Stage("a", QWEN3_8_27B_NVFP4_FULL_A, tuple(range(56, 64)), True),
}


@dataclass(slots=True)
class Target:
    """One physical object rewritten to NVFP4 and the logical parts it carries."""

    object_id: str
    shape: tuple[int, int]
    parts: tuple[tuple[str, str], ...]  # (logical parameter, checkpoint matrix)
    source: str  # "mlp" or "head"
    previous_format: str = ""
    previous_bytes: int = 0
    report: dict = field(default_factory=dict)


def _mlp_matrix(layer: int, leaf: str) -> str:
    projection = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}[leaf]
    return f"model.language_model.layers.{layer}.mlp.{projection}.weight"


def _whole_object(directory, logical: str) -> tuple[str, int, int]:
    binding = directory.bindings.get(logical)
    if binding is None:
        raise RewriteError(f"base artifact has no binding {logical}")
    if "object" in binding:
        return binding["object"], -1, -1
    if len(binding["parts"]) != 1:
        raise RewriteError(f"{logical}: binding spans several objects")
    part = binding["parts"][0]
    return part["object"], int(part["range"][0]), int(part["range"][1])


def _float_matrix(source: SourceCheckpoint, name: str) -> tuple[torch.Tensor, str]:
    """Return FP32 values of a checkpoint matrix and a label of how they were obtained."""

    dtype = source.dtype(name)
    if dtype == "BF16":
        return source.get(name).float(), "bf16"
    if dtype == "F8_E4M3":
        base = name[: -len(".weight")]
        codes = source.get(name)
        scales = source.get(base + ".weight_scale")
        if scales.dtype != torch.bfloat16 or scales.numel() != codes.shape[0]:
            raise RewriteError(f"{name}: FP8 row scale signature mismatch")
        return codes.float() * scales.float().reshape(-1, 1), "fp8_row_dequantized"
    raise RewriteError(f"{name}: no NVFP4 route for source dtype {dtype or 'missing'}")


def stage_objects(
    base: Artifact, stage: Stage
) -> list[tuple[str, tuple[tuple[str, str], ...], str]]:
    """Objects a stage rewrites: (object id, (logical parameter, checkpoint matrix)..., source)."""

    directory = base.directory
    result = []
    for layer in stage.mlp_layers:
        prefix = f"text/layers/{layer}/mlp/"
        gate_object, gate_begin, gate_end = _whole_object(directory, prefix + "gate")
        up_object, up_begin, up_end = _whole_object(directory, prefix + "up")
        obj = base.object(gate_object)
        if up_object != gate_object or gate_begin != 0 or up_begin != gate_end:
            raise RewriteError(f"{prefix}gate_up: gate and up are not one fused object")
        if up_end != obj.shape[0] * obj.shape[1]:
            raise RewriteError(f"{prefix}gate_up: gate and up do not cover the object")
        parts = (
            (prefix + "gate", _mlp_matrix(layer, "gate")),
            (prefix + "up", _mlp_matrix(layer, "up")),
        )
        result.append((gate_object, parts, "mlp"))
        down_object, _, _ = _whole_object(directory, prefix + "down")
        result.append((down_object, ((prefix + "down", _mlp_matrix(layer, "down")),), "mlp"))
    if stage.output_head:
        head_object, _, _ = _whole_object(directory, OUTPUT_HEAD)
        result.append((head_object, ((OUTPUT_HEAD, OUTPUT_HEAD_SOURCE),), "head"))
    return result


class Rewriter:
    def __init__(
        self,
        base: Artifact,
        stage: Stage,
        mlp_source: SourceCheckpoint,
        head_source: SourceCheckpoint,
    ) -> None:
        self.base = base
        self.stage = stage
        self.mlp_source = mlp_source
        self.head_source = head_source
        self.targets: dict[str, Target] = {}
        for object_id, parts, source in stage_objects(base, stage):
            self._add_target(object_id, parts, source)
        self._plan_uses()

    def _add_target(self, object_id: str, parts, source: str) -> None:
        obj = self.base.object(object_id)
        if not isinstance(obj, TensorObject) or len(obj.shape) != 2:
            raise RewriteError(f"{object_id}: expected a rank-two tensor object")
        if obj.format != FP8_FORMAT:
            raise RewriteError(f"{object_id}: expected {FP8_FORMAT}, base has {obj.format}")
        self.targets[object_id] = Target(
            object_id,
            (int(obj.shape[0]), int(obj.shape[1])),
            tuple(parts),
            source,
            previous_format=obj.format,
            previous_bytes=obj.bytes,
        )

    # -- directory -------------------------------------------------------------

    def _neighbor_divisor(self, parameter: str) -> tuple[float, str]:
        """Input divisor of the same projection in the nearest calibrated NVFP4 layer."""

        match = re.match(r"^text/layers/(\d+)/mlp/(gate|up|down)$", parameter)
        if not match:
            return UNIT_INPUT_DIVISOR, "unit placeholder"
        layer, leaf = int(match.group(1)), match.group(2)
        for candidate in range(layer - 1, -1, -1):
            name = f"text/layers/{candidate}/mlp/{leaf}"
            for use in self.base.directory.uses:
                aux = use.get("auxiliaries", {}).get(DIVISOR_ROLE)
                if use["parameter"] == name and aux is not None:
                    raw = self.base.read_object(aux["object"])
                    (value,) = struct.unpack("<f", raw)
                    return value, f"copied from {name}"
        raise RewriteError(f"{parameter}: no calibrated NVFP4 neighbor for the input divisor")

    def _plan_uses(self) -> None:
        directory = self.base.directory
        aux_numbers = [
            int(obj.id.split("/")[1])
            for obj in self.base.objects
            if obj.id.startswith("auxiliary/") and obj.id.split("/")[1].isdigit()
        ]
        next_number = max(aux_numbers, default=-1) + 1
        rewritten = {
            logical for target in self.targets.values() for logical, _ in target.parts
        }
        self.aux_values: dict[str, float] = {}
        self.aux_labels: dict[str, str] = {}
        shared: dict[str, str] = {}
        self.uses = []
        for use in directory.uses:
            use = copy.deepcopy(use)
            parameter = use["parameter"]
            if parameter in rewritten:
                if DIVISOR_ROLE in use.get("auxiliaries", {}):
                    raise RewriteError(f"{parameter}: base Use already has an input divisor")
                # One divisor per MLP leaf Use (as layers 0-55). The output head is read by
                # the text, MTP and DFlash2 Uses, which share one divisor object.
                aux_id = shared.get(parameter)
                if aux_id is None:
                    aux_id = f"auxiliary/{next_number:06d}"
                    next_number += 1
                    value, label = self._neighbor_divisor(parameter)
                    self.aux_values[aux_id] = value
                    self.aux_labels[aux_id] = f"{parameter}: {label}"
                    if parameter == OUTPUT_HEAD:
                        shared[parameter] = aux_id
                use["activation_policy"] = "AllowA4"
                use.setdefault("auxiliaries", {})[DIVISOR_ROLE] = {"object": aux_id}
            self.uses.append(use)
        covered = {
            use["parameter"] for use in self.uses if DIVISOR_ROLE in use.get("auxiliaries", {})
        }
        missing = rewritten - covered
        if missing:
            raise RewriteError(f"rewritten parameters without a Use: {sorted(missing)}")

    def specs(self) -> list:
        specs = []
        for obj in self.base.objects:
            if isinstance(obj, TensorObject):
                if obj.id in self.targets:
                    specs.append(TensorSpec(obj.id, tuple(obj.shape), NVFP4_FORMAT, NVFP4_LAYOUT))
                else:
                    specs.append(TensorSpec(obj.id, tuple(obj.shape), obj.format, obj.layout))
            else:
                specs.append(ResourceSpec(obj.id, obj.bytes, obj.encoding))
        for aux_id in self.aux_values:
            specs.append(TensorSpec(aux_id, (), AUX_FORMAT, AUX_LAYOUT))
        return specs

    # -- payloads --------------------------------------------------------------

    def _quantize(self, target: Target) -> bytes:
        started = time.perf_counter()
        checkpoint = self.head_source if target.source == "head" else self.mlp_source
        values, origins = [], []
        for _, matrix in target.parts:
            tensor, origin = _float_matrix(checkpoint, matrix)
            values.append(tensor)
            origins.append(origin)
        rows = sum(tensor.shape[0] for tensor in values)
        if (rows, values[0].shape[1]) != target.shape:
            raise RewriteError(f"{target.object_id}: source geometry differs from the object")
        amax = nvfp4_quantize.tensor_amax(values)
        words = nvfp4_quantize.quantize_fused(values, divisor=nvfp4_quantize.global_divisor(amax))
        payload = words.encode(target.shape)
        error = _relative_rms_error(values, words)
        target.report = {
            "object": target.object_id,
            "parameters": [logical for logical, _ in target.parts],
            "checkpoint": str(checkpoint.model_dir),
            "matrices": [matrix for _, matrix in target.parts],
            "source_values": sorted(set(origins)),
            "shape": list(target.shape),
            "amax": amax,
            "weight_divisor": float(words.divisor),
            "relative_rms_error": error,
            "bytes_before": target.previous_bytes,
            "format_before": target.previous_format,
            "bytes_after": len(payload),
            "seconds": round(time.perf_counter() - started, 1),
        }
        return payload

    def payload(self, obj) -> bytes | Iterator[bytes]:
        if obj.id in self.targets:
            return self._quantize(self.targets[obj.id])
        return self.base.iter_object(obj.id)


def _relative_rms_error(values: Sequence[torch.Tensor], words: nvfp4_quantize.Nvfp4Words) -> float:
    """RMS(dequantized - source) / RMS(source) over the fused matrix, in FP64."""

    error_sum = 0.0
    value_sum = 0.0
    row = 0
    chunk = 4096
    for tensor in values:
        for begin in range(0, tensor.shape[0], chunk):
            end = min(begin + chunk, tensor.shape[0])
            source = tensor[begin:end].double()
            decoded = nvfp4_quantize.dequantize(
                words.packed[row + begin : row + end],
                words.scales[row + begin : row + end],
                words.divisor,
            ).double()
            error_sum += float(((decoded - source) ** 2).sum())
            value_sum += float((source**2).sum())
        row += tensor.shape[0]
    return (error_sum / value_sum) ** 0.5 if value_sum else 0.0


def _source_record(checkpoint: SourceCheckpoint, label: str | None) -> dict:
    return {
        "label": label or str(checkpoint.model_dir),
        "path": str(checkpoint.model_dir.resolve()),
    }


def rewrite(rewriter: Rewriter, out_path: Path, labels: dict[str, str | None]) -> dict:
    base = rewriter.base
    directory = base.directory
    stage = rewriter.stage
    mlp_values = sorted(
        {
            _float_matrix_dtype(rewriter.mlp_source, matrix)
            for target in rewriter.targets.values()
            if target.source == "mlp"
            for _, matrix in target.parts
        }
    )
    provenance = copy.deepcopy(directory.provenance)
    provenance["recipe"] = stage.recipe
    provenance["rewrite"] = {
        "tool": "tools.convert.qwen3_8_27b.rewrite_nvfp4",
        "stage": stage.name,
        "base_artifact_id": base.artifact_id.hex(),
        "base_recipe": directory.provenance.get("recipe"),
        "encoder": nvfp4_quantize.ENCODER_PROFILE,
        "sources": {
            "mlp": _source_record(rewriter.mlp_source, labels.get("mlp")),
            "output_head": _source_record(rewriter.head_source, labels.get("head")),
        },
        "roles": {
            "mlp_layers": list(stage.mlp_layers),
            "mlp_source_values": mlp_values,
            "mlp_double_quantization": "fp8_row_dequantized" in mlp_values,
            "output_head": stage.output_head,
        },
        "activation_input_divisors": {
            "note": (
                "not calibrated; sm_70 runs NVFP4 with A16Only activations and never scales by "
                "these values"
            ),
            "values": {
                aux_id: {"value": value, "origin": rewriter.aux_labels[aux_id]}
                for aux_id, value in rewriter.aux_values.items()
            },
        },
        "unchanged": "every other object is copied byte for byte from the base artifact",
    }
    specs = rewriter.specs()
    started = time.perf_counter()
    writer = ArtifactWriter(
        out_path,
        specs,
        components=directory.components,
        bindings=directory.bindings,
        uses=rewriter.uses,
        metadata=directory.metadata,
        provenance=provenance,
    )
    try:
        total = len(specs)
        index = 0
        for obj in base.objects:
            index += 1
            generated = obj.id in rewriter.targets
            writer.write_object(obj.id, rewriter.payload(obj))
            if generated:
                report = rewriter.targets[obj.id].report
                print(
                    f"[{index}/{total}] nvfp4 {obj.id} {','.join(report['parameters'])} "
                    f"amax={report['amax']:.6g} divisor={report['weight_divisor']:.6g} "
                    f"rel_rms={report['relative_rms_error']:.5f} "
                    f"bytes {report['bytes_before']}->{report['bytes_after']} "
                    f"({report['seconds']}s)",
                    flush=True,
                )
        for aux_id, value in rewriter.aux_values.items():
            index += 1
            writer.write_object(aux_id, struct.pack("<f", value))
            print(
                f"[{index}/{total}] aux   {aux_id} {rewriter.aux_labels[aux_id]} = {value}",
                flush=True,
            )
        writer.finish()
    except BaseException:
        writer.abort()
        raise
    finally:
        rewriter.mlp_source.close()
        rewriter.head_source.close()
    elapsed = time.perf_counter() - started
    report = {
        "artifact": str(out_path),
        "bytes": out_path.stat().st_size,
        "base": str(base.path),
        "base_bytes": base.file_bytes,
        "stage": stage.name,
        "recipe": stage.recipe,
        "objects": [target.report for target in rewriter.targets.values()],
        "auxiliaries": provenance["rewrite"]["activation_input_divisors"]["values"],
        "provenance": provenance,
        "elapsed_seconds": round(elapsed, 1),
    }
    report_path = Path(str(out_path) + ".rewrite.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"complete: {report['bytes']} bytes in {elapsed:.1f}s; report={report_path}", flush=True)
    return report


def _float_matrix_dtype(checkpoint: SourceCheckpoint, name: str) -> str:
    dtype = checkpoint.dtype(name)
    return {"BF16": "bf16", "F8_E4M3": "fp8_row_dequantized"}.get(dtype, dtype or "missing")


# -- verification ------------------------------------------------------------------


def _object_digest(artifact: Artifact, object_id: str) -> str:
    digest = hashlib.sha256()
    for chunk in artifact.iter_object(object_id):
        digest.update(chunk)
    return digest.hexdigest()


def _use_key(use: dict) -> tuple[str, str]:
    return use["parameter"], use["input"]


def verify_output(base_path: Path, out_path: Path, stage: Stage) -> int:
    """Check a rewritten artifact against its base without a GPU.

    Exactly the stage objects change format (to NVFP4, decodable); every other base
    object is present and byte-identical; bindings are unchanged; the Uses of
    untouched parameters are unchanged; each rewritten Use has ``AllowA4`` and a
    positive scalar input divisor that is a new auxiliary object; the Uses of the
    output head share one divisor object; and every new object is such a divisor.
    """

    failures = 0

    def fail(message: str) -> None:
        nonlocal failures
        failures += 1
        print(f"FAIL {message}", flush=True)

    with Artifact(base_path) as base, Artifact(out_path) as out:
        targets = {object_id: parts for object_id, parts, _ in stage_objects(base, stage)}
        rewritten = {logical for parts in targets.values() for logical, _ in parts}
        base_ids = {obj.id for obj in base.objects}
        out_ids = {obj.id for obj in out.objects}
        new_ids = out_ids - base_ids

        if out.directory.provenance.get("recipe") != stage.recipe:
            fail(f"recipe {out.directory.provenance.get('recipe')!r}")
        if out.directory.bindings != base.directory.bindings:
            fail("bindings differ from base")
        if not base_ids <= out_ids:
            fail(f"objects missing from the output: {sorted(base_ids - out_ids)[:8]}")

        changed = set()
        for obj in out.objects:
            if obj.id in new_ids:
                continue
            before = base.object(obj.id)
            if isinstance(obj, TensorObject) and obj.format != before.format:
                changed.add(obj.id)
                if obj.id not in targets or obj.format != NVFP4_FORMAT:
                    fail(f"{obj.id} changed {before.format}->{obj.format} outside the stage")
                    continue
                try:
                    _, _, divisor = decode_nvfp4_words(out.read_object(obj.id), obj.shape)
                except ValueError as error:
                    fail(f"{obj.id} does not decode as NVFP4: {error}")
                    continue
                print(f"OK   {obj.id} {before.format}->{obj.format} divisor={float(divisor):.6g}")
                continue
            if _object_digest(out, obj.id) != _object_digest(base, obj.id):
                fail(f"{obj.id} differs from base")
        if changed != set(targets):
            fail(f"stage objects left unconverted: {sorted(set(targets) - changed)}")

        base_uses = {_use_key(use): use for use in base.directory.uses}
        out_uses = {_use_key(use): use for use in out.directory.uses}
        if set(base_uses) != set(out_uses):
            fail("the set of Uses differs from base")
        referenced: set[str] = set()
        head_divisors: set[str] = set()
        for key, use in out_uses.items():
            if key not in base_uses:
                continue
            parameter = key[0]
            if parameter not in rewritten:
                if use != base_uses[key]:
                    fail(f"Use {key} of an untouched parameter changed")
                continue
            aux = use.get("auxiliaries", {}).get(DIVISOR_ROLE, {}).get("object")
            if use.get("activation_policy") != "AllowA4" or aux is None:
                fail(f"Use {key} lacks AllowA4 or an input divisor")
                continue
            if aux not in new_ids:
                fail(f"Use {key} divisor {aux} is not a new auxiliary object")
                continue
            referenced.add(aux)
            if parameter == OUTPUT_HEAD:
                head_divisors.add(aux)
        if stage.output_head and len(head_divisors) != 1:
            fail(f"the output head Uses name {len(head_divisors)} divisor objects, expected one")

        for object_id in sorted(new_ids):
            obj = out.object(object_id)
            raw = out.read_object(object_id)
            if (
                not object_id.startswith("auxiliary/")
                or not isinstance(obj, TensorObject)
                or obj.format != AUX_FORMAT
                or tuple(obj.shape) != ()
                or len(raw) != 4
            ):
                fail(f"new object {object_id} is not a scalar FP32 auxiliary")
                continue
            (value,) = struct.unpack("<f", raw)
            if not value > 0.0 or value == float("inf"):
                fail(f"new {object_id} = {value} is not a positive finite divisor")
            elif object_id not in referenced:
                fail(f"new {object_id} is not referenced by a rewritten Use")
            else:
                print(f"OK   new {object_id} = {value}", flush=True)
    print(f"verify: {failures} failures", flush=True)
    return 1 if failures else 0


def check_idempotence(source: SourceCheckpoint, layers: Sequence[int]) -> int:
    """Re-quantize NVFP4 source matrices with their own divisor; require identical words."""

    failures = 0
    for layer in layers:
        groups = (("gate_proj", "up_proj"), ("down_proj",))
        for group in groups:
            packed, scales, divisors = [], [], set()
            for projection in group:
                base = f"model.language_model.layers.{layer}.mlp.{projection}"
                if source.dtype(base + ".weight_packed") != "U8":
                    raise RewriteError(f"{base}: not an NVFP4 source matrix")
                packed.append(source.get(base + ".weight_packed"))
                scales.append(source.get(base + ".weight_scale").view(torch.uint8))
                divisors.add(float(source.get(base + ".weight_global_scale").float().reshape(())))
            if len(divisors) != 1:
                raise RewriteError(f"layer {layer} {group}: fused parts have different divisors")
            divisor = torch.tensor(divisors.pop(), dtype=torch.float32)
            values = [
                nvfp4_quantize.dequantize(part_packed, part_scales, divisor)
                for part_packed, part_scales in zip(packed, scales)
            ]
            words = nvfp4_quantize.quantize_fused(values, divisor=divisor)
            reference_packed = torch.cat(packed)
            reference_scales = torch.cat(scales)
            code_diff = int((words.packed != reference_packed).sum())
            scale_diff = int((words.scales != reference_scales).sum())
            # Fresh divisor from the dequantized amax, for information only: the
            # source divisor was computed from the unquantized weights.
            fresh = float(nvfp4_quantize.global_divisor(nvfp4_quantize.tensor_amax(values)))
            ok = code_diff == 0 and scale_diff == 0
            failures += 0 if ok else 1
            print(
                f"{'OK  ' if ok else 'DIFF'} layer {layer} {'+'.join(group)}: "
                f"{reference_packed.numel()} code bytes ({code_diff} differ), "
                f"{reference_scales.numel()} scales ({scale_diff} differ); "
                f"divisor {float(divisor)} (fresh from dequantized amax {fresh:.6g})",
                flush=True,
            )
    source.close()
    print(f"idempotence: {failures} mismatching matrices", flush=True)
    return 1 if failures else 0


def _fp32_word(value: torch.Tensor) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def check_reproduction(
    source: SourceCheckpoint, reference: SourceCheckpoint, layers: Sequence[int]
) -> int:
    """Quantize BF16 source MLP matrices from scratch and compare every word with a
    published NVFP4 checkpoint of the same weights: the global divisor bit for bit,
    then the packed codes and the scale words."""

    failures = 0
    for layer in layers:
        for group in (("gate_proj", "up_proj"), ("down_proj",)):
            prefix = f"model.language_model.layers.{layer}.mlp."
            values = [_float_matrix(source, prefix + name + ".weight")[0] for name in group]
            packed = torch.cat([reference.get(prefix + name + ".weight_packed") for name in group])
            scales = torch.cat(
                [reference.get(prefix + name + ".weight_scale").view(torch.uint8) for name in group]
            )
            published = {
                _fp32_word(reference.get(prefix + name + ".weight_global_scale")) for name in group
            }
            if len(published) != 1:
                raise RewriteError(f"layer {layer} {group}: fused parts have different divisors")
            (published_word,) = published
            words = nvfp4_quantize.quantize_fused(values)
            divisor_word = _fp32_word(words.divisor)
            code_diff = int((words.packed != packed).sum())
            scale_diff = int((words.scales != scales).sum())
            ok = divisor_word == published_word and code_diff == 0 and scale_diff == 0
            failures += 0 if ok else 1
            print(
                f"{'OK  ' if ok else 'DIFF'} layer {layer} {'+'.join(group)}: divisor "
                f"{float(words.divisor)} (0x{divisor_word:08x}) vs published "
                f"0x{published_word:08x}; {code_diff}/{packed.numel()} code bytes and "
                f"{scale_diff}/{scales.numel()} scales differ",
                flush=True,
            )
    source.close()
    reference.close()
    print(f"reproduction: {failures} matrices with differing words", flush=True)
    return 1 if failures else 0


def _layers(text: str) -> list[int]:
    result = []
    for item in text.split(","):
        if "-" in item:
            begin, end = item.split("-")
            result.extend(range(int(begin), int(end) + 1))
        else:
            result.append(int(item))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", type=Path, help="mixed NVFP4/FP8 NInfer artifact")
    parser.add_argument("--source", required=True, type=Path, help="checkpoint for the output head")
    parser.add_argument(
        "--mlp-source",
        type=Path,
        default=None,
        help="checkpoint for the MLP matrices (default: --source); BF16 or FP8 row-scaled",
    )
    parser.add_argument("--source-label", default=None)
    parser.add_argument("--mlp-source-label", default=None)
    parser.add_argument("--stage", default="a", choices=sorted(STAGES))
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--verify", action="store_true", help="after writing, compare the output with --base"
    )
    parser.add_argument(
        "--check-idempotence",
        metavar="LAYERS",
        default=None,
        help="re-quantize the NVFP4 MLP of these source layers (e.g. 0,27,55) and exit",
    )
    parser.add_argument(
        "--check-reproduction",
        metavar="LAYERS",
        default=None,
        help=(
            "quantize these BF16 --source MLP layers from scratch and require the published "
            "NVFP4 words of --reference (divisor, codes, scales), then exit"
        ),
    )
    parser.add_argument("--reference", type=Path, help="published NVFP4 checkpoint")
    arguments = parser.parse_args(argv)
    if arguments.check_reproduction is not None:
        if arguments.reference is None:
            parser.error("--check-reproduction needs --reference")
        return check_reproduction(
            SourceCheckpoint(arguments.source),
            SourceCheckpoint(arguments.reference),
            _layers(arguments.check_reproduction),
        )
    if arguments.check_idempotence is not None:
        return check_idempotence(
            SourceCheckpoint(arguments.source), _layers(arguments.check_idempotence)
        )
    if arguments.base is None or arguments.out is None:
        parser.error("--base and --out are required unless --check-idempotence is set")
    stage = STAGES[arguments.stage]
    mlp_path = arguments.mlp_source or arguments.source
    with Artifact(arguments.base) as base:
        rewriter = Rewriter(
            base,
            stage,
            mlp_source=SourceCheckpoint(mlp_path),
            head_source=SourceCheckpoint(arguments.source),
        )
        rewrite(
            rewriter,
            arguments.out,
            {
                "head": arguments.source_label,
                "mlp": arguments.mlp_source_label
                or (arguments.source_label if arguments.mlp_source is None else None),
            },
        )
    if arguments.verify:
        return verify_output(arguments.base, arguments.out, stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
