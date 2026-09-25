"""Re-encode the NVFP4 MLP objects of a Qwen3.8-27B artifact with a calibrated checkpoint.

A single-source NVFP4 artifact (``--base``) carries the MLP of its NVFP4 layers as the words of its
quantized source, often round-to-nearest with max block scales. This tool rewrites those objects in
place from a calibrated NVFP4 checkpoint of the same weights (``--donor``), for example NVIDIA Model
Optimizer's Local-Hessian checkpoint. Every re-encoded object keeps its id, format, layout, shape,
bindings and Uses; every other object is copied byte for byte, and the recipe is unchanged, so the
runtime loads the result through the same profile, kernels and memory plan as the base.

Per projection, the new words come from one of two routes:

* donor words (default): the donor's packed E2M1 codes and E4M3 block scales, as they are;
* rounded (``--round down``): the donor's block and tensor scales with codes rounded to nearest
  (ties to even, saturating at 6) from the unquantized weights of the base model (``--weights``),
  for projections whose weights differ from the donor's, such as the residual writers of an
  abliterated fine-tune. A nonzero weight in a block whose donor scale is zero is refused.

Donor layouts:

* ModelOpt: ``weight`` (packed), ``weight_scale`` (E4M3), ``weight_scale_2``;
  value = e2m1(code) * e4m3(scale) * weight_scale_2;
* compressed-tensors: ``weight_packed``, ``weight_scale``, ``weight_global_scale``;
  value = e2m1(code) * e4m3(scale) / weight_global_scale.

The artifact stores one FP32 weight divisor per object (value = e2m1 * e4m3 / divisor). A
compressed-tensors global scale is that divisor. For ModelOpt, the divisor is ``fl32(1 /
weight_scale_2)``: when its FP32 reciprocal is ``weight_scale_2`` again, the FP32 decode routes,
which multiply the block scales by ``1.0f / divisor``, reproduce the donor's values bit for bit;
routes that fold the divisor into FP16 constants or divide agree to their own rounding. When it is
not, no FP32 word is nearer and the object is reported with ``divisor_exact: false``. Rounded codes
use the same decode step, e4m3 * (1 / divisor). Gate and up are one fused object with one divisor:
their donor tensor scales must be equal.

Before the output is created, every selected object is checked against the donor's safetensors
headers and tensor scales (tensors present, NVFP4 dtypes and geometry, finite positive divisors,
one tensor scale for gate and up) and against the ``--weights`` headers. While encoding, every
re-encoded object is compared with the base object's decoded values, and with ``--weights`` when
given; the tool refuses an object whose relative RMS error exceeds ``--max-error`` (a donor, or
weights, of other weights) and ``--weights`` rows that are not finite. ``--verify`` re-reads the
output: re-encoded objects must hold the encoded words and every other object and the directory
must match the base; a failed verification removes the output.

The artifact's provenance names the donor and the weights by their required labels (a repository
id, for example), never by path, and drops the path members of the base's provenance, listed under
``removed_base_paths``. The report ``<out>.reencode.json`` records, per object, the donor layout,
the divisor and whether it is exact, the errors, the counts of saturated and zeroed rounded values,
and the SHA-256 of the donor tensors and ``--weights`` matrices read and of the payload written.

Example (layers 0-55 of an abliterated build, gate/up from NVIDIA's Local-Hessian words, down
rounded from the abliterated BF16 weights under NVIDIA's scales)::

    python3 -m tools.convert.qwen3_8_27b.reencode_nvfp4 \\
      --base <dir>/qwen3_8_27b_orca_nvfp4.ninfer \\
      --donor <checkpoint>/nvidia-Qwen3.8-27B-NVFP4 --donor-label nvidia/Qwen3.8-27B-NVFP4 \\
      --weights <checkpoint>/Qwen3.8-27B-Uncensored-BF16 \\
      --weights-label orcarouter/Qwen3.8-27B-Uncensored --round down \\
      --out <dir>/qwen3_8_27b_orca_nvfp4_g2.ninfer --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time
from typing import Iterator, Sequence

import torch

from tools.artifact.codecs.fp8_row import dequantize_fp8_row_scaled
from tools.artifact.codecs.nvfp4 import decode_nvfp4_words, encode_nvfp4
from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorObject, TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.convert.provenance import input_label, local_name, strip_local_paths
from tools.convert.qwen3_8_27b.graft_single_source import SourceCheckpoint
from tools.convert.qwen3_8_27b.reencode_nvfp4_numeric import (
    RelativeError,
    dequantize_words,
    divisor_word,
    round_to_nearest,
    same_codes,
    tensor_sha256,
    update_digest,
)
from tools.convert.qwen3_8_27b.reencode_nvfp4_plan import (
    FP8_ROW_FORMAT,
    NVFP4_FORMAT,
    NVFP4_LAYOUT,
    ReencodeError,
    Target,
    check_base,
    nvfp4_layers,
    plan_targets,
)


TOOL = "tools.convert.qwen3_8_27b.reencode_nvfp4"
PROJECTIONS = ("gate", "up", "down")
DEFAULT_MAX_ERROR = 0.3
ROW_CHUNK = 4096


class Encoder:
    def __init__(self, base: Artifact, donor: SourceCheckpoint, weights: SourceCheckpoint | None,
                 max_error: float = DEFAULT_MAX_ERROR) -> None:
        self.base = base
        self.donor = donor
        self.weights = weights
        self.max_error = max_error

    def payload(self, target: Target) -> tuple[bytes, dict]:
        """Encoded object payload and its report entry."""

        started = time.perf_counter()
        packed, scales = [], []
        donor_sha256 = {}
        for item in target.donors:
            codes_name, scales_name, scale_name = item.tensors
            codes = self.donor.get(codes_name)
            block_scales = self.donor.get(scales_name).view(torch.uint8)
            packed.append(codes)
            scales.append(block_scales)
            for name, tensor in ((codes_name, codes), (scales_name, block_scales),
                                 (scale_name, self.donor.get(scale_name))):
                donor_sha256[name] = tensor_sha256(tensor)
        report = {
            "object": target.object_id,
            "parameters": list(target.parameters),
            "donor_matrices": [item.module for item in target.donors],
            "donor_layout": target.donors[0].layout,
            "codes": "rounded from --weights" if target.rounded else "donor",
            "shape": list(target.shape),
            "weight_divisor": float(target.divisor),
            "weight_divisor_word": f"0x{divisor_word(target.divisor):08x}",
            "divisor_exact": all(item.exact for item in target.donors),
            "donor_sha256": donor_sha256,
        }
        block_scales = torch.cat(scales)
        words = self._check(target, torch.cat(packed), block_scales, report)
        payload = encode_nvfp4(words, block_scales, target.divisor, target.shape)
        report["payload_sha256"] = hashlib.sha256(payload).hexdigest()
        report["seconds"] = round(time.perf_counter() - started, 1)
        return payload, report

    def _check(self, target: Target, packed: torch.Tensor, scales: torch.Tensor,
               report: dict) -> torch.Tensor:
        """Packed codes of the object, compared with the base object and with --weights."""

        if target.converts:
            base_codes, base_scales, base_divisor = None, None, None
            base_values = dequantize_fp8_row_scaled(
                self.base.read_object(target.object_id), target.shape)
        else:
            base_codes, base_scales, base_divisor = decode_nvfp4_words(
                self.base.read_object(target.object_id), target.shape)
        against_base, against_weights = RelativeError(), RelativeError()
        rounded, same, saturated, zeroed = [], 0, 0, 0
        weights_sha256: dict[str, str] = {}
        for begin, end, weights in self._row_chunks(target, weights_sha256):
            codes = packed[begin:end]
            if weights is not None:
                result = round_to_nearest(weights, scales[begin:end], target.divisor)
                if target.rounded and result.unscaled:
                    raise ReencodeError(f"{target.object_id}: {result.unscaled} nonzero --weights "
                                        f"values in rows [{begin}, {end}) have a zero donor block "
                                        "scale and would decode to zero")
                if target.rounded:
                    codes = result.packed
                    rounded.append(codes)
                    saturated += result.saturated
                    zeroed += result.zeroed
                else:
                    same += same_codes(codes, result.packed)
            values = dequantize_words(codes, scales[begin:end], target.divisor)
            if weights is not None:
                against_weights.add(values, weights)
            if target.converts:
                against_base.add(values, base_values[begin:end])
            else:
                against_base.add(values, dequantize_words(base_codes[begin:end],
                                                          base_scales[begin:end], base_divisor))
        report["relative_rms_error_vs_base"] = against_base.value()
        self._refuse(target, against_base.value(), "the base object's values")
        if self.weights is None:
            return packed
        report["relative_rms_error_vs_weights"] = against_weights.value()
        report["weights_sha256"] = weights_sha256
        if target.rounded:
            report["saturated"] = saturated
            report["zeroed"] = zeroed
        else:
            report["codes_equal_to_rounded_weights"] = same / (target.shape[0] * target.shape[1])
        self._refuse(target, against_weights.value(), "--weights")
        return torch.cat(rounded) if target.rounded else packed

    def _refuse(self, target: Target, error: float, reference: str) -> None:
        if not math.isfinite(error) or error > self.max_error:
            raise ReencodeError(f"{target.object_id}: relative RMS error {error:.3f} against "
                                f"{reference} exceeds --max-error {self.max_error}; does the "
                                "donor quantize these weights?")

    def _row_chunks(self, target: Target,
                    digests: dict[str, str]) -> Iterator[tuple[int, int, torch.Tensor | None]]:
        """Row ranges of the object with their finite --weights rows, when given."""

        if self.weights is None:
            for begin in range(0, target.shape[0], ROW_CHUNK):
                yield begin, min(begin + ROW_CHUNK, target.shape[0]), None
            return
        row = 0
        for item in target.donors:
            name = item.module + ".weight"
            digest = hashlib.sha256()
            for begin in range(0, item.rows, ROW_CHUNK):
                end = min(begin + ROW_CHUNK, item.rows)
                stored = self.weights.rows(name, begin, end)
                update_digest(digest, stored)
                values = stored.float()
                if not bool(torch.isfinite(values).all()):
                    raise ReencodeError(f"{name}: rows [{begin}, {end}) of --weights hold "
                                        "non-finite values")
                yield row + begin, row + end, values
            digests[name] = digest.hexdigest()
            row += item.rows


def _specs(base: Artifact, targets: Sequence["Target"], aux_values: dict | None = None) -> list:
    """Output specs: the base's, converted targets switched to NVFP4, plus divisor auxes."""
    converted = {t.object_id: t for t in targets if t.converts}
    specs = []
    for obj in base.objects:
        if isinstance(obj, TensorObject):
            if obj.id in converted:
                specs.append(TensorSpec(obj.id, tuple(converted[obj.id].shape),
                                        NVFP4_FORMAT, NVFP4_LAYOUT))
            else:
                specs.append(TensorSpec(obj.id, tuple(obj.shape), obj.format, obj.layout))
        else:
            specs.append(ResourceSpec(obj.id, obj.bytes, obj.encoding))
    for aux_id in aux_values or {}:
        specs.append(TensorSpec(aux_id, (), AUX_FORMAT, AUX_LAYOUT))
    return specs


def _neighbor_divisor(base: Artifact, parameter: str) -> tuple[float, str]:
    """Input divisor of the same MLP leaf in the nearest calibrated NVFP4 layer below."""
    import re as _re
    match = _re.match(r"^text/layers/(\d+)/mlp/(gate|up|down)$", parameter)
    if not match:
        return 1.0, "unit placeholder"
    layer, leaf = int(match.group(1)), match.group(2)
    for candidate in range(layer - 1, -1, -1):
        name = f"text/layers/{candidate}/mlp/{leaf}"
        for use in base.directory.uses:
            aux = use.get("auxiliaries", {}).get(DIVISOR_ROLE)
            if use["parameter"] == name and aux is not None:
                (value,) = struct.unpack("<f", base.read_object(aux["object"]))
                return value, f"copied from {name}"
    raise ReencodeError(f"{parameter}: no calibrated NVFP4 neighbor for the input divisor")


def _plan_conversion_uses(base: Artifact, targets: Sequence["Target"]):
    """Uses of the output: converted MLP leaves gain AllowA4 and an activation input divisor
    auxiliary copied from the nearest calibrated NVFP4 layer, as the full-a loader expects."""
    import copy as _copy
    aux_numbers = [int(obj.id.split("/")[1]) for obj in base.objects
                   if obj.id.startswith("auxiliary/") and obj.id.split("/")[1].isdigit()]
    next_number = max(aux_numbers, default=-1) + 1
    converted_params = {logical for target in targets if target.converts
                        for logical in target.parameters}
    aux_values: dict[str, float] = {}
    aux_labels: dict[str, str] = {}
    shared: dict[str, str] = {}
    uses = []
    for use in base.directory.uses:
        use = _copy.deepcopy(use)
        parameter = use["parameter"]
        if parameter in converted_params:
            aux_id = shared.get(parameter)
            if aux_id is None:
                aux_id = f"auxiliary/{next_number:06d}"
                next_number += 1
                value, label = _neighbor_divisor(base, parameter)
                aux_values[aux_id] = value
                aux_labels[aux_id] = f"{parameter}: {label}"
                shared[parameter] = aux_id
            use["activation_policy"] = "AllowA4"
            if DIVISOR_ROLE not in use.get("auxiliaries", {}):
                use.setdefault("auxiliaries", {})[DIVISOR_ROLE] = {"object": aux_id}
        uses.append(use)
    return aux_values, aux_labels, uses


def _record(arguments, base: Artifact, targets: Sequence[Target], removed: list[str]) -> dict:
    record = {
        "tool": TOOL,
        "base": local_name(arguments.base),
        "base_artifact_id": base.artifact_id.hex(),
        "donor": {"label": arguments.donor_label},
        "layers": sorted({target.layer for target in targets}),
        "codes": {projection: ("rounded from weights" if projection in arguments.round else "donor")
                  for projection in PROJECTIONS},
        "divisor": "donor global scale, or the FP32 word whose FP32 reciprocal is the ModelOpt "
                   "weight_scale_2 (nearest when none is exact)",
        "max_relative_rms_error": arguments.max_error,
        "unchanged": "every other object, the bindings, the Uses and the recipe",
    }
    if arguments.weights is not None:
        record["weights"] = {"label": arguments.weights_label}
    if removed:
        record["removed_base_paths"] = removed
    return record


FULL_A_RECIPE = "qwen3_8_27b_nvfp4-full-a"


DIVISOR_ROLE = "activation_input_divisor"
AUX_FORMAT = "fp32"
AUX_LAYOUT = "contiguous_le_v1"


def reencode(arguments, base: Artifact, targets: Sequence[Target], encoder: Encoder) -> dict:
    directory = base.directory
    provenance, removed = strip_local_paths(directory.provenance)
    record = _record(arguments, base, targets, removed)
    provenance["reencode"] = record
    if any(target.converts for target in targets):
        # Layers 56-63 leave as NVFP4: the engine resolves the weights profile from the
        # recipe, and only the full-a profile binds those layers as NVFP4.
        if provenance.get("recipe") != FULL_A_RECIPE:
            record["base_recipe"] = provenance.get("recipe")
            provenance["recipe"] = FULL_A_RECIPE
    aux_values, aux_labels, uses = _plan_conversion_uses(base, targets)
    if aux_labels:
        record["activation_input_divisors"] = aux_labels
    by_id = {target.object_id: target for target in targets}
    started = time.perf_counter()
    writer = ArtifactWriter(
        arguments.out,
        _specs(base, targets, aux_values),
        components=directory.components,
        bindings=directory.bindings,
        uses=uses,
        metadata=directory.metadata,
        provenance=provenance,
    )
    reports = []
    try:
        total = len(base.objects)
        for index, obj in enumerate(base.objects, start=1):
            target = by_id.get(obj.id)
            if target is None:
                writer.write_object(obj.id, base.iter_object(obj.id))
                continue
            payload, report = encoder.payload(target)
            writer.write_object(obj.id, payload)
            reports.append(report)
            error = report.get("relative_rms_error_vs_weights")
            print(f"[{index}/{total}] {obj.id} {','.join(report['parameters'])} "
                  f"{report['codes']} divisor={report['weight_divisor']:.9g}"
                  f"{'' if report['divisor_exact'] else ' (inexact reciprocal)'} "
                  f"vs_base={report['relative_rms_error_vs_base']:.5f}"
                  f"{'' if error is None else f' vs_weights={error:.5f}'} ({report['seconds']}s)",
                  flush=True)
        for aux_id, value in aux_values.items():
            writer.write_object(aux_id, struct.pack("<f", value))
        writer.finish()
    except BaseException:
        writer.abort()
        raise
    return {
        "artifact": local_name(arguments.out),
        "bytes": arguments.out.stat().st_size,
        "recipe": provenance.get("recipe"),
        "reencode_record_sha256": hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "reencode": record,
        "objects": reports,
        "inexact_divisors": sum(not item["divisor_exact"] for item in reports),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }


def _digest(chunks) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def verify_output(base_path: Path, out_path: Path, expected: dict[str, str],
                   converted: set[str] | None = None) -> int:
    """0 when re-encoded objects hold the digests in ``expected`` and all else matches the base.

    Objects in ``converted`` keep their payload digest check but are allowed a different object
    record: an FP8 layer converted to NVFP4 changes format and layout while keeping its id, shape
    and position."""

    converted = converted or set()
    failures = 0
    with Artifact(base_path) as base, Artifact(out_path) as out:
        for field in ("components", "bindings", "uses", "metadata"):
            if getattr(out.directory, field) != getattr(base.directory, field):
                print(f"DIFF directory {field}", flush=True)
                failures += 1
        expected_recipe = (FULL_A_RECIPE if converted
                           else base.directory.provenance.get("recipe"))
        if out.directory.provenance.get("recipe") != expected_recipe:
            print("DIFF provenance recipe", flush=True)
            failures += 1
        for base_obj, out_obj in zip(base.objects, out.objects):
            if base_obj.id != out_obj.id:
                print(f"DIFF object order at {base_obj.id}", flush=True)
                return 1
            if base_obj.id in converted:
                continue
            # A converted object changes byte size, so every object stored after one shifts its
            # offset: records compare without the offset (order is enforced above, sizes below).
            base_record = {k: v for k, v in base_obj.to_json().items() if k != "offset"}
            out_record = {k: v for k, v in out_obj.to_json().items() if k != "offset"}
            if base_record != out_record:
                print(f"DIFF object record {base_obj.id}", flush=True)
                return 1
        for obj in base.objects:
            want = expected.get(obj.id) or _digest(base.iter_object(obj.id))
            if _digest(out.iter_object(obj.id)) != want:
                kind = "re-encoded" if obj.id in expected else "copied"
                print(f"DIFF {obj.id} ({kind} object)", flush=True)
                failures += 1
        for obj in base.objects:
            want = expected.get(obj.id) or _digest(base.iter_object(obj.id))
            if _digest(out.iter_object(obj.id)) != want:
                kind = "re-encoded" if obj.id in expected else "copied"
                print(f"DIFF {obj.id} ({kind} object)", flush=True)
                failures += 1
    print(f"verified {len(expected)} re-encoded objects and the copied rest: {failures} "
          f"mismatches", flush=True)
    return 1 if failures else 0


def _remove_output(out_path: Path) -> None:
    with Artifact(out_path) as out:
        parts = [out_path.parent / item.path for item in out.directory.files[1:]]
    for path in [out_path, *parts]:
        path.unlink(missing_ok=True)


def _layers(value: str) -> list[int]:
    layers: list[int] = []
    for item in value.split(","):
        first, _, last = item.partition("-")
        begin, end = int(first), int(last or first)
        if end < begin:
            raise argparse.ArgumentTypeError(f"layer range {item} is reversed")
        layers.extend(range(begin, end + 1))
    return sorted(set(layers))


def _max_error(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError(f"--max-error {value} must be finite and positive")
    return number


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", required=True, type=Path, help="artifact to re-encode")
    parser.add_argument("--donor", required=True, type=Path,
                        help="calibrated NVFP4 checkpoint of the same weights")
    parser.add_argument("--donor-label", required=True, type=input_label,
                        help="repository id or name of --donor recorded in the artifact")
    parser.add_argument("--weights", type=Path,
                        help="unquantized checkpoint of the base model (needed by --round)")
    parser.add_argument("--weights-label", type=input_label,
                        help="repository id or name of --weights (required with --weights)")
    parser.add_argument("--round", nargs="+", default=[], choices=PROJECTIONS,
                        help="projections whose codes are rounded from --weights under the "
                             "donor's scales")
    parser.add_argument("--layers", type=_layers,
                        help="text MLP layers, e.g. 0-55 (default: every NVFP4 MLP layer)")
    parser.add_argument("--max-error", type=_max_error, default=DEFAULT_MAX_ERROR,
                        help="largest relative RMS error of a re-encoded object against the "
                             "base's decoded values and against --weights (default 0.3)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--verify", action="store_true",
                        help="re-read the output and compare it with the encoded words and the "
                             "base; remove it on mismatch")
    arguments = parser.parse_args(argv)
    arguments.round = set(arguments.round)
    if arguments.round and arguments.weights is None:
        parser.error("--round needs --weights")
    if (arguments.weights is None) != (arguments.weights_label is None):
        parser.error("--weights and --weights-label go together")
    if arguments.out.resolve() == arguments.base.resolve():
        parser.error("--out must differ from --base")
    donor = SourceCheckpoint(arguments.donor)
    weights = SourceCheckpoint(arguments.weights) if arguments.weights else None
    try:
        with Artifact(arguments.base) as base:
            check_base(base)
            layers = arguments.layers if arguments.layers is not None else nvfp4_layers(base)
            targets = plan_targets(base, layers, arguments.round, donor, weights)
            report = reencode(arguments, base, targets,
                              Encoder(base, donor, weights, arguments.max_error))
    finally:
        donor.close()
        if weights is not None:
            weights.close()
    status = 0
    if arguments.verify:
        expected = {item["object"]: item["payload_sha256"] for item in report["objects"]}
        status = verify_output(arguments.base, arguments.out, expected,
                               converted={t.object_id for t in targets if t.converts})
        report["verified"] = status == 0
        if status:
            _remove_output(arguments.out)
            report["removed"] = True
    report_path = Path(str(arguments.out) + ".reencode.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{'complete' if status == 0 else 'FAILED'}: {report['bytes']} bytes, {len(targets)} "
          f"objects re-encoded ({report['inexact_divisors']} inexact divisors) in "
          f"{report['elapsed_seconds']}s; report={report_path}", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
