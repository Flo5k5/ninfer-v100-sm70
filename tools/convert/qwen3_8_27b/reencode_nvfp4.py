"""Re-encode the NVFP4 roles of a Qwen3.8-27B artifact with calibrated checkpoints.

A single-source NVFP4 artifact (``--base``) carries the MLP of its NVFP4 layers as the words of its
quantized source, often round-to-nearest with max block scales, and keeps the other projections
in row-scaled FP8. This tool rewrites objects from calibrated NVFP4 checkpoints of the same
weights: the text MLP of ``--layers`` from ``--donor`` (for example NVIDIA Model Optimizer's
Local-Hessian checkpoint), the output head from ``--donor`` too when a role converts, and with
``--input-donor`` the attention and GDN input projections of every text layer (for example a QAT
checkpoint). An NVFP4 object keeps its id, format, layout and shape; an FP8 object keeps its id
and shape and is converted to NVFP4. Every other object is copied byte for byte, and the bindings
are unchanged.

The engine resolves the weights profile from the output's recipe:

* nothing converted: the base's recipe, ``qwen3_8_27b_nvfp4``;
* ``qwen3_8_27b_nvfp4-full-a``: every text MLP layer and the output head in NVFP4;
* ``qwen3_8_27b_nvfp4-full-b``: full-a plus the attention and GDN input projections
  (``--input-donor``); the attention and GDN output projections stay FP8, GDN a/b BF16.

A conversion must select every FP8 MLP layer of the base. The Uses of a converted leaf take
``AllowA4`` and an ``activation_input_divisor`` auxiliary, one object per leaf: the input donor's
``input_global_scale`` for an input projection (compressed-tensors stores the divisor itself, and
the fused matrices must share it), the divisor of the same leaf in the nearest NVFP4 layer below
for an MLP leaf, and 1.0 for the output head (sm_70 runs NVFP4 with 16-bit activations).

An object's parameters are its bindings in row order; each takes its donor matrix and rows from
the artifact converter's source routes. The attention object is [query | key | gate | value]:
q_proj stores each of the 24 heads as 256 query rows then 256 gate rows, so query and gate are
de-interleaved from it, around k_proj; the GDN object is [query | key | value | z], in_proj_qkv
then in_proj_z in checkpoint order.

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
use the same decode step, e4m3 * (1 / divisor). The matrices of one fused object share one
divisor: their donor tensor scales must be equal.

Before the output is created, every selected object is checked against the donors' safetensors
headers and scalars (tensors present, NVFP4 dtypes and geometry, rows matching the object's
bindings, finite positive divisors, one tensor scale per fused object) and against the
``--weights`` headers. While encoding, every object is compared with the base object's decoded
values (the dequantized FP8 values for a conversion), and with ``--weights`` when given; the tool
refuses an object whose relative RMS error exceeds ``--max-error`` against either, and
``--weights`` rows that are not finite. The base comparison is the one guard on row order: rows
taken in a wrong order match ``--weights`` read in that same order, not the base object.
``--verify`` re-reads the output: written objects must hold the encoded payloads, every other
object the base's, and the directory must differ from the base only as the conversion planned; a
failed verification removes the output.

The artifact's provenance names the donors and the weights by their required labels (a repository
id, for example), never by path, and drops the path members of the base's provenance, listed under
``removed_base_paths``. The report ``<out>.reencode.json`` records, per object, its role, the donor
layout, the divisor and whether it is exact, the errors, the counts of saturated and zeroed rounded
values, the SHA-256 of the donor tensors read, of the ``--weights`` rows read (in object row order,
so the stored matrix when it is read whole) and of the payload written, and each new auxiliary
divisor with its payload SHA-256.

Example (full-b: every MLP layer and the head from NVIDIA's Local-Hessian words, down rounded from
the abliterated BF16 weights under NVIDIA's scales, the input projections from QUASAR's QAT
words)::

    python3 -m tools.convert.qwen3_8_27b.reencode_nvfp4 \\
      --base <dir>/qwen3_8_27b_orca_nvfp4.ninfer \\
      --donor <checkpoint>/nvidia-Qwen3.8-27B-NVFP4 --donor-label nvidia/Qwen3.8-27B-NVFP4 \\
      --input-donor <checkpoint>/Qwen3.8-27B-QUASAR-NVFP4 \\
      --input-donor-label QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 \\
      --weights <checkpoint>/Qwen3.8-27B-Uncensored-BF16 \\
      --weights-label orcarouter/Qwen3.8-27B-Uncensored --round down --layers 0-63 \\
      --out <dir>/qwen3_8_27b_orca_nvfp4_full_b.ninfer --verify
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
from tools.convert.qwen3_8_27b.graft_single_source import SourceCheckpoint, _select_rows
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
    INPUT_GLOBAL_SCALE,
    NVFP4_FORMAT,
    NVFP4_LAYOUT,
    ReencodeError,
    Target,
    check_base,
    nvfp4_layers,
    output_recipe,
    plan_targets,
    plan_uses,
)
from tools.convert.qwen3_8_27b.reencode_nvfp4_verify import remove_output, verify_output


TOOL = "tools.convert.qwen3_8_27b.reencode_nvfp4"
PROJECTIONS = ("gate", "up", "down")
DEFAULT_MAX_ERROR = 0.3
ROW_CHUNK = 4096
AUX_FORMAT = "fp32"
AUX_LAYOUT = "contiguous_le_v1"


class Encoder:
    def __init__(self, base: Artifact, weights: SourceCheckpoint | None,
                 max_error: float = DEFAULT_MAX_ERROR) -> None:
        self.base = base
        self.weights = weights
        self.max_error = max_error

    def payload(self, target: Target) -> tuple[bytes, dict]:
        """Encoded object payload and its report entry."""

        started = time.perf_counter()
        packed, scales = [], []
        donor_sha256 = {}
        for item in target.donors:
            codes_name, scales_name, scale_name = item.tensors
            codes = target.donor.get(codes_name)
            block_scales = target.donor.get(scales_name).view(torch.uint8)
            packed.append(_select_rows(codes, item.ranges))
            scales.append(_select_rows(block_scales, item.ranges))
            read = [(codes_name, codes), (scales_name, block_scales),
                    (scale_name, target.donor.get(scale_name))]
            if target.input_divisor is not None:
                name = item.module + INPUT_GLOBAL_SCALE
                read.append((name, target.donor.get(name)))
            for name, tensor in read:
                donor_sha256[name] = tensor_sha256(tensor)
        report = {
            "object": target.object_id,
            "role": target.role,
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
        digests: dict = {}
        for begin, end, weights in self._row_chunks(target, digests):
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
        # Blocking for conversions too: --weights rows are read in the object's row order, so
        # only the base object shows rows taken in a wrong order (a relative error near 1.4). A
        # conversion compares two quantizations of the same weights, 0.09 to 0.13 on the model.
        self._refuse(target, against_base.value(), "the base object's values")
        if self.weights is None:
            return packed
        report["relative_rms_error_vs_weights"] = against_weights.value()
        report["weights_sha256"] = {name: digest.hexdigest() for name, digest in digests.items()}
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
                                "donor quantize these weights, in this row order?")

    def _row_chunks(self, target: Target,
                    digests: dict) -> Iterator[tuple[int, int, torch.Tensor | None]]:
        """Row ranges of the object with their finite --weights rows, when given; ``digests``
        accumulates, per --weights matrix, the SHA-256 of the rows read in that order."""

        if self.weights is None:
            for begin in range(0, target.shape[0], ROW_CHUNK):
                yield begin, min(begin + ROW_CHUNK, target.shape[0]), None
            return
        row = 0
        for item in target.donors:
            name = item.module + ".weight"
            digest = digests.setdefault(name, hashlib.sha256())
            for first, last in item.ranges or ((0, item.rows),):
                for begin in range(first, last, ROW_CHUNK):
                    end = min(begin + ROW_CHUNK, last)
                    stored = self.weights.rows(name, begin, end)
                    update_digest(digest, stored)
                    values = stored.float()
                    if not bool(torch.isfinite(values).all()):
                        raise ReencodeError(f"{name}: rows [{begin}, {end}) of --weights hold "
                                            "non-finite values")
                    yield row, row + end - begin, values
                    row += end - begin


def _specs(base: Artifact, targets: Sequence[Target], aux_values: dict[str, float]) -> list:
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
    for aux_id in aux_values:
        specs.append(TensorSpec(aux_id, (), AUX_FORMAT, AUX_LAYOUT))
    return specs


def _record(arguments, base: Artifact, targets: Sequence[Target], removed: list[str]) -> dict:
    record = {
        "tool": TOOL,
        "base": local_name(arguments.base),
        "base_artifact_id": base.artifact_id.hex(),
        "donor": {"label": arguments.donor_label},
    }
    if arguments.input_donor is not None:
        record["input_donor"] = {"label": arguments.input_donor_label}
    record.update({
        "layers": sorted({target.layer for target in targets}),
        "codes": {projection: ("rounded from weights" if projection in arguments.round else "donor")
                  for projection in PROJECTIONS},
        "divisor": "donor global scale, or the FP32 word whose FP32 reciprocal is the ModelOpt "
                   "weight_scale_2 (nearest when none is exact)",
        "max_relative_rms_error": arguments.max_error,
        "unchanged": "every other object and the bindings",
    })
    if arguments.weights is not None:
        record["weights"] = {"label": arguments.weights_label}
    if removed:
        record["removed_base_paths"] = removed
    return record


def reencode(arguments, base: Artifact, targets: Sequence[Target], encoder: Encoder) -> dict:
    directory = base.directory
    provenance, removed = strip_local_paths(directory.provenance)
    record = _record(arguments, base, targets, removed)
    provenance["reencode"] = record
    recipe = output_recipe(targets)
    if provenance.get("recipe") != recipe:
        # The engine resolves the weights profile from the recipe: only the full-a and full-b
        # profiles bind the converted roles as NVFP4.
        record["base_recipe"] = provenance.get("recipe")
        provenance["recipe"] = recipe
    aux_values, aux_labels, uses = plan_uses(base, targets)
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
    reports, auxiliaries = [], []
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
            payload = struct.pack("<f", value)
            writer.write_object(aux_id, payload)
            auxiliaries.append({"object": aux_id, "value": value,
                                "payload_sha256": hashlib.sha256(payload).hexdigest()})
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
        "auxiliaries": auxiliaries,
        "inexact_divisors": sum(not item["divisor_exact"] for item in reports),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }


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
                        help="calibrated NVFP4 checkpoint of the same weights (MLP and head)")
    parser.add_argument("--donor-label", required=True, type=input_label,
                        help="repository id or name of --donor recorded in the artifact")
    parser.add_argument("--input-donor", type=Path,
                        help="calibrated compressed-tensors NVFP4 checkpoint whose words convert "
                             "the attention and GDN input projections of every text layer "
                             "(recipe qwen3_8_27b_nvfp4-full-b)")
    parser.add_argument("--input-donor-label", type=input_label,
                        help="repository id or name of --input-donor (required with it)")
    parser.add_argument("--weights", type=Path,
                        help="unquantized checkpoint of the base model (needed by --round)")
    parser.add_argument("--weights-label", type=input_label,
                        help="repository id or name of --weights (required with --weights)")
    parser.add_argument("--round", nargs="+", default=[], choices=PROJECTIONS,
                        help="MLP projections whose codes are rounded from --weights under the "
                             "donor's scales")
    parser.add_argument("--layers", type=_layers,
                        help="text MLP layers, e.g. 0-55 (default: every NVFP4 MLP layer); a "
                             "conversion needs every FP8 MLP layer")
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
    if (arguments.input_donor is None) != (arguments.input_donor_label is None):
        parser.error("--input-donor and --input-donor-label go together")
    if arguments.out.resolve() == arguments.base.resolve():
        parser.error("--out must differ from --base")
    donor = SourceCheckpoint(arguments.donor)
    input_donor = SourceCheckpoint(arguments.input_donor) if arguments.input_donor else None
    weights = SourceCheckpoint(arguments.weights) if arguments.weights else None
    try:
        with Artifact(arguments.base) as base:
            check_base(base)
            layers = arguments.layers if arguments.layers is not None else nvfp4_layers(base)
            targets = plan_targets(base, layers, arguments.round, donor, weights, input_donor)
            report = reencode(arguments, base, targets,
                              Encoder(base, weights, arguments.max_error))
    finally:
        for checkpoint in (donor, input_donor, weights):
            if checkpoint is not None:
                checkpoint.close()
    status = 0
    if arguments.verify:
        expected = {item["object"]: item["payload_sha256"]
                    for item in (*report["objects"], *report["auxiliaries"])}
        status = verify_output(arguments.base, arguments.out, expected,
                               converted={target.object_id: target.parameters
                                          for target in targets if target.converts},
                               recipe=output_recipe(targets))
        report["verified"] = status == 0
        if status:
            remove_output(arguments.out)
            report["removed"] = True
    report_path = Path(str(arguments.out) + ".reencode.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{'complete' if status == 0 else 'FAILED'}: {report['bytes']} bytes, recipe "
          f"{report['recipe']}, {len(targets)} objects re-encoded ({report['inexact_divisors']} "
          f"inexact divisors) in {report['elapsed_seconds']}s; report={report_path}", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
