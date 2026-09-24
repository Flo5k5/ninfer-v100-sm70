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
  from the unquantized weights of the base model (``--weights``), for projections whose weights
  differ from the donor's, such as the residual writers of an abliterated fine-tune.

Donor layouts:

* ModelOpt: ``weight`` (packed), ``weight_scale`` (E4M3), ``weight_scale_2``;
  value = e2m1(code) * e4m3(scale) * weight_scale_2;
* compressed-tensors: ``weight_packed``, ``weight_scale``, ``weight_global_scale``;
  value = e2m1(code) * e4m3(scale) / weight_global_scale.

The artifact stores one FP32 weight divisor per object (value = e2m1 * e4m3 / divisor). A
compressed-tensors global scale is that divisor. For ModelOpt, the stored divisor is the FP32 word
whose FP32 reciprocal is exactly ``weight_scale_2``, searched around ``fl(1 / weight_scale_2)``, so
the FP32 decode routes, which multiply the block scales by ``1.0f / divisor``, reproduce the donor's
values bit for bit; routes that fold the divisor into FP16 constants or divide agree to their own
rounding. When no FP32 word has that exact reciprocal, the nearest one is stored and the object is
reported with ``divisor_exact: false``. Rounded codes use the same decode step, e4m3 * (1 /
divisor). Gate and up are one fused object with one divisor: their donor tensor scales must be equal.

Example (layers 0-55 of an abliterated build, gate/up from NVIDIA's Local-Hessian words, down
rounded from the abliterated BF16 weights under NVIDIA's scales)::

    python3 -m tools.convert.qwen3_8_27b.reencode_nvfp4 \\
      --base <dir>/qwen3_8_27b_orca_nvfp4.ninfer \\
      --donor <checkpoint>/nvidia-Qwen3.8-27B-NVFP4 \\
      --weights <checkpoint>/Qwen3.8-27B-Uncensored-BF16 --round down \\
      --out <dir>/qwen3_8_27b_orca_nvfp4_g2.ninfer --verify

With ``--weights``, every re-encoded object is compared with the unquantized weights and the tool
refuses an object whose relative RMS error exceeds ``--max-error`` (a donor of other weights).
``--verify`` re-reads the output: re-encoded objects must hold the encoded words and every other
object and the directory must match the base; a failed verification removes the output. The report
``<out>.reencode.json`` records, per object, the donor layout, the divisor and whether it is exact,
the relative RMS error against ``--weights`` and the payload digest.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Iterable, Iterator, Sequence

import torch

from tools.artifact.codecs.nvfp4 import encode_nvfp4
from tools.artifact.formats import _E2M1_MAGNITUDES
from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorObject, TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.convert.qwen3_8_27b.graft_single_source import SourceCheckpoint
from tools.convert.qwen3_8_27b.inventory_nvfp4 import MODEL_ID


NVFP4_FORMAT = "nvfp4"
BASE_RECIPE = "qwen3_8_27b_nvfp4"
PROJECTIONS = ("gate", "up", "down")
RECIPROCAL_SEARCH_ULPS = 8
DEFAULT_MAX_ERROR = 0.3
ROW_CHUNK = 4096
_E2M1 = torch.tensor(_E2M1_MAGNITUDES, dtype=torch.float32)
# Midpoints between consecutive E2M1 magnitudes; a value on a midpoint rounds to the level with an
# even mantissa: up at 0.75, 1.75 and 3.5, down at 0.25, 1.25, 2.5 and 5.
_MIDPOINTS = torch.tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), dtype=torch.float32)
_TIES_UP = torch.tensor((0.75, 1.75, 3.5), dtype=torch.float32)


class ReencodeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DonorWords:
    """NVFP4 words of one donor matrix and the FP32 divisor the artifact stores for them."""

    packed: torch.Tensor  # uint8 [N, K/2]
    scales: torch.Tensor  # uint8 E4M3FN words [N, K/16]
    divisor: torch.Tensor  # FP32 scalar
    exact: bool  # the stored divisor reproduces the donor's tensor scale exactly
    layout: str


@dataclass(frozen=True, slots=True)
class Target:
    """One NVFP4 object of the base and the donor matrices that re-encode it, in row order."""

    object_id: str
    shape: tuple[int, int]
    layer: int
    parts: tuple[tuple[str, str], ...]  # (logical parameter, donor module)
    rounded: bool


def _fp32(value: float) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32)


def reciprocal_divisor(scale_2: float) -> tuple[torch.Tensor, bool]:
    """FP32 divisor whose FP32 reciprocal is ``scale_2``, or nearest to it, and whether exact."""

    target = _fp32(scale_2)
    if not bool(torch.isfinite(target)) or float(target) <= 0.0:
        raise ReencodeError(f"tensor scale {scale_2!r} must be finite and positive")
    one = _fp32(1.0)
    start = int((one / target).view(torch.int32))
    best: tuple[float, torch.Tensor] | None = None
    for delta in sorted(range(-RECIPROCAL_SEARCH_ULPS, RECIPROCAL_SEARCH_ULPS + 1), key=abs):
        candidate = torch.tensor(start + delta, dtype=torch.int32).view(torch.float32)
        if not bool(torch.isfinite(candidate)) or float(candidate) <= 0.0:
            continue
        error = abs(float(one / candidate) - float(target))
        if error == 0.0:
            return candidate, True
        if best is None or error < best[0]:
            best = (error, candidate)
    assert best is not None
    return best[1], False


def read_donor_words(donor: SourceCheckpoint, module: str) -> DonorWords:
    """Words of ``module`` in a ModelOpt or compressed-tensors NVFP4 checkpoint."""

    if donor.dtype(module + ".weight_packed") == "U8":
        global_scale = donor.get(module + ".weight_global_scale")
        if global_scale.dtype != torch.float32 or global_scale.numel() != 1:
            raise ReencodeError(f"{module}.weight_global_scale must be FP32[1]")
        packed = donor.get(module + ".weight_packed")
        divisor, exact, layout = global_scale.reshape(()).clone(), True, "compressed-tensors"
    elif donor.dtype(module + ".weight") == "U8" and donor.dtype(module + ".weight_scale_2"):
        scale_2 = donor.get(module + ".weight_scale_2")
        if scale_2.dtype != torch.float32 or scale_2.numel() != 1:
            raise ReencodeError(f"{module}.weight_scale_2 must be one FP32 value")
        packed = donor.get(module + ".weight")
        divisor, exact = reciprocal_divisor(float(scale_2.reshape(())))
        layout = "modelopt"
    else:
        raise ReencodeError(f"{module}: no NVFP4 words in the donor checkpoint")
    if donor.dtype(module + ".weight_scale") != "F8_E4M3":
        raise ReencodeError(f"{module}.weight_scale must be F8_E4M3 block scales")
    scales = donor.get(module + ".weight_scale").view(torch.uint8)
    if packed.dim() != 2 or scales.shape != (packed.shape[0], packed.shape[1] // 8):
        raise ReencodeError(f"{module}: codes {tuple(packed.shape)} and scales "
                            f"{tuple(scales.shape)} do not describe one NVFP4 matrix")
    return DonorWords(packed.contiguous(), scales.contiguous(), divisor, exact, layout)


def code_steps(scales: torch.Tensor, divisor: torch.Tensor) -> torch.Tensor:
    """FP32 value of one E2M1 unit per block as the FP32 decode routes form it: e4m3 * (1/d)."""

    return scales.view(torch.float8_e4m3fn).float() * (_fp32(1.0) / divisor.float())


def round_to_nearest(values: torch.Tensor, scales: torch.Tensor,
                     divisor: torch.Tensor) -> torch.Tensor:
    """Packed E2M1 codes of FP32 ``values`` [N, K] under given block scales and divisor.

    Each value goes to the nearest E2M1 level of ``value / step`` with ties to even and saturation
    at 6, like ModelOpt and compressed-tensors; a block whose scale is zero decodes to zero.
    """

    rows, columns = values.shape
    step = code_steps(scales, divisor).unsqueeze(2)
    blocks = values.float().reshape(rows, columns // 16, 16)
    scaled = torch.where(step > 0, blocks / torch.where(step > 0, step, 1.0), 0.0)
    magnitude = scaled.abs().clamp(max=6.0)
    index = torch.bucketize(magnitude, _MIDPOINTS, right=False) + torch.isin(magnitude, _TIES_UP)
    codes = (index | (torch.signbit(scaled).to(index.dtype) << 3)).to(torch.uint8)
    codes = codes.reshape(rows, columns)
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()


def dequantize_words(packed: torch.Tensor, scales: torch.Tensor,
                     divisor: torch.Tensor) -> torch.Tensor:
    rows = packed.shape[0]
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=2).reshape(rows, -1).long()
    values = torch.where((codes & 0x8) != 0, -_E2M1[codes & 0x7], _E2M1[codes & 0x7])
    return (values.reshape(rows, -1, 16) * code_steps(scales, divisor).unsqueeze(2)).reshape(rows, -1)


def _object_of(directory, logical: str) -> tuple[str, int, int]:
    binding = directory.bindings.get(logical)
    if binding is None:
        raise ReencodeError(f"base artifact has no binding {logical}")
    if "object" in binding:
        return binding["object"], -1, -1
    if len(binding["parts"]) != 1:
        raise ReencodeError(f"{logical}: binding spans several objects")
    part = binding["parts"][0]
    return part["object"], int(part["range"][0]), int(part["range"][1])


def _nvfp4_object(base: Artifact, object_id: str, logical: str) -> TensorObject:
    obj = base.object(object_id)
    if not isinstance(obj, TensorObject) or len(obj.shape) != 2 or obj.format != NVFP4_FORMAT:
        raise ReencodeError(f"{logical}: object {object_id} is not an NVFP4 matrix in the base")
    return obj


def check_base(base: Artifact) -> None:
    """The base must be an original Qwen3.8-27B NVFP4 artifact (not an earlier re-encode)."""

    name = base.directory.metadata.get("name")
    recipe = base.directory.provenance.get("recipe")
    if name != MODEL_ID or recipe != BASE_RECIPE:
        raise ReencodeError(f"base is {name}/{recipe}, expected {MODEL_ID}/{BASE_RECIPE}")
    if "reencode" in base.directory.provenance:
        raise ReencodeError("base is already re-encoded; start from the original artifact")


def plan_targets(base: Artifact, layers: Iterable[int], rounded: set[str]) -> list[Target]:
    """Fused gate/up and down objects of the selected text MLP layers."""

    if bool({"gate", "up"} & rounded) and not {"gate", "up"} <= rounded:
        raise ReencodeError("gate and up share one object: round both or neither")
    directory = base.directory
    targets = []
    for layer in layers:
        prefix = f"text/layers/{layer}/mlp/"
        donor = f"model.language_model.layers.{layer}.mlp."
        gate_id, gate_begin, gate_end = _object_of(directory, prefix + "gate")
        up_id, up_begin, up_end = _object_of(directory, prefix + "up")
        gate_up = _nvfp4_object(base, gate_id, prefix + "gate")
        elements = gate_up.shape[0] * gate_up.shape[1]
        if up_id != gate_id or gate_begin != 0 or up_begin != gate_end or up_end != elements:
            raise ReencodeError(f"{prefix}gate_up: gate and up are not one fused object")
        targets.append(
            Target(gate_id, tuple(gate_up.shape), layer,
                   ((prefix + "gate", donor + "gate_proj"), (prefix + "up", donor + "up_proj")),
                   "gate" in rounded)
        )
        down_id, _, _ = _object_of(directory, prefix + "down")
        down = _nvfp4_object(base, down_id, prefix + "down")
        targets.append(
            Target(down_id, tuple(down.shape), layer, ((prefix + "down", donor + "down_proj"),),
                   "down" in rounded)
        )
    if not targets:
        raise ReencodeError("no MLP layer selected")
    return targets


def nvfp4_layers(base: Artifact) -> list[int]:
    """Text MLP layers whose gate/up and down objects are NVFP4 in the base."""

    layers = []
    layer = 0
    while f"text/layers/{layer}/mlp/gate" in base.directory.bindings:
        objects = [_object_of(base.directory, f"text/layers/{layer}/mlp/{leaf}")[0]
                   for leaf in ("gate", "down")]
        if all(base.object(object_id).format == NVFP4_FORMAT for object_id in objects):
            layers.append(layer)
        layer += 1
    return layers


def _same_codes(left: torch.Tensor, right: torch.Tensor) -> int:
    """Number of equal E2M1 codes in two packed matrices, counting +0 and -0 as equal."""

    count = 0
    for shift in (0, 4):
        a = (left >> shift) & 0x0F
        b = (right >> shift) & 0x0F
        count += int(((a == b) | (((a & 0x7) == 0) & ((b & 0x7) == 0))).sum())
    return count


class Encoder:
    def __init__(self, donor: SourceCheckpoint, weights: SourceCheckpoint | None,
                 max_error: float = DEFAULT_MAX_ERROR) -> None:
        self.donor = donor
        self.weights = weights
        self.max_error = max_error

    def payload(self, target: Target) -> tuple[bytes, dict]:
        """Encoded object payload and its report entry."""

        started = time.perf_counter()
        donors = [read_donor_words(self.donor, module) for _, module in target.parts]
        if len({int(item.divisor.view(torch.int32)) for item in donors}) != 1:
            raise ReencodeError(f"{target.object_id}: fused donor matrices have different "
                                "tensor scales")
        divisor = donors[0].divisor
        packed = torch.cat([item.packed for item in donors])
        scales = torch.cat([item.scales for item in donors])
        if (packed.shape[0], packed.shape[1] * 2) != target.shape:
            raise ReencodeError(f"{target.object_id}: donor geometry "
                                f"{(packed.shape[0], packed.shape[1] * 2)} != {target.shape}")
        report = {
            "object": target.object_id,
            "parameters": [logical for logical, _ in target.parts],
            "donor_matrices": [module for _, module in target.parts],
            "donor_layout": donors[0].layout,
            "codes": "rounded from --weights" if target.rounded else "donor",
            "shape": list(target.shape),
            "weight_divisor": float(divisor),
            "weight_divisor_word": f"0x{int(divisor.view(torch.int32)) & 0xFFFFFFFF:08x}",
            "divisor_exact": all(item.exact for item in donors),
        }
        if target.rounded or self.weights is not None:
            packed = self._compare_or_round(target, packed, scales, divisor, report)
        payload = encode_nvfp4(packed, scales, divisor.reshape(()), target.shape)
        report["payload_sha256"] = hashlib.sha256(payload).hexdigest()
        report["seconds"] = round(time.perf_counter() - started, 1)
        return payload, report

    def _compare_or_round(self, target: Target, packed: torch.Tensor, scales: torch.Tensor,
                          divisor: torch.Tensor, report: dict) -> torch.Tensor:
        error = norm = 0.0
        same = 0
        rounded = []
        for row, weights in self._weight_rows(target):
            end = row + weights.shape[0]
            codes = round_to_nearest(weights, scales[row:end], divisor)
            chosen = codes if target.rounded else packed[row:end]
            if target.rounded:
                rounded.append(codes)
            else:
                same += _same_codes(chosen, codes)
            decoded = dequantize_words(chosen, scales[row:end], divisor)
            error += float((decoded.double() - weights.double()).square().sum())
            norm += float(weights.double().square().sum())
        report["relative_rms_error"] = (error / norm) ** 0.5 if norm else 0.0
        if not target.rounded:
            report["codes_equal_to_rounded_weights"] = same / (target.shape[0] * target.shape[1])
        if report["relative_rms_error"] > self.max_error:
            raise ReencodeError(f"{target.object_id}: relative RMS error "
                                f"{report['relative_rms_error']:.3f} against --weights exceeds "
                                f"{self.max_error}; does the donor quantize these weights?")
        return torch.cat(rounded) if target.rounded else packed

    def _weight_rows(self, target: Target) -> Iterator[tuple[int, torch.Tensor]]:
        if self.weights is None:
            raise ReencodeError(f"{target.object_id}: rounding needs --weights")
        row = 0
        for _, module in target.parts:
            name = module + ".weight"
            if self.weights.dtype(name) not in ("BF16", "F16", "F32"):
                raise ReencodeError(f"{name}: expected unquantized weights in --weights")
            rows, columns = self.weights.meta[name][1]
            if columns != target.shape[1]:
                raise ReencodeError(f"{name}: {columns} columns, object has {target.shape[1]}")
            for begin in range(0, rows, ROW_CHUNK):
                end = min(begin + ROW_CHUNK, rows)
                yield row + begin, self.weights.rows(name, begin, end).float()
            row += rows
        if row != target.shape[0]:
            raise ReencodeError(f"{target.object_id}: weights have {row} rows, object has "
                                f"{target.shape[0]}")


def _specs(base: Artifact) -> list:
    specs = []
    for obj in base.objects:
        if isinstance(obj, TensorObject):
            specs.append(TensorSpec(obj.id, tuple(obj.shape), obj.format, obj.layout))
        else:
            specs.append(ResourceSpec(obj.id, obj.bytes, obj.encoding))
    return specs


def _record(arguments, base: Artifact, targets: Sequence[Target]) -> dict:
    record = {
        "tool": "tools.convert.qwen3_8_27b.reencode_nvfp4",
        "base": str(arguments.base.resolve()),
        "base_artifact_id": base.artifact_id.hex(),
        "donor": {"label": arguments.donor_label or str(arguments.donor),
                  "path": str(arguments.donor.resolve())},
        "layers": sorted({target.layer for target in targets}),
        "codes": {projection: ("rounded from weights" if projection in arguments.round else "donor")
                  for projection in PROJECTIONS},
        "divisor": "donor global scale, or the FP32 word whose FP32 reciprocal is the ModelOpt "
                   "weight_scale_2 (nearest when none is exact)",
        "unchanged": "every other object, the bindings, the Uses and the recipe",
    }
    if arguments.weights is not None:
        record["weights"] = {"label": arguments.weights_label or str(arguments.weights),
                             "path": str(arguments.weights.resolve())}
    return record


def reencode(arguments, base: Artifact, targets: Sequence[Target], encoder: Encoder) -> dict:
    directory = base.directory
    provenance = copy.deepcopy(directory.provenance)
    record = _record(arguments, base, targets)
    provenance["reencode"] = record
    by_id = {target.object_id: target for target in targets}
    started = time.perf_counter()
    writer = ArtifactWriter(
        arguments.out,
        _specs(base),
        components=directory.components,
        bindings=directory.bindings,
        uses=directory.uses,
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
            error = report.get("relative_rms_error")
            print(f"[{index}/{total}] {obj.id} {','.join(report['parameters'])} "
                  f"{report['codes']} divisor={report['weight_divisor']:.9g}"
                  f"{'' if report['divisor_exact'] else ' (inexact reciprocal)'}"
                  f"{'' if error is None else f' rel_rms={error:.5f}'} ({report['seconds']}s)",
                  flush=True)
        writer.finish()
    except BaseException:
        writer.abort()
        raise
    return {
        "artifact": str(arguments.out),
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


def verify_output(base_path: Path, out_path: Path, expected: dict[str, str]) -> int:
    """0 when re-encoded objects hold the digests in ``expected`` and all else matches the base."""

    failures = 0
    with Artifact(base_path) as base, Artifact(out_path) as out:
        for field in ("components", "bindings", "uses", "metadata"):
            if getattr(out.directory, field) != getattr(base.directory, field):
                print(f"DIFF directory {field}", flush=True)
                failures += 1
        if out.directory.provenance.get("recipe") != base.directory.provenance.get("recipe"):
            print("DIFF provenance recipe", flush=True)
            failures += 1
        if [obj.to_json() for obj in out.objects] != [obj.to_json() for obj in base.objects]:
            print("DIFF object records", flush=True)
            return 1
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", required=True, type=Path, help="artifact to re-encode")
    parser.add_argument("--donor", required=True, type=Path,
                        help="calibrated NVFP4 checkpoint of the same weights")
    parser.add_argument("--weights", type=Path,
                        help="unquantized checkpoint of the base model (needed by --round)")
    parser.add_argument("--round", nargs="*", default=[], choices=PROJECTIONS,
                        help="projections whose codes are rounded from --weights under the "
                             "donor's scales")
    parser.add_argument("--layers", type=_layers,
                        help="text MLP layers, e.g. 0-55 (default: every NVFP4 MLP layer)")
    parser.add_argument("--max-error", type=float, default=DEFAULT_MAX_ERROR,
                        help="largest relative RMS error against --weights (default 0.3)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--donor-label", default=None)
    parser.add_argument("--weights-label", default=None)
    parser.add_argument("--verify", action="store_true",
                        help="re-read the output and compare it with the encoded words and the "
                             "base; remove it on mismatch")
    arguments = parser.parse_args(argv)
    arguments.round = set(arguments.round)
    if arguments.round and arguments.weights is None:
        parser.error("--round needs --weights")
    if arguments.out.resolve() == arguments.base.resolve():
        parser.error("--out must differ from --base")
    encoder = Encoder(SourceCheckpoint(arguments.donor),
                      SourceCheckpoint(arguments.weights) if arguments.weights else None,
                      arguments.max_error)
    with Artifact(arguments.base) as base:
        check_base(base)
        layers = arguments.layers if arguments.layers is not None else nvfp4_layers(base)
        targets = plan_targets(base, layers, arguments.round)
        report = reencode(arguments, base, targets, encoder)
    status = 0
    if arguments.verify:
        expected = {item["object"]: item["payload_sha256"] for item in report["objects"]}
        status = verify_output(arguments.base, arguments.out, expected)
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
