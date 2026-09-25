"""Re-encode the NVFP4 roles of a Qwen3.8-27B artifact with calibrated checkpoints.

A single-source NVFP4 artifact (``--base``) carries the MLP of its NVFP4 layers as the words of its
quantized source, often round-to-nearest with max block scales, and keeps the other projections
in row-scaled FP8. This tool rewrites objects from calibrated NVFP4 checkpoints of the same
weights, each role from its own donor (``--role-donor ROLE=PATH`` with ``--role-donor-label
ROLE=LABEL``):

* ``mlp``: the text MLP of ``--layers`` (for example NVIDIA Model Optimizer's Local-Hessian
  checkpoint, or a QAT checkpoint);
* ``head``: the output head, when a role converts;
* ``attention_input`` and ``gdn_input``: the attention and GDN input projections of every text
  layer;
* ``attention_output`` and ``gdn_output``: the attention and GDN output projections of every text
  layer.

``--donor`` is the donor of the mlp and head roles, ``--input-donor`` the donor of both input
roles. One pass writes every object, each from its role's donor. An NVFP4 object keeps its id,
format, layout and shape; an FP8 object keeps its id and shape and is converted to NVFP4. Every
other object is copied byte for byte, and the bindings are unchanged.

The engine resolves the weights profile from the output's recipe:

* nothing converted: the base's recipe, ``qwen3_8_27b_nvfp4``;
* ``qwen3_8_27b_nvfp4-full-a``: every text MLP layer and the output head in NVFP4;
* ``qwen3_8_27b_nvfp4-full-b``: full-a plus the attention and GDN input projections;
* ``qwen3_8_27b_nvfp4-full-c``: full-b plus the attention and GDN output projections.

The GDN a/b projection stays BF16 and the token embedding FP8. A conversion must select every FP8
MLP layer of the base, and convert the attention and GDN roles of one recipe. The Uses of a
converted leaf take ``AllowA4`` and an ``activation_input_divisor`` auxiliary, one object per
leaf: the donor's ``input_global_scale`` for an attention or GDN projection (compressed-tensors
stores the divisor itself, and the fused matrices must share it), the divisor of the same leaf in
the nearest NVFP4 layer below for an MLP leaf, and 1.0 for the output head (sm_70 runs NVFP4 with
16-bit activations).

An object's parameters are its bindings in row order; each takes its donor matrix and rows from
the artifact converter's source routes. The attention input object is [query | key | gate |
value]: q_proj stores each of the 24 heads as 256 query rows then 256 gate rows, so query and gate
are de-interleaved from it, around k_proj; the GDN input object is [query | key | value | z],
in_proj_qkv then in_proj_z in checkpoint order. An output object is one matrix, o_proj or out_proj.

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

Before the output is created, every selected object is checked against its donor's safetensors
header and scalars (tensors present, NVFP4 dtypes and geometry, rows matching the object's
bindings, finite positive divisors, one tensor scale per fused object) and against the
``--weights`` headers. Row order rests on the bindings: an object's parameters must tile it in
their order, and each takes its donor rows from the converter's source routes. While encoding,
every parameter's rows are compared with the base object's decoded values (the dequantized FP8
values for a conversion), and the object with ``--weights`` when given; the tool refuses a
parameter, or an object, whose relative RMS error exceeds ``--max-error``, and ``--weights`` rows
that are not finite. The comparison with the base, per parameter so that a small misplaced block
is not diluted in its object, is the numeric backstop on row order: rows taken in a wrong order
match ``--weights`` read in that same order, not the base object. ``--verify`` re-reads the
output: written objects must hold the encoded payloads and every other object the base's, the
Uses must be the planned ones, and the directory must differ from the base only as the conversion
planned; a failed verification removes the output.

The artifact's provenance names each role's donor and the weights by their required labels (a
repository id, for example), never by path, and drops the path members of the base's provenance,
listed under ``removed_base_paths``. The report ``<out>.reencode.json`` records, per object, its
role, the donor layout, the divisor and whether it is exact, the errors (against the base per
parameter too), the counts of saturated and zeroed rounded values, the SHA-256 of the donor
tensors read, of the ``--weights`` rows read (in object row order, the stored matrix when read
whole) and of the payload written, and each new auxiliary divisor with its payload SHA-256.

Example (full-c: every quantized matrix from QUASAR's QAT words, the output head, which QUASAR
keeps unquantized, from NVIDIA's words of the same base model; ``--weights`` checks them all)::

    python3 -m tools.convert.qwen3_8_27b.reencode_nvfp4 \\
      --base <dir>/qwen3_8_27b_nvfp4.ninfer \\
      --role-donor head=<checkpoint>/nvidia-Qwen3.8-27B-NVFP4 \\
      --role-donor-label head=nvidia/Qwen3.8-27B-NVFP4 \\
      --role-donor mlp=<qat> --role-donor-label mlp=<qat-label> \\
      --input-donor <qat> --input-donor-label <qat-label> \\
      --role-donor attention_output=<qat> --role-donor-label attention_output=<qat-label> \\
      --role-donor gdn_output=<qat> --role-donor-label gdn_output=<qat-label> \\
      --weights <checkpoint>/Qwen3.8-27B --weights-label Qwen/Qwen3.8-27B --layers 0-63 \\
      --out <dir>/qwen3_8_27b_nvfp4_full_c.ninfer --verify

with ``<qat>`` the QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 checkpoint and ``<qat-label>`` its
repository id. A full-b build gives ``--donor`` and ``--input-donor`` only.
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
from typing import Sequence

from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorObject, TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.convert.provenance import input_label, local_name, strip_local_paths
from tools.convert.qwen3_8_27b.graft_single_source import SourceCheckpoint
from tools.convert.qwen3_8_27b.reencode_nvfp4_encode import DEFAULT_MAX_ERROR, Encoder
from tools.convert.qwen3_8_27b.reencode_nvfp4_plan import (
    AUX_FORMAT,
    AUX_LAYOUT,
    DONOR_ROLES,
    HEAD_ROLE,
    NVFP4_FORMAT,
    NVFP4_LAYOUT,
    Target,
    UsesPlan,
    check_base,
    nvfp4_layers,
    output_recipe,
    plan_targets,
    plan_uses,
)
from tools.convert.qwen3_8_27b.reencode_nvfp4_verify import remove_output, verify_output


TOOL = "tools.convert.qwen3_8_27b.reencode_nvfp4"
PROJECTIONS = ("gate", "up", "down")
# The donor options that give several roles one donor.
SHORTHANDS = {"donor": ("mlp", "head"), "input_donor": ("attention_input", "gdn_input")}


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


def _donors_record(labels: dict[str, str], targets: Sequence[Target]) -> dict:
    """Label of each donor role that re-encoded objects, with their text layers."""

    record = {}
    for donor_role, roles in DONOR_ROLES.items():
        layers = sorted({target.layer for target in targets if target.role in roles})
        if not layers:
            continue
        record[donor_role] = {"label": labels[donor_role]}
        if HEAD_ROLE not in roles:
            record[donor_role]["layers"] = layers
    return record


def _record(arguments, labels: dict[str, str], base: Artifact, targets: Sequence[Target],
            removed: list[str]) -> dict:
    record = {
        "tool": TOOL,
        "base": local_name(arguments.base),
        "base_artifact_id": base.artifact_id.hex(),
        "donors": _donors_record(labels, targets),
        "codes": {projection: ("rounded from weights" if projection in arguments.round else "donor")
                  for projection in PROJECTIONS},
        "divisor": "donor global scale, or the FP32 word whose FP32 reciprocal is the ModelOpt "
                   "weight_scale_2 (nearest when none is exact)",
        "max_relative_rms_error": arguments.max_error,
        "unchanged": "every other object and the bindings",
    }
    if arguments.weights is not None:
        record["weights"] = {"label": arguments.weights_label}
    if removed:
        record["removed_base_paths"] = removed
    return record


def reencode(arguments, labels: dict[str, str], base: Artifact, targets: Sequence[Target],
             plan: UsesPlan, recipe: str, encoder: Encoder) -> dict:
    directory = base.directory
    provenance, removed = strip_local_paths(directory.provenance)
    record = _record(arguments, labels, base, targets, removed)
    provenance["reencode"] = record
    if provenance.get("recipe") != recipe:
        # The engine resolves the weights profile from the recipe: only the conversion profiles
        # bind the converted roles as NVFP4.
        record["base_recipe"] = provenance.get("recipe")
        provenance["recipe"] = recipe
    if plan.labels:
        record["activation_input_divisors"] = plan.labels
    by_id = {target.object_id: target for target in targets}
    started = time.perf_counter()
    writer = ArtifactWriter(
        arguments.out,
        _specs(base, targets, plan.divisors),
        components=directory.components,
        bindings=directory.bindings,
        uses=plan.uses,
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
            leaf = max(report["relative_rms_error_vs_base_by_parameter"].values())
            print(f"[{index}/{total}] {obj.id} {','.join(report['parameters'])} "
                  f"{report['codes']} divisor={report['weight_divisor']:.9g}"
                  f"{'' if report['divisor_exact'] else ' (inexact reciprocal)'} "
                  f"vs_base={report['relative_rms_error_vs_base']:.5f} (worst parameter "
                  f"{leaf:.5f}){'' if error is None else f' vs_weights={error:.5f}'} "
                  f"({report['seconds']}s)", flush=True)
        for aux_id, value in plan.divisors.items():
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


def _role_value(value: str) -> tuple[str, str]:
    """ROLE and VALUE of ROLE=VALUE; a missing VALUE is empty, which its option refuses."""

    role, _, rest = value.partition("=")
    if role not in DONOR_ROLES:
        raise argparse.ArgumentTypeError(f"{value!r} is not ROLE=VALUE with ROLE one of "
                                         f"{', '.join(DONOR_ROLES)}")
    return role, rest


def _role_path(value: str) -> tuple[str, Path]:
    role, path = _role_value(value)
    if not path:
        raise argparse.ArgumentTypeError(f"{value!r} names no checkpoint")
    return role, Path(path)


def _role_label(value: str) -> tuple[str, str]:
    role, label = _role_value(value)
    return role, input_label(label)


def _role_donors(parser: argparse.ArgumentParser,
                 arguments) -> tuple[dict[str, Path], dict[str, str]]:
    """Donor checkpoint and label of each role, from --role-donor, --role-donor-label and the
    shorthands; a role takes one donor, and the mlp role needs one."""

    paths: dict[str, Path] = {}
    labels: dict[str, str] = {}
    for option, pairs, given in (("--role-donor", arguments.role_donor, paths),
                                 ("--role-donor-label", arguments.role_donor_label, labels)):
        for role, value in pairs:
            if role in given:
                parser.error(f"{option} gives the {role} role twice")
            given[role] = value
    for name, roles in SHORTHANDS.items():
        path, label = getattr(arguments, name), getattr(arguments, name + "_label")
        flag = "--" + name.replace("_", "-")
        if (path is None) != (label is None):
            parser.error(f"{flag} and {flag}-label go together")
        for role in roles if path is not None else ():
            if role in paths or role in labels:
                parser.error(f"the {role} role has a donor from {flag} and from --role-donor")
            paths[role], labels[role] = path, label
    unmatched = sorted(paths.keys() ^ labels.keys())
    if unmatched:
        parser.error("--role-donor and --role-donor-label must name the same roles, not "
                     f"{', '.join(unmatched)} in one of them only")
    if "mlp" not in paths:
        parser.error("the mlp role needs a donor: --donor or --role-donor mlp=PATH")
    return paths, labels


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", required=True, type=Path, help="artifact to re-encode")
    parser.add_argument("--role-donor", action="append", default=[], type=_role_path,
                        metavar="ROLE=PATH",
                        help="calibrated NVFP4 checkpoint of the same weights for one role: "
                             f"{', '.join(DONOR_ROLES)}")
    parser.add_argument("--role-donor-label", action="append", default=[], type=_role_label,
                        metavar="ROLE=LABEL",
                        help="repository id or name of a role's donor recorded in the artifact "
                             "(required with it)")
    parser.add_argument("--donor", type=Path, help="donor of the mlp and head roles")
    parser.add_argument("--donor-label", type=input_label,
                        help="repository id or name of --donor (required with it)")
    parser.add_argument("--input-donor", type=Path,
                        help="donor of the attention_input and gdn_input roles")
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
                        help="largest relative RMS error of each parameter against the base's "
                             "decoded values, and of each object against --weights (default 0.3)")
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
    paths, labels = _role_donors(parser, arguments)
    if arguments.out.resolve() == arguments.base.resolve():
        parser.error("--out must differ from --base")
    donors = {role: SourceCheckpoint(path) for role, path in paths.items()}
    weights = SourceCheckpoint(arguments.weights) if arguments.weights else None
    try:
        with Artifact(arguments.base) as base:
            check_base(base)
            layers = arguments.layers if arguments.layers is not None else nvfp4_layers(base)
            targets = plan_targets(base, layers, arguments.round, donors, weights)
            recipe = output_recipe(targets)
            plan = plan_uses(base, targets)
            report = reencode(arguments, labels, base, targets, plan, recipe,
                              Encoder(base, weights, arguments.max_error))
    finally:
        for checkpoint in (*donors.values(), weights):
            if checkpoint is not None:
                checkpoint.close()
    status = 0
    if arguments.verify:
        expected = {item["object"]: item["payload_sha256"]
                    for item in (*report["objects"], *report["auxiliaries"])}
        status = verify_output(arguments.base, arguments.out, expected,
                               converted={target.object_id: target.parameters
                                          for target in targets if target.converts},
                               uses=plan.uses, recipe=recipe)
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
