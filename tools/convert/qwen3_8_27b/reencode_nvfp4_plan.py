"""Objects that ``reencode_nvfp4`` rewrites and the Uses it gives them, checked before writing.

A target is one object of the base with the donor matrices that re-encode it in row order: a text
MLP object (fused gate/up, or down), the output head, or the fused input projection of a text
layer (attention query/key/gate/value, or GDN query/key/value/z). Its parameters are the base's
bindings of the object in row order, and each one maps to its donor matrix and rows through the
source routes of ``graft_single_source.logical_matrix``: the attention query and gate rows come
from the head-interleaved ``q_proj`` exactly as the artifact's converter takes them.

Planning reads only the base directory and auxiliary divisors, the donors' and ``--weights``'
safetensors headers and the donors' FP32 scalars, and refuses what the encoding pass could only
discover while writing: missing tensors, dtypes or shapes that are not the object's, row counts
that differ from the object's bindings, fused matrices whose tensor scales (or, for an input
projection, input global scales) differ, scalars that are not finite and positive, and a
conversion that leaves an FP8 MLP layer, which no official recipe describes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import struct
from typing import Sequence

import torch

from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorObject
from tools.convert.official_recipes import QWEN3_8_27B_NVFP4_FULL_A, QWEN3_8_27B_NVFP4_FULL_B
from tools.convert.qwen3_8_27b.graft_single_source import SourceCheckpoint, logical_matrix
from tools.convert.qwen3_8_27b.reencode_nvfp4_numeric import divisor_word, reciprocal_divisor


NVFP4_FORMAT = "nvfp4"
FP8_ROW_FORMAT = "fp8_e4m3fn_row_bf16"
# Layout of the NVFP4 matrices; the FP8 rows of an original artifact carry row_scale_v1.
NVFP4_LAYOUT = "block_scale_k16_m128x4_v1"
BASE_NAME = "qwen3.8-27b"
BASE_RECIPE = "qwen3_8_27b_nvfp4"
UNQUANTIZED_DTYPES = ("BF16", "F16", "F32")
# Tensor suffixes per donor layout: packed codes, E4M3 block scales, FP32 tensor scale.
DONOR_TENSORS = {
    "compressed-tensors": (".weight_packed", ".weight_scale", ".weight_global_scale"),
    "modelopt": (".weight", ".weight_scale", ".weight_scale_2"),
}
# compressed-tensors stores the activation input divisor itself under this suffix.
INPUT_GLOBAL_SCALE = ".input_global_scale"
DIVISOR_ROLE = "activation_input_divisor"
AUX_FORMAT = "fp32"
AUX_LAYOUT = "contiguous_le_v1"
OUTPUT_HEAD = "text/output_head"
HEAD_ROLE = "output_head"
# Leaves of the per-layer roles, in the row order of their object.
MLP_ROLES = {"mlp/gate_up": ("mlp/gate", "mlp/up"), "mlp/down": ("mlp/down",)}
# The fused input projections the engine binds as attention/query_key_gate_value and
# gdn/query_key_value_z; every text layer has exactly one of them.
INPUT_ROLES = {
    "attention/input": ("attention/query", "attention/key", "attention/gate", "attention/value"),
    "gdn/input": ("gdn/query", "gdn/key", "gdn/value", "gdn/z"),
}


class ReencodeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DonorMatrix:
    """One donor matrix checked from its safetensors header, the rows taken from it, and the
    divisor of its tensor scale."""

    module: str
    layout: str
    tensors: tuple[str, str, str]  # packed codes, block scales, tensor scale
    matrix_rows: int
    ranges: tuple[tuple[int, int], ...] | None  # donor rows taken in order; None: every row
    rows: int  # rows taken
    divisor: torch.Tensor  # FP32 scalar stored in the artifact
    exact: bool  # the divisor reproduces the donor's tensor scale exactly


@dataclass(frozen=True, slots=True)
class Target:
    """One object of the base and the donor matrices that re-encode it, in row order."""

    object_id: str
    shape: tuple[int, int]
    layer: int  # -1 for the output head
    role: str
    parameters: tuple[str, ...]
    donor: SourceCheckpoint
    donors: tuple[DonorMatrix, ...]
    rounded: bool
    # Format of the base object this target replaces: NVFP4 for a re-encode of words already in
    # NVFP4, the FP8 row format for an object the original artifact keeps in FP8 (converted here).
    base_format: str
    # Input projections only: the donor's input global scale, the activation input divisor of
    # the converted Uses.
    input_divisor: torch.Tensor | None

    @property
    def converts(self) -> bool:
        return self.base_format != NVFP4_FORMAT

    @property
    def divisor(self) -> torch.Tensor:
        return self.donors[0].divisor


def _positive_scalar(donor: SourceCheckpoint, name: str) -> torch.Tensor:
    if name not in donor.meta:
        raise ReencodeError(f"the donor checkpoint is missing {name}")
    if donor.dtype(name) != "F32" or math.prod(donor.meta[name][1]) != 1:
        raise ReencodeError(f"{name} must be one FP32 value")
    value = donor.get(name).reshape(())
    if not bool(torch.isfinite(value)) or float(value) <= 0.0:
        raise ReencodeError(f"{name} = {float(value)!r} must be finite and positive")
    return value


def check_donor_matrix(donor: SourceCheckpoint, module: str, columns: int,
                       ranges: tuple[tuple[int, int], ...] | None) -> DonorMatrix:
    """Check one donor matrix from its header and the rows taken from it (``ranges``, None for
    every row), and read the divisor of its tensor scale."""

    if donor.dtype(module + ".weight_packed") == "U8":
        layout = "compressed-tensors"
    elif donor.dtype(module + ".weight") == "U8":
        layout = "modelopt"
    else:
        raise ReencodeError(f"{module}: no NVFP4 words in the donor checkpoint")
    packed, scales, scale = (module + suffix for suffix in DONOR_TENSORS[layout])
    missing = [name for name in (scales, scale) if name not in donor.meta]
    if missing:
        raise ReencodeError(f"the donor checkpoint is missing {', '.join(missing)}")
    if donor.dtype(scales) != "F8_E4M3":
        raise ReencodeError(f"{scales} must be F8_E4M3 block scales, not {donor.dtype(scales)}")
    code_shape, scale_shape = donor.meta[packed][1], donor.meta[scales][1]
    rows = code_shape[0] if code_shape else 0
    if rows < 1 or code_shape != (rows, columns // 2) or scale_shape != (rows, columns // 16):
        raise ReencodeError(f"{module}: codes {code_shape} and scales {scale_shape} are not an "
                            f"NVFP4 matrix of {columns} columns")
    if ranges is not None and not all(0 <= begin < end <= rows for begin, end in ranges):
        raise ReencodeError(f"{module}: its {rows} rows do not hold the rows taken from it "
                            f"(up to {max(end for _, end in ranges)})")
    taken = rows if ranges is None else sum(end - begin for begin, end in ranges)
    value = _positive_scalar(donor, scale)
    if layout == "compressed-tensors":
        return DonorMatrix(module, layout, (packed, scales, scale), rows, ranges, taken,
                           value.clone(), True)
    try:
        divisor, exact = reciprocal_divisor(value)
    except ValueError as error:
        raise ReencodeError(f"{scale}: {error}") from None
    return DonorMatrix(module, layout, (packed, scales, scale), rows, ranges, taken, divisor,
                       exact)


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


def _matrix_object(base: Artifact, object_id: str, logical: str) -> TensorObject:
    """One matrix of the base: NVFP4, or FP8 row-scaled for the roles the original artifact keeps
    in FP8 (converted here into NVFP4 with the donor's calibrated words)."""
    obj = base.object(object_id)
    if not isinstance(obj, TensorObject) or len(obj.shape) != 2 or \
            obj.format not in (NVFP4_FORMAT, FP8_ROW_FORMAT):
        raise ReencodeError(f"{logical}: object {object_id} is neither an NVFP4 nor an FP8 "
                            "row-scaled matrix in the base")
    return obj


def _tiled_object(base: Artifact, parameters: Sequence[str]) -> tuple[TensorObject, list[int]]:
    """The one base matrix that ``parameters`` tile end to end in this order, and their rows."""

    directory = base.directory
    obj = _matrix_object(base, _object_of(directory, parameters[0])[0], parameters[0])
    columns = obj.shape[1]
    elements = obj.shape[0] * columns
    untiled = ReencodeError(f"{obj.id}: {', '.join(parameters)} do not tile it in this order")
    rows, cursor = [], 0
    for parameter in parameters:
        object_id, begin, end = _object_of(directory, parameter)
        if begin < 0:
            begin, end = 0, elements
        if object_id != obj.id or begin != cursor or end <= begin or (end - begin) % columns:
            raise untiled
        rows.append((end - begin) // columns)
        cursor = end
    if cursor != elements:
        raise untiled
    return obj, rows


def check_base(base: Artifact) -> None:
    """The base must be an original Qwen3.8-27B NVFP4 artifact (not an earlier re-encode)."""

    name = base.directory.metadata.get("name")
    recipe = base.directory.provenance.get("recipe")
    if name != BASE_NAME or recipe != BASE_RECIPE:
        raise ReencodeError(f"base is {name}/{recipe}, expected {BASE_NAME}/{BASE_RECIPE}")
    if "reencode" in base.directory.provenance:
        raise ReencodeError("base is already re-encoded; start from the original artifact")


def _input_divisor(object_id: str, donor: SourceCheckpoint,
                   donors: Sequence[DonorMatrix]) -> torch.Tensor:
    """The input global scale that the fused donor matrices share, as ``graft_single_source``
    imports it: compressed-tensors stores the activation input divisor itself."""

    values = [_positive_scalar(donor, item.module + INPUT_GLOBAL_SCALE) for item in donors]
    if len({divisor_word(value) for value in values}) != 1:
        raise ReencodeError(f"{object_id}: fused donor matrices have different input global "
                            "scales")
    return values[0].clone()


def _target(base: Artifact, role: str, layer: int, parameters: tuple[str, ...],
            donor: SourceCheckpoint, weights: SourceCheckpoint | None, rounded: bool) -> Target:
    obj, leaf_rows = _tiled_object(base, parameters)
    shape = tuple(obj.shape)
    donors = []
    for parameter, rows in zip(parameters, leaf_rows):
        matrix = logical_matrix(parameter)
        item = check_donor_matrix(donor, matrix.source.removesuffix(".weight"), shape[1],
                                  matrix.rows)
        if item.rows != rows:
            raise ReencodeError(f"{obj.id}: donor matrices have {item.rows} rows for {parameter}, "
                                f"object has {rows}")
        donors.append(item)
    if len({divisor_word(item.divisor) for item in donors}) != 1:
        raise ReencodeError(f"{obj.id}: fused donor matrices have different tensor scales")
    for item in donors if weights is not None else ():
        name = item.module + ".weight"
        if weights.dtype(name) not in UNQUANTIZED_DTYPES:
            raise ReencodeError(f"{name}: expected unquantized weights in --weights, got "
                                f"{weights.dtype(name) or 'nothing'}")
        if weights.meta[name][1] != (item.matrix_rows, shape[1]):
            raise ReencodeError(f"{name}: --weights shape {weights.meta[name][1]} differs from "
                                f"the donor matrix {(item.matrix_rows, shape[1])}")
    input_divisor = _input_divisor(obj.id, donor, donors) if role in INPUT_ROLES else None
    return Target(obj.id, shape, layer, role, parameters, donor, tuple(donors), rounded,
                  obj.format, input_divisor)


def _layer_parameters(layer: int, leaves: Sequence[str]) -> tuple[str, ...]:
    return tuple(f"text/layers/{layer}/{leaf}" for leaf in leaves)


def _text_layers(base: Artifact) -> range:
    count = 0
    while f"text/layers/{count}/mlp/gate" in base.directory.bindings:
        count += 1
    return range(count)


def _input_role(base: Artifact, layer: int) -> tuple[str, tuple[str, ...]]:
    bound = [(role, leaves) for role, leaves in INPUT_ROLES.items()
             if _layer_parameters(layer, leaves[:1])[0] in base.directory.bindings]
    if len(bound) != 1:
        raise ReencodeError(f"text layer {layer} binds {len(bound)} input projections, not one")
    return bound[0]


def _require_every_mlp_layer(base: Artifact, layers: Sequence[int]) -> None:
    """A conversion moves the recipe to full-a or full-b, whose profiles bind every text MLP
    layer as NVFP4: no FP8 MLP layer of the base may stay out of the selection."""

    nvfp4 = set(nvfp4_layers(base))
    kept = [layer for layer in _text_layers(base) if layer not in nvfp4 and layer not in layers]
    if kept:
        raise ReencodeError(f"text MLP layers {kept} would stay FP8, but a converted role moves "
                            "the recipe to a profile that binds every MLP layer as NVFP4: select "
                            "them with --layers")


def plan_targets(base: Artifact, layers: Sequence[int], rounded: set[str],
                 donor: SourceCheckpoint, weights: SourceCheckpoint | None,
                 input_donor: SourceCheckpoint | None) -> list[Target]:
    """Targets of the selected text MLP layers, of every input projection when ``input_donor`` is
    given, and of the output head when a role converts, all checked before writing."""

    if bool({"gate", "up"} & rounded) and not {"gate", "up"} <= rounded:
        raise ReencodeError("gate and up share one object: round both or neither")
    targets = []
    for layer in layers:
        for role, leaves in MLP_ROLES.items():
            projection = leaves[0].removeprefix("mlp/")
            targets.append(_target(base, role, layer, _layer_parameters(layer, leaves), donor,
                                   weights, projection in rounded))
    if not targets:
        raise ReencodeError("no MLP layer selected")
    if input_donor is not None:
        for layer in _text_layers(base):
            role, leaves = _input_role(base, layer)
            targets.append(_target(base, role, layer, _layer_parameters(layer, leaves),
                                   input_donor, weights, False))
    if any(target.converts for target in targets):
        _require_every_mlp_layer(base, layers)
        # The full-a and full-b profiles bind the output head as NVFP4 too: convert it with the
        # donor's calibrated lm_head words (orca's head is the base model's, so the words
        # transpose as-is).
        targets.append(_target(base, HEAD_ROLE, -1, (OUTPUT_HEAD,), donor, weights, False))
    return targets


def output_recipe(targets: Sequence[Target]) -> str:
    """The recipe of the output: the base's without a conversion, full-b when input projections
    convert, full-a otherwise (``plan_targets`` converted every FP8 MLP layer and the head)."""

    converted = {target.role for target in targets if target.converts}
    if not converted:
        return BASE_RECIPE
    return QWEN3_8_27B_NVFP4_FULL_B if INPUT_ROLES.keys() & converted else QWEN3_8_27B_NVFP4_FULL_A


def nvfp4_layers(base: Artifact) -> list[int]:
    """Text MLP layers whose gate/up and down objects are NVFP4 in the base."""

    layers = []
    for layer in _text_layers(base):
        objects = [_object_of(base.directory, f"text/layers/{layer}/mlp/{leaf}")[0]
                   for leaf in ("gate", "down")]
        if all(base.object(object_id).format == NVFP4_FORMAT for object_id in objects):
            layers.append(layer)
    return layers


def _neighbor_divisor(base: Artifact, layer: int, leaf: str) -> tuple[float, str]:
    """Input divisor of the same MLP leaf in the nearest calibrated NVFP4 layer below."""

    for candidate in range(layer - 1, -1, -1):
        name = f"text/layers/{candidate}/{leaf}"
        for use in base.directory.uses:
            aux = use.get("auxiliaries", {}).get(DIVISOR_ROLE)
            if use["parameter"] == name and aux is not None:
                (value,) = struct.unpack("<f", base.read_object(aux["object"]))
                return value, f"copied from {name}"
    raise ReencodeError(f"text/layers/{layer}/{leaf}: no calibrated NVFP4 neighbor for the "
                        "input divisor")


def _activation_divisor(base: Artifact, target: Target, parameter: str) -> tuple[float, str]:
    if target.role in INPUT_ROLES:
        return float(target.input_divisor), "input_global_scale of the input donor"
    if target.role == HEAD_ROLE:
        # As in full-a: the sm_70 routes run NVFP4 with 16-bit activations and never read it,
        # and the donor's head input scale (a ModelOpt input_scale) is not imported.
        return 1.0, "unit placeholder"
    return _neighbor_divisor(base, target.layer,
                             parameter.removeprefix(f"text/layers/{target.layer}/"))


@dataclass(frozen=True, slots=True)
class UsesPlan:
    """The Uses of the output and the auxiliary divisor objects they add."""

    uses: list[dict]
    divisors: dict[str, float]  # new auxiliary object id -> divisor
    labels: dict[str, str]  # new auxiliary object id -> "parameter: where the divisor comes from"


def plan_uses(base: Artifact, targets: Sequence[Target]) -> UsesPlan:
    """The Uses of the output and the auxiliary divisor objects they add.

    Every Use of a converted leaf takes ``AllowA4`` and an activation input divisor auxiliary, as
    the full-a and full-b profiles require of NVFP4 leaves. A Use that names a divisor keeps it;
    each leaf with Uses that do not gets one new auxiliary object of its own. The divisor is the
    input donor's input global scale for an input projection, the one of the same MLP leaf in the
    nearest NVFP4 layer below for an MLP leaf, and 1.0 for the output head."""

    converted = {parameter: target for target in targets if target.converts
                 for parameter in target.parameters}
    numbers = [int(obj.id.split("/")[1]) for obj in base.objects
               if obj.id.startswith("auxiliary/") and obj.id.split("/")[1].isdigit()]
    # Imported divisors are numbered last, so the auxiliaries of the full-a roles keep the ids
    # that a full-a conversion of the same base gives them.
    parameters = sorted(dict.fromkeys(use["parameter"] for use in base.directory.uses
                                      if use["parameter"] in converted
                                      and DIVISOR_ROLE not in use.get("auxiliaries", {})),
                        key=lambda parameter: converted[parameter].role in INPUT_ROLES)
    aux_ids: dict[str, str] = {}
    divisors: dict[str, float] = {}
    labels: dict[str, str] = {}
    for number, parameter in enumerate(parameters, start=max(numbers, default=-1) + 1):
        aux_id = f"auxiliary/{number:06d}"
        value, label = _activation_divisor(base, converted[parameter], parameter)
        aux_ids[parameter] = aux_id
        divisors[aux_id] = value
        labels[aux_id] = f"{parameter}: {label}"
    uses = []
    for use in base.directory.uses:
        use = copy.deepcopy(use)
        if use["parameter"] in converted:
            use["activation_policy"] = "AllowA4"
            if DIVISOR_ROLE not in use.get("auxiliaries", {}):
                aux_id = aux_ids[use["parameter"]]
                use.setdefault("auxiliaries", {})[DIVISOR_ROLE] = {"object": aux_id}
        uses.append(use)
    return UsesPlan(uses, divisors, labels)
