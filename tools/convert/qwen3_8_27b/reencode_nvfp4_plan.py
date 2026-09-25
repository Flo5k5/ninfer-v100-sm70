"""Objects that ``reencode_nvfp4`` rewrites, checked before any output is created.

A target is one NVFP4 MLP object of the base (fused gate/up, or down) with the donor matrices that
re-encode it in row order. Planning reads only the base directory, the donor's and ``--weights``'
safetensors headers and the donors' FP32 tensor scales, and refuses what the encoding pass could
only discover while writing: missing tensors, dtypes or shapes that are not the object's, gate and
up tensor scales that differ, and tensor scales without a finite positive divisor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch

from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorObject
from tools.convert.qwen3_8_27b.graft_single_source import SourceCheckpoint
from tools.convert.qwen3_8_27b.reencode_nvfp4_numeric import divisor_word, reciprocal_divisor


NVFP4_FORMAT = "nvfp4"
BASE_NAME = "qwen3.8-27b"
BASE_RECIPE = "qwen3_8_27b_nvfp4"
UNQUANTIZED_DTYPES = ("BF16", "F16", "F32")
# Tensor suffixes per donor layout: packed codes, E4M3 block scales, FP32 tensor scale.
DONOR_TENSORS = {
    "compressed-tensors": (".weight_packed", ".weight_scale", ".weight_global_scale"),
    "modelopt": (".weight", ".weight_scale", ".weight_scale_2"),
}


class ReencodeError(ValueError):
    pass


FP8_ROW_FORMAT = "fp8_e4m3fn_row_bf16"
# Layout of the NVFP4 text-MLP objects; the FP8 rows of an original artifact carry row_scale_v1.
NVFP4_LAYOUT = "block_scale_k16_m128x4_v1"


@dataclass(frozen=True, slots=True)
class DonorMatrix:
    """One donor matrix checked from its safetensors header, and the divisor of its tensor scale."""

    module: str
    layout: str
    tensors: tuple[str, str, str]  # packed codes, block scales, tensor scale
    rows: int
    divisor: torch.Tensor  # FP32 scalar stored in the artifact
    exact: bool  # the divisor reproduces the donor's tensor scale exactly


@dataclass(frozen=True, slots=True)
class Target:
    """One NVFP4 object of the base and the donor matrices that re-encode it, in row order."""

    object_id: str
    shape: tuple[int, int]
    layer: int
    parameters: tuple[str, ...]
    donors: tuple[DonorMatrix, ...]
    rounded: bool
    # Format of the base object this target replaces: NVFP4 for a re-encode of words already in
    # NVFP4, the FP8 row format for a layer the original artifact keeps in FP8 (converted here).
    base_format: str = NVFP4_FORMAT

    @property
    def converts(self) -> bool:
        return self.base_format != NVFP4_FORMAT

    @property
    def divisor(self) -> torch.Tensor:
        return self.donors[0].divisor


def check_donor_matrix(donor: SourceCheckpoint, module: str, columns: int) -> DonorMatrix:
    """Check one donor matrix from its header and read the divisor of its tensor scale."""

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
    if donor.dtype(scale) != "F32" or math.prod(donor.meta[scale][1]) != 1:
        raise ReencodeError(f"{scale} must be one FP32 value")
    value = donor.get(scale).reshape(())
    if not bool(torch.isfinite(value)) or float(value) <= 0.0:
        raise ReencodeError(f"{scale} = {float(value)!r} must be finite and positive")
    if layout == "compressed-tensors":
        return DonorMatrix(module, layout, (packed, scales, scale), rows, value.clone(), True)
    try:
        divisor, exact = reciprocal_divisor(value)
    except ValueError as error:
        raise ReencodeError(f"{scale}: {error}") from None
    return DonorMatrix(module, layout, (packed, scales, scale), rows, divisor, exact)


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


def _mlp_object(base: Artifact, object_id: str, logical: str) -> TensorObject:
    """One text-MLP matrix of the base: NVFP4, or FP8 row-scaled for the layers the original
    artifact keeps in FP8 (re-encoded here into NVFP4 with the donor's calibrated words)."""
    obj = base.object(object_id)
    if not isinstance(obj, TensorObject) or len(obj.shape) != 2 or \
            obj.format not in (NVFP4_FORMAT, FP8_ROW_FORMAT):
        raise ReencodeError(f"{logical}: object {object_id} is neither an NVFP4 nor an FP8 "
                            "row-scaled matrix in the base")
    return obj


def check_base(base: Artifact) -> None:
    """The base must be an original Qwen3.8-27B NVFP4 artifact (not an earlier re-encode)."""

    name = base.directory.metadata.get("name")
    recipe = base.directory.provenance.get("recipe")
    if name != BASE_NAME or recipe != BASE_RECIPE:
        raise ReencodeError(f"base is {name}/{recipe}, expected {BASE_NAME}/{BASE_RECIPE}")
    if "reencode" in base.directory.provenance:
        raise ReencodeError("base is already re-encoded; start from the original artifact")


def _target(obj: TensorObject, layer: int, parameters: list[str], modules: list[str],
            rounded: bool, donor: SourceCheckpoint, weights: SourceCheckpoint | None) -> Target:
    shape = tuple(obj.shape)
    donors = tuple(check_donor_matrix(donor, module, shape[1]) for module in modules)
    if len({divisor_word(item.divisor) for item in donors}) != 1:
        raise ReencodeError(f"{obj.id}: fused donor matrices have different tensor scales")
    rows = sum(item.rows for item in donors)
    if rows != shape[0]:
        raise ReencodeError(f"{obj.id}: donor matrices have {rows} rows, object has {shape[0]}")
    for item in donors if weights is not None else ():
        name = item.module + ".weight"
        if weights.dtype(name) not in UNQUANTIZED_DTYPES:
            raise ReencodeError(f"{name}: expected unquantized weights in --weights, got "
                                f"{weights.dtype(name) or 'nothing'}")
        if weights.meta[name][1] != (item.rows, shape[1]):
            raise ReencodeError(f"{name}: --weights shape {weights.meta[name][1]} differs from "
                                f"the donor matrix {(item.rows, shape[1])}")
    return Target(obj.id, shape, layer, tuple(parameters), donors, rounded, obj.format)


def plan_targets(base: Artifact, layers: Iterable[int], rounded: set[str],
                 donor: SourceCheckpoint, weights: SourceCheckpoint | None) -> list[Target]:
    """Fused gate/up and down objects of the selected text MLP layers, checked before writing."""

    if bool({"gate", "up"} & rounded) and not {"gate", "up"} <= rounded:
        raise ReencodeError("gate and up share one object: round both or neither")
    directory = base.directory
    targets = []
    for layer in layers:
        prefix = f"text/layers/{layer}/mlp/"
        module = f"model.language_model.layers.{layer}.mlp."
        gate_id, gate_begin, gate_end = _object_of(directory, prefix + "gate")
        up_id, up_begin, up_end = _object_of(directory, prefix + "up")
        gate_up = _mlp_object(base, gate_id, prefix + "gate")
        elements = gate_up.shape[0] * gate_up.shape[1]
        if up_id != gate_id or gate_begin != 0 or up_begin != gate_end or up_end != elements:
            raise ReencodeError(f"{prefix}gate_up: gate and up are not one fused object")
        targets.append(_target(gate_up, layer, [prefix + "gate", prefix + "up"],
                               [module + "gate_proj", module + "up_proj"], "gate" in rounded,
                               donor, weights))
        down = _mlp_object(base, _object_of(directory, prefix + "down")[0], prefix + "down")
        targets.append(_target(down, layer, [prefix + "down"], [module + "down_proj"],
                               "down" in rounded, donor, weights))
    if not targets:
        raise ReencodeError("no MLP layer selected")
    if any(target.converts for target in targets):
        # The full-a profile binds the output head as NVFP4 too (stage A): convert it with the
        # donor's calibrated lm_head words — orca's head is the base model's, so the words
        # transpose as-is.
        head_object = _mlp_object(base, _object_of(directory, "text/output_head")[0],
                                  "text/output_head")
        targets.append(_target(head_object, -1, ("text/output_head",),
                               ("lm_head",), False, donor, weights))
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
