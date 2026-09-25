"""reencode_nvfp4 on the attention and GDN input projections: the full-b recipe in one pass.

The miniature base adds a GDN input projection (layer 0) and an attention input projection
(layer 1) to the MLP fixture of test_reencode_nvfp4, with the real Qwen3.8-27B row counts and 64
columns. The input donor holds them as a compressed-tensors checkpoint stores them: q_proj with
the query and gate rows of each head interleaved, in_proj_qkv then in_proj_z.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import struct

import pytest
import torch

from tools.artifact.codecs.fp8_row import encode_fp8_row_scaled
from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorObject, TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.convert.quantization.fp8_row import quantize_bf16_rows
from tools.convert.qwen3_8_27b import reencode_nvfp4

from .test_reencode_nvfp4 import (
    HIDDEN,
    NVFP4_LAYOUT,
    Fixture,
    _arguments,
    _assert_nothing_written,
    _block_words,
    _build,
    _nearest_codes,
    _pack,
    _report,
    _save,
    _stored_sha256,
    _steps,
    _word,
    _words,
)


INPUT_LABEL = "org/qat-NVFP4"
HEADS, HEAD_ROWS = 24, 256
# Leaves of each input projection in the object's row order, with their rows.
LEAVES = {
    "attention": (("query", 6144), ("key", 1024), ("gate", 6144), ("value", 1024)),
    "gdn": (("query", 2048), ("key", 2048), ("value", 6144), ("z", 6144)),
}
# The attention key and value rows have 0.4 times the norm of the others: swapped, they move so
# little of their object's norm that the whole-object error stays under --max-error.
SMALL_LEAVES = {("attention", "key"), ("attention", "value")}
# (weight_global_scale, input_global_scale) shared by the donor matrices of each projection.
SCALES = {"attention": (5120.5, 0.10498046875), "gdn": (6144.25, 0.1171875)}
ROLES = {0: "gdn", 1: "attention"}
OBJECTS = {"gdn": "weight/gdn0", "attention": "weight/attention1"}
# Checkpoints that store an input projection in a wrong row order, and what each misplaces.
MISPLACED = {
    "naive q_proj": ("attention", "query|gate"),  # all query rows, then all gate rows
    "swapped query and key": ("gdn", "query|key"),  # in_proj_qkv as key, query, value
    "swapped key and value": ("attention", "key|value"),  # k_proj holds v_proj's rows
}
FULL_A, FULL_B = "qwen3_8_27b_nvfp4-full-a", "qwen3_8_27b_nvfp4-full-b"


@dataclass
class InputFixture:
    mlp: Fixture  # the MLP fixture, its base replaced by the base with input projections
    donor_dir: Path
    donor: dict[str, torch.Tensor]
    expected: dict[str, tuple[torch.Tensor, torch.Tensor]]  # object -> codes, scale words


def _hf_matrices(kind: str, leaves: dict[str, torch.Tensor],
                 misplaced: str | None) -> dict[str, torch.Tensor]:
    """One layer's input projection as the checkpoint stores it, or as ``misplaced`` stores it."""

    if kind == "attention":
        if misplaced == "naive q_proj":
            q_proj = torch.cat([leaves["query"], leaves["gate"]])
        else:
            q_proj = torch.stack([leaves["query"].reshape(HEADS, HEAD_ROWS, -1),
                                  leaves["gate"].reshape(HEADS, HEAD_ROWS, -1)], dim=1)
        key, value = ("value", "key") if misplaced == "swapped key and value" else ("key", "value")
        return {"self_attn.q_proj": q_proj.reshape(2 * HEADS * HEAD_ROWS, -1),
                "self_attn.k_proj": leaves[key], "self_attn.v_proj": leaves[value]}
    first, second = ("key", "query") if misplaced == "swapped query and key" else ("query", "key")
    return {"linear_attn.in_proj_qkv": torch.cat([leaves[first], leaves[second],
                                                   leaves["value"]]),
            "linear_attn.in_proj_z": leaves["z"]}


def _object_rows(kind: str, matrices: dict[str, torch.Tensor]) -> torch.Tensor:
    """Oracle of the row mapping: the fused object's rows from the checkpoint matrices."""

    if kind == "attention":
        heads = matrices["self_attn.q_proj"].reshape(HEADS, 2, HEAD_ROWS, -1)
        return torch.cat([heads[:, 0].reshape(HEADS * HEAD_ROWS, -1), matrices["self_attn.k_proj"],
                          heads[:, 1].reshape(HEADS * HEAD_ROWS, -1), matrices["self_attn.v_proj"]])
    return torch.cat([matrices["linear_attn.in_proj_qkv"], matrices["linear_attn.in_proj_z"]])


def specs_of(artifact: Artifact, shapes: dict[str, tuple[int, ...]] | None = None) -> list:
    """Specs of an artifact's objects, with some tensor shapes replaced."""

    return [TensorSpec(obj.id, tuple((shapes or {}).get(obj.id, obj.shape)), obj.format,
                       obj.layout) if isinstance(obj, TensorObject)
            else ResourceSpec(obj.id, obj.bytes, obj.encoding) for obj in artifact.objects]


def _build_inputs(tmp_path, *, roles=None, misplaced: str | None = None,
                  order: tuple[str, ...] | None = None, short: str | None = None) -> InputFixture:
    """The MLP fixture (layer 1's down in FP8) plus FP8 input projections placed first, as in
    the real artifact, their Uses first too, and the compressed-tensors input donor.

    ``order`` stores and binds the attention leaves in another row order; ``short`` (a leaf such
    as ``gdn/z``) leaves its binding's last row unbound."""

    fixture = _build(tmp_path, layer1_down_format="fp8")
    generator = torch.Generator().manual_seed(11)
    with Artifact(fixture.base) as base:
        directory = base.directory
        specs = specs_of(base)
        payloads = {obj.id: base.read_object(obj.id) for obj in base.objects}
        bindings, uses = dict(directory.bindings), list(directory.uses)
    donor, expected, input_specs, input_uses = {}, {}, [], []
    for layer, kind in (ROLES if roles is None else roles).items():
        object_id = OBJECTS[kind]
        leaves = {name: (torch.randn(rows, HIDDEN, generator=generator) * 0.02
                         * (0.4 if (kind, name) in SMALL_LEAVES else 1.0)).to(torch.bfloat16)
                  for name, rows in LEAVES[kind]}
        stored = order if kind == "attention" and order else [name for name, _ in LEAVES[kind]]
        values = torch.cat([leaves[name] for name in stored])
        fp8 = quantize_bf16_rows(values)
        payloads[object_id] = encode_fp8_row_scaled(fp8.codes, fp8.scales, tuple(values.shape))
        input_specs.append(TensorSpec(object_id, tuple(values.shape), "fp8_e4m3fn_row_bf16",
                                      "row_scale_v1"))
        cursor = 0
        for name in stored:
            parameter = f"text/layers/{layer}/{kind}/{name}"
            end = cursor + leaves[name].shape[0] * HIDDEN
            bindings[parameter] = {"parts": [{"object": object_id, "range": [
                cursor, end - HIDDEN if short == f"{kind}/{name}" else end]}]}
            cursor = end
        for name, _ in LEAVES[kind]:
            input_uses.append({"parameter": f"text/layers/{layer}/{kind}/{name}",
                               "input": f"text/layers/{layer}/mixer_input",
                               "activation_policy": "AllowA8"})
        weight_scale, input_scale = SCALES[kind]
        codes, words = {}, {}
        wrong = misplaced if misplaced and MISPLACED[misplaced][0] == kind else None
        for suffix, matrix in _hf_matrices(kind, leaves, wrong).items():
            module = f"model.language_model.layers.{layer}.{suffix}"
            shift = torch.randint(-1, 5, (matrix.shape[0], HIDDEN // 16), generator=generator,
                                  dtype=torch.int16)
            words[suffix] = _block_words(matrix.float(), weight_scale, shift)
            codes[suffix] = _nearest_codes(matrix.float(), _steps(words[suffix],
                                                                  divisor=weight_scale))
            donor[module + ".weight_packed"] = _pack(codes[suffix])
            donor[module + ".weight_scale"] = words[suffix].view(torch.float8_e4m3fn)
            donor[module + ".weight_global_scale"] = torch.tensor([weight_scale])
            donor[module + ".input_global_scale"] = torch.tensor([input_scale])
            fixture.weights[module + ".weight"] = matrix.contiguous()
        expected[object_id] = (_object_rows(kind, codes), _object_rows(kind, words))
    path = tmp_path / "base-inputs.ninfer"
    with Artifact(fixture.base) as base:
        writer = ArtifactWriter(path, input_specs + specs, components=base.directory.components,
                                bindings=bindings, uses=input_uses + uses,
                                metadata=base.directory.metadata,
                                provenance=base.directory.provenance)
    for object_id, payload in payloads.items():
        writer.write_object(object_id, payload)
    writer.finish()
    donor_dir = tmp_path / "input-donor"
    donor_dir.mkdir()
    _save(donor_dir, donor)
    _save(fixture.weights_dir, fixture.weights)
    return InputFixture(replace(fixture, base=path), donor_dir, donor, expected)


def full_b(fixture: InputFixture, out_path: Path, *extra: str, weights: bool = True) -> list[str]:
    rounding = ("--round", "down") if weights else ()
    return _arguments(fixture.mlp, out_path, "--layers", "0-1", *rounding,
                      "--input-donor", str(fixture.donor_dir), "--input-donor-label", INPUT_LABEL,
                      *extra, weights=weights)


def _divisor_uses(out: Artifact) -> dict[str, tuple[str, str, int]]:
    """Parameter -> (policy, divisor object, divisor FP32 word) for the Uses with a divisor."""

    result = {}
    for use in out.directory.uses:
        aux = use.get("auxiliaries", {}).get("activation_input_divisor")
        if aux is not None:
            (word,) = struct.unpack("<I", out.read_object(aux["object"]))
            result[use["parameter"]] = (use["activation_policy"], aux["object"], word)
    return result


def test_full_b_converts_the_input_projections_with_the_input_donor_words(tmp_path,
                                                                           monkeypatch) -> None:
    # Row chunks that divide neither a head's 256 rows nor a leaf: chunks cross both boundaries.
    monkeypatch.setattr(reencode_nvfp4, "ROW_CHUNK", 100)
    fixture = _build_inputs(tmp_path)
    out_path = tmp_path / "out.ninfer"
    assert reencode_nvfp4.main(full_b(fixture, out_path, "--verify")) == 0
    report = _report(tmp_path)
    assert report["verified"] is True and report["recipe"] == FULL_B
    with Artifact(out_path) as out:
        record = out.directory.provenance["reencode"]
        assert out.directory.provenance["recipe"] == FULL_B
        assert record["base_recipe"] == "qwen3_8_27b_nvfp4"
        assert record["input_donor"] == {"label": INPUT_LABEL, "layers": [0, 1]}
        assert record["mlp_layers"] == [0, 1] and record["output_head"] is True
        divisors = _divisor_uses(out)
        for object_id in OBJECTS.values():
            assert (out.object(object_id).format, out.object(object_id).layout) == (
                "nvfp4", NVFP4_LAYOUT)
        labels = record["activation_input_divisors"]
    entries = {item["object"]: item for item in report["objects"]}
    stored = _stored_sha256(fixture.mlp.weights_dir)
    for layer, kind in ROLES.items():
        object_id = OBJECTS[kind]
        rows = sum(count for _, count in LEAVES[kind])
        codes, scales, divisor, _ = _words(out_path, object_id, (rows, HIDDEN))
        # The donor's codes and block scales, as they are, in the object's row order.
        assert torch.equal(codes, fixture.expected[object_id][0])
        assert torch.equal(scales, fixture.expected[object_id][1])
        weight_scale, input_scale = SCALES[kind]
        assert _word(float(divisor)) == _word(weight_scale)
        # Every leaf Use: AllowA4 and its own auxiliary holding the donor's input global scale.
        parameters = [f"text/layers/{layer}/{kind}/{name}" for name, _ in LEAVES[kind]]
        leaf_divisors = [divisors[parameter] for parameter in parameters]
        assert {(policy, word) for policy, _, word in leaf_divisors} == {
            ("AllowA4", _word(input_scale))}
        assert len({aux for _, aux, _ in leaf_divisors}) == len(parameters)
        assert {labels[aux] for _, aux, _ in leaf_divisors} == {
            f"{parameter}: input_global_scale of the input donor" for parameter in parameters}
        entry = entries[object_id]
        assert entry["role"] == f"{kind}/input" and entry["codes"] == "donor"
        assert list(entry["relative_rms_error_vs_base_by_parameter"]) == parameters
        assert max(entry["relative_rms_error_vs_base_by_parameter"].values()) < 0.3
        assert entry["relative_rms_error_vs_weights"] < 0.2
        assert entry["codes_equal_to_rounded_weights"] == 1.0
        module = f"model.language_model.layers.{layer}."
        assert all(name.startswith(module) for name in entry["donor_sha256"])
        assert {name for name in entry["donor_sha256"] if name.endswith(".input_global_scale")}
    # --weights digests follow the rows read: whole matrices read in order keep their stored
    # digest; q_proj is read as its query rows, then its gate rows.
    gdn = entries[OBJECTS["gdn"]]["weights_sha256"]
    assert gdn == {name: stored[name] for name in gdn}
    q_proj = fixture.mlp.weights["model.language_model.layers.1.self_attn.q_proj.weight"]
    heads = q_proj.reshape(HEADS, 2, HEAD_ROWS, HIDDEN)
    order = torch.cat([heads[:, 0], heads[:, 1]]).contiguous()
    assert entries[OBJECTS["attention"]]["weights_sha256"][
        "model.language_model.layers.1.self_attn.q_proj.weight"] == hashlib.sha256(
        order.view(torch.uint8).numpy().tobytes()).hexdigest()


def test_full_b_is_full_a_plus_the_input_projections(tmp_path) -> None:
    """One pass with both donors gives the full-a conversion of the same base, object for object
    and auxiliary id for auxiliary id, plus the converted input projections."""

    fixture = _build_inputs(tmp_path)
    full_a_path, full_b_path = tmp_path / "full-a.ninfer", tmp_path / "full-b.ninfer"
    assert reencode_nvfp4.main(_arguments(fixture.mlp, full_a_path, "--layers", "0-1", "--round",
                                          "down", weights=True)) == 0
    assert reencode_nvfp4.main(full_b(fixture, full_b_path)) == 0
    inputs = set(OBJECTS.values())
    with Artifact(full_a_path) as full_a, Artifact(full_b_path) as full_b_out:
        assert full_a.directory.provenance["recipe"] == FULL_A
        assert "input_donor" not in full_a.directory.provenance["reencode"]
        a_ids, b_ids = [obj.id for obj in full_a.objects], [obj.id for obj in full_b_out.objects]
        assert b_ids[:len(a_ids)] == a_ids
        for object_id in a_ids:
            if object_id not in inputs:
                assert full_b_out.read_object(object_id) == full_a.read_object(object_id)
        a_divisors, b_divisors = _divisor_uses(full_a), _divisor_uses(full_b_out)
        assert {parameter: b_divisors[parameter] for parameter in a_divisors} == a_divisors
        new = {aux for parameter, (_, aux, _) in b_divisors.items() if parameter not in a_divisors}
        assert set(b_ids[len(a_ids):]) == new and len(new) == 8
        # The output head keeps full-a's unit divisor; the input leaves never take it.
        assert a_divisors["text/output_head"][2] == _word(1.0)
        assert all(word != _word(1.0) for parameter, (_, _, word) in b_divisors.items()
                   if parameter not in a_divisors)
        a_uses = {(use["parameter"], use["input"]): use for use in full_a.directory.uses}
        for use in full_b_out.directory.uses:
            if "/attention/" not in use["parameter"] and "/gdn/" not in use["parameter"]:
                assert a_uses[(use["parameter"], use["input"])] == use


@pytest.mark.parametrize("weights", [True, False])
@pytest.mark.parametrize("misplaced", list(MISPLACED))
def test_rows_in_a_wrong_order_are_refused_against_the_base(tmp_path, misplaced, weights) -> None:
    """vs_base blocking regression: rows taken in a wrong order match --weights stored the same
    way, so only the base comparison can refuse them; it was reported but not enforced for
    converted objects, and is now made per parameter."""

    fixture = _build_inputs(tmp_path, misplaced=misplaced)
    kind, leaves = MISPLACED[misplaced]
    layer = next(layer for layer, role in ROLES.items() if role == kind)
    with pytest.raises(reencode_nvfp4.ReencodeError,
                       match=rf"{OBJECTS[kind]}: relative RMS error 1\.\d+ against the base "
                             rf"object's values of text/layers/{layer}/{kind}/({leaves}) exceeds "
                             "--max-error 0.3"):
        reencode_nvfp4.main(full_b(fixture, tmp_path / "out.ninfer", weights=weights))
    _assert_nothing_written(tmp_path)


def test_a_small_misplaced_parameter_is_not_diluted_in_its_object(tmp_path) -> None:
    """The key and value rows are 1/7 of the attention object and have 0.4 times the norm of
    the others: swapped, the whole object stays under 0.3, the swapped parameters do not."""

    fixture = _build_inputs(tmp_path, misplaced="swapped key and value")
    assert reencode_nvfp4.main(full_b(fixture, tmp_path / "out.ninfer", "--max-error", "5")) == 0
    entry = next(item for item in _report(tmp_path)["objects"]
                 if item["object"] == OBJECTS["attention"])
    by_parameter = entry["relative_rms_error_vs_base_by_parameter"]
    assert 0.2 < entry["relative_rms_error_vs_base"] < 0.3
    assert min(by_parameter["text/layers/1/attention/key"],
               by_parameter["text/layers/1/attention/value"]) > 1.0


def test_provenance_records_each_role_s_layers(tmp_path) -> None:
    """--layers 1: layer 0's MLP is NVFP4 and keeps its words, the input projections of both
    layers convert, and the record keeps the two lists apart."""

    fixture = _build_inputs(tmp_path)
    arguments = full_b(fixture, tmp_path / "out.ninfer")
    arguments[arguments.index("--layers") + 1] = "1"
    assert reencode_nvfp4.main(arguments) == 0
    with Artifact(tmp_path / "out.ninfer") as out:
        record = out.directory.provenance["reencode"]
        assert out.directory.provenance["recipe"] == FULL_B
    assert record["mlp_layers"] == [1] and record["output_head"] is True
    assert record["input_donor"] == {"label": INPUT_LABEL, "layers": [0, 1]}


def _edit_input_donor(name: str, value):
    def mutate(fixture: InputFixture) -> None:
        if value is None:
            del fixture.donor[name]
        else:
            fixture.donor[name] = value(fixture.donor[name])
        _save(fixture.donor_dir, fixture.donor)
    return mutate


def _cut_rows(module: str, rows: int):
    def mutate(fixture: InputFixture) -> None:
        for suffix in (".weight_packed", ".weight_scale"):
            fixture.donor[module + suffix] = fixture.donor[module + suffix][:rows].contiguous()
        _save(fixture.donor_dir, fixture.donor)
    return mutate


def _grow_rows(module: str, rows: int):
    def mutate(fixture: InputFixture) -> None:
        for suffix in (".weight_packed", ".weight_scale"):
            tensor = fixture.donor[module + suffix]
            fixture.donor[module + suffix] = tensor.repeat(rows // tensor.shape[0], 1)
        _save(fixture.donor_dir, fixture.donor)
    return mutate


ATTENTION = "model.language_model.layers.1.self_attn."
GDN = "model.language_model.layers.0.linear_attn."
UNTILED = "do not tile it in this order"


@pytest.mark.parametrize(
    "build, mutate, extra, match",
    [
        ({}, _edit_input_donor(ATTENTION + "k_proj.weight_global_scale",
                               lambda t: t * 2), (),
         "weight/attention1: fused donor matrices have different tensor scales"),
        ({}, _edit_input_donor(GDN + "in_proj_z.input_global_scale", lambda t: t * 2), (),
         "weight/gdn0: fused donor matrices have different input global scales"),
        ({}, _edit_input_donor(ATTENTION + "v_proj.input_global_scale", None), (),
         "the donor checkpoint is missing .*v_proj.input_global_scale$"),
        *[({}, _edit_input_donor(ATTENTION + "q_proj.input_global_scale",
                                 lambda t, v=v: torch.tensor([v])), (),
           r"q_proj.input_global_scale = .* must be finite and positive")
          for v in (0.0, float("nan"))],
        ({}, _edit_input_donor(GDN + "in_proj_qkv.input_global_scale", lambda t: t.double()), (),
         "in_proj_qkv.input_global_scale must be one FP32 value"),
        ({}, _cut_rows(ATTENTION + "q_proj", 6144), (),
         r"q_proj: its 6144 rows do not hold the rows taken from it \(up to 12032\)"),
        ({}, _cut_rows(GDN + "in_proj_qkv", 8192), (),
         r"in_proj_qkv: its 8192 rows do not hold the rows taken from it \(up to 10240\)"),
        ({}, _grow_rows(ATTENTION + "k_proj", 2048), (),
         "weight/attention1: donor matrices have 2048 rows for text/layers/1/attention/key, "
         "object has 1024"),
        # The base stores and binds attention as [query | gate | key | value].
        ({"order": ("query", "gate", "key", "value")}, None, (),
         f"weight/attention1: .* {UNTILED}"),
        # A one-row gap between the query and key bindings, then after the last binding.
        ({"short": "gdn/query"}, None, (), f"weight/gdn0: .* {UNTILED}"),
        ({"short": "gdn/z"}, None, (), f"weight/gdn0: .* {UNTILED}"),
        # Layer 1 keeps its FP8 down projection: full-b binds every MLP layer as NVFP4.
        ({}, None, ("--layers", "0"), r"text MLP layers \[1\] would stay FP8"),
        ({"roles": {1: "attention"}}, None, (), "text layer 0 binds 0 input projections, not one"),
    ],
)
def test_input_projection_refusals_before_writing(tmp_path, monkeypatch, build, mutate, extra,
                                                  match) -> None:
    def no_writer(*arguments, **keywords):
        raise AssertionError("the output was created before the inputs were checked")

    fixture = _build_inputs(tmp_path, **build)
    if mutate is not None:
        mutate(fixture)
    monkeypatch.setattr(reencode_nvfp4, "ArtifactWriter", no_writer)
    arguments = full_b(fixture, tmp_path / "out.ninfer")
    if extra:
        arguments[arguments.index("--layers") + 1] = extra[1]
    with pytest.raises(reencode_nvfp4.ReencodeError, match=match):
        reencode_nvfp4.main(arguments)


@pytest.mark.parametrize(
    "options",
    [
        ("--input-donor", "input-donor"),
        ("--input-donor-label", INPUT_LABEL),
        ("--input-donor", "input-donor", "--input-donor-label", "~/models/qat"),
    ],
)
def test_input_donor_arguments(tmp_path, options) -> None:
    fixture = _build_inputs(tmp_path)
    resolved = [str(tmp_path / item) if item == "input-donor" else item for item in options]
    with pytest.raises(SystemExit):
        reencode_nvfp4.main(_arguments(fixture.mlp, tmp_path / "out.ninfer", "--layers", "0-1",
                                       *resolved))
    _assert_nothing_written(tmp_path)
