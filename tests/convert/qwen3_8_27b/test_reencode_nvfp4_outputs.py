"""reencode_nvfp4 on the attention and GDN output projections: the full-c recipe, per-role donors.

The miniature base adds one FP8 output projection per text layer (GDN in layer 0, attention in
layer 1) to the input-projection fixture of test_reencode_nvfp4_inputs, with the real 6144 columns
and 128 rows (NVFP4 objects need a multiple of 128 rows). Each output Use follows its layer's input
Uses, as in the real artifact, so a divisor numbering that ranked outputs with inputs would move
full-b's ids. The QAT donor holds the MLP, input and output words in compressed-tensors, as the
QAT checkpoint does; the head donor holds only ModelOpt lm_head words, the head that the QAT
checkpoint does not quantize.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch

from tools.artifact.codecs.fp8_row import encode_fp8_row_scaled
from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.convert.quantization.fp8_row import quantize_bf16_rows
from tools.convert.qwen3_8_27b import reencode_nvfp4, reencode_nvfp4_encode
from tools.convert.qwen3_8_27b.reencode_nvfp4_plan import ReencodeError

from .test_reencode_nvfp4 import (
    DONOR_LABEL,
    GLOBAL_SCALES,
    HIDDEN,
    INTERMEDIATE,
    NVFP4_LAYOUT,
    VOCAB,
    WEIGHTS_LABEL,
    _assert_nothing_written,
    _block_words,
    _nearest_codes,
    _pack,
    _report,
    _save,
    _steps,
    _unpack,
    _word,
    _words,
)
from .test_reencode_nvfp4_inputs import (
    FULL_B,
    INPUT_LABEL,
    LEAVES,
    OBJECTS,
    ROLES,
    InputFixture,
    _build_inputs,
    _divisor_uses,
    specs_of,
)
from .test_reencode_nvfp4_verify import DIVISORS, USES, _rewrite


FULL_C = "qwen3_8_27b_nvfp4-full-c"
OUTPUT_SHAPE = (128, 6144)
OUTPUT_OBJECTS = {"gdn": "weight/gdn0-output", "attention": "weight/attention1-output"}
OUTPUT_MODULES = {"gdn": "linear_attn.out_proj", "attention": "self_attn.o_proj"}
# (weight_global_scale, input_global_scale) of each output donor matrix.
OUTPUT_SCALES = {"gdn": (1988.75, 263.75), "attention": (1869.875, 197.25)}
# Tensor name fragments of each donor role.
ROLE_TENSORS = {
    "mlp": (".mlp.",),
    "head": ("lm_head.",),
    "attention_input": (".q_proj.", ".k_proj.", ".v_proj."),
    "gdn_input": (".in_proj_qkv.", ".in_proj_z."),
    "attention_output": (".o_proj.",),
    "gdn_output": (".out_proj.",),
}
QAT_ROLES = ("mlp", "attention_input", "gdn_input", "attention_output", "gdn_output")


@dataclass
class OutputFixture:
    inputs: InputFixture  # the input fixture, its base replaced by the base with outputs
    tensors: dict[str, torch.Tensor]  # every donor tensor, QAT and head
    expected: dict[str, tuple[torch.Tensor, torch.Tensor]]  # output object -> codes, scale words

    @property
    def base(self) -> Path:
        return self.inputs.mlp.base

    def donor_dir(self, name: str, roles) -> Path:
        """A donor checkpoint holding the tensors of ``roles`` only."""

        directory = self.base.parent / name
        directory.mkdir(exist_ok=True)
        _save(directory, {tensor_name: tensor for tensor_name, tensor in self.tensors.items()
                          if any(fragment in tensor_name for role in roles
                                 for fragment in ROLE_TENSORS[role])})
        return directory


def _build_outputs(tmp_path, *, outputs=None, swapped: bool = False) -> OutputFixture:
    """The input fixture (MLP words in compressed-tensors) plus the FP8 output projections of
    ``outputs`` (every layer by default); ``swapped`` stores each layer's output words under the
    other layer's name: same shapes, other weights."""

    fixture = _build_inputs(tmp_path, layout="compressed-tensors")
    generator = torch.Generator().manual_seed(13)
    with Artifact(fixture.mlp.base) as base:
        directory = base.directory
        specs = specs_of(base)
        payloads = {obj.id: base.read_object(obj.id) for obj in base.objects}
        bindings, base_uses = dict(directory.bindings), list(directory.uses)
    outputs = ROLES if outputs is None else outputs
    values = {kind: (torch.randn(*OUTPUT_SHAPE, generator=generator) * 0.02).to(torch.bfloat16)
              for kind in ROLES.values()}
    expected = {}
    for layer, kind in outputs.items():
        object_id = OUTPUT_OBJECTS[kind]
        fp8 = quantize_bf16_rows(values[kind])
        payloads[object_id] = encode_fp8_row_scaled(fp8.codes, fp8.scales, OUTPUT_SHAPE)
        specs.append(TensorSpec(object_id, OUTPUT_SHAPE, "fp8_e4m3fn_row_bf16", "row_scale_v1"))
        bindings[f"text/layers/{layer}/{kind}/output"] = {"object": object_id}
    for layer, kind in ROLES.items():
        module = f"model.language_model.layers.{layer}.{OUTPUT_MODULES[kind]}"
        stored = values[ROLES[1 - layer] if swapped else kind].float()
        weight_scale, input_scale = OUTPUT_SCALES[kind]
        shift = torch.randint(-1, 5, (OUTPUT_SHAPE[0], OUTPUT_SHAPE[1] // 16),
                              generator=generator, dtype=torch.int16)
        words = _block_words(stored, weight_scale, shift)
        codes = _nearest_codes(stored, _steps(words, divisor=weight_scale))
        fixture.donor[module + ".weight_packed"] = _pack(codes)
        fixture.donor[module + ".weight_scale"] = words.view(torch.float8_e4m3fn)
        fixture.donor[module + ".weight_global_scale"] = torch.tensor([weight_scale])
        fixture.donor[module + ".input_global_scale"] = torch.tensor([input_scale])
        fixture.mlp.weights[module + ".weight"] = values[kind]
        expected[OUTPUT_OBJECTS[kind]] = (codes, words)
    uses = []
    for use in base_uses:
        uses.append(use)
        for layer, kind in outputs.items():
            if use["parameter"] == f"text/layers/{layer}/{kind}/{LEAVES[kind][-1][0]}":
                uses.append({"parameter": f"text/layers/{layer}/{kind}/output",
                             "input": f"text/layers/{layer}/{kind}/gated_output",
                             "activation_policy": "AllowA8"})
    path = tmp_path / "base-outputs.ninfer"
    writer = ArtifactWriter(path, specs, components=directory.components, bindings=bindings,
                            uses=uses, metadata=directory.metadata,
                            provenance=directory.provenance)
    for spec in specs:
        writer.write_object(spec.id, payloads[spec.id])
    writer.finish()
    _save(fixture.mlp.weights_dir, fixture.mlp.weights)
    inputs = replace(fixture, mlp=replace(fixture.mlp, base=path))
    return OutputFixture(inputs, {**fixture.donor, **fixture.mlp.donor}, expected)


def role_donors(fixture: OutputFixture, roles=QAT_ROLES, *, split: bool = False) -> list[str]:
    """--role-donor options: the QAT roles from one checkpoint (or each from its own with
    ``split``) and the head from its own."""

    arguments = []
    shared = None if split else fixture.donor_dir("qat", QAT_ROLES)
    for role in roles:
        directory = fixture.donor_dir(f"donor-{role}", (role,)) if split else shared
        arguments += ["--role-donor", f"{role}={directory}",
                      "--role-donor-label", f"{role}={INPUT_LABEL}"]
    return arguments + ["--role-donor", f"head={fixture.donor_dir('head', ('head',))}",
                        "--role-donor-label", f"head={DONOR_LABEL}"]


def full_c(fixture: OutputFixture, out_path: Path, *extra: str, roles=QAT_ROLES,
           split: bool = False, weights: bool = True) -> list[str]:
    arguments = ["--base", str(fixture.base), "--out", str(out_path), "--layers", "0-1",
                 *role_donors(fixture, roles, split=split)]
    if weights:
        arguments += ["--weights", str(fixture.inputs.mlp.weights_dir),
                      "--weights-label", WEIGHTS_LABEL]
    return arguments + list(extra)


def test_full_c_converts_the_output_projections_with_their_donor_words(tmp_path,
                                                                        monkeypatch) -> None:
    # Row chunks that divide no parameter: chunks cross every boundary.
    monkeypatch.setattr(reencode_nvfp4_encode, "ROW_CHUNK", 100)
    fixture = _build_outputs(tmp_path)
    out_path = tmp_path / "out.ninfer"
    assert reencode_nvfp4.main(full_c(fixture, out_path, "--verify")) == 0
    report = _report(tmp_path)
    assert report["verified"] is True and report["recipe"] == FULL_C
    with Artifact(out_path) as out:
        record = out.directory.provenance["reencode"]
        assert out.directory.provenance["recipe"] == FULL_C
        assert record["base_recipe"] == "qwen3_8_27b_nvfp4"
        assert record["donors"] == {
            "mlp": {"label": INPUT_LABEL, "layers": [0, 1]}, "head": {"label": DONOR_LABEL},
            "attention_input": {"label": INPUT_LABEL, "layers": [1]},
            "gdn_input": {"label": INPUT_LABEL, "layers": [0]},
            "attention_output": {"label": INPUT_LABEL, "layers": [1]},
            "gdn_output": {"label": INPUT_LABEL, "layers": [0]}}
        for object_id in OUTPUT_OBJECTS.values():
            assert (out.object(object_id).format, out.object(object_id).layout) == (
                "nvfp4", NVFP4_LAYOUT)
        divisors = _divisor_uses(out)
        labels = record["activation_input_divisors"]
    entries = {item["object"]: item for item in report["objects"]}
    for layer, kind in ROLES.items():
        object_id, parameter = OUTPUT_OBJECTS[kind], f"text/layers/{layer}/{kind}/output"
        codes, scales, divisor, _ = _words(out_path, object_id, OUTPUT_SHAPE)
        assert torch.equal(codes, fixture.expected[object_id][0])
        assert torch.equal(scales, fixture.expected[object_id][1])
        weight_scale, input_scale = OUTPUT_SCALES[kind]
        assert _word(float(divisor)) == _word(weight_scale)
        # The output Use: AllowA4 and its own auxiliary holding the donor's input global scale.
        policy, aux, word = divisors[parameter]
        assert (policy, word) == ("AllowA4", _word(input_scale))
        assert labels[aux] == f"{parameter}: input_global_scale of the {kind}_output donor"
        entry = entries[object_id]
        assert entry["role"] == f"{kind}/output" and entry["codes"] == "donor"
        assert list(entry["relative_rms_error_vs_base_by_parameter"]) == [parameter]
        assert entry["relative_rms_error_vs_base"] < 0.3
        assert entry["relative_rms_error_vs_weights"] < 0.2
        assert entry["codes_equal_to_rounded_weights"] == 1.0
        module = f"model.language_model.layers.{layer}.{OUTPUT_MODULES[kind]}"
        assert set(entry["donor_sha256"]) == {module + suffix for suffix in (
            ".weight_packed", ".weight_scale", ".weight_global_scale", ".input_global_scale")}
    # The MLP holds the QAT donor's compressed-tensors words, the head the head donor's.
    mlp = fixture.inputs.mlp.expected
    codes, _, divisor, _ = _words(out_path, "weight/000000", (2 * INTERMEDIATE, HIDDEN))
    layer0 = "model.language_model.layers.0.mlp."
    assert torch.equal(codes, torch.cat([mlp[layer0 + "gate_proj"][0], mlp[layer0 + "up_proj"][0]]))
    assert _word(float(divisor)) == _word(GLOBAL_SCALES["gate_proj"])
    codes, _, _, _ = _words(out_path, "weight/head", (VOCAB, HIDDEN))
    assert torch.equal(codes, _unpack(fixture.tensors["lm_head.weight"]))


def test_each_role_reads_its_own_donor(tmp_path) -> None:
    """Every role from a checkpoint that holds its tensors only: a target reading another role's
    donor finds nothing. The output equals the build from one QAT checkpoint."""

    fixture = _build_outputs(tmp_path)
    split, shared = tmp_path / "split.ninfer", tmp_path / "shared.ninfer"
    assert reencode_nvfp4.main(full_c(fixture, split, split=True)) == 0
    assert reencode_nvfp4.main(full_c(fixture, shared)) == 0
    with Artifact(split) as one, Artifact(shared) as other:
        assert one.directory.uses == other.directory.uses
        assert one.directory.provenance == other.directory.provenance
        assert [obj.id for obj in one.objects] == [obj.id for obj in other.objects]
        for obj in one.objects:
            assert one.read_object(obj.id) == other.read_object(obj.id), obj.id


def test_full_c_is_full_b_plus_the_output_projections(tmp_path) -> None:
    """The full-b conversion of the same base, object for object and auxiliary id for auxiliary
    id, plus the converted output projections and their divisors, numbered last."""

    fixture = _build_outputs(tmp_path)
    full_b_path, full_c_path = tmp_path / "full-b.ninfer", tmp_path / "full-c.ninfer"
    assert reencode_nvfp4.main(full_c(fixture, full_b_path,
                                      roles=("mlp", "attention_input", "gdn_input"))) == 0
    assert reencode_nvfp4.main(full_c(fixture, full_c_path)) == 0
    outputs = set(OUTPUT_OBJECTS.values())
    with Artifact(full_b_path) as full_b_out, Artifact(full_c_path) as full_c_out:
        assert full_b_out.directory.provenance["recipe"] == FULL_B
        assert full_b_out.object(OBJECTS["attention"]).format == "nvfp4"
        assert {full_b_out.object(object_id).format for object_id in outputs} == {
            "fp8_e4m3fn_row_bf16"}
        b_ids, c_ids = [obj.id for obj in full_b_out.objects], [obj.id for obj in full_c_out.objects]
        assert c_ids[:len(b_ids)] == b_ids
        for object_id in b_ids:
            if object_id not in outputs:
                assert full_c_out.read_object(object_id) == full_b_out.read_object(object_id)
        b_divisors, c_divisors = _divisor_uses(full_b_out), _divisor_uses(full_c_out)
        assert {parameter: c_divisors[parameter] for parameter in b_divisors} == b_divisors
        new = {aux for parameter, (_, aux, _) in c_divisors.items() if parameter not in b_divisors}
        assert c_ids[len(b_ids):] == sorted(new) and len(new) == 2
        b_uses = {use["parameter"]: use for use in full_b_out.directory.uses}
        for use in full_c_out.directory.uses:
            if not use["parameter"].endswith("/output"):
                assert b_uses[use["parameter"]] == use


@pytest.mark.parametrize("weights", [True, False])
def test_output_words_of_other_weights_are_refused_against_the_base(tmp_path, weights) -> None:
    """Each layer's output words stored under the other layer's name: the right shape, the wrong
    weights. The base comparison refuses the first one, with or without --weights."""

    fixture = _build_outputs(tmp_path, swapped=True)
    with pytest.raises(ReencodeError,
                       match=r"weight/gdn0-output: relative RMS error 1\.\d+ against the base "
                             r"object's values of text/layers/0/gdn/output exceeds"):
        reencode_nvfp4.main(full_c(fixture, tmp_path / "out.ninfer", weights=weights))
    _assert_nothing_written(tmp_path)


def _edit_tensors(module: str, suffixes: tuple[str, ...], value):
    def mutate(fixture: OutputFixture) -> None:
        for name in (module + suffix for suffix in suffixes):
            if value is None:
                del fixture.tensors[name]
            else:
                fixture.tensors[name] = value(fixture.tensors[name])
    return mutate


O_PROJ = "model.language_model.layers.1.self_attn.o_proj"
NO_RECIPE = "no recipe converts the {} projections alone"


@pytest.mark.parametrize(
    "build, mutate, roles, match",
    [
        ({}, None, ("mlp", "attention_input"), NO_RECIPE.format("attention/input")),
        ({}, None, ("mlp", "attention_output", "gdn_output"),
         NO_RECIPE.format("attention/output, gdn/output")),
        ({}, None, ("mlp", "attention_input", "gdn_input", "attention_output"),
         NO_RECIPE.format("attention/input, attention/output, gdn/input")),
        ({"outputs": {1: "attention"}}, None, QAT_ROLES,
         "text layer 0 binds 0 output projections, not one"),
        ({}, _edit_tensors(O_PROJ, (".input_global_scale",), None), QAT_ROLES,
         "the donor checkpoint is missing .*o_proj.input_global_scale$"),
        ({}, _edit_tensors(O_PROJ, (".weight_packed",), lambda t: t[:, :1536].contiguous()),
         QAT_ROLES, r"o_proj: codes \(128, 1536\) and scales \(128, 384\) are not an NVFP4 "
                    "matrix of 6144 columns"),
        ({}, _edit_tensors(O_PROJ, (".weight_packed", ".weight_scale"), lambda t: t.repeat(2, 1)),
         QAT_ROLES, "weight/attention1-output: donor matrices have 256 rows for "
                    "text/layers/1/attention/output, object has 128"),
    ],
)
def test_output_refusals_before_writing(tmp_path, monkeypatch, build, mutate, roles,
                                        match) -> None:
    def no_writer(*arguments, **keywords):
        raise AssertionError("the output was created before the inputs were checked")

    fixture = _build_outputs(tmp_path, **build)
    if mutate is not None:
        mutate(fixture)
    monkeypatch.setattr(reencode_nvfp4, "ArtifactWriter", no_writer)
    with pytest.raises(ReencodeError, match=match):
        reencode_nvfp4.main(full_c(fixture, tmp_path / "out.ninfer", roles=roles))


def test_a_conversion_needs_a_head_donor(tmp_path) -> None:
    fixture = _build_outputs(tmp_path)
    arguments = full_c(fixture, tmp_path / "out.ninfer")
    head = arguments.index(f"head={fixture.donor_dir('head', ('head',))}")
    del arguments[head - 1:head + 3]
    with pytest.raises(ReencodeError, match="the head role needs a donor"):
        reencode_nvfp4.main(arguments)
    _assert_nothing_written(tmp_path)


@pytest.fixture
def full_c_output(tmp_path):
    """A verified full-c output and a verify of it against the plan, or tampered Uses."""

    fixture = _build_outputs(tmp_path)
    out_path = tmp_path / "out.ninfer"
    assert reencode_nvfp4.main(full_c(fixture, out_path, "--verify")) == 0
    report = _report(tmp_path)
    expected = {item["object"]: item["payload_sha256"]
                for item in (*report["objects"], *report["auxiliaries"])}
    with Artifact(fixture.base) as base, Artifact(out_path) as out:
        converted = {item["object"]: tuple(item["parameters"]) for item in report["objects"]
                     if base.object(item["object"]).format != "nvfp4"}
        uses = list(out.directory.uses)
    assert set(OUTPUT_OBJECTS.values()) <= set(converted)

    def verify(path=out_path, *, uses=uses, recipe=FULL_C) -> int:
        return reencode_nvfp4.verify_output(fixture.base, path, expected, converted=converted,
                                            uses=uses, recipe=recipe)

    return verify, out_path, uses


def test_verify_expects_the_full_c_recipe(full_c_output) -> None:
    verify, _, _ = full_c_output
    assert verify() == 0
    assert verify(recipe=FULL_B) == 1


OUTPUT = "text/layers/1/attention/output"


def _drop_divisor(use: dict, others: list[dict]) -> None:
    del use["auxiliaries"]


def _query_divisor(use: dict, others: list[dict]) -> None:
    query = next(item for item in others if item["parameter"] == "text/layers/1/attention/query")
    use["auxiliaries"] = dict(query["auxiliaries"])


@pytest.mark.parametrize(
    "tamper, planned, diff",
    [
        (_drop_divisor, False, USES),
        (_query_divisor, False, USES),
        # In the plan too: the output leaf shares the query's divisor, its own is left unused.
        (_query_divisor, True, DIVISORS),
    ],
)
def test_verify_refuses_output_uses_that_differ(full_c_output, tmp_path, capsys, tamper,
                                                planned, diff) -> None:
    verify, out_path, uses = full_c_output
    tampered_uses = copy.deepcopy(uses)
    tamper(next(use for use in tampered_uses if use["parameter"] == OUTPUT), tampered_uses)
    tampered = tmp_path / "tampered.ninfer"
    _rewrite(out_path, tampered, uses=tampered_uses)
    capsys.readouterr()
    assert verify(tampered, uses=tampered_uses if planned else uses) == 1
    printed = capsys.readouterr().out
    assert diff in printed and "DIFF object record" not in printed
