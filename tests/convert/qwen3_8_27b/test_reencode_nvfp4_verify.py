"""--verify of a full-b output: each check refuses its own kind of divergence from the plan.

Every tampered output is a byte-for-byte copy of a verified full-b output with one directory
field changed, so exactly one check can refuse it.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tools.artifact.reader import Artifact
from tools.artifact.writer import ArtifactWriter
from tools.convert.qwen3_8_27b import reencode_nvfp4

from .test_reencode_nvfp4 import _report
from .test_reencode_nvfp4_inputs import FULL_A, FULL_B, OBJECTS, _build_inputs, full_b, specs_of


DIVISOR = "activation_input_divisor"
QUERY = "text/layers/1/attention/query"


def _rewrite(source: Path, target: Path, *, uses: list[dict] | None = None,
             shapes: dict[str, tuple[int, ...]] | None = None, drop: str | None = None) -> None:
    """Copy of an artifact with other Uses, other object shapes or without one object, payloads
    byte for byte."""

    with Artifact(source) as artifact:
        directory = artifact.directory
        specs = [spec for spec in specs_of(artifact, shapes) if spec.id != drop]
        writer = ArtifactWriter(target, specs,
                                components=directory.components, bindings=directory.bindings,
                                uses=directory.uses if uses is None else uses,
                                metadata=directory.metadata, provenance=directory.provenance)
        for spec in specs:
            writer.write_object(spec.id, artifact.read_object(spec.id))
    writer.finish()


@pytest.fixture
def plan(tmp_path):
    """A verified full-b output, its written payload digests, converted objects and Uses."""

    fixture = _build_inputs(tmp_path)
    out_path = tmp_path / "out.ninfer"
    assert reencode_nvfp4.main(full_b(fixture, out_path, "--verify")) == 0
    report = _report(tmp_path)
    expected = {item["object"]: item["payload_sha256"]
                for item in (*report["objects"], *report["auxiliaries"])}
    with Artifact(fixture.mlp.base) as base, Artifact(out_path) as out:
        converted = {item["object"]: tuple(item["parameters"]) for item in report["objects"]
                     if base.object(item["object"]).format != "nvfp4"}
        uses = list(out.directory.uses)
    assert set(converted) == {*OBJECTS.values(), "weight/000003", "weight/head"}

    def verify(path=out_path, *, expected=expected, converted=converted, uses=uses,
               recipe=FULL_B) -> int:
        return reencode_nvfp4.verify_output(fixture.mlp.base, path, expected,
                                            converted=converted, uses=uses, recipe=recipe)

    return verify, out_path, report, uses


def test_verify_expects_the_recipe_and_objects_of_the_conversion(plan) -> None:
    verify, out_path, report, _ = plan
    assert verify() == 0
    assert verify(recipe=FULL_A) == 1
    # An auxiliary object that nothing planned is refused.
    aux = report["auxiliaries"][-1]["object"]
    expected = {item["object"]: item["payload_sha256"]
                for item in (*report["objects"], *report["auxiliaries"])}
    assert verify(expected={key: value for key, value in expected.items() if key != aux}) == 1
    # A corrupted input divisor is refused.
    with Artifact(out_path) as out:
        offset = out.payload_offset + out.object(aux).offset
    with out_path.open("r+b") as handle:
        handle.seek(offset)
        byte = handle.read(1)[0]
        handle.seek(offset)
        handle.write(bytes([byte ^ 0x01]))
    assert verify() == 1


def _set_policy(use: dict, others: list[dict]) -> None:
    use["activation_policy"] = "AllowA8"


def _drop_divisor(use: dict, others: list[dict]) -> None:
    del use["auxiliaries"]


def _other_leaf_divisor(use: dict, others: list[dict]) -> None:
    key = next(item for item in others if item["parameter"] == "text/layers/1/attention/key")
    use["auxiliaries"] = copy.deepcopy(key["auxiliaries"])


USES, DIVISORS = "DIFF directory uses", "DIFF divisor objects of the converted leaves"


@pytest.mark.parametrize(
    "tamper, planned, diff",
    [
        (_set_policy, False, USES),
        (_drop_divisor, False, USES),
        (_other_leaf_divisor, False, USES),
        # The plan has it too: it no longer differs from the base as a conversion does, or two
        # leaves share one divisor object and query's own is left unused.
        (_set_policy, True, USES),
        (_drop_divisor, True, USES),
        (_other_leaf_divisor, True, DIVISORS),
    ],
)
def test_verify_refuses_converted_uses_that_differ(plan, tmp_path, capsys, tamper, planned,
                                                   diff) -> None:
    """A converted leaf's Use set back to AllowA8, without its divisor, or naming another leaf's
    divisor object, in the output only or in the plan too, with every record intact."""

    verify, out_path, _, uses = plan
    tampered_uses = copy.deepcopy(uses)
    tamper(next(use for use in tampered_uses if use["parameter"] == QUERY), tampered_uses)
    tampered = tmp_path / "tampered.ninfer"
    _rewrite(out_path, tampered, uses=tampered_uses)
    capsys.readouterr()
    assert verify(tampered, uses=tampered_uses if planned else uses) == 1
    printed = capsys.readouterr().out
    assert diff in printed and "DIFF object record" not in printed


def test_verify_refuses_two_leaves_sharing_one_new_divisor(plan, tmp_path, capsys) -> None:
    """Query names key's divisor object in the plan and the output, and query's own object is
    gone, so no new object is left unused: only the sharing shows."""

    verify, out_path, report, uses = plan
    tampered_uses = copy.deepcopy(uses)
    query = next(use for use in tampered_uses if use["parameter"] == QUERY)
    own = query["auxiliaries"][DIVISOR]["object"]
    _other_leaf_divisor(query, tampered_uses)
    tampered = tmp_path / "tampered.ninfer"
    _rewrite(out_path, tampered, uses=tampered_uses, drop=own)
    expected = {item["object"]: item["payload_sha256"]
                for item in (*report["objects"], *report["auxiliaries"]) if item["object"] != own}
    capsys.readouterr()
    assert verify(tampered, expected=expected, uses=tampered_uses) == 1
    printed = capsys.readouterr().out
    assert DIVISORS in printed and USES not in printed


@pytest.mark.parametrize("kind", ["converted object", "auxiliary object"])
def test_verify_refuses_records_that_differ(plan, tmp_path, capsys, kind) -> None:
    """A converted object reshaped to the same element count and NVFP4 byte size, or a divisor
    stored as FP32[1]: payloads and digests unchanged, Uses intact."""

    verify, out_path, report, _ = plan
    if kind == "converted object":
        object_id, shape, diff = OBJECTS["attention"], (7168, 128), "DIFF object record"
    else:
        object_id, shape, diff = report["auxiliaries"][-1]["object"], (1,), "DIFF auxiliary record"
    tampered = tmp_path / "tampered.ninfer"
    _rewrite(out_path, tampered, shapes={object_id: shape})
    capsys.readouterr()
    assert verify(tampered) == 1
    printed = capsys.readouterr().out
    assert f"{diff} {object_id}" in printed and "DIFF directory uses" not in printed
