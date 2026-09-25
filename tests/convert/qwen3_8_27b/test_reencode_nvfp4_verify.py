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
             shapes: dict[str, tuple[int, ...]] | None = None) -> None:
    """Copy of an artifact with other Uses or other object shapes, payloads byte for byte."""

    with Artifact(source) as artifact:
        directory = artifact.directory
        writer = ArtifactWriter(target, specs_of(artifact, shapes),
                                components=directory.components, bindings=directory.bindings,
                                uses=directory.uses if uses is None else uses,
                                metadata=directory.metadata, provenance=directory.provenance)
        for obj in artifact.objects:
            writer.write_object(obj.id, artifact.read_object(obj.id))
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


@pytest.mark.parametrize(
    "tamper, planned",
    [
        (_set_policy, False),
        (_drop_divisor, False),
        (_other_leaf_divisor, False),
        # The plan has it too: it no longer differs from the base as a conversion does.
        (_set_policy, True),
        (_drop_divisor, True),
    ],
)
def test_verify_refuses_converted_uses_that_differ(plan, tmp_path, capsys, tamper,
                                                   planned) -> None:
    """A converted leaf's Use set back to AllowA8, without its divisor, or naming another leaf's
    divisor object where the plan does not, with every record intact."""

    verify, out_path, _, uses = plan
    tampered_uses = copy.deepcopy(uses)
    tamper(next(use for use in tampered_uses if use["parameter"] == QUERY), tampered_uses)
    tampered = tmp_path / "tampered.ninfer"
    _rewrite(out_path, tampered, uses=tampered_uses)
    capsys.readouterr()
    assert verify(tampered, uses=tampered_uses if planned else uses) == 1
    printed = capsys.readouterr().out
    assert "DIFF directory uses" in printed and "DIFF object record" not in printed


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
