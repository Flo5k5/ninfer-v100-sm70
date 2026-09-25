"""The donor options of reencode_nvfp4: one donor per role with --role-donor and
--role-donor-label, and the --donor and --input-donor shorthands that full-a and full-b builds use.
"""

from __future__ import annotations

import pytest

from tools.artifact.reader import Artifact
from tools.convert.qwen3_8_27b import reencode_nvfp4

from .test_reencode_nvfp4 import DONOR_LABEL, WEIGHTS_LABEL, _assert_nothing_written
from .test_reencode_nvfp4_inputs import INPUT_LABEL, _build_inputs, full_b


def _role(role: str, path, label: str) -> list[str]:
    return ["--role-donor", f"{role}={path}", "--role-donor-label", f"{role}={label}"]


def test_shorthands_equal_their_role_donors(tmp_path) -> None:
    """--donor is the mlp and head roles' donor, --input-donor both input roles': the full-b
    build of the shorthands equals the build that names each role, directory and payloads."""

    fixture = _build_inputs(tmp_path)
    shorthands, roles = tmp_path / "shorthands.ninfer", tmp_path / "roles.ninfer"
    assert reencode_nvfp4.main(full_b(fixture, shorthands)) == 0
    assert reencode_nvfp4.main([
        "--base", str(fixture.mlp.base), "--out", str(roles), "--layers", "0-1",
        "--round", "down", "--weights", str(fixture.mlp.weights_dir),
        "--weights-label", WEIGHTS_LABEL,
        *_role("head", fixture.mlp.donor_dir, DONOR_LABEL),
        *_role("gdn_input", fixture.donor_dir, INPUT_LABEL),
        *_role("mlp", fixture.mlp.donor_dir, DONOR_LABEL),
        *_role("attention_input", fixture.donor_dir, INPUT_LABEL)]) == 0
    with Artifact(shorthands) as one, Artifact(roles) as other:
        for field in ("components", "bindings", "uses", "metadata", "provenance"):
            assert getattr(one.directory, field) == getattr(other.directory, field), field
        assert [obj.to_json() for obj in one.objects] == [obj.to_json() for obj in other.objects]
        for obj in one.objects:
            assert one.read_object(obj.id) == other.read_object(obj.id), obj.id


@pytest.mark.parametrize(
    "options, message",
    [
        ((), "the mlp role needs a donor"),
        (("HEAD",), "the mlp role needs a donor"),
        (("--role-donor", "mlp=DONOR"), "must name the same roles, not mlp in one of them only"),
        (("--role-donor-label", f"mlp={DONOR_LABEL}"),
         "must name the same roles, not mlp in one of them only"),
        (("MLP", "--role-donor", "mlp=DONOR"), "--role-donor gives the mlp role twice"),
        (("MLP", "--role-donor-label", f"mlp={DONOR_LABEL}"),
         "--role-donor-label gives the mlp role twice"),
        (("--donor", "DONOR", "--donor-label", DONOR_LABEL, "HEAD"),
         "the head role has a donor from --donor and from --role-donor"),
        (("--donor", "DONOR", "--donor-label", DONOR_LABEL,
          "--role-donor-label", f"mlp={DONOR_LABEL}"),
         "the mlp role has a donor from --donor and from --role-donor"),
        (("MLP", "--input-donor", "INPUTS", "--input-donor-label", INPUT_LABEL,
          "--role-donor", "gdn_input=INPUTS", "--role-donor-label", f"gdn_input={INPUT_LABEL}"),
         "the gdn_input role has a donor from --input-donor and from --role-donor"),
        (("MLP", "--role-donor", "attention=INPUTS"), "is not ROLE=VALUE with ROLE one of mlp, "
         "head, attention_input, gdn_input, attention_output, gdn_output"),
        (("--role-donor", "mlp=", "--role-donor-label", f"mlp={DONOR_LABEL}"),
         "'mlp=' names no checkpoint"),
        (("--role-donor", "mlp", "--role-donor-label", f"mlp={DONOR_LABEL}"),
         "'mlp' names no checkpoint"),
        (("--role-donor", "mlp=DONOR", "--role-donor-label", "mlp=/models/donor"),
         "is not an input label"),
        (("--role-donor", "mlp=DONOR", "--role-donor-label", "mlp"), "is not an input label"),
    ],
)
def test_role_donor_argument_checks(tmp_path, capsys, options, message) -> None:
    """Each check refuses its own case: MLP and HEAD stand for a full --role-donor and
    --role-donor-label pair of that role, DONOR and INPUTS for the fixture's two donors."""

    fixture = _build_inputs(tmp_path)
    paths = {"DONOR": str(fixture.mlp.donor_dir), "INPUTS": str(fixture.donor_dir)}
    arguments = ["--base", str(fixture.mlp.base), "--out", str(tmp_path / "out.ninfer")]
    for option in options:
        if option in ("MLP", "HEAD"):
            arguments += _role(option.lower(), fixture.mlp.donor_dir, DONOR_LABEL)
        else:
            for placeholder, path in paths.items():
                option = option.replace(placeholder, path)
            arguments.append(option)
    capsys.readouterr()
    with pytest.raises(SystemExit):
        reencode_nvfp4.main(arguments)
    assert message in capsys.readouterr().err
    _assert_nothing_written(tmp_path)
