"""Re-read a ``reencode_nvfp4`` output and compare it with the written payloads and the base.

The output must keep the base's components, bindings and metadata, carry the recipe of its
conversion and the planned Uses, and hold, for every object, the payload the tool wrote
(re-encoded objects and new FP32 scalar auxiliary objects) or the base's bytes (everything else).
Converted objects change format and layout only, and the planned Uses of their leaves differ from
the base's only by ``AllowA4`` and an activation input divisor, each converted leaf naming its own
new auxiliary object; every other record and Use is the base's.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorObject
from tools.convert.qwen3_8_27b.reencode_nvfp4_plan import (
    AUX_FORMAT,
    AUX_LAYOUT,
    DIVISOR_ROLE,
    NVFP4_FORMAT,
    NVFP4_LAYOUT,
)


def _digest(chunks: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _expected_uses(base_uses: Sequence[dict], planned_uses: Sequence[dict],
                   converted_parameters: set[str]) -> list[dict]:
    """The base's Uses as a conversion leaves them, with the divisor objects of the plan."""

    expected = []
    for base_use, planned_use in zip(base_uses, planned_uses):
        if base_use["parameter"] in converted_parameters:
            divisor = planned_use.get("auxiliaries", {}).get(DIVISOR_ROLE)
            base_use = {**base_use, "activation_policy": "AllowA4",
                        "auxiliaries": {**base_use.get("auxiliaries", {}), DIVISOR_ROLE: divisor}}
        expected.append(base_use)
    return expected


def _new_divisors_match(base_uses: Sequence[dict], planned_uses: Sequence[dict],
                        converted_parameters: set[str], new_objects: set[str]) -> bool:
    """Every converted leaf whose base Uses had no divisor names one new auxiliary object in all
    its Uses (the output head's Uses share one), no two leaves the same, and together all of the
    new auxiliary objects."""

    by_parameter: dict[str, set] = {}
    for base_use, planned_use in zip(base_uses, planned_uses):
        if planned_use["parameter"] in converted_parameters and \
                DIVISOR_ROLE not in base_use.get("auxiliaries", {}):
            divisor = planned_use.get("auxiliaries", {}).get(DIVISOR_ROLE) or {}
            by_parameter.setdefault(planned_use["parameter"], set()).add(divisor.get("object"))
    named = [objects.pop() for objects in by_parameter.values() if len(objects) == 1]
    return len(named) == len(by_parameter) and len(set(named)) == len(named) and \
        set(named) == new_objects


def _record(obj) -> dict:
    # A converted object changes byte size, so every object stored after one shifts its offset:
    # records compare without it (the order is checked on its own, sizes through the digests).
    return {key: value for key, value in obj.to_json().items() if key != "offset"}


def verify_output(base_path: Path, out_path: Path, expected: Mapping[str, str], *,
                  converted: Mapping[str, Sequence[str]], uses: Sequence[dict],
                  recipe: str) -> int:
    """0 when the output holds the written payloads, Uses and recipe and all else is the base's.

    ``expected`` maps every object the tool wrote (re-encoded objects and new auxiliary objects)
    to its payload SHA-256; ``converted`` maps the objects converted from another format to their
    parameters; ``uses`` are the planned Uses of the output; ``recipe`` is the recipe the output
    must carry."""

    converted_parameters = {parameter for parameters in converted.values()
                            for parameter in parameters}
    failures = 0
    with Artifact(base_path) as base, Artifact(out_path) as out:
        for field in ("components", "bindings", "metadata"):
            if getattr(out.directory, field) != getattr(base.directory, field):
                print(f"DIFF directory {field}", flush=True)
                failures += 1
        base_uses, planned = list(base.directory.uses), list(uses)
        if len(planned) != len(base_uses) or list(out.directory.uses) != planned or \
                planned != _expected_uses(base_uses, planned, converted_parameters):
            print("DIFF directory uses", flush=True)
            failures += 1
        new_objects = set(expected) - {obj.id for obj in base.objects}
        if not _new_divisors_match(base_uses, planned, converted_parameters, new_objects):
            print("DIFF divisor objects of the converted leaves", flush=True)
            failures += 1
        if out.directory.provenance.get("recipe") != recipe:
            print(f"DIFF provenance recipe (expected {recipe})", flush=True)
            failures += 1
        if len(out.objects) < len(base.objects):
            print(f"DIFF object count {len(out.objects)} < {len(base.objects)}", flush=True)
            return 1
        for base_obj, out_obj in zip(base.objects, out.objects):
            if base_obj.id != out_obj.id:
                print(f"DIFF object order at {base_obj.id}", flush=True)
                return 1
            want = _record(base_obj)
            if base_obj.id in converted:
                want.update(format=NVFP4_FORMAT, layout=NVFP4_LAYOUT, bytes=out_obj.bytes)
            if _record(out_obj) != want:
                print(f"DIFF object record {base_obj.id}", flush=True)
                return 1
        for obj in out.objects[len(base.objects):]:
            if obj.id not in expected:
                print(f"DIFF unexpected object {obj.id}", flush=True)
                return 1
            if not isinstance(obj, TensorObject) or \
                    (obj.format, obj.layout, tuple(obj.shape)) != (AUX_FORMAT, AUX_LAYOUT, ()):
                print(f"DIFF auxiliary record {obj.id}", flush=True)
                return 1
        for obj in out.objects:
            want = expected.get(obj.id) or _digest(base.iter_object(obj.id))
            if _digest(out.iter_object(obj.id)) != want:
                kind = "written" if obj.id in expected else "copied"
                print(f"DIFF {obj.id} ({kind} object)", flush=True)
                failures += 1
    print(f"verified {len(expected)} written objects and the copied rest: {failures} mismatches",
          flush=True)
    return 1 if failures else 0


def remove_output(out_path: Path) -> None:
    with Artifact(out_path) as out:
        parts = [out_path.parent / item.path for item in out.directory.files[1:]]
    for path in [out_path, *parts]:
        path.unlink(missing_ok=True)
