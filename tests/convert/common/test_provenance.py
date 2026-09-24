from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from tools.convert.common.provenance import (
    file_record,
    input_label,
    is_local_path,
    local_name,
    strip_local_paths,
)

from ..path_fragments import assert_no_path_fragments


# Values an older tool could have written where a repository id or artifact id was expected --
# the graft tool once recorded the raw --source argument as the label when none was given.
LOCAL_PATHS = [
    "/data/models/donor",
    "~/models/donor",
    "\\\\server\\share\\donor",
    "C:\\models\\donor",
    "../checkpoints/orca-NVFP4",
    "./checkpoints/orca-NVFP4",
    "checkpoints/../orca-NVFP4",
    "checkpoints/./orca-NVFP4",
    "..\\checkpoints\\orca-NVFP4",
    "..",
    "../",
]

# Labels that must survive both is_local_path and strip_local_paths untouched: Hugging Face
# repository ids, an artifact id (hex digest), and other short opaque names.
NON_PATH_LABELS = [
    "org/fine-tune-NVFP4",
    "nvidia/Qwen3.8-27B-NVFP4",
    "orcarouter/Qwen3.8-27B-Uncensored",
    "00112233445566778899aabbccddeeff",
    "qwen3_8_27b_nvfp4",
    "base.ninfer",
    "org/name-v1.2",
]

# Paths a converter receives for an input or output, and the name it records for each instead.
NAMED_PATHS = [
    ("out/qwen3_8_27b.ninfer", "qwen3_8_27b.ninfer"),
    ("/data/models/Qwen3.8-27B/", "Qwen3.8-27B"),
    ("~/models/Qwen3.8-27B-DFlash2", "Qwen3.8-27B-DFlash2"),
    ("models/Qwen3.8-27B/base-hf-bf16/..", "Qwen3.8-27B"),
    ("models/./base-hf-bf16", "base-hf-bf16"),
]


@pytest.mark.parametrize("value", LOCAL_PATHS)
def test_is_local_path_recognizes_absolute_and_relative_paths(value: str) -> None:
    assert is_local_path(value)


@pytest.mark.parametrize("value", NON_PATH_LABELS)
def test_is_local_path_keeps_repository_ids_and_artifact_ids(value: str) -> None:
    assert not is_local_path(value)


@pytest.mark.parametrize("value", LOCAL_PATHS)
def test_input_label_refuses_paths(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        input_label(value)


@pytest.mark.parametrize("value", NON_PATH_LABELS)
def test_input_label_accepts_repository_ids_and_artifact_ids(value: str) -> None:
    assert input_label(value) == value


def test_strip_local_paths_removes_a_relative_label_inherited_from_an_older_artifact() -> None:
    # Before input_label existed, the graft tool could record --source itself as the label.
    # Once it is removed, "single" and then "sources" hold nothing else and collapse away too.
    provenance = {"sources": {"single": {"label": "../checkpoints/orca-NVFP4"}}}
    stripped, removed = strip_local_paths(provenance)
    assert stripped == {}
    assert removed == ["sources.single.label"]


def test_strip_local_paths_removes_only_the_relative_label_when_a_sibling_survives() -> None:
    # A sibling member of "single" that is not a path keeps the container in the result.
    provenance = {"sources": {"single": {"label": "../checkpoints/orca-NVFP4",
                                         "repository": "org/fine-tune-NVFP4"}}}
    stripped, removed = strip_local_paths(provenance)
    assert stripped == {"sources": {"single": {"repository": "org/fine-tune-NVFP4"}}}
    assert removed == ["sources.single.label"]


def test_strip_local_paths_removes_a_relative_path_in_a_list() -> None:
    provenance = {"history": ["org/fine-tune-NVFP4", "../checkpoints/orca-NVFP4"]}
    stripped, removed = strip_local_paths(provenance)
    assert stripped == {"history": ["org/fine-tune-NVFP4"]}
    assert removed == ["history[1]"]


def test_strip_local_paths_keeps_repository_ids_and_artifact_ids() -> None:
    provenance = {
        "sources": {"single": {"label": "org/fine-tune-NVFP4"}},
        "base_artifact_id": "00112233445566778899aabbccddeeff",
        "recipe": "qwen3_8_27b_nvfp4",
    }
    stripped, removed = strip_local_paths(provenance)
    assert stripped == provenance
    assert removed == []


@pytest.mark.parametrize(("path", "name"), NAMED_PATHS)
def test_local_name_records_the_last_component_only(path: str, name: str) -> None:
    assert local_name(path) == local_name(Path(path)) == name
    assert not is_local_path(name)


def test_local_name_names_the_directory_a_dot_component_denotes(tmp_path, monkeypatch) -> None:
    # A bare "." or ".." names no file of its own (Path.name gives "" and ".."): record the name
    # of the directory it stands for.
    working = tmp_path / "checkpoints" / "Qwen3.8-27B"
    working.mkdir(parents=True)
    monkeypatch.chdir(working)
    assert local_name(".") == "Qwen3.8-27B"
    assert local_name("..") == "checkpoints"


def test_local_name_refuses_a_path_without_a_name() -> None:
    with pytest.raises(ValueError, match="no file or directory name"):
        local_name("/")


def test_file_record_names_the_file_and_digests_its_contents(tmp_path) -> None:
    data = bytes(range(256)) * 1024
    path = tmp_path / "fixtures" / "ranking.train.counts.i64"
    path.parent.mkdir()
    path.write_bytes(data)

    record = file_record(path)

    assert record == {
        "name": "ranking.train.counts.i64",
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    assert_no_path_fragments(json.dumps(record), tmp_path)
