from __future__ import annotations

import argparse

import pytest

from tools.convert.common.provenance import input_label, is_local_path, strip_local_paths


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
