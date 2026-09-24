"""Every converter's <out>.conversion.json names its inputs and artifact without a local path.

The report travels apart from the artifact and from the machine that wrote it. Each test builds a
converter's report from inputs laid out under tmp_path, as the converter does once the artifact is
written, then checks the names it records and that no path fragment remains anywhere in its JSON.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from tools.artifact.container import ResourceSpec, plan_objects
from tools.convert.qwen3_6.common.recipe import SourcePreflight
from tools.convert.qwen3_6_27b.draft_head import DEFAULT_RANKING

from .path_fragments import CHECKOUT, assert_no_path_fragments


# Each converter here imports artifact codecs from where tools.artifact kept them before its v3
# module (2235303e), so none of them imports yet. Strict: once they do, these tests must pass and
# this marker must go.
pytestmark = pytest.mark.xfail(
    raises=ImportError,
    strict=True,
    reason="the converters import artifact codecs from their pre-v3 location",
)

RANKING = bytes(range(256)) * 64
DEVICE = "cpu"
FINAL_BYTES = 4096
SOURCE = SourcePreflight(
    recipe_count=1,
    source_tensor_count=1,
    source_shard_count=1,
    source_dtype_counts={"BF16": 1},
)


def _objects():
    return plan_objects((ResourceSpec("frontend/tokenizer.json", "raw-bytes-v1", 2),))


def _ranking(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A ranking file under tmp_path, and the record a report must hold for it."""

    path = tmp_path / "fixtures" / "ranking.train.counts.i64"
    path.parent.mkdir()
    path.write_bytes(RANKING)
    return path, {"name": path.name, "sha256": hashlib.sha256(RANKING).hexdigest()}


def _repository_ranking() -> dict[str, str]:
    """The record of the ranking fixture that the NVFP4 converters read from the repository."""

    path = CHECKOUT / DEFAULT_RANKING
    return {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _assert_path_free(
    report: dict,
    tmp_path: Path,
    *,
    source: dict,
    arguments: dict,
    artifact: str,
) -> None:
    assert report["source"] == source
    assert report["arguments"] == {**arguments, "device": DEVICE}
    assert report["artifact"] == {"name": artifact, "bytes": FINAL_BYTES}
    assert_no_path_fragments(json.dumps(report), tmp_path)


def test_qwen3_6_27b_report_names_inputs_without_paths(tmp_path) -> None:
    from tools.convert.qwen3_6_27b import convert

    ranking, ranking_record = _ranking(tmp_path)
    report = convert.build_conversion_report(
        model_dir=tmp_path / "models" / "base-hf-bf16",
        out_path=tmp_path / "out" / "qwen3_6_27b.ninfer",
        requested_device=DEVICE,
        config_summary={},
        source_preflight=SOURCE,
        objects=_objects(),
        elapsed_seconds=1.0,
        final_bytes=FINAL_BYTES,
        device=torch.device(DEVICE),
        ranking_path=ranking,
        revision="test-revision",
        environment={"python": "test"},
    )

    _assert_path_free(
        report,
        tmp_path,
        source={"base": {"name": "base-hf-bf16"}, "ranking": ranking_record},
        arguments={"model": "base-hf-bf16", "out": "qwen3_6_27b.ninfer"},
        artifact="qwen3_6_27b.ninfer",
    )


def test_qwen3_6_27b_nvfp4_report_names_inputs_without_paths(tmp_path) -> None:
    from tools.convert.qwen3_6_27b import convert_nvfp4

    recipe = convert_nvfp4.recipe
    preflight = convert_nvfp4.ConversionPreflight(
        base_dir=tmp_path / "models" / "base-hf-bf16",
        nvfp4_dir=tmp_path / "models" / "vllm-nvfp4-bf16",
        config_summary={},
        base_source=SOURCE,
        nvfp4_dtype_counts={},
        resources=(),
        draft=None,
        object_plan=None,
    )
    report = convert_nvfp4.build_conversion_report(
        preflight=preflight,
        output=tmp_path / "out" / "qwen3_6_27b_nvfp4.ninfer",
        requested_device=DEVICE,
        objects=_objects(),
        elapsed_seconds=1.0,
        final_bytes=FINAL_BYTES,
        device=torch.device(DEVICE),
    )

    _assert_path_free(
        report,
        tmp_path,
        source={
            "base": {
                "repository": recipe.BASE_REPOSITORY,
                "revision": recipe.BASE_REVISION,
                "name": "base-hf-bf16",
            },
            "nvfp4": {
                "repository": recipe.NVFP4_REPOSITORY,
                "revision": recipe.NVFP4_REVISION,
                "name": "vllm-nvfp4-bf16",
            },
            "ranking": _repository_ranking(),
        },
        arguments={
            "model": "base-hf-bf16",
            "nvfp4_model": "vllm-nvfp4-bf16",
            "out": "qwen3_6_27b_nvfp4.ninfer",
        },
        artifact="qwen3_6_27b_nvfp4.ninfer",
    )


def test_qwen3_8_27b_report_names_inputs_without_paths(tmp_path) -> None:
    from tools.convert.qwen3_8_27b import convert

    ranking, ranking_record = _ranking(tmp_path)
    report = convert.build_conversion_report(
        model_dir=tmp_path / "models" / "Qwen3.8-27B",
        dflash2_model_dir=tmp_path / "models" / "Qwen3.8-27B-DFlash2",
        out_path=tmp_path / "out" / "qwen3_8_27b.ninfer",
        requested_device=DEVICE,
        base_config_summary={},
        dflash2_config_summary={},
        base_source_preflight=SOURCE,
        dflash2_source_preflight=SOURCE,
        objects=_objects(),
        elapsed_seconds=1.0,
        final_bytes=FINAL_BYTES,
        device=torch.device(DEVICE),
        ranking_path=ranking,
    )

    dflash2 = convert.dflash2_recipe
    _assert_path_free(
        report,
        tmp_path,
        source={
            "base": {
                "repository": convert.BASE_REPOSITORY,
                "revision": convert.BASE_REVISION,
                "name": "Qwen3.8-27B",
            },
            "dflash2": {
                "repository": dflash2.REPOSITORY,
                "revision": dflash2.REVISION,
                "name": "Qwen3.8-27B-DFlash2",
            },
            "ranking": ranking_record,
        },
        arguments={
            "model": "Qwen3.8-27B",
            "dflash2_model": "Qwen3.8-27B-DFlash2",
            "out": "qwen3_8_27b.ninfer",
        },
        artifact="qwen3_8_27b.ninfer",
    )


def test_qwen3_8_27b_nvfp4_report_names_inputs_without_paths(tmp_path) -> None:
    from tools.convert.qwen3_8_27b import convert_nvfp4

    recipe = convert_nvfp4.recipe
    dflash2 = convert_nvfp4.dflash2_recipe
    preflight = convert_nvfp4.ConversionPreflight(
        official_dir=tmp_path / "models" / "base-hf-bf16",
        quantized_dir=tmp_path / "models" / "vllm-nvfp4-fp8",
        dflash2_model_dir=tmp_path / "models" / "Qwen3.8-27B-DFlash2",
        base_config_summary={},
        dflash2_config_summary={},
        official_source=SOURCE,
        quantized_source=SOURCE,
        dflash2_source=SOURCE,
        resources=(),
        draft=None,
        object_plan=None,
    )
    report = convert_nvfp4.build_conversion_report(
        preflight=preflight,
        output=tmp_path / "out" / "qwen3_8_27b_nvfp4.ninfer",
        requested_device=DEVICE,
        objects=_objects(),
        elapsed_seconds=1.0,
        final_bytes=FINAL_BYTES,
        device=torch.device(DEVICE),
    )

    _assert_path_free(
        report,
        tmp_path,
        source={
            "official": {
                "repository": recipe.BASE_REPOSITORY,
                "revision": recipe.BASE_REVISION,
                "name": "base-hf-bf16",
            },
            "quantized": {
                "repository": recipe.QUANTIZED_REPOSITORY,
                "revision": recipe.QUANTIZED_REVISION,
                "name": "vllm-nvfp4-fp8",
            },
            "dflash2": {
                "repository": dflash2.REPOSITORY,
                "revision": dflash2.REVISION,
                "name": "Qwen3.8-27B-DFlash2",
            },
            "ranking": _repository_ranking(),
        },
        arguments={
            "model": "base-hf-bf16",
            "quantized_model": "vllm-nvfp4-fp8",
            "dflash2_model": "Qwen3.8-27B-DFlash2",
            "out": "qwen3_8_27b_nvfp4.ninfer",
        },
        artifact="qwen3_8_27b_nvfp4.ninfer",
    )


def test_qwen3_6_35b_a3b_report_names_inputs_without_paths(tmp_path) -> None:
    from tools.convert.qwen3_6_35b_a3b import convert

    ranking, ranking_record = _ranking(tmp_path)
    report = convert.build_conversion_report(
        model_dir=tmp_path / "models" / "base-hf-bf16",
        dflash_model_dir=tmp_path / "models" / "dflash-bf16",
        out_path=tmp_path / "out" / "qwen3_6_35b_a3b.ninfer",
        requested_device=DEVICE,
        base_config_summary={},
        dflash_config_summary={},
        base_source_preflight=SOURCE,
        dflash_source_preflight=SOURCE,
        objects=_objects(),
        elapsed_seconds=1.0,
        final_bytes=FINAL_BYTES,
        device=torch.device(DEVICE),
        ranking_path=ranking,
        revision="test-revision",
        environment={"python": "test"},
    )

    _assert_path_free(
        report,
        tmp_path,
        source={
            "base": {"name": "base-hf-bf16"},
            "dflash": {"name": "dflash-bf16"},
            "gguf_evidence": {"name": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"},
            "ranking": ranking_record,
        },
        arguments={
            "model": "base-hf-bf16",
            "dflash_model": "dflash-bf16",
            "out": "qwen3_6_35b_a3b.ninfer",
        },
        artifact="qwen3_6_35b_a3b.ninfer",
    )
