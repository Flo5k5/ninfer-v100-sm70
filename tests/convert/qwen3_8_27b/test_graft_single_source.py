from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import pytest
from safetensors.torch import save_file
import torch

from tools.artifact.codecs.direct import encode_direct
from tools.artifact.codecs.nvfp4 import decode_nvfp4_words, encode_nvfp4
from tools.artifact.framing import HEADER
from tools.artifact.layouts import encoded_size
from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.convert.qwen3_8_27b import graft_single_source

from ..path_fragments import assert_no_path_fragments


HIDDEN = 64
INTERMEDIATE = 128
GATE_UP_SHAPE = (2 * INTERMEDIATE, HIDDEN)
NVFP4_LAYOUT = "block_scale_k16_m128x4_v1"
SOURCE_LABEL = "org/fine-tune-NVFP4"
MODULE = "model.language_model.layers.0.mlp."


def _build(tmp_path: Path):
    """A one-layer template whose provenance carries local paths, and a single source for it."""

    generator = torch.Generator().manual_seed(3)
    source, words = {}, {}
    for name in ("gate_proj", "up_proj"):
        packed = torch.randint(0, 256, (INTERMEDIATE, HIDDEN // 2), generator=generator,
                               dtype=torch.uint8)
        scales = torch.randint(0x20, 0x50, (INTERMEDIATE, HIDDEN // 16), generator=generator,
                               dtype=torch.uint8)
        source[MODULE + name + ".weight_packed"] = packed
        source[MODULE + name + ".weight_scale"] = scales.view(torch.float8_e4m3fn)
        source[MODULE + name + ".weight_global_scale"] = torch.tensor([4133.75])
        source[MODULE + name + ".input_global_scale"] = torch.tensor([40.0])
        words[name] = (packed, scales)
    source["model.language_model.norm.weight"] = torch.randn(HIDDEN, generator=generator).to(
        torch.bfloat16)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    save_file(source, str(source_dir / "model.safetensors"))

    half = INTERMEDIATE * HIDDEN
    specs = [TensorSpec("weight/000000", GATE_UP_SHAPE, "nvfp4", NVFP4_LAYOUT),
             TensorSpec("auxiliary/000000", (), "fp32", "contiguous_le_v1"),
             TensorSpec("auxiliary/000001", (), "fp32", "contiguous_le_v1"),
             TensorSpec("weight/000001", (HIDDEN,), "bf16", "contiguous_le_v1"),
             ResourceSpec("resource/tokenizer", 37)]
    bindings = {
        "text/layers/0/mlp/gate": {"parts": [{"object": "weight/000000", "range": [0, half]}]},
        "text/layers/0/mlp/up": {"parts": [{"object": "weight/000000",
                                            "range": [half, 2 * half]}]},
        "text/final_norm": {"object": "weight/000001"},
    }
    uses = [{"parameter": f"text/layers/0/mlp/{leaf}", "input": "text/layers/0/ffn_input",
             "activation_policy": "AllowA4",
             "auxiliaries": {"activation_input_divisor": {"object": f"auxiliary/00000{index}"}}}
            for index, leaf in enumerate(("gate", "up"))]
    provenance = {
        "converter": "ninfer-v3",
        "recipe": "qwen3_8_27b_nvfp4",
        "sources": {"base": {"path": str(tmp_path / "base-hf-bf16")},
                    "quantized": {"repository": "org/quantized-NVFP4",
                                  "path": "~/models/quantized"}},
        "ranking": str(tmp_path / "ranking.counts.i64"),
    }
    template_path = tmp_path / "template.ninfer"
    writer = ArtifactWriter(template_path, specs, components={"text": {"config": {}}},
                            bindings=bindings, uses=uses, metadata={"name": "qwen3.8-27b"},
                            provenance=provenance)
    # Template payloads of other weights: the graft replaces every non-resource object.
    writer.write_object("weight/000000", encode_nvfp4(
        torch.zeros(GATE_UP_SHAPE[0], HIDDEN // 2, dtype=torch.uint8),
        torch.full((GATE_UP_SHAPE[0], HIDDEN // 16), 0x38, dtype=torch.uint8), torch.tensor(1.0),
        GATE_UP_SHAPE))
    for index in range(2):
        writer.write_object(f"auxiliary/00000{index}", struct.pack("<f", 1.0))
    writer.write_object("weight/000001", encode_direct(torch.zeros(HIDDEN, dtype=torch.bfloat16),
                                                       "bf16"))
    writer.write_object("resource/tokenizer", bytes(range(37)))
    writer.finish()
    return template_path, source_dir, words


def _arguments(template_path, source_dir, out_path, *extra: str) -> list[str]:
    return ["--template", str(template_path), "--source", str(source_dir), "--out", str(out_path),
            *extra]


def test_graft_names_the_source_by_label_and_drops_local_paths(tmp_path) -> None:
    template_path, source_dir, words = _build(tmp_path)
    out_path = tmp_path / "out.ninfer"
    assert graft_single_source.main(
        _arguments(template_path, source_dir, out_path, "--source-label", SOURCE_LABEL)) == 0
    with Artifact(template_path) as template, Artifact(out_path) as out:
        assert out.directory.provenance == {
            "converter": "ninfer-v3",
            "recipe": "qwen3_8_27b_nvfp4",
            "sources": {"single": {"label": SOURCE_LABEL}},
            "graft": {
                "tool": "tools.convert.qwen3_8_27b.graft_single_source",
                "template": "template.ninfer",
                "template_artifact_id": template.artifact_id.hex(),
                "copied": list(graft_single_source.COPY_PREFIXES),
                "template_sources": {"quantized": {"repository": "org/quantized-NVFP4"}},
                "removed_template_paths": ["sources.base.path", "sources.quantized.path",
                                           "ranking"],
            },
        }
        codes, scales, divisor = decode_nvfp4_words(out.read_object("weight/000000"),
                                                    GATE_UP_SHAPE)
    parts = [words[name] for name in ("gate_proj", "up_proj")]
    assert torch.equal(codes, torch.cat([part[0] for part in parts]))
    assert torch.equal(scales, torch.cat([part[1] for part in parts]))
    assert float(divisor) == 4133.75

    with out_path.open("rb") as handle:
        _, json_bytes, _ = HEADER.unpack(handle.read(HEADER.size))
        text = handle.read(json_bytes).decode("utf-8")
    assert_no_path_fragments(text, tmp_path)


def test_graft_report_names_files_by_name_only(tmp_path) -> None:
    """The side report (<out>.graft.json) travels separately from the artifact: it must be just
    as free of local paths, naming the artifact, template and source by file name only."""

    template_path, source_dir, _ = _build(tmp_path)
    out_path = tmp_path / "out.ninfer"
    assert graft_single_source.main(
        _arguments(template_path, source_dir, out_path, "--source-label", SOURCE_LABEL)) == 0
    report = json.loads(Path(f"{out_path}.graft.json").read_text())
    assert report["artifact"] == out_path.name
    assert report["template"] == template_path.name
    assert report["source"] == source_dir.name
    assert report["fp8_bf16_encoder"] == "MAXABS_BF16S_RECIP_E4M3FN_RNE_V1"
    assert report["fp8_bf16_method"] == "fp8_row_maxabs"
    for value in (report["artifact"], report["template"], report["source"]):
        assert "/" not in value and "\\" not in value
    assert_no_path_fragments(json.dumps(report), tmp_path)


@pytest.mark.parametrize("extra", [(), ("--source-label", "/data/models/fine-tune"),
                                   ("--source-label", "~/models/fine-tune")])
def test_graft_requires_a_source_label_that_is_not_a_path(tmp_path, extra) -> None:
    template_path, source_dir, _ = _build(tmp_path)
    out_path = tmp_path / "out.ninfer"
    with pytest.raises(SystemExit):
        graft_single_source.main(_arguments(template_path, source_dir, out_path, *extra))
    assert not out_path.exists()


# One object per route on which the graft encodes BF16 source values itself: the grouped
# quantizers, including a fused parent and K that is not a multiple of 128, the in-memory FP8 row
# encoder, and the chunked one that matrices of 65,536 rows or more take.
ENCODED = (
    ("weight/000000", ("text/layers/0/mlp/gate", "text/layers/0/mlp/up"), "q4_g64_fp16", (16, 192)),
    ("weight/000001", ("text/layers/0/mlp/down",), "q5_g64_fp16", (8, 192)),
    ("weight/000002", ("text/layers/0/attention/output",), "q6_g64_fp16", (8, 192)),
    ("weight/000003", ("text/layers/0/attention/value",), "q8_g32_fp16", (8, 96)),
    ("weight/000004", ("text/layers/0/attention/key",), "fp8_e4m3fn_row_bf16", (8, 64)),
    ("weight/000005", ("text/token_embedding",), "fp8_e4m3fn_row_bf16", (65536, 16)),
)
# SHA-256 of the payloads that master's graft wrote for these inputs, before it moved onto
# tools.convert.quantization. A change to these bytes changes every artifact the graft produces.
ENCODED_SHA256 = {
    "weight/000000": "76bfd0466b677fc07aa85372e88ebe3609234f15860903b476f0f6f0264efc7d",
    "weight/000001": "8675199f707abee2a2f16f7480878db78634884f041d48253dbee7d4aed7f44c",
    "weight/000002": "2bb9506ce50f2b50926aab1c4cf722b08bb98fa33dc0b5aa945863642d29735a",
    "weight/000003": "cdca047878fc49b58dc5796b578b1b4d52d30fc088789a228c130b4fceeadac5",
    "weight/000004": "e7fc0b86ca5d61447eede45a002ce6ca4e35fbc9f501f1616d5051849c381477",
    "weight/000005": "7c1cada73f251e0bb7a25007593c2913522134ee19285a1c3676647f12c223bd",
}


def _source_values(generator: torch.Generator, rows: int, columns: int) -> torch.Tensor:
    """Row magnitudes from 1e-4 to 1e3, an all-zero row, and a few exact ties and signed zeros."""

    values = torch.randn(rows, columns, generator=generator)
    values *= torch.logspace(-4, 3, rows).unsqueeze(1)
    values[0, :4] = torch.tensor([0.0, -0.0, 0.5, -1.5])
    values[-1] = 0.0
    return values.to(torch.bfloat16)


def _encoded_payload_digests(tmp_path: Path) -> dict[str, str]:
    generator = torch.Generator().manual_seed(11)
    source, specs, bindings = {}, [], {}
    for object_id, logicals, format, shape in ENCODED:
        layout = "row_scale_v1" if format.startswith("fp8") else "row_split_k128_v1"
        specs.append(TensorSpec(object_id, shape, format, layout))
        rows = shape[0] // len(logicals)
        for index, logical in enumerate(logicals):
            name = graft_single_source.logical_matrix(logical).source
            source[name] = _source_values(generator, rows, shape[1])
            if len(logicals) == 1:
                bindings[logical] = {"object": object_id}
            else:
                span = rows * shape[1]
                bindings[logical] = {"parts": [{"object": object_id,
                                                "range": [index * span, (index + 1) * span]}]}
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    save_file(source, str(source_dir / "model.safetensors"))
    template_path = tmp_path / "template.ninfer"
    writer = ArtifactWriter(template_path, specs, components={"text": {"config": {}}},
                            bindings=bindings, uses=[], metadata={"name": "qwen3.8-27b"},
                            provenance={"converter": "ninfer-v3", "recipe": "qwen3_8_27b_nvfp4"})
    for spec in specs:
        writer.write_object(spec.id, bytes(encoded_size(spec.layout, spec.format, spec.shape)))
    writer.finish()

    out_path = tmp_path / "out.ninfer"
    assert graft_single_source.main(
        _arguments(template_path, source_dir, out_path, "--source-label", SOURCE_LABEL)) == 0
    with Artifact(out_path) as out:
        return {object_id: hashlib.sha256(out.read_object(object_id)).hexdigest()
                for object_id, *_ in ENCODED}


def test_graft_encodes_bf16_sources_to_the_bytes_master_wrote(tmp_path) -> None:
    assert _encoded_payload_digests(tmp_path) == ENCODED_SHA256
