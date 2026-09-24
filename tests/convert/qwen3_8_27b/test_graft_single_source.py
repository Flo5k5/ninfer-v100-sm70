from __future__ import annotations

from pathlib import Path
import re
import struct
import tempfile

import pytest
from safetensors.torch import save_file
import torch

from tools.artifact.codecs.direct import encode_direct
from tools.artifact.codecs.nvfp4 import decode_nvfp4_words, encode_nvfp4
from tools.artifact.framing import HEADER
from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.convert.qwen3_8_27b import graft_single_source


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
    for fragment in (str(tmp_path), str(Path.home()), tempfile.gettempdir(), "/home/", "/tmp/",
                     "/private/", "/Users/", "/data/", "/var/"):
        assert fragment not in text, fragment
    # A JSON string (member name or value) that starts like an absolute or home path.
    assert not re.findall(r'"(?:/|~|\\\\|[A-Za-z]:)', text)


@pytest.mark.parametrize("extra", [(), ("--source-label", "/data/models/fine-tune"),
                                   ("--source-label", "~/models/fine-tune")])
def test_graft_requires_a_source_label_that_is_not_a_path(tmp_path, extra) -> None:
    template_path, source_dir, _ = _build(tmp_path)
    out_path = tmp_path / "out.ninfer"
    with pytest.raises(SystemExit):
        graft_single_source.main(_arguments(template_path, source_dir, out_path, *extra))
    assert not out_path.exists()
