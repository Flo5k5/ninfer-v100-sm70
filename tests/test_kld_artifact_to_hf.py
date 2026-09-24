from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from tools.artifact.codecs.direct import encode_direct
from tools.artifact.codecs.fp8_row import dequantize_fp8_row_scaled, encode_fp8_row_scaled
from tools.artifact.schema import TensorObject
from tools.convert.quantization.fp8_row import quantize_bf16_rows

from tools.kld import artifact_to_hf


class _FakeDecoder:
    def __init__(self, values: dict[str, torch.Tensor]) -> None:
        self.values = values

    def logical(self, name: str) -> tuple[torch.Tensor, list[str]]:
        return self.values[name], ["fp8_e4m3fn_row_bf16"]


def test_assemble_interleaves_selected_rows_and_transposes_the_convolution() -> None:
    source = torch.zeros(8, 2)
    query = torch.full((4, 2), 1.0)
    gate = torch.full((4, 2), 2.0)
    decoder = _FakeDecoder({"query": query, "gate": gate})
    heads = [((0, 2), (4, 6)), ((2, 4), (6, 8))]
    values, formats = artifact_to_hf.assemble(
        source, [("query", heads[0]), ("gate", heads[1])], decoder)
    assert values[:, 0].tolist() == [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]
    assert formats == ["fp8_e4m3fn_row_bf16"]

    kernel = torch.arange(12, dtype=torch.float32).reshape(4, 3)  # [taps, channels]
    decoder = _FakeDecoder({"text/layers/0/gdn/convolution": kernel})
    values, _ = artifact_to_hf.assemble(
        torch.zeros(3, 1, 4), [("text/layers/0/gdn/convolution", None)], decoder)
    assert values[:, 0, :].tolist() == kernel.transpose(0, 1).tolist()


class _FakeArtifact:
    """The reader interface artifact_to_hf uses, over in-memory objects."""

    def __init__(self) -> None:
        self.path = "fake.ninfer"
        self.by_id: dict[str, TensorObject] = {}
        self.payloads: dict[str, bytes] = {}
        self.directory = SimpleNamespace(bindings={})
        self.reads: Counter[str] = Counter()

    def add(self, object_id: str, fmt: str, shape: tuple[int, ...], payload: bytes) -> None:
        layout = "row_scale_v1" if fmt == "fp8_e4m3fn_row_bf16" else "contiguous_le_v1"
        self.by_id[object_id] = TensorObject(object_id, shape, fmt, layout, 0, len(payload))
        self.payloads[object_id] = payload

    def read_object(self, object_id: str) -> bytes:
        self.reads[object_id] += 1
        return self.payloads[object_id]


def _fp8_object(values: torch.Tensor) -> bytes:
    words = quantize_bf16_rows(values.to(torch.bfloat16))
    return encode_fp8_row_scaled(words.codes, words.scales, tuple(values.shape))


def _checkpoint(tmp_path):
    generator = torch.Generator().manual_seed(5)
    gate = torch.randn(4, 32, generator=generator).to(torch.bfloat16)
    up = torch.randn(4, 32, generator=generator).to(torch.bfloat16)
    input_norm = torch.randn(32, generator=generator).to(torch.bfloat16)
    final_norm = torch.randn(32, generator=generator).to(torch.bfloat16)
    source = tmp_path / "source"
    source.mkdir()
    prefix = "model.language_model."
    save_file({prefix + "layers.0.mlp.gate_proj.weight": gate,
               prefix + "layers.0.mlp.up_proj.weight": up,
               prefix + "layers.0.input_layernorm.weight": input_norm},
              str(source / "model-00001-of-00002.safetensors"), metadata={"format": "pt"})
    save_file({prefix + "norm.weight": final_norm,
               "model.visual.patch_embed.weight": torch.ones(2, 2, dtype=torch.bfloat16)},
              str(source / "model-00002-of-00002.safetensors"), metadata={"format": "pt"})
    (source / "config.json").write_text(json.dumps({"architectures": ["Test"]}))

    artifact = _FakeArtifact()
    fused = torch.cat([gate, up]).float()
    artifact.add("weight/0", "fp8_e4m3fn_row_bf16", (8, 32), _fp8_object(fused))
    artifact.add("weight/1", "bf16", (32,), encode_direct(input_norm, "bf16"))
    artifact.add("weight/2", "bf16", (32,), encode_direct(final_norm + 0.5, "bf16"))
    artifact.directory.bindings.update({
        "text/layers/0/mlp/gate": {"parts": [{"object": "weight/0", "range": [0, 128]}]},
        "text/layers/0/mlp/up": {"parts": [{"object": "weight/0", "range": [128, 256]}]},
        "text/layers/0/input_norm": {"object": "weight/1"},
        "text/final_norm": {"object": "weight/2"},
    })
    return source, artifact, gate, up, input_norm, final_norm


def test_decoder_slices_fused_objects_and_caches_them_fifo(tmp_path) -> None:
    _, artifact, _, _, _, _ = _checkpoint(tmp_path)
    decoded = dequantize_fp8_row_scaled(artifact.payloads["weight/0"], (8, 32))
    decoder = artifact_to_hf.Decoder(artifact)

    gate, gate_formats = decoder.logical("text/layers/0/mlp/gate")
    up, _ = decoder.logical("text/layers/0/mlp/up")
    assert torch.equal(gate, decoded[:4]) and torch.equal(up, decoded[4:])
    assert gate_formats == ["fp8_e4m3fn_row_bf16"]
    assert artifact.reads["weight/0"] == 1  # both parts come from one decode

    for index in range(artifact_to_hf.Decoder.CACHED_OBJECTS):
        object_id = f"weight/{10 + index}"
        artifact.add(object_id, "bf16", (32,), encode_direct(torch.zeros(32, dtype=torch.bfloat16), "bf16"))
        artifact.directory.bindings[f"text/layers/{index + 1}/input_norm"] = {"object": object_id}
        decoder.logical(f"text/layers/{index + 1}/input_norm")
    decoder.logical("text/layers/0/mlp/up")
    assert artifact.reads["weight/0"] == 2  # the oldest object was evicted first


def test_convert_writes_decoded_text_weights_and_passes_the_rest_through(tmp_path) -> None:
    source, artifact, gate, up, input_norm, final_norm = _checkpoint(tmp_path)
    out = tmp_path / "out"
    report = artifact_to_hf.convert_artifact(artifact, source, out, torch.float16)

    first = load_file(str(out / "model-00001-of-00002.safetensors"))
    second = load_file(str(out / "model-00002-of-00002.safetensors"))
    decoded = dequantize_fp8_row_scaled(artifact.payloads["weight/0"], (8, 32))
    prefix = "model.language_model."
    assert first[prefix + "layers.0.mlp.gate_proj.weight"].dtype == torch.float16
    assert torch.equal(first[prefix + "layers.0.mlp.gate_proj.weight"],
                       decoded[:4].to(torch.float16))
    assert torch.equal(first[prefix + "layers.0.mlp.up_proj.weight"],
                       decoded[4:].to(torch.float16))
    # A direct-format parameter equal to the source keeps its exact source tensor and dtype.
    assert first[prefix + "layers.0.input_layernorm.weight"].dtype == torch.bfloat16
    assert torch.equal(first[prefix + "layers.0.input_layernorm.weight"], input_norm)
    # A direct-format parameter that differs is written in the requested dtype.
    assert second[prefix + "norm.weight"].dtype == torch.float16
    assert torch.equal(second[prefix + "norm.weight"], (final_norm + 0.5).to(torch.float16))
    assert torch.equal(second["model.visual.patch_embed.weight"],
                       torch.ones(2, 2, dtype=torch.bfloat16))
    assert json.loads((out / "config.json").read_text()) == {"architectures": ["Test"]}
    assert report["tensors"][prefix + "layers.0.mlp.gate_proj.weight"]["formats"] == [
        "fp8_e4m3fn_row_bf16"]
    assert report["tensors"][prefix + "layers.0.input_layernorm.weight"][
        "relative_rms_difference"] == 0.0


def test_convert_refuses_a_source_without_a_routed_tensor(tmp_path) -> None:
    source, artifact, _, _, _, _ = _checkpoint(tmp_path)
    artifact.add("weight/3", "bf16", (4 * 32,), encode_direct(torch.zeros(4 * 32, dtype=torch.bfloat16), "bf16"))
    artifact.directory.bindings["text/layers/0/mlp/down"] = {"object": "weight/3"}
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="lacks 1 routed tensors"):
        artifact_to_hf.convert_artifact(artifact, source, out, torch.float16)
    assert not out.exists()  # checked before anything is written
