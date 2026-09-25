from __future__ import annotations

import json
import struct

from safetensors.torch import save_file
import torch

from tools.artifact.codecs.fp8_row import encode_fp8_row_scaled
from tools.artifact.codecs.nvfp4 import decode_nvfp4_words
from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.convert.common import nvfp4_quantize
from tools.convert.qwen3_8_27b import rewrite_nvfp4


# Miniature geometry: the tool reads shapes from the base artifact, not from constants.
HIDDEN = 64
INTERMEDIATE = 128
VOCAB = 256
FIRST_LAYER = 55
LAYERS = range(FIRST_LAYER, 64)


def _fp8_words(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scales = (values.abs().amax(dim=1) / 448.0).to(torch.bfloat16)
    codes = (values / scales.float().unsqueeze(1)).to(torch.float8_e4m3fn)
    return codes, scales


def _build(tmp_path):
    generator = torch.Generator().manual_seed(11)
    checkpoint: dict[str, torch.Tensor] = {}
    specs, payloads, bindings, uses = [], {}, {}, []
    aux_number = 0
    for index, layer in enumerate(LAYERS):
        source = f"model.language_model.layers.{layer}.mlp."
        prefix = f"text/layers/{layer}/mlp/"
        gate = torch.randn(INTERMEDIATE, HIDDEN, generator=generator)
        up = torch.randn(INTERMEDIATE, HIDDEN, generator=generator)
        down = torch.randn(HIDDEN * 2, INTERMEDIATE, generator=generator)
        gate_up_id, down_id = f"weight/{2 * index:06d}", f"weight/{2 * index + 1:06d}"
        gate_up_shape = (2 * INTERMEDIATE, HIDDEN)
        down_shape = tuple(down.shape)
        if layer == FIRST_LAYER:
            gate_up = nvfp4_quantize.quantize_fused([gate, up])
            down_words = nvfp4_quantize.quantize_fused([down])
            specs += [
                TensorSpec(gate_up_id, gate_up_shape, "nvfp4", "block_scale_k16_m128x4_v1"),
                TensorSpec(down_id, down_shape, "nvfp4", "block_scale_k16_m128x4_v1"),
            ]
            payloads[gate_up_id] = gate_up.encode(gate_up_shape)
            payloads[down_id] = down_words.encode(down_shape)
        else:
            for name, values in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
                codes, scales = _fp8_words(values)
                checkpoint[source + name + ".weight"] = codes
                checkpoint[source + name + ".weight_scale"] = scales.reshape(-1, 1)
            codes, scales = zip(*(_fp8_words(item) for item in (gate, up)))
            specs += [
                TensorSpec(gate_up_id, gate_up_shape, "fp8_e4m3fn_row_bf16", "row_scale_v1"),
                TensorSpec(down_id, down_shape, "fp8_e4m3fn_row_bf16", "row_scale_v1"),
            ]
            payloads[gate_up_id] = encode_fp8_row_scaled(
                torch.cat(codes).view(torch.uint8), torch.cat(scales), gate_up_shape
            )
            down_codes, down_scales = _fp8_words(down)
            payloads[down_id] = encode_fp8_row_scaled(
                down_codes.view(torch.uint8), down_scales, down_shape
            )
        half = INTERMEDIATE * HIDDEN
        bindings[prefix + "gate"] = {"parts": [{"object": gate_up_id, "range": [0, half]}]}
        bindings[prefix + "up"] = {"parts": [{"object": gate_up_id, "range": [half, 2 * half]}]}
        bindings[prefix + "down"] = {"object": down_id}
        leaves = (("gate", "ffn_input"), ("up", "ffn_input"), ("down", "mlp/product"))
        for leaf, source_input in leaves:
            use = {
                "parameter": prefix + leaf,
                "input": f"text/layers/{layer}/{source_input}",
                "activation_policy": "AllowA8",
            }
            if layer == FIRST_LAYER:
                aux_id = f"auxiliary/{aux_number:06d}"
                aux_number += 1
                use["activation_policy"] = "AllowA4"
                use["auxiliaries"] = {"activation_input_divisor": {"object": aux_id}}
                payloads[aux_id] = struct.pack("<f", 42.0 + (leaf == "down"))
            uses.append(use)
    head = torch.randn(VOCAB, HIDDEN, generator=generator).to(torch.bfloat16)
    checkpoint["lm_head.weight"] = head
    head_codes, head_scales = _fp8_words(head.float())
    specs.append(TensorSpec("weight/head", (VOCAB, HIDDEN), "fp8_e4m3fn_row_bf16", "row_scale_v1"))
    payloads["weight/head"] = encode_fp8_row_scaled(
        head_codes.view(torch.uint8), head_scales, (VOCAB, HIDDEN)
    )
    bindings["text/output_head"] = {"object": "weight/head"}
    for source_input in ("text/final_hidden", "mtp/final_hidden", "dflash2/final_hidden"):
        uses.append(
            {"parameter": "text/output_head", "input": source_input, "activation_policy": "AllowA8"}
        )
    for index in range(aux_number):
        specs.append(TensorSpec(f"auxiliary/{index:06d}", (), "fp32", "contiguous_le_v1"))

    base_path = tmp_path / "base.ninfer"
    writer = ArtifactWriter(
        base_path,
        specs,
        components={"text": {"config": {}}},
        bindings=bindings,
        uses=uses,
        metadata={"name": "qwen3.8-27b"},
        provenance={"recipe": "qwen3_8_27b_nvfp4"},
    )
    for object_id, payload in payloads.items():
        writer.write_object(object_id, payload)
    writer.finish()
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    save_file(checkpoint, str(source_dir / "model.safetensors"))
    return base_path, source_dir, checkpoint


def test_stage_a_rewrites_fp8_roles_and_copies_the_rest(tmp_path) -> None:
    base_path, source_dir, checkpoint = _build(tmp_path)
    out_path = tmp_path / "out.ninfer"
    code = rewrite_nvfp4.main(
        ["--base", str(base_path), "--source", str(source_dir), "--out", str(out_path), "--verify"]
    )
    assert code == 0

    with Artifact(base_path) as base, Artifact(out_path) as out:
        provenance = out.directory.provenance
        assert provenance["recipe"] == "qwen3_8_27b_nvfp4-full-a"
        assert provenance["rewrite"]["roles"]["mlp_double_quantization"] is True
        assert out.directory.bindings == base.directory.bindings
        formats = {obj.id: obj.format for obj in out.objects}
        assert formats["weight/head"] == "nvfp4"
        assert formats["weight/000000"] == "nvfp4"  # layer 55 kept as it was
        assert out.read_object("weight/000000") == base.read_object("weight/000000")
        for index in range(1, len(LAYERS)):
            assert formats[f"weight/{2 * index:06d}"] == "nvfp4"
            assert formats[f"weight/{2 * index + 1:06d}"] == "nvfp4"

        uses = {(use["parameter"], use["input"]): use for use in out.directory.uses}
        head_aux = {
            use["auxiliaries"]["activation_input_divisor"]["object"]
            for (parameter, _), use in uses.items()
            if parameter == "text/output_head"
        }
        assert len(head_aux) == 1
        (head_divisor,) = struct.unpack("<f", out.read_object(head_aux.pop()))
        assert head_divisor == 1.0
        gate = uses[("text/layers/60/mlp/gate", "text/layers/60/ffn_input")]
        down = uses[("text/layers/60/mlp/down", "text/layers/60/mlp/product")]
        assert gate["activation_policy"] == "AllowA4"
        for use, expected in ((gate, 42.0), (down, 43.0)):
            aux = use["auxiliaries"]["activation_input_divisor"]["object"]
            assert struct.unpack("<f", out.read_object(aux)) == (expected,)

        # The gate/up object is one NVFP4 matrix with a divisor shared by both halves,
        # computed from the dequantized FP8 source.
        codes, scales, divisor = decode_nvfp4_words(
            out.read_object("weight/000010"), (2 * INTERMEDIATE, HIDDEN)
        )
        source = "model.language_model.layers.60.mlp."
        fp32 = [
            checkpoint[source + name + ".weight"].float()
            * checkpoint[source + name + ".weight_scale"].float()
            for name in ("gate_proj", "up_proj")
        ]
        expected = nvfp4_quantize.quantize_fused(fp32)
        assert struct.unpack("<I", expected.divisor_word)[0] & 0xFFFF == 0
        assert float(divisor) == float(expected.divisor)
        assert torch.equal(codes, expected.packed)
        assert torch.equal(scales, expected.scales)

    report = json.loads((tmp_path / "out.ninfer.rewrite.json").read_text())
    assert len(report["objects"]) == 2 * (len(LAYERS) - 1) + 1
    assert all(item["relative_rms_error"] < 0.15 for item in report["objects"])


def test_verify_rejects_a_corrupted_copied_object(tmp_path) -> None:
    base_path, source_dir, _ = _build(tmp_path)
    out_path = tmp_path / "out.ninfer"
    arguments = ["--base", str(base_path), "--source", str(source_dir), "--out", str(out_path)]
    assert rewrite_nvfp4.main(arguments) == 0
    stage = rewrite_nvfp4.STAGES["a"]
    assert rewrite_nvfp4.verify_output(base_path, out_path, stage) == 0

    with Artifact(out_path) as out:
        # Layer 55 gate/up stays NVFP4 and is copied from the base.
        offset = out.payload_offset + out.object("weight/000000").offset + 17
    with out_path.open("r+b") as handle:
        handle.seek(offset)
        byte = handle.read(1)[0]
        handle.seek(offset)
        handle.write(bytes([byte ^ 0x01]))
    assert rewrite_nvfp4.verify_output(base_path, out_path, stage) == 1
