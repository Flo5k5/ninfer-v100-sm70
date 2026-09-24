from __future__ import annotations

import json
import struct

import pytest
from safetensors.torch import save_file
import torch

from tools.artifact.codecs.fp8_row import encode_fp8_row_scaled
from tools.artifact.codecs.nvfp4 import decode_nvfp4_words, dequantize_nvfp4, encode_nvfp4
from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.convert.qwen3_8_27b import reencode_nvfp4
from tools.convert.qwen3_8_27b.graft_single_source import SourceCheckpoint


# Miniature geometry: the tool reads shapes from the base artifact, not from constants.
HIDDEN = 64
INTERMEDIATE = 128
DOWN_ROWS = 128  # NVFP4 objects need a multiple of 128 rows
VOCAB = 256
LAYERS = (0, 1)
NVFP4_LAYOUT = "block_scale_k16_m128x4_v1"
MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# weight_scale_2 values: the first has an exact FP32 reciprocal, the second (0x397dbe11) has none.
EXACT_SCALE_2 = 2.0 ** -12
INEXACT_SCALE_2 = struct.unpack("<f", struct.pack("<I", 0x397DBE11))[0]
PROJECTIONS_INDEX = {"gate": 0, "up": 1, "down": 2}


def _nearest_codes(values: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
    """Exact oracle: nearest signed E2M1 level of value / step in float64, ties to the even level."""

    rows, columns = values.shape
    blocks = values.double().reshape(rows, columns // 16, 16, 1)
    levels = torch.tensor(MAGNITUDES, dtype=torch.float64) * steps.double().reshape(rows, -1, 1, 1)
    distance = (blocks.abs() - levels).abs()
    best = distance.min(dim=-1, keepdim=True).values
    candidates = distance == best
    even = candidates & (torch.arange(8) % 2 == 0)
    index = torch.where(even.any(dim=-1), even.int().argmax(dim=-1), candidates.int().argmax(dim=-1))
    codes = (index | (torch.signbit(blocks.squeeze(-1)).long() << 3)).to(torch.uint8)
    return codes.reshape(rows, columns)


def _pack(codes: torch.Tensor) -> torch.Tensor:
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()


def _unpack(packed: torch.Tensor) -> torch.Tensor:
    return torch.stack((packed & 0x0F, packed >> 4), dim=2).reshape(packed.shape[0], -1)


def _e4m3_words(values: torch.Tensor) -> torch.Tensor:
    return values.to(torch.float8_e4m3fn).view(torch.uint8)


def _block_words(values: torch.Tensor, divisor: float, shift: torch.Tensor | None = None):
    """E4M3 block scale words for max scaling under ``divisor``, optionally moved by some codes."""

    rows, columns = values.shape
    amax = values.abs().reshape(rows, columns // 16, 16).amax(dim=2)
    words = _e4m3_words(amax / 6.0 * divisor).to(torch.int16)
    if shift is not None:
        words = words + shift
    return words.clamp(1, 0x7E).to(torch.uint8)


def _dequantize(codes: torch.Tensor, words: torch.Tensor, multiplier: float) -> torch.Tensor:
    rows, columns = codes.shape
    magnitude = torch.tensor(MAGNITUDES, dtype=torch.float32)[(codes & 0x7).long()]
    values = torch.where((codes & 0x8) != 0, -magnitude, magnitude)
    step = words.view(torch.float8_e4m3fn).float() * torch.tensor(multiplier, dtype=torch.float32)
    return (values.reshape(rows, -1, 16) * step.unsqueeze(2)).reshape(rows, columns)


def _build(tmp_path, *, up_scale_2: float | None = None, layer1_down_format: str = "nvfp4"):
    generator = torch.Generator().manual_seed(7)
    donor: dict[str, torch.Tensor] = {}
    weights: dict[str, torch.Tensor] = {}
    specs, payloads, bindings, uses = [], {}, {}, []
    expected = {}
    for index, layer in enumerate(LAYERS):
        source = f"model.language_model.layers.{layer}.mlp."
        prefix = f"text/layers/{layer}/mlp/"
        gate = torch.randn(INTERMEDIATE, HIDDEN, generator=generator).to(torch.bfloat16) * 0.02
        up = torch.randn(INTERMEDIATE, HIDDEN, generator=generator).to(torch.bfloat16) * 0.02
        down = torch.randn(DOWN_ROWS, INTERMEDIATE, generator=generator).to(torch.bfloat16) * 0.02
        # The model edits down after quantization of its original: donor and local weights differ.
        edited_down = (down.float() + 0.001 * torch.randn(down.shape, generator=generator)).to(
            torch.bfloat16)
        weights[source + "gate_proj.weight"] = gate
        weights[source + "up_proj.weight"] = up
        weights[source + "down_proj.weight"] = edited_down

        # Donor: ModelOpt words with block scales moved away from max scaling (a calibrated choice).
        scale_2 = {"gate_proj": EXACT_SCALE_2, "up_proj": up_scale_2 or EXACT_SCALE_2,
                   "down_proj": INEXACT_SCALE_2 if layer == 1 else EXACT_SCALE_2}
        for name, values in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
            shift = torch.randint(-1, 5, (values.shape[0], values.shape[1] // 16),
                                  generator=generator, dtype=torch.int16)
            words = _block_words(values.float(), 1.0 / scale_2[name], shift)
            step = words.view(torch.float8_e4m3fn).float() * torch.tensor(scale_2[name])
            codes = _nearest_codes(values.float(), step)
            donor[source + name + ".weight"] = _pack(codes)
            donor[source + name + ".weight_scale"] = words.view(torch.float8_e4m3fn)
            donor[source + name + ".weight_scale_2"] = torch.tensor(scale_2[name])
            expected[source + name] = (codes, words, scale_2[name])

        # Base: round-to-nearest words with max block scales, one divisor for gate and up.
        gate_up_id, down_id = f"weight/{2 * index:06d}", f"weight/{2 * index + 1:06d}"
        gate_up_shape, down_shape = (2 * INTERMEDIATE, HIDDEN), tuple(down.shape)
        fused = torch.cat([gate.float(), up.float()])
        divisor = 2688.0 / float(fused.abs().max())
        words = _block_words(fused, divisor)
        codes = _nearest_codes(fused, words.view(torch.float8_e4m3fn).float() / divisor)
        payloads[gate_up_id] = encode_nvfp4(_pack(codes), words, torch.tensor(divisor), gate_up_shape)
        down_divisor = 2688.0 / float(edited_down.float().abs().max())
        down_words = _block_words(edited_down.float(), down_divisor)
        down_codes = _nearest_codes(edited_down.float(),
                                    down_words.view(torch.float8_e4m3fn).float() / down_divisor)
        payloads[down_id] = encode_nvfp4(_pack(down_codes), down_words, torch.tensor(down_divisor),
                                         down_shape)
        down_format = layer1_down_format if layer == 1 else "nvfp4"
        specs.append(TensorSpec(gate_up_id, gate_up_shape, "nvfp4", NVFP4_LAYOUT))
        if down_format == "nvfp4":
            specs.append(TensorSpec(down_id, down_shape, "nvfp4", NVFP4_LAYOUT))
        else:
            codes8 = edited_down.float().to(torch.float8_e4m3fn).view(torch.uint8)
            payloads[down_id] = encode_fp8_row_scaled(
                codes8, torch.ones(DOWN_ROWS, dtype=torch.bfloat16), down_shape)
            specs.append(TensorSpec(down_id, down_shape, "fp8_e4m3fn_row_bf16", "row_scale_v1"))
        half = INTERMEDIATE * HIDDEN
        bindings[prefix + "gate"] = {"parts": [{"object": gate_up_id, "range": [0, half]}]}
        bindings[prefix + "up"] = {"parts": [{"object": gate_up_id, "range": [half, 2 * half]}]}
        bindings[prefix + "down"] = {"object": down_id}
        for leaf, source_input in (("gate", "ffn_input"), ("up", "ffn_input"),
                                   ("down", "mlp/product")):
            aux_id = f"auxiliary/{3 * index + PROJECTIONS_INDEX[leaf]:06d}"
            uses.append({"parameter": prefix + leaf, "input": f"text/layers/{layer}/{source_input}",
                         "activation_policy": "AllowA4",
                         "auxiliaries": {"activation_input_divisor": {"object": aux_id}}})
            specs.append(TensorSpec(aux_id, (), "fp32", "contiguous_le_v1"))
            payloads[aux_id] = struct.pack("<f", 40.0 + 3 * index + PROJECTIONS_INDEX[leaf])
    head = torch.randn(VOCAB, HIDDEN, generator=generator)
    specs.append(TensorSpec("weight/head", (VOCAB, HIDDEN), "fp8_e4m3fn_row_bf16", "row_scale_v1"))
    payloads["weight/head"] = encode_fp8_row_scaled(
        head.to(torch.float8_e4m3fn).view(torch.uint8), torch.ones(VOCAB, dtype=torch.bfloat16),
        (VOCAB, HIDDEN))
    bindings["text/output_head"] = {"object": "weight/head"}
    uses.append({"parameter": "text/output_head", "input": "text/final_hidden",
                 "activation_policy": "AllowA8"})
    specs.append(ResourceSpec("resource/tokenizer", 37))
    payloads["resource/tokenizer"] = bytes(range(37))

    base_path = tmp_path / "base.ninfer"
    writer = ArtifactWriter(base_path, specs, components={"text": {"config": {}}},
                            bindings=bindings, uses=uses, metadata={"name": "qwen3.8-27b"},
                            provenance={"recipe": "qwen3_8_27b_nvfp4"})
    for object_id, payload in payloads.items():
        writer.write_object(object_id, payload)
    writer.finish()
    donor_dir, weights_dir = tmp_path / "donor", tmp_path / "weights"
    donor_dir.mkdir()
    weights_dir.mkdir()
    save_file(donor, str(donor_dir / "model.safetensors"))
    save_file(weights, str(weights_dir / "model.safetensors"))
    return base_path, donor_dir, weights_dir, weights, expected



def _run(base_path, donor_dir, weights_dir, out_path, *extra: str) -> int:
    return reencode_nvfp4.main([
        "--base", str(base_path), "--donor", str(donor_dir), "--weights", str(weights_dir),
        "--round", "down", "--out", str(out_path), *extra])


def test_reencodes_mlp_objects_and_copies_everything_else(tmp_path) -> None:
    base_path, donor_dir, weights_dir, weights, expected = _build(tmp_path)
    out_path = tmp_path / "out.ninfer"
    assert _run(base_path, donor_dir, weights_dir, out_path, "--verify") == 0

    with Artifact(base_path) as base, Artifact(out_path) as out:
        for field in ("components", "bindings", "uses", "metadata"):
            assert getattr(out.directory, field) == getattr(base.directory, field)
        assert out.directory.provenance["recipe"] == "qwen3_8_27b_nvfp4"
        assert out.directory.provenance["reencode"]["layers"] == list(LAYERS)
        assert [obj.to_json() for obj in out.objects] == [obj.to_json() for obj in base.objects]
        reencoded = {f"weight/{i:06d}" for i in range(2 * len(LAYERS))}
        for obj in base.objects:
            if obj.id not in reencoded:
                assert out.read_object(obj.id) == base.read_object(obj.id), obj.id

        for index, layer in enumerate(LAYERS):
            source = f"model.language_model.layers.{layer}.mlp."
            # Gate/up: the donor's words as they are, decoding to the donor's values.
            gate_up = f"weight/{2 * index:06d}"
            shape = (2 * INTERMEDIATE, HIDDEN)
            codes, scales, divisor = decode_nvfp4_words(out.read_object(gate_up), shape)
            donor_codes = torch.cat([expected[source + name][0] for name in ("gate_proj", "up_proj")])
            donor_words = torch.cat([expected[source + name][1] for name in ("gate_proj", "up_proj")])
            assert torch.equal(_unpack(codes), donor_codes)
            assert torch.equal(scales, donor_words)
            assert float(torch.tensor(1.0) / divisor) == EXACT_SCALE_2
            donor_values = torch.cat([_dequantize(*expected[source + name])
                                      for name in ("gate_proj", "up_proj")])
            torch.testing.assert_close(dequantize_nvfp4(out.read_object(gate_up), shape),
                                       donor_values, rtol=2.0 ** -21, atol=0.0)

            # Down: the donor's scales, codes rounded from the edited local weights.
            down_id = f"weight/{2 * index + 1:06d}"
            shape = (DOWN_ROWS, INTERMEDIATE)
            codes, scales, divisor = decode_nvfp4_words(out.read_object(down_id), shape)
            _, donor_down_words, scale_2 = expected[source + "down_proj"]
            assert torch.equal(scales, donor_down_words)
            steps = scales.view(torch.float8_e4m3fn).float() * (torch.tensor(1.0) / divisor)
            local = weights[source + "down_proj.weight"].float()
            assert torch.equal(_unpack(codes), _nearest_codes(local, steps))
            if scale_2 == EXACT_SCALE_2:
                assert float(torch.tensor(1.0) / divisor) == scale_2

    report = json.loads((tmp_path / "out.ninfer.reencode.json").read_text())
    assert report["inexact_reciprocals"] == 1
    assert report["recipe"] == "qwen3_8_27b_nvfp4"
    gate_up_report = report["objects"][0]
    assert gate_up_report["codes_equal_to_rounded_weights"] == 1.0
    assert all(item["relative_rms_error"] < 0.2 for item in report["objects"])


def test_inexact_reciprocal_is_the_nearest_fp32_word() -> None:
    divisor, exact = reencode_nvfp4.reciprocal_divisor(INEXACT_SCALE_2)
    assert not exact
    target = torch.tensor(INEXACT_SCALE_2)
    one = torch.tensor(1.0)
    chosen = abs(float(one / divisor) - float(target))
    start = int((one / target).view(torch.int32))
    for delta in range(-64, 65):
        candidate = torch.tensor(start + delta, dtype=torch.int32).view(torch.float32)
        assert float(one / candidate) != float(target)
        assert abs(float(one / candidate) - float(target)) >= chosen
    divisor, exact = reencode_nvfp4.reciprocal_divisor(EXACT_SCALE_2)
    assert exact and float(one / divisor) == EXACT_SCALE_2


def test_verify_rejects_corrupted_copied_and_reencoded_objects(tmp_path) -> None:
    base_path, donor_dir, weights_dir, _, _ = _build(tmp_path)
    out_path = tmp_path / "out.ninfer"
    assert _run(base_path, donor_dir, weights_dir, out_path) == 0
    with Artifact(base_path) as base:
        targets = reencode_nvfp4.plan_targets(base, LAYERS, {"down"})
    encoder = reencode_nvfp4.Encoder(SourceCheckpoint(donor_dir), SourceCheckpoint(weights_dir))
    assert reencode_nvfp4.verify_output(base_path, out_path, targets, encoder) == 0
    for object_id in ("weight/head", "weight/000001"):
        with Artifact(out_path) as out:
            offset = out.payload_offset + out.object(object_id).offset + 5
        with out_path.open("r+b") as handle:
            handle.seek(offset)
            byte = handle.read(1)[0]
            handle.seek(offset)
            handle.write(bytes([byte ^ 0x10]))
        assert reencode_nvfp4.verify_output(base_path, out_path, targets, encoder) == 1
        with out_path.open("r+b") as handle:
            handle.seek(offset)
            handle.write(bytes([byte]))


def test_refuses_fused_parts_with_different_tensor_scales(tmp_path) -> None:
    base_path, donor_dir, weights_dir, _, _ = _build(tmp_path, up_scale_2=2.0 ** -11)
    out_path = tmp_path / "out.ninfer"
    with pytest.raises(reencode_nvfp4.ReencodeError, match="different tensor scales"):
        _run(base_path, donor_dir, weights_dir, out_path)
    assert not out_path.exists()


def test_refuses_a_layer_whose_mlp_is_not_nvfp4(tmp_path) -> None:
    base_path, donor_dir, weights_dir, _, _ = _build(tmp_path, layer1_down_format="fp8")
    with Artifact(base_path) as base:
        assert reencode_nvfp4.nvfp4_layers(base) == [0]
        with pytest.raises(reencode_nvfp4.ReencodeError, match="not an NVFP4 matrix"):
            reencode_nvfp4.plan_targets(base, LAYERS, {"down"})
