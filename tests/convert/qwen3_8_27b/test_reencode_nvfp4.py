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


# Miniature geometry: the tool reads shapes from the base artifact, not from constants.
HIDDEN = 64
INTERMEDIATE = 128
DOWN_ROWS = 128  # NVFP4 objects need a multiple of 128 rows
VOCAB = 256
LAYERS = (0, 1)
NVFP4_LAYOUT = "block_scale_k16_m128x4_v1"
MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# ModelOpt weight_scale_2 values: the first has an exact FP32 reciprocal, 0x397dbe11 has none.
EXACT_SCALE_2 = 2.0 ** -12
INEXACT_SCALE_2 = struct.unpack("<f", struct.pack("<I", 0x397DBE11))[0]
# compressed-tensors global scales (divisors), not powers of two.
GLOBAL_SCALES = {"gate_proj": 4133.75, "up_proj": 4133.75, "down_proj": 3999.125}
LEAF_INDEX = {"gate": 0, "up": 1, "down": 2}


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


def _block_words(values: torch.Tensor, divisor: float, shift: torch.Tensor | None = None):
    """E4M3 block scale words for max scaling under ``divisor``, optionally moved by some codes."""

    rows, columns = values.shape
    amax = values.abs().reshape(rows, columns // 16, 16).amax(dim=2)
    words = (amax / 6.0 * divisor).to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int16)
    if shift is not None:
        words = words + shift
    return words.clamp(1, 0x7E).to(torch.uint8)


def _steps(words: torch.Tensor, *, multiplier: float | None = None,
           divisor: float | None = None) -> torch.Tensor:
    scale = words.view(torch.float8_e4m3fn).float()
    if multiplier is not None:
        return scale * torch.tensor(multiplier, dtype=torch.float32)
    return scale / torch.tensor(divisor, dtype=torch.float32)


def _values(codes: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
    rows, columns = codes.shape
    magnitude = torch.tensor(MAGNITUDES, dtype=torch.float32)[(codes & 0x7).long()]
    signed = torch.where((codes & 0x8) != 0, -magnitude, magnitude)
    return (signed.reshape(rows, -1, 16) * steps.unsqueeze(2)).reshape(rows, columns)


def _build(tmp_path, *, layout: str = "modelopt", up_scale_2: float | None = None,
           layer1_down_format: str = "nvfp4", mismatched_gate: bool = False):
    generator = torch.Generator().manual_seed(7)
    donor: dict[str, torch.Tensor] = {}
    weights: dict[str, torch.Tensor] = {}
    specs, payloads, bindings, uses = [], {}, {}, []
    expected = {}
    for index, layer in enumerate(LAYERS):
        source = f"model.language_model.layers.{layer}.mlp."
        prefix = f"text/layers/{layer}/mlp/"
        gate = (torch.randn(INTERMEDIATE, HIDDEN, generator=generator) * 0.02).to(torch.bfloat16)
        up = (torch.randn(INTERMEDIATE, HIDDEN, generator=generator) * 0.02).to(torch.bfloat16)
        down = (torch.randn(DOWN_ROWS, INTERMEDIATE, generator=generator) * 0.02).to(torch.bfloat16)
        # The model edited down after the donor quantized the original: they differ slightly.
        edited_down = (down.float() + 0.001 * torch.randn(down.shape, generator=generator)).to(
            torch.bfloat16)
        weights[source + "gate_proj.weight"] = gate.flip(0) if mismatched_gate else gate
        weights[source + "up_proj.weight"] = up
        weights[source + "down_proj.weight"] = edited_down

        # Donor: words with block scales moved away from max scaling (a calibrated choice).
        scale_2 = {"gate_proj": EXACT_SCALE_2, "up_proj": up_scale_2 or EXACT_SCALE_2,
                   "down_proj": INEXACT_SCALE_2 if layer == 1 else EXACT_SCALE_2}
        for name, values in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
            shift = torch.randint(-1, 5, (values.shape[0], values.shape[1] // 16),
                                  generator=generator, dtype=torch.int16)
            if layout == "modelopt":
                words = _block_words(values.float(), 1.0 / scale_2[name], shift)
                steps = _steps(words, multiplier=scale_2[name])
                donor[source + name + ".weight_scale_2"] = torch.tensor(scale_2[name])
            else:
                words = _block_words(values.float(), GLOBAL_SCALES[name], shift)
                steps = _steps(words, divisor=GLOBAL_SCALES[name])
                donor[source + name + ".weight_global_scale"] = torch.tensor([GLOBAL_SCALES[name]])
            codes = _nearest_codes(values.float(), steps)
            key = ".weight" if layout == "modelopt" else ".weight_packed"
            donor[source + name + key] = _pack(codes)
            donor[source + name + ".weight_scale"] = words.view(torch.float8_e4m3fn)
            expected[source + name] = (codes, words, steps)

        # Base: round-to-nearest words with max block scales, one divisor for gate and up.
        gate_up_id, down_id = f"weight/{2 * index:06d}", f"weight/{2 * index + 1:06d}"
        gate_up_shape, down_shape = (2 * INTERMEDIATE, HIDDEN), tuple(down.shape)
        for object_id, values, shape in ((gate_up_id, torch.cat([gate.float(), up.float()]),
                                          gate_up_shape),
                                         (down_id, edited_down.float(), down_shape)):
            divisor = 2688.0 / float(values.abs().max())
            words = _block_words(values, divisor)
            codes = _nearest_codes(values, _steps(words, divisor=divisor))
            payloads[object_id] = encode_nvfp4(_pack(codes), words, torch.tensor(divisor), shape)
        specs.append(TensorSpec(gate_up_id, gate_up_shape, "nvfp4", NVFP4_LAYOUT))
        if layer == 1 and layer1_down_format != "nvfp4":
            codes8 = edited_down.float().to(torch.float8_e4m3fn).view(torch.uint8)
            payloads[down_id] = encode_fp8_row_scaled(
                codes8, torch.ones(DOWN_ROWS, dtype=torch.bfloat16), down_shape)
            specs.append(TensorSpec(down_id, down_shape, "fp8_e4m3fn_row_bf16", "row_scale_v1"))
        else:
            specs.append(TensorSpec(down_id, down_shape, "nvfp4", NVFP4_LAYOUT))
        half = INTERMEDIATE * HIDDEN
        bindings[prefix + "gate"] = {"parts": [{"object": gate_up_id, "range": [0, half]}]}
        bindings[prefix + "up"] = {"parts": [{"object": gate_up_id, "range": [half, 2 * half]}]}
        bindings[prefix + "down"] = {"object": down_id}
        for leaf, source_input in (("gate", "ffn_input"), ("up", "ffn_input"),
                                   ("down", "mlp/product")):
            aux_id = f"auxiliary/{3 * index + LEAF_INDEX[leaf]:06d}"
            uses.append({"parameter": prefix + leaf, "input": f"text/layers/{layer}/{source_input}",
                         "activation_policy": "AllowA4",
                         "auxiliaries": {"activation_input_divisor": {"object": aux_id}}})
            specs.append(TensorSpec(aux_id, (), "fp32", "contiguous_le_v1"))
            payloads[aux_id] = struct.pack("<f", 40.0 + 3 * index + LEAF_INDEX[leaf])
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


def _arguments(base_path, donor_dir, out_path, *extra) -> list[str]:
    return ["--base", str(base_path), "--donor", str(donor_dir), "--out", str(out_path), *extra]


def _check_copies(base_path, out_path) -> None:
    with Artifact(base_path) as base, Artifact(out_path) as out:
        for field in ("components", "bindings", "uses", "metadata"):
            assert getattr(out.directory, field) == getattr(base.directory, field)
        assert out.directory.provenance["recipe"] == "qwen3_8_27b_nvfp4"
        assert [obj.to_json() for obj in out.objects] == [obj.to_json() for obj in base.objects]
        reencoded = {f"weight/{i:06d}" for i in range(2 * len(LAYERS))}
        for obj in base.objects:
            if obj.id not in reencoded:
                assert out.read_object(obj.id) == base.read_object(obj.id), obj.id


def _words(out_path, object_id: str, shape: tuple[int, int]):
    with Artifact(out_path) as out:
        payload = out.read_object(object_id)
    codes, scales, divisor = decode_nvfp4_words(payload, shape)
    return _unpack(codes), scales, divisor, payload


@pytest.mark.parametrize("layout", ["modelopt", "compressed-tensors"])
def test_reencodes_mlp_objects_and_copies_everything_else(tmp_path, monkeypatch, layout) -> None:
    # Row chunks that divide neither part: offsets across chunk and gate/up boundaries are exercised.
    monkeypatch.setattr(reencode_nvfp4, "ROW_CHUNK", 48)
    base_path, donor_dir, weights_dir, weights, expected = _build(tmp_path, layout=layout)
    out_path = tmp_path / "out.ninfer"
    arguments = _arguments(base_path, donor_dir, out_path, "--weights", str(weights_dir),
                           "--round", "down", "--verify")
    assert reencode_nvfp4.main(arguments) == 0
    _check_copies(base_path, out_path)
    with Artifact(out_path) as out:
        assert out.directory.provenance["reencode"]["layers"] == list(LAYERS)

    for index, layer in enumerate(LAYERS):
        source = f"model.language_model.layers.{layer}.mlp."
        # Gate/up: the donor's words as they are, decoding to the donor's values.
        shape = (2 * INTERMEDIATE, HIDDEN)
        codes, scales, divisor, payload = _words(out_path, f"weight/{2 * index:06d}", shape)
        parts = [expected[source + name] for name in ("gate_proj", "up_proj")]
        assert torch.equal(codes, torch.cat([part[0] for part in parts]))
        assert torch.equal(scales, torch.cat([part[1] for part in parts]))
        donor_values = torch.cat([_values(part[0], part[2]) for part in parts])
        if layout == "modelopt":
            assert float(torch.tensor(1.0) / divisor) == EXACT_SCALE_2
            torch.testing.assert_close(dequantize_nvfp4(payload, shape), donor_values,
                                       rtol=2.0 ** -21, atol=0.0)
        else:
            assert float(divisor) == GLOBAL_SCALES["gate_proj"]
            assert torch.equal(dequantize_nvfp4(payload, shape), donor_values)

        # Down: the donor's scales, codes rounded from the edited local weights.
        shape = (DOWN_ROWS, INTERMEDIATE)
        codes, scales, divisor, _ = _words(out_path, f"weight/{2 * index + 1:06d}", shape)
        assert torch.equal(scales, expected[source + "down_proj"][1])
        steps = scales.view(torch.float8_e4m3fn).float() * (torch.tensor(1.0) / divisor)
        local = weights[source + "down_proj.weight"].float()
        assert torch.equal(codes, _nearest_codes(local, steps))

    report = json.loads((tmp_path / "out.ninfer.reencode.json").read_text())
    assert report["verified"] is True
    assert report["inexact_divisors"] == (1 if layout == "modelopt" else 0)
    assert report["recipe"] == "qwen3_8_27b_nvfp4"
    assert report["objects"][0]["codes_equal_to_rounded_weights"] == 1.0
    assert all(item["relative_rms_error"] < 0.2 for item in report["objects"])


def test_donor_words_everywhere_without_weights(tmp_path) -> None:
    base_path, donor_dir, _, _, expected = _build(tmp_path)
    out_path = tmp_path / "out.ninfer"
    assert reencode_nvfp4.main(_arguments(base_path, donor_dir, out_path, "--verify")) == 0
    _check_copies(base_path, out_path)
    codes, scales, _, _ = _words(out_path, "weight/000001", (DOWN_ROWS, INTERMEDIATE))
    down = expected["model.language_model.layers.0.mlp.down_proj"]
    assert torch.equal(codes, down[0]) and torch.equal(scales, down[1])


def test_rounds_gate_and_up_together(tmp_path) -> None:
    base_path, donor_dir, weights_dir, weights, expected = _build(tmp_path)
    out_path = tmp_path / "out.ninfer"
    arguments = _arguments(base_path, donor_dir, out_path, "--weights", str(weights_dir),
                           "--round", "gate", "up")
    assert reencode_nvfp4.main(arguments) == 0
    codes, scales, divisor, _ = _words(out_path, "weight/000000", (2 * INTERMEDIATE, HIDDEN))
    source = "model.language_model.layers.0.mlp."
    local = torch.cat([weights[source + name + ".weight"].float() for name in ("gate_proj", "up_proj")])
    steps = scales.view(torch.float8_e4m3fn).float() * (torch.tensor(1.0) / divisor)
    assert torch.equal(codes, _nearest_codes(local, steps))
    down = expected[source + "down_proj"]
    down_codes, _, _, _ = _words(out_path, "weight/000001", (DOWN_ROWS, INTERMEDIATE))
    assert torch.equal(down_codes, down[0])


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
    arguments = _arguments(base_path, donor_dir, out_path, "--weights", str(weights_dir),
                           "--round", "down")
    assert reencode_nvfp4.main(arguments) == 0
    report = json.loads((tmp_path / "out.ninfer.reencode.json").read_text())
    expected = {item["object"]: item["payload_sha256"] for item in report["objects"]}
    assert reencode_nvfp4.verify_output(base_path, out_path, expected) == 0
    for object_id in ("weight/head", "weight/000001"):
        with Artifact(out_path) as out:
            offset = out.payload_offset + out.object(object_id).offset + 5
        with out_path.open("r+b") as handle:
            handle.seek(offset)
            byte = handle.read(1)[0]
            handle.seek(offset)
            handle.write(bytes([byte ^ 0x10]))
        assert reencode_nvfp4.verify_output(base_path, out_path, expected) == 1
        with out_path.open("r+b") as handle:
            handle.seek(offset)
            handle.write(bytes([byte]))


def test_failed_verification_removes_the_output(tmp_path, monkeypatch) -> None:
    base_path, donor_dir, weights_dir, _, _ = _build(tmp_path)
    out_path = tmp_path / "out.ninfer"
    monkeypatch.setattr(reencode_nvfp4, "verify_output", lambda *arguments: 1)
    arguments = _arguments(base_path, donor_dir, out_path, "--weights", str(weights_dir),
                           "--round", "down", "--verify")
    assert reencode_nvfp4.main(arguments) == 1
    assert not out_path.exists()
    report = json.loads((tmp_path / "out.ninfer.reencode.json").read_text())
    assert report["verified"] is False and report["removed"] is True


@pytest.mark.parametrize(
    "build, extra, match",
    [
        ({"up_scale_2": 2.0 ** -11}, ("--round", "down"), "different tensor scales"),
        ({"mismatched_gate": True}, ("--round", "down"), "does the donor quantize"),
        ({}, ("--round", "gate"), "round both or neither"),
        ({"layer1_down_format": "fp8"}, ("--round", "down", "--layers", "0-1"),
         "not an NVFP4 matrix"),
    ],
)
def test_refusals_leave_no_output(tmp_path, build, extra, match) -> None:
    base_path, donor_dir, weights_dir, _, _ = _build(tmp_path, **build)
    out_path = tmp_path / "out.ninfer"
    with pytest.raises(reencode_nvfp4.ReencodeError, match=match):
        reencode_nvfp4.main(_arguments(base_path, donor_dir, out_path, "--weights",
                                       str(weights_dir), *extra))
    assert not out_path.exists()


def test_argument_and_base_checks(tmp_path) -> None:
    base_path, donor_dir, weights_dir, _, _ = _build(tmp_path, layer1_down_format="fp8")
    out_path = tmp_path / "out.ninfer"
    for extra in (("--round", "down"), ("--layers", "1-0")):
        with pytest.raises(SystemExit):
            reencode_nvfp4.main(_arguments(base_path, donor_dir, out_path, *extra))
    with pytest.raises(SystemExit):
        reencode_nvfp4.main(_arguments(base_path, donor_dir, base_path))
    with Artifact(base_path) as base:
        assert reencode_nvfp4.nvfp4_layers(base) == [0]
    # An earlier re-encode is not a valid base.
    first = tmp_path / "first.ninfer"
    assert reencode_nvfp4.main(_arguments(base_path, donor_dir, first)) == 0
    with pytest.raises(reencode_nvfp4.ReencodeError, match="already re-encoded"):
        reencode_nvfp4.main(_arguments(first, donor_dir, tmp_path / "second.ninfer"))
