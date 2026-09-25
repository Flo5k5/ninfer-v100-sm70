from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import struct

import pytest
from safetensors.torch import save_file
import torch

from tools.artifact.codecs.fp8_row import encode_fp8_row_scaled
from tools.artifact.codecs.nvfp4 import decode_nvfp4_words, dequantize_nvfp4, encode_nvfp4
from tools.artifact.framing import HEADER
from tools.artifact.reader import Artifact, TensorObject
from tools.artifact.schema import ResourceSpec, TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.convert.qwen3_8_27b import reencode_nvfp4

from ..path_fragments import assert_no_path_fragments


# Miniature geometry: the tool reads shapes from the base artifact, not from constants.
HIDDEN = 64
INTERMEDIATE = 128
DOWN_ROWS = 128  # NVFP4 objects need a multiple of 128 rows
VOCAB = 256
LAYERS = (0, 1)
NVFP4_LAYOUT = "block_scale_k16_m128x4_v1"
MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# ModelOpt weight_scale_2 words and the divisor word the artifact must store for each: the first
# two have exact FP32 reciprocals that BF16 cannot represent; 0x397DBE11 has none, so the nearest
# reciprocal is stored and reported inexact.
GATE_UP_SCALE_2 = 0x39A2877F
DOWN_SCALE_2 = {0: 0x3A03126F, 1: 0x397DBE11}
HEAD_SCALE_2 = 0x39A2877F
DIVISOR_WORDS = {0x39A2877F: 0x45499CE7, 0x3A03126F: 0x44F9FFFF, 0x397DBE11: 0x4581238A}
INEXACT_SCALES_2 = {0x397DBE11}
# compressed-tensors global scales are the divisors themselves, not powers of two.
GLOBAL_SCALES = {"gate_proj": 4133.75, "up_proj": 4133.75, "down_proj": 3999.125}
DONOR_SUFFIXES = {"modelopt": (".weight", ".weight_scale", ".weight_scale_2"),
                  "compressed-tensors": (".weight_packed", ".weight_scale", ".weight_global_scale")}
LEAF_INDEX = {"gate": 0, "up": 1, "down": 2}
DONOR_LABEL = "org/calibrated-NVFP4"
WEIGHTS_LABEL = "org/fine-tune"
# The base's provenance as a graft leaves it, with local paths, and what the output keeps of it.
INHERITED_PROVENANCE = {
    "converter": "ninfer-v3",
    "recipe": "qwen3_8_27b_nvfp4",
    "sources": {"single": {"label": "org/fine-tune-NVFP4"}},
    "graft": {"tool": "tools.convert.qwen3_8_27b.graft_single_source",
              "template_artifact_id": "00112233445566778899aabbccddeeff",
              "copied": ["resource/", "vision/"]},
}
REMOVED_BASE_PATHS = ["sources.single.path", "ranking", "graft.template",
                      "graft.template_sources.base.path"]


@dataclass
class Fixture:
    base: Path
    donor_dir: Path
    weights_dir: Path
    donor: dict[str, torch.Tensor]
    weights: dict[str, torch.Tensor]
    expected: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def _f32(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", word))[0]


def _word(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _scale_2_word(layer: int, name: str) -> int:
    return DOWN_SCALE_2[layer] if name == "down_proj" else GATE_UP_SCALE_2


def _divisor_word(layout: str, layer: int, name: str) -> int:
    """The divisor word the artifact must store, from the donor's tensor scale alone."""

    if layout == "modelopt":
        return DIVISOR_WORDS[_scale_2_word(layer, name)]
    return _word(GLOBAL_SCALES[name])


def _decode_steps(scales: torch.Tensor, divisor_word: int) -> torch.Tensor:
    """One E2M1 unit per block as the FP32 decode routes form it: e4m3 * (1.0f / divisor)."""

    reciprocal = torch.tensor(1.0) / torch.tensor(_f32(divisor_word), dtype=torch.float32)
    return scales.view(torch.float8_e4m3fn).float() * reciprocal


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


def _save(directory: Path, tensors: dict[str, torch.Tensor]) -> None:
    save_file(tensors, str(directory / "model.safetensors"))


def _base_provenance(tmp_path: Path) -> dict:
    return {
        "converter": "ninfer-v3",
        "recipe": "qwen3_8_27b_nvfp4",
        "sources": {"single": {"label": "org/fine-tune-NVFP4",
                               "path": str(tmp_path / "fine-tune-NVFP4")}},
        "ranking": str(tmp_path / "ranking.counts.i64"),
        "graft": {"tool": "tools.convert.qwen3_8_27b.graft_single_source",
                  "template": str(tmp_path / "template.ninfer"),
                  "template_artifact_id": "00112233445566778899aabbccddeeff",
                  "template_sources": {"base": {"path": "~/models/base-hf-bf16"}},
                  "copied": ["resource/", "vision/"]},
    }


def _build(tmp_path, *, layout: str = "modelopt", up_scale_2: int | None = None,
           layer1_down_format: str = "nvfp4", mismatched_gate: bool = False) -> Fixture:
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
        scale_2 = {name: _scale_2_word(layer, name) for name in ("gate_proj", "down_proj")}
        scale_2["up_proj"] = up_scale_2 or GATE_UP_SCALE_2
        for name, values in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
            shift = torch.randint(-1, 5, (values.shape[0], values.shape[1] // 16),
                                  generator=generator, dtype=torch.int16)
            if layout == "modelopt":
                multiplier = _f32(scale_2[name])
                words = _block_words(values.float(), 1.0 / multiplier, shift)
                steps = _steps(words, multiplier=multiplier)
                donor[source + name + ".weight_scale_2"] = torch.tensor(multiplier)
            else:
                words = _block_words(values.float(), GLOBAL_SCALES[name], shift)
                steps = _steps(words, divisor=GLOBAL_SCALES[name])
                donor[source + name + ".weight_global_scale"] = torch.tensor([GLOBAL_SCALES[name]])
            codes = _nearest_codes(values.float(), steps)
            donor[source + name + DONOR_SUFFIXES[layout][0]] = _pack(codes)
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
        if layer == 1 and layer1_down_format == "bf16":
            payloads[down_id] = edited_down.to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()
            specs.append(TensorSpec(down_id, down_shape, "bf16", "contiguous_le_v1"))
        elif layer == 1 and layer1_down_format == "fp8":
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
    head = (torch.randn(VOCAB, HIDDEN, generator=generator) * 0.02).to(torch.bfloat16)
    # Donor lm_head words (calibrated), so a conversion can graft the head as the loader
    # full-a profile requires.
    head_multiplier = _f32(HEAD_SCALE_2)
    head_words = _block_words(head.float(), 1.0 / head_multiplier,
                              torch.randint(-1, 5, (VOCAB, HIDDEN // 16),
                                            generator=generator, dtype=torch.int16))
    head_steps = _steps(head_words, multiplier=head_multiplier)
    donor["lm_head.weight"] = _pack(_nearest_codes(head.float(), head_steps))
    donor["lm_head.weight_scale"] = head_words.view(torch.float8_e4m3fn)
    donor["lm_head.weight_scale_2"] = torch.tensor(head_multiplier)
    weights["lm_head.weight"] = head
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
                            provenance=_base_provenance(tmp_path))
    for object_id, payload in payloads.items():
        writer.write_object(object_id, payload)
    writer.finish()
    donor_dir, weights_dir = tmp_path / "donor", tmp_path / "weights"
    donor_dir.mkdir()
    weights_dir.mkdir()
    _save(donor_dir, donor)
    _save(weights_dir, weights)
    return Fixture(base_path, donor_dir, weights_dir, donor, weights, expected)


def _arguments(fixture: Fixture, out_path: Path, *extra: str, weights: bool = False) -> list[str]:
    arguments = ["--base", str(fixture.base), "--donor", str(fixture.donor_dir),
                 "--donor-label", DONOR_LABEL, "--out", str(out_path)]
    if weights:
        arguments += ["--weights", str(fixture.weights_dir), "--weights-label", WEIGHTS_LABEL]
    return arguments + list(extra)


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


def _stored_sha256(directory: Path) -> dict[str, str]:
    """SHA-256 of every tensor's stored bytes, read from the safetensors file itself."""

    data = (directory / "model.safetensors").read_bytes()
    (length,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8:8 + length])
    start = 8 + length
    return {name: hashlib.sha256(data[start + meta["data_offsets"][0]:
                                      start + meta["data_offsets"][1]]).hexdigest()
            for name, meta in header.items() if name != "__metadata__"}


def _assert_no_local_paths(artifact: Path, tmp_path: Path) -> None:
    """No directory JSON string is a filesystem path or names a home or temporary directory."""

    with artifact.open("rb") as handle:
        _, json_bytes, _ = HEADER.unpack(handle.read(HEADER.size))
        text = handle.read(json_bytes).decode("utf-8")
    assert_no_path_fragments(text, tmp_path)


def _report(tmp_path) -> dict:
    return json.loads((tmp_path / "out.ninfer.reencode.json").read_text())


def _assert_nothing_written(tmp_path) -> None:
    assert not (tmp_path / "out.ninfer").exists()
    assert not list(tmp_path.glob(".out.ninfer.*"))


@pytest.mark.parametrize("layout", ["modelopt", "compressed-tensors"])
def test_reencodes_mlp_objects_and_copies_everything_else(tmp_path, monkeypatch, layout) -> None:
    # Row chunks that divide neither part: offsets across chunk and gate/up boundaries are exercised.
    monkeypatch.setattr(reencode_nvfp4, "ROW_CHUNK", 48)
    fixture = _build(tmp_path, layout=layout)
    out_path = tmp_path / "out.ninfer"
    arguments = _arguments(fixture, out_path, "--round", "down", "--verify", weights=True)
    assert reencode_nvfp4.main(arguments) == 0
    _check_copies(fixture.base, out_path)
    report = _report(tmp_path)
    assert report["verified"] is True
    assert report["recipe"] == "qwen3_8_27b_nvfp4"
    assert report["reencode"]["mlp_layers"] == list(LAYERS)
    assert report["reencode"]["output_head"] is False
    assert report["inexact_divisors"] == (1 if layout == "modelopt" else 0)
    entries = {item["object"]: item for item in report["objects"]}
    donor_sha256, weights_sha256 = _stored_sha256(fixture.donor_dir), _stored_sha256(
        fixture.weights_dir)

    for index, layer in enumerate(LAYERS):
        source = f"model.language_model.layers.{layer}.mlp."
        # Gate/up: the donor's words as they are, under the divisor of the donor's tensor scale.
        gate_up_id, shape = f"weight/{2 * index:06d}", (2 * INTERMEDIATE, HIDDEN)
        word = _divisor_word(layout, layer, "gate_proj")
        codes, scales, divisor, payload = _words(out_path, gate_up_id, shape)
        assert _word(float(divisor)) == word
        parts = [fixture.expected[source + name] for name in ("gate_proj", "up_proj")]
        assert torch.equal(codes, torch.cat([part[0] for part in parts]))
        assert torch.equal(scales, torch.cat([part[1] for part in parts]))
        donor_values = torch.cat([_values(part[0], part[2]) for part in parts])
        if layout == "modelopt":
            # The FP32 decode route reproduces the donor's values bit for bit; division is close.
            assert torch.equal(_values(codes, _decode_steps(scales, word)), donor_values)
            torch.testing.assert_close(dequantize_nvfp4(payload, shape), donor_values,
                                       rtol=2.0 ** -21, atol=0.0)
        else:
            assert torch.equal(dequantize_nvfp4(payload, shape), donor_values)

        # Down: the donor's scales and divisor, codes rounded from the edited local weights.
        down_id, shape = f"weight/{2 * index + 1:06d}", (DOWN_ROWS, INTERMEDIATE)
        word = _divisor_word(layout, layer, "down_proj")
        codes, scales, divisor, _ = _words(out_path, down_id, shape)
        assert _word(float(divisor)) == word
        assert torch.equal(scales, fixture.expected[source + "down_proj"][1])
        steps = _decode_steps(scales, word)
        local = fixture.weights[source + "down_proj.weight"].float()
        oracle = _nearest_codes(local, steps)
        assert torch.equal(codes, oracle)
        clipped = local.double().abs().reshape(DOWN_ROWS, -1, 16) > 6.0 * steps.double().unsqueeze(2)
        assert entries[down_id]["saturated"] == int(clipped.sum()) > 0
        assert entries[down_id]["zeroed"] == int((((oracle & 0x7) == 0) & (local != 0)).sum()) > 0

        for object_id, names in ((gate_up_id, ("gate_proj", "up_proj")), (down_id, ("down_proj",))):
            entry = entries[object_id]
            word = _divisor_word(layout, layer, names[0])
            assert entry["weight_divisor_word"] == f"0x{word:08x}"
            assert entry["divisor_exact"] is (
                layout != "modelopt" or _scale_2_word(layer, names[0]) not in INEXACT_SCALES_2)
            assert entry["relative_rms_error_vs_base"] < 0.3
            by_parameter = entry["relative_rms_error_vs_base_by_parameter"]
            assert list(by_parameter) == entry["parameters"]
            assert max(by_parameter.values()) < 0.3
            assert entry["relative_rms_error_vs_weights"] < 0.2
            tensors = [source + name + suffix for name in names for suffix in DONOR_SUFFIXES[layout]]
            assert entry["donor_sha256"] == {name: donor_sha256[name] for name in tensors}
            matrices = [source + name + ".weight" for name in names]
            assert entry["weights_sha256"] == {name: weights_sha256[name] for name in matrices}
    assert entries["weight/000000"]["codes_equal_to_rounded_weights"] == 1.0


def test_output_metadata_names_inputs_without_paths(tmp_path) -> None:
    fixture = _build(tmp_path)
    out_path = tmp_path / "out.ninfer"
    assert reencode_nvfp4.main(_arguments(fixture, out_path, "--round", "down", weights=True)) == 0
    with Artifact(out_path) as out:
        provenance = dict(out.directory.provenance)
    record = provenance.pop("reencode")
    assert provenance == INHERITED_PROVENANCE
    assert record["donor"] == {"label": DONOR_LABEL}
    assert record["weights"] == {"label": WEIGHTS_LABEL}
    assert record["base"] == "base.ninfer"
    assert record["removed_base_paths"] == REMOVED_BASE_PATHS

    _assert_no_local_paths(out_path, tmp_path)


def test_reencode_report_names_the_artifact_by_name_only(tmp_path) -> None:
    """<out>.reencode.json travels separately from the artifact: it must stay path-free too,
    naming the output by file name only."""

    fixture = _build(tmp_path)
    out_path = tmp_path / "out.ninfer"
    assert reencode_nvfp4.main(_arguments(fixture, out_path, "--round", "down", weights=True)) == 0
    report = _report(tmp_path)
    assert report["artifact"] == out_path.name
    assert "/" not in report["artifact"] and "\\" not in report["artifact"]
    assert_no_path_fragments(json.dumps(report), tmp_path)


def test_donor_words_everywhere_without_weights(tmp_path) -> None:
    fixture = _build(tmp_path)
    out_path = tmp_path / "out.ninfer"
    assert reencode_nvfp4.main(_arguments(fixture, out_path, "--verify")) == 0
    _check_copies(fixture.base, out_path)
    for index, layer in enumerate(LAYERS):
        down = fixture.expected[f"model.language_model.layers.{layer}.mlp.down_proj"]
        codes, scales, divisor, _ = _words(out_path, f"weight/{2 * index + 1:06d}",
                                           (DOWN_ROWS, INTERMEDIATE))
        assert torch.equal(codes, down[0]) and torch.equal(scales, down[1])
        assert _word(float(divisor)) == DIVISOR_WORDS[DOWN_SCALE_2[layer]]
    report = _report(tmp_path)
    assert "weights" not in report["reencode"]
    for entry in report["objects"]:
        assert entry["relative_rms_error_vs_base"] < 0.3
        assert "relative_rms_error_vs_weights" not in entry and "weights_sha256" not in entry


def test_donor_of_other_weights_is_refused_without_weights(tmp_path) -> None:
    fixture = _build(tmp_path)
    # Layers swapped: valid NVFP4 words of the right shapes, but of other weights.
    swap = {"layers.0.": "layers.1.", "layers.1.": "layers.0."}
    _save(fixture.donor_dir, {re.sub(r"layers\.[01]\.", lambda match: swap[match.group(0)], name):
                              tensor for name, tensor in fixture.donor.items()})
    with pytest.raises(reencode_nvfp4.ReencodeError,
                       match=r"weight/000000: relative RMS error 1\.\d+ against the base"):
        reencode_nvfp4.main(_arguments(fixture, tmp_path / "out.ninfer", "--verify"))
    _assert_nothing_written(tmp_path)


def test_rounds_gate_and_up_together(tmp_path) -> None:
    fixture = _build(tmp_path)
    out_path = tmp_path / "out.ninfer"
    assert reencode_nvfp4.main(_arguments(fixture, out_path, "--round", "gate", "up",
                                          weights=True)) == 0
    codes, scales, divisor, _ = _words(out_path, "weight/000000", (2 * INTERMEDIATE, HIDDEN))
    word = DIVISOR_WORDS[GATE_UP_SCALE_2]
    assert _word(float(divisor)) == word
    source = "model.language_model.layers.0.mlp."
    local = torch.cat([fixture.weights[source + name + ".weight"].float()
                       for name in ("gate_proj", "up_proj")])
    assert torch.equal(codes, _nearest_codes(local, _decode_steps(scales, word)))
    down_codes, _, _, _ = _words(out_path, "weight/000001", (DOWN_ROWS, INTERMEDIATE))
    assert torch.equal(down_codes, fixture.expected[source + "down_proj"][0])


def _set_weight(name: str, row: int, column: int, value: float):
    def mutate(fixture: Fixture) -> None:
        fixture.weights[name] = fixture.weights[name].clone()
        fixture.weights[name][row, column] = value
        _save(fixture.weights_dir, fixture.weights)
    return mutate


def _zero_weights(*names: str):
    def mutate(fixture: Fixture) -> None:
        for name in names:
            fixture.weights[name] = torch.zeros_like(fixture.weights[name])
        _save(fixture.weights_dir, fixture.weights)
    return mutate


def _zero_block_scale(name: str):
    def mutate(fixture: Fixture) -> None:
        words = fixture.donor[name].view(torch.uint8).clone()
        words[0, 0] = 0
        fixture.donor[name] = words.view(torch.float8_e4m3fn)
        _save(fixture.donor_dir, fixture.donor)
    return mutate


LAYER0 = "model.language_model.layers.0.mlp."


@pytest.mark.parametrize(
    "build, mutate, extra, match",
    [
        ({"mismatched_gate": True}, None, ("--round", "down"),
         "against --weights exceeds --max-error 0.3; does the donor quantize"),
        ({}, _set_weight(LAYER0 + "down_proj.weight", 3, 5, float("nan")), ("--round", "down"),
         r"down_proj.weight: rows \[0, 128\) of --weights hold non-finite values"),
        ({}, _set_weight(LAYER0 + "gate_proj.weight", 7, 1, float("inf")), (),
         "gate_proj.weight: rows .* hold non-finite values"),
        ({}, _zero_weights(LAYER0 + "gate_proj.weight", LAYER0 + "up_proj.weight"), (),
         "relative RMS error inf against --weights"),
        ({}, _zero_block_scale(LAYER0 + "down_proj.weight_scale"), ("--round", "down"),
         "weight/000001: 16 nonzero --weights values .* zero donor block scale"),
    ],
)
def test_refusals_while_encoding_leave_no_output(tmp_path, build, mutate, extra, match) -> None:
    fixture = _build(tmp_path, **build)
    if mutate is not None:
        mutate(fixture)
    with pytest.raises(reencode_nvfp4.ReencodeError, match=match):
        reencode_nvfp4.main(_arguments(fixture, tmp_path / "out.ninfer", *extra, weights=True))
    _assert_nothing_written(tmp_path)


def _edit_donor(name: str, value):
    def mutate(fixture: Fixture) -> None:
        if value is None:
            del fixture.donor[name]
        else:
            fixture.donor[name] = value(fixture.donor[name])
        _save(fixture.donor_dir, fixture.donor)
    return mutate


def _edit_weights(name: str, value):
    def mutate(fixture: Fixture) -> None:
        fixture.weights[name] = value(fixture.weights[name])
        _save(fixture.weights_dir, fixture.weights)
    return mutate


def _both_rows(rows: int):
    def mutate(fixture: Fixture) -> None:
        for suffix in (".weight", ".weight_scale"):
            name = LAYER0 + "down_proj" + suffix
            fixture.donor[name] = fixture.donor[name][:rows].contiguous()
        _save(fixture.donor_dir, fixture.donor)
    return mutate


LAYER1 = "model.language_model.layers.1.mlp."


@pytest.mark.parametrize(
    "build, mutate, extra, match",
    [
        ({}, _edit_donor(LAYER1 + "down_proj.weight_scale", None), (),
         "donor checkpoint is missing .*layers.1.mlp.down_proj.weight_scale$"),
        ({}, _edit_donor(LAYER1 + "up_proj.weight_scale_2", None), (),
         "missing .*layers.1.mlp.up_proj.weight_scale_2"),
        ({}, _edit_donor(LAYER1 + "down_proj.weight", None), (),
         "layers.1.mlp.down_proj: no NVFP4 words in the donor checkpoint"),
        ({}, _edit_donor(LAYER1 + "down_proj.weight", lambda t: t[:64].contiguous()), (),
         r"codes \(64, 64\) and scales \(128, 8\) are not an NVFP4 matrix of 128 columns"),
        ({}, _both_rows(64), (),
         "weight/000001: donor matrices have 64 rows for text/layers/0/mlp/down, object has 128"),
        ({}, _edit_donor(LAYER1 + "down_proj.weight_scale", lambda t: t.view(torch.uint8)), (),
         "weight_scale must be F8_E4M3 block scales, not U8"),
        ({"up_scale_2": 0x39800000}, None, (), "weight/000000: fused donor matrices have different"),
        *[({}, _edit_donor(LAYER1 + "down_proj.weight_scale_2", lambda t, v=v: torch.tensor(v)),
           (), "down_proj.weight_scale_2 = .* must be finite and positive")
          for v in (float("nan"), float("inf"), 0.0, -2.0 ** -12)],
        ({}, _edit_donor(LAYER1 + "down_proj.weight_scale_2", lambda t: t.half()), (),
         "down_proj.weight_scale_2 must be one FP32 value"),
        ({}, _edit_donor(LAYER1 + "down_proj.weight_scale_2", lambda t: t.repeat(2)), (),
         "down_proj.weight_scale_2 must be one FP32 value"),
        ({}, _edit_donor(LAYER1 + "down_proj.weight_scale_2", lambda t: torch.tensor(1e-45)), (),
         "has no finite positive FP32 divisor"),
        ({}, _edit_weights(LAYER1 + "down_proj.weight", lambda t: torch.cat([t, t])),
         ("--weights",), r"--weights shape \(256, 128\) differs from the donor matrix \(128, 128\)"),
        ({}, _edit_weights(LAYER1 + "up_proj.weight", lambda t: t.to(torch.float8_e4m3fn)),
         ("--weights",), "up_proj.weight: expected unquantized weights in --weights, got F8_E4M3"),
        # An FP8 base object is now CONVERTED (donor words, or scales rounded on --weights),
        # so the refusal suite keeps a genuinely foreign format instead:
        ({"layer1_down_format": "bf16"}, None, ("--layers", "0-1"),
         "neither an NVFP4 nor an FP8 row-scaled matrix"),
        ({}, None, ("--weights", "--round", "gate"), "round both or neither"),
    ],
)
def test_refusals_before_writing(tmp_path, monkeypatch, build, mutate, extra, match) -> None:
    def no_writer(*arguments, **keywords):
        raise AssertionError("the output was created before the inputs were checked")

    monkeypatch.setattr(reencode_nvfp4, "ArtifactWriter", no_writer)
    fixture = _build(tmp_path, **build)
    if mutate is not None:
        mutate(fixture)
    weights = "--weights" in extra
    options = [item for item in extra if item != "--weights"]
    with pytest.raises(reencode_nvfp4.ReencodeError, match=match):
        reencode_nvfp4.main(_arguments(fixture, tmp_path / "out.ninfer", *options,
                                       weights=weights))


@pytest.mark.parametrize(
    "extra",
    [
        ("--round", "down"),
        ("--round",),
        ("--layers", "1-0"),
        *[("--max-error", value) for value in ("nan", "inf", "0", "-0.1")],
        ("--weights-label", WEIGHTS_LABEL),
        *[("--donor-label", label) for label in ("/data/models/donor", "~/models/donor",
                                                 "C:\\models\\donor", "../donor", "", " org/x")],
    ],
)
def test_argument_checks(tmp_path, extra) -> None:
    fixture = _build(tmp_path)
    with pytest.raises(SystemExit):
        reencode_nvfp4.main(_arguments(fixture, tmp_path / "out.ninfer", *extra))
    _assert_nothing_written(tmp_path)


def test_labels_are_required(tmp_path) -> None:
    fixture = _build(tmp_path)
    out = str(tmp_path / "out.ninfer")
    common = ["--base", str(fixture.base), "--donor", str(fixture.donor_dir), "--out", out]
    for arguments in (common, [*common, "--donor-label", DONOR_LABEL, "--weights",
                               str(fixture.weights_dir)]):
        with pytest.raises(SystemExit):
            reencode_nvfp4.main(arguments)
    with pytest.raises(SystemExit):
        reencode_nvfp4.main(_arguments(fixture, fixture.base))  # --out is the base
    _assert_nothing_written(tmp_path)


def test_base_checks(tmp_path) -> None:
    fixture = _build(tmp_path, layer1_down_format="fp8")
    with Artifact(fixture.base) as base:
        assert reencode_nvfp4.nvfp4_layers(base) == [0]
    # An earlier re-encode is not a valid base.
    first = tmp_path / "first.ninfer"
    assert reencode_nvfp4.main(_arguments(fixture, first)) == 0
    with pytest.raises(reencode_nvfp4.ReencodeError, match="already re-encoded"):
        reencode_nvfp4.main(_arguments(replace(fixture, base=first), tmp_path / "second.ninfer"))


def test_verify_rejects_corrupted_copied_and_reencoded_objects(tmp_path) -> None:
    fixture = _build(tmp_path)
    out_path = tmp_path / "out.ninfer"
    assert reencode_nvfp4.main(_arguments(fixture, out_path, "--round", "down",
                                          weights=True)) == 0
    expected = {item["object"]: item["payload_sha256"] for item in _report(tmp_path)["objects"]}
    with Artifact(fixture.base) as base:
        keywords = {"converted": {}, "uses": list(base.directory.uses),
                    "recipe": "qwen3_8_27b_nvfp4"}
    assert reencode_nvfp4.verify_output(fixture.base, out_path, expected, **keywords) == 0
    for object_id in ("weight/head", "weight/000001"):
        with Artifact(out_path) as out:
            offset = out.payload_offset + out.object(object_id).offset + 5
        with out_path.open("r+b") as handle:
            handle.seek(offset)
            byte = handle.read(1)[0]
            handle.seek(offset)
            handle.write(bytes([byte ^ 0x10]))
        assert reencode_nvfp4.verify_output(fixture.base, out_path, expected, **keywords) == 1
        with out_path.open("r+b") as handle:
            handle.seek(offset)
            handle.write(bytes([byte]))


def test_failed_verification_removes_the_output(tmp_path, monkeypatch) -> None:
    fixture = _build(tmp_path)
    out_path = tmp_path / "out.ninfer"
    monkeypatch.setattr(reencode_nvfp4, "verify_output", lambda *arguments, **keywords: 1)
    assert reencode_nvfp4.main(_arguments(fixture, out_path, "--round", "down", "--verify",
                                          weights=True)) == 1
    assert not out_path.exists()
    report = _report(tmp_path)
    assert report["verified"] is False and report["removed"] is True


def test_converted_layer_keeps_following_object_records(tmp_path) -> None:
    """A converted object changes byte size, shifting the offsets of every object stored after
    it: verification must compare records without the offset (found on the real G23 build,
    where weight/000156 shrinks and weight/000157 shifts)."""
    fixture = _build(tmp_path, layer1_down_format="fp8")
    out_path = tmp_path / "g23-offset.ninfer"
    assert reencode_nvfp4.main(_arguments(fixture, out_path, "--layers", "0-1",
                                          "--round", "down", weights=True)) == 0
    with Artifact(fixture.base) as base, Artifact(out_path) as out:
        converted = {"weight/000003", "weight/head"}
        shifted = 0
        for base_obj, out_obj in zip(base.objects, out.objects):
            assert base_obj.id == out_obj.id
            if base_obj.offset != out_obj.offset:
                shifted += 1
                # A shifted object that was not converted keeps its own record intact apart
                # from the offset; converted objects legitimately change format and layout.
                assert base_obj.id in converted or (
                    {k: v for k, v in base_obj.to_json().items() if k != "offset"}
                    == {k: v for k, v in out_obj.to_json().items() if k != "offset"})
        assert shifted > 0, "the fixture must store objects after the converted one"


def test_converts_an_fp8_layer_to_calibrated_nvfp4(tmp_path) -> None:
    """G23's shape: layer 1's down object is FP8 in the base and leaves as NVFP4, donor scales
    rounded on the local weights; the object's format and layout change in the output."""
    fixture = _build(tmp_path, layer1_down_format="fp8")
    out_path = tmp_path / "g23.ninfer"
    assert reencode_nvfp4.main(_arguments(fixture, out_path, "--layers", "0-1",
                                          "--round", "down", weights=True)) == 0
    source = "model.language_model.layers.1.mlp."
    with Artifact(out_path) as out:
        converted = out.object("weight/000003")
        assert isinstance(converted, TensorObject)
        assert converted.format == "nvfp4"
        assert converted.layout == NVFP4_LAYOUT
        assert tuple(converted.shape) == (DOWN_ROWS, INTERMEDIATE)
        # A conversion switches the recipe so the engine resolves the full-a weights profile.
        assert out.directory.provenance["recipe"] == "qwen3_8_27b_nvfp4-full-a"
        assert out.directory.provenance["reencode"]["base_recipe"] == "qwen3_8_27b_nvfp4"
        # Converted leaves carry AllowA4 and an activation input divisor auxiliary.
        down_use = next(use for use in out.directory.uses
                        if use["parameter"] == "text/layers/1/mlp/down")
        assert down_use["activation_policy"] == "AllowA4"
        aux_id = down_use["auxiliaries"]["activation_input_divisor"]["object"]
        assert out.object(aux_id).format == "fp32"
        # The output head converts too (stage A): NVFP4 object, AllowA4, one shared divisor.
        head = out.object("weight/head")
        assert head.format == "nvfp4" and head.layout == NVFP4_LAYOUT
        head_use = next(use for use in out.directory.uses
                        if use["parameter"] == "text/output_head")
        assert head_use["activation_policy"] == "AllowA4"
        assert out.object(head_use["auxiliaries"]["activation_input_divisor"]
                          ["object"]).format == "fp32"
    codes, scales, divisor, _ = _words(out_path, "weight/000003", (DOWN_ROWS, INTERMEDIATE))
    local = fixture.weights[source + "down_proj.weight"].float()
    divisor_word = DIVISOR_WORDS[DOWN_SCALE_2[1]]
    assert _word(float(divisor)) == divisor_word
    steps = _decode_steps(scales, divisor_word)
    assert torch.equal(codes, _nearest_codes(local, steps))
    values = _values(codes, steps)
    # The conversion stays close to the FP8 values it replaces and to the local weights.
    from tools.artifact.codecs.fp8_row import dequantize_fp8_row_scaled
    with Artifact(fixture.base) as base:
        fp8_values = dequantize_fp8_row_scaled(
            base.read_object("weight/000003"), (DOWN_ROWS, INTERMEDIATE)).float()
    error = (values - fp8_values).pow(2).mean().sqrt() / fp8_values.pow(2).mean().sqrt()
    assert error < 0.30, error
    error_weights = (values - local).pow(2).mean().sqrt() / local.pow(2).mean().sqrt()
    assert error_weights < 0.13, error_weights
