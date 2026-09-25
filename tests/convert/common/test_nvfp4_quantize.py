from __future__ import annotations

import struct

import pytest
import torch

from tools.artifact.codecs.nvfp4 import decode_nvfp4_words
from tools.convert.common import nvfp4_quantize


def _well_formed_words(rows: int, columns: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Random NVFP4 words where every nonzero block holds a magnitude-6 code.

    An RTN quantizer always maps the block maximum to the E2M1 maximum, so these
    are exactly the words it can produce; the all-zero block carries the epsilon
    scale word a zero scale is replaced by.
    """

    generator = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (rows, columns), generator=generator, dtype=torch.uint8)
    codes = codes.reshape(rows, columns // 16, 16)
    codes[:, :, 0] = (codes[:, :, 0] & 0x8) | 0x7
    scales = torch.randint(1, 0x7F, (rows, columns // 16), generator=generator, dtype=torch.uint8)
    codes[0, 0] = 0
    scales[0, 0] = nvfp4_quantize.E4M3_EPS_WORD
    codes = codes.reshape(rows, columns)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.contiguous(), scales


def test_requantizing_decoded_words_is_bit_identical() -> None:
    packed, scales = _well_formed_words(256, 128, seed=3)
    divisor = torch.tensor(2752.0)
    values = nvfp4_quantize.dequantize(packed, scales, divisor)
    words = nvfp4_quantize.quantize_fused([values], divisor=divisor, chunk_rows=64)
    # Signed zero codes (0x8) survive because the sign comes from signbit.
    assert torch.equal(words.packed, packed)
    assert torch.equal(words.scales, scales)


def test_zero_blocks_get_the_epsilon_scale_word() -> None:
    row = torch.zeros(2, 32)
    row[1, 16] = 1.0
    packed, scales = nvfp4_quantize.quantize_rows(row, torch.tensor(2688.0))
    assert scales[:, 0].tolist() == [0x20, 0x20]
    assert scales[1, 1].item() != 0x20
    assert int(packed[0].abs().sum()) == 0


def test_e2m1_rounding_is_nearest_with_ties_to_even() -> None:
    magnitudes = (0.0, 0.2, 0.25, 0.3, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.1, 7.0)
    expected = (0, 0, 0, 1, 2, 2, 4, 4, 6, 6, 7, 7)
    # One block whose maximum is 6 so the block multiplier is exactly one.
    row = torch.zeros(1, 16)
    row[0, : len(magnitudes)] = torch.tensor(magnitudes)
    row[0, 15] = 6.0
    divisor = torch.tensor(448.0)
    packed, scales = nvfp4_quantize.quantize_rows(row, divisor)
    assert scales.view(torch.float8_e4m3fn).float().item() == 448.0
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=2).reshape(-1)
    assert codes[: len(magnitudes)].tolist() == list(expected)
    negative, _ = nvfp4_quantize.quantize_rows(-row, divisor)
    negative_codes = torch.stack((negative & 0x0F, negative >> 4), dim=2).reshape(-1)
    assert negative_codes[: len(magnitudes)].tolist() == [code | 0x8 for code in expected]


def test_fused_parts_share_the_amax_divisor_and_encode() -> None:
    generator = torch.Generator().manual_seed(7)
    gate = torch.randn(128, 64, generator=generator)
    up = 3.0 * torch.randn(128, 64, generator=generator)
    words = nvfp4_quantize.quantize_fused([gate, up])
    amax = max(float(gate.abs().max()), float(up.abs().max()))
    assert float(words.divisor) == pytest.approx(448.0 * 6.0 / amax, rel=1e-2)
    # The divisor is evaluated in BF16, so its FP32 word has a zero low half.
    assert struct.unpack("<I", words.divisor_word)[0] & 0xFFFF == 0
    assert words.packed.shape == (256, 32)
    assert words.scales.shape == (256, 4)
    assert int(words.scales.max()) <= 0x7E

    payload = words.encode((256, 64))
    codes, scales, divisor = decode_nvfp4_words(payload, (256, 64))
    assert torch.equal(codes, words.packed)
    assert torch.equal(scales, words.scales)
    assert float(divisor) == float(words.divisor)

    decoded = nvfp4_quantize.dequantize(words.packed, words.scales, words.divisor)
    source = torch.cat((gate, up))
    error = (decoded - source).norm() / source.norm()
    assert float(error) < 0.15


def test_quantizer_rejects_nonfinite_and_bad_geometry() -> None:
    with pytest.raises(ValueError, match="NaN or infinity"):
        nvfp4_quantize.quantize_fused([torch.tensor([[float("nan")] * 16])])
    with pytest.raises(TypeError, match="K divisible"):
        nvfp4_quantize.quantize_rows(torch.ones(2, 8), torch.tensor(1.0))
    with pytest.raises(ValueError, match="positive amax"):
        nvfp4_quantize.global_divisor(0.0)
