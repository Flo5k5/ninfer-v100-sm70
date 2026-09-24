from __future__ import annotations

import struct

import pytest
import torch

from tools.artifact.layouts import (
    encoded_size,
    row_scale_geometry,
)
from tools.artifact.codecs.fp8_row import (
    decode_fp8_row_scaled_words,
    dequantize_fp8_row_scaled,
    encode_fp8_row_scaled,
)


def _bf16_words(*words: int) -> torch.Tensor:
    signed = [word if word < 0x8000 else word - 0x10000 for word in words]
    return torch.tensor(signed, dtype=torch.int16).view(torch.bfloat16)


def test_fp8_weight_and_bf16_row_scale_word_validity():
    finite_codes = [word for word in range(0x100) if word not in (0x7F, 0xFF)]
    signed_zeros = [0x00, 0x80] * (len(finite_codes) // 2)
    codes = torch.tensor([finite_codes] * 3 + [signed_zeros], dtype=torch.uint8)
    scales = _bf16_words(0x0001, 0x3F80, 0x7F7F, 0x0000)
    shape = tuple(codes.shape)
    decoded_codes, decoded_scales = decode_fp8_row_scaled_words(
        encode_fp8_row_scaled(codes, scales, shape), shape
    )
    assert torch.equal(decoded_codes, codes)
    assert torch.equal(decoded_scales.view(torch.int16), scales.view(torch.int16))

    row_shape = (1, len(finite_codes))
    for word in (0x7F, 0xFF):
        invalid_codes = codes[:1].clone()
        invalid_codes[0, 0] = word
        with pytest.raises(ValueError, match="finite E4M3FN"):
            encode_fp8_row_scaled(invalid_codes, _bf16_words(0x3F80), row_shape)
    for word in (0x8000, 0xBF80, 0x7F80, 0x7FC0, 0xFF80, 0xFFC0):
        with pytest.raises(ValueError, match="nonnegative finite BF16"):
            encode_fp8_row_scaled(codes[:1], _bf16_words(word), row_shape)

    stored = encode_fp8_row_scaled(codes[:1], _bf16_words(0x3F80), row_shape)
    geometry = row_scale_geometry("fp8_e4m3fn_row_bf16", row_shape)
    for offset, word, message in (
        (0, b"\x7f", "finite E4M3FN"),
        (geometry.scale_plane_offset, b"\x00\x80", "nonnegative finite BF16"),
    ):
        corrupted = bytearray(stored)
        corrupted[offset : offset + len(word)] = word
        with pytest.raises(ValueError, match=message):
            decode_fp8_row_scaled_words(bytes(corrupted), row_shape)


def test_row_scale_layout_known_words_padding_and_reconstruction():
    shape = (2, 4)
    geometry = row_scale_geometry("fp8_e4m3fn_row_bf16", shape)
    assert (
        geometry.code_plane_bytes,
        geometry.scale_plane_offset,
        geometry.scale_plane_bytes,
        geometry.payload_bytes,
    ) == (8, 256, 4, 260)
    assert encoded_size("row_scale_v1", "fp8_e4m3fn_row_bf16", shape) == 260

    codes = torch.tensor(
        [
            [0x00, 0x80, 0x38, 0xB8],
            [0x40, 0xC0, 0x7E, 0xFE],
        ],
        dtype=torch.uint8,
    )
    scales = _bf16_words(0x3F00, 0x4000)  # 0.5, 2.0
    payload = encode_fp8_row_scaled(codes, scales, shape)

    assert payload[:8] == bytes(codes.reshape(-1).tolist())
    assert payload[8:256] == bytes(248)
    assert payload[256:] == struct.pack("<HH", 0x3F00, 0x4000)

    decoded_codes, decoded_scales = decode_fp8_row_scaled_words(payload, shape)
    assert torch.equal(decoded_codes, codes)
    assert torch.equal(decoded_scales.view(torch.int16), scales.view(torch.int16))
    assert torch.equal(
        dequantize_fp8_row_scaled(payload, shape),
        torch.tensor(
            [
                [0.0, -0.0, 0.5, -0.5],
                [4.0, -4.0, 896.0, -896.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_row_scaled_fp8_rejects_invalid_words_and_signatures():
    zero_codes = torch.zeros((1, 2), dtype=torch.uint8)
    positive_scale = _bf16_words(0x3F80)

    invalid_codes = zero_codes.clone()
    invalid_codes[0, 0] = 0x7F
    with pytest.raises(ValueError, match="finite E4M3FN"):
        encode_fp8_row_scaled(invalid_codes, positive_scale, (1, 2))

    with pytest.raises(ValueError, match="nonnegative finite BF16"):
        encode_fp8_row_scaled(zero_codes, _bf16_words(0x8000), (1, 2))

    nonzero_codes = zero_codes.clone()
    nonzero_codes[0, 0] = 0x38
    with pytest.raises(ValueError, match="zero row scale"):
        encode_fp8_row_scaled(nonzero_codes, _bf16_words(0x0000), (1, 2))

    with pytest.raises(TypeError, match="uint8"):
        encode_fp8_row_scaled(zero_codes.to(torch.int8), positive_scale, (1, 2))
    with pytest.raises(TypeError, match="BF16"):
        encode_fp8_row_scaled(zero_codes, positive_scale.float(), (1, 2))
    with pytest.raises(ValueError, match="rank 2"):
        encoded_size("row_scale_v1", "fp8_e4m3fn_row_bf16", (2,))
    with pytest.raises(ValueError, match="does not accept"):
        encoded_size("row_scale_v1", "nvfp4", (128, 64))
