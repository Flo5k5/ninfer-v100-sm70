from __future__ import annotations

import struct

import pytest
import torch

from tools.artifact.layouts import (
    block_scale_geometry,
    encoded_size,
    row_split_geometry,
)
from tools.artifact.codecs.direct import decode_direct, encode_direct
from tools.artifact.codecs.nvfp4 import decode_nvfp4_words, encode_nvfp4
from tools.artifact.codecs.row_split import (
    assemble_row_planes,
    decode_row_split_codes,
    encode_row_split,
    split_row_planes,
)


def _signed_word(word: int, bits: int) -> int:
    return word if word < 1 << (bits - 1) else word - (1 << bits)


@pytest.mark.parametrize(
    ("format_name", "tensor", "words", "word_format", "word_view"),
    [
        (
            "bf16",
            torch.tensor(
                [_signed_word(word, 16) for word in (0x0000, 0x8000, 0x0001, 0x7FC1)],
                dtype=torch.int16,
            ).view(torch.bfloat16),
            (0x0000, 0x8000, 0x0001, 0x7FC1),
            "H",
            torch.int16,
        ),
        (
            "fp32",
            torch.tensor(
                [
                    _signed_word(word, 32)
                    for word in (0x00000000, 0x80000000, 0x00000001, 0x7FC01234)
                ],
                dtype=torch.int32,
            ).view(torch.float32),
            (0x00000000, 0x80000000, 0x00000001, 0x7FC01234),
            "I",
            torch.int32,
        ),
        (
            "int32",
            torch.tensor((0, -1, -(1 << 31), (1 << 31) - 1), dtype=torch.int32),
            (0, -1, -(1 << 31), (1 << 31) - 1),
            "i",
            torch.int32,
        ),
    ],
)
def test_direct_layout_preserves_exact_little_endian_words(
    format_name, tensor, words, word_format, word_view
):
    expected = struct.pack("<" + word_format * len(words), *words)
    payload = encode_direct(tensor, format_name)
    assert payload == expected
    decoded = decode_direct(payload, format_name, tensor.shape)
    assert torch.equal(decoded.view(word_view), tensor.view(word_view))

    if format_name == "bf16":
        with pytest.raises(TypeError):
            encode_direct(tensor.float(), format_name)


def test_row_split_geometry_and_encoded_size_are_derived_from_format_and_shape():
    geometry = row_split_geometry("q5_g64_fp16", (2, 130))
    assert (
        geometry.k_pad,
        geometry.groups_per_row,
        geometry.base_bytes,
        geometry.high_offset,
        geometry.high_bytes,
        geometry.scale_offset,
        geometry.scale_bytes,
        geometry.payload_bytes,
    ) == (256, 4, 256, 256, 64, 512, 16, 528)
    assert encoded_size("row_split_k128_v1", "q5_g64_fp16", (2, 130)) == 528

    q4 = row_split_geometry("q4_g64_fp16", (1, 4304))
    q8 = row_split_geometry("q8_g32_fp16", (1, 4304))
    assert (q4.k_pad, q4.groups_per_row, q4.base_row_bytes, q4.high_row_bytes) == (
        4352,
        68,
        2176,
        0,
    )
    assert (q8.k_pad, q8.groups_per_row, q8.base_row_bytes, q8.high_row_bytes) == (
        4352,
        136,
        4352,
        0,
    )


@pytest.mark.parametrize(
    (
        "format_name",
        "k",
        "group_size",
        "k_pad",
        "scale_offset",
        "prefix",
        "base_prefix",
        "high_prefix",
    ),
    [
        pytest.param(
            "q4_g64_fp16",
            65,
            64,
            128,
            256,
            (-8, -7, -1, 0, 1, 7),
            b"\x98\x0f\x71",
            b"",
            id="q4",
        ),
        pytest.param(
            "q5_g64_fp16",
            130,
            64,
            256,
            512,
            (-16, -15, -1, 0, 1, 15),
            b"\x10\x0f\xf1",
            b"\x07",
            id="q5",
        ),
        pytest.param(
            "q6_g64_fp16",
            65,
            64,
            128,
            512,
            (-32, -31, -17, -16, -1, 0, 15, 31),
            b"\x10\x0f\x0f\xff",
            b"\xea\x43",
            id="q6",
        ),
        pytest.param(
            "q8_g32_fp16",
            33,
            32,
            128,
            256,
            (-127, -1, 0, 1, 127),
            b"\x81\xff\x00\x01\x7f",
            b"",
            id="q8",
        ),
    ],
)
def test_row_split_matches_known_packed_bytes(
    format_name, k, group_size, k_pad, scale_offset, prefix, base_prefix, high_prefix
):
    groups = k_pad // group_size
    codes = torch.zeros((1, groups, group_size), dtype=torch.int8)
    codes[0, 0, : len(prefix)] = torch.tensor(prefix, dtype=torch.int8)
    scales = torch.zeros((1, groups), dtype=torch.float16)
    scales[0, :2] = torch.tensor([1.5, 0.25], dtype=torch.float16)

    # Literal wire positions and words come from the format, independently of layout helpers.
    expected = bytearray(scale_offset + groups * 2)
    expected[: len(base_prefix)] = base_prefix
    expected[256 : 256 + len(high_prefix)] = high_prefix
    expected[scale_offset:] = struct.pack(
        "<" + "H" * groups, 0x3E00, 0x3400, *([0] * (groups - 2))
    )
    assert encode_row_split(codes, scales, format_name, (1, k)) == expected
    decoded_scales, decoded_codes = decode_row_split_codes(
        expected, format_name, (1, k)
    )
    assert torch.equal(decoded_scales, scales)
    assert torch.equal(decoded_codes, codes)


def test_consecutive_row_views_and_standalone_assembly():
    format_name = "q5_g64_fp16"
    shape = (4, 130)
    geometry = row_split_geometry(format_name, shape)
    codes = (
        torch.arange(geometry.n * geometry.groups_per_row * 64, dtype=torch.int32)
        .remainder(31)
        .sub(16)
        .to(torch.int8)
        .reshape(geometry.n, geometry.groups_per_row, 64)
    )
    codes.reshape(geometry.n, geometry.k_pad)[:, geometry.k :] = 0
    scales = torch.tensor(
        [
            [0.25, 0.5, 1.0, 0.0],
            [0.5, 1.0, 1.5, 0.0],
            [1.0, 1.5, 2.0, 0.0],
            [1.5, 2.0, 2.5, 0.0],
        ],
        dtype=torch.float16,
    )
    payload = encode_row_split(codes, scales, format_name, shape)

    consecutive = split_row_planes(payload, geometry, 1, 2)
    assert isinstance(consecutive.base, memoryview)
    assert consecutive.base.obj is payload
    standalone = assemble_row_planes(consecutive, format_name, shape[1])
    consecutive_scales, consecutive_codes = decode_row_split_codes(
        standalone, format_name, (2, shape[1])
    )
    assert torch.equal(consecutive_scales, scales[1:3])
    assert torch.equal(consecutive_codes, codes[1:3])


def test_nvfp4_known_vector_geometry_swizzle_tail_and_round_trip():
    shape = (128, 64)
    geometry = block_scale_geometry("nvfp4", shape)
    assert (
        geometry.code_plane_bytes,
        geometry.scale_plane_offset,
        geometry.scale_plane_bytes,
        geometry.weight_divisor_offset,
        geometry.payload_bytes,
    ) == (4096, 4096, 512, 4608, 4612)

    packed = (
        torch.arange(geometry.code_plane_bytes, dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(128, 32)
    )
    packed[0, 0] = 0x10
    scales = (
        torch.arange(128 * 4, dtype=torch.int64)
        .remainder(0x7F)
        .to(torch.uint8)
        .reshape(128, 4)
    )
    divisor = struct.pack("<f", 2.5)
    payload = encode_nvfp4(packed, scales, divisor, shape)

    assert len(payload) == 4612
    assert payload[0] == 0x10  # low nibble is K=0; high nibble is K=1.
    for row, lane in ((0, 0), (31, 3), (32, 0), (127, 3)):
        offset = geometry.scale_plane_offset + (row % 32) * 16 + (row // 32) * 4 + lane
        assert payload[offset] == int(scales[row, lane])
    assert payload[geometry.weight_divisor_offset :] == divisor

    decoded_packed, decoded_scales, decoded_divisor = decode_nvfp4_words(payload, shape)
    assert torch.equal(decoded_packed, packed)
    assert torch.equal(decoded_scales, scales)
    assert bytes(decoded_divisor.reshape(1).view(torch.uint8).numpy()) == divisor


@pytest.mark.parametrize(
    ("layout", "format_name", "shape", "message"),
    [
        ("block_scale_k16_m128x4_v1", "nvfp4", (128,), "rank 2"),
        ("block_scale_k16_m128x4_v1", "nvfp4", (64, 64), "N divisible by 128"),
        ("block_scale_k16_m128x4_v1", "nvfp4", (128, 32), "K divisible by 64"),
        ("block_scale_k16_m128x4_v1", "q4_g64_fp16", (128, 64), "does not accept"),
        ("row_split_k128_v1", "nvfp4", (128, 64), "does not accept"),
    ],
)
def test_nvfp4_layout_rejects_out_of_contract_signatures(
    layout, format_name, shape, message
):
    with pytest.raises(ValueError, match=message):
        encoded_size(layout, format_name, shape)
