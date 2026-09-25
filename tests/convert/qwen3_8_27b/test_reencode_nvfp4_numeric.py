from __future__ import annotations

import math
import struct

import pytest
import torch

from tools.convert.qwen3_8_27b.reencode_nvfp4_numeric import (
    RelativeError,
    reciprocal_divisor,
    round_to_nearest,
)


def _f32(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", word))[0]


def _word(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


@pytest.mark.parametrize(
    "scale_2, divisor, exact",
    [
        (0x39A2877F, 0x45499CE7, True),  # exact reciprocal that BF16 cannot represent
        (0x3A03126F, 0x44F9FFFF, True),
        (0x39800000, 0x45800000, True),  # 2^-12
        (0x397DBE11, 0x4581238A, False),  # no FP32 word has this reciprocal
    ],
)
def test_reciprocal_divisor_words(scale_2, divisor, exact) -> None:
    word, is_exact = reciprocal_divisor(torch.tensor(_f32(scale_2)))
    assert (_word(float(word)), is_exact) == (divisor, exact)
    assert (_word(float(torch.tensor(1.0) / word)) == scale_2) is exact


def test_inexact_reciprocal_is_the_nearest_fp32_word() -> None:
    target = torch.tensor(_f32(0x397DBE11))
    divisor, exact = reciprocal_divisor(target)
    assert not exact
    one = torch.tensor(1.0)
    chosen = abs(float(one / divisor) - float(target))
    start = int(divisor.view(torch.int32))
    for delta in range(-64, 65):
        candidate = torch.tensor(start + delta, dtype=torch.int32).view(torch.float32)
        assert float(one / candidate) != float(target)
        assert abs(float(one / candidate) - float(target)) >= chosen


@pytest.mark.parametrize("scale_2", [float("nan"), float("inf"), 0.0, -2.0 ** -12, 1e-45])
def test_reciprocal_divisor_refuses_scales_without_a_finite_divisor(scale_2) -> None:
    with pytest.raises(ValueError, match="no finite positive FP32 divisor"):
        reciprocal_divisor(scale_2)


def test_round_to_nearest_ties_saturation_and_counts() -> None:
    # Divisor 1 and block scale 1.0 (E4M3 0x38): one E2M1 unit is exactly 1.0. The second block
    # has a zero scale.
    first = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, 6.5, -7.0, 0.1, -0.1, 0.0, -0.0, 0.26, 2.6]
    second = [0.0] * 15 + [0.5]
    result = round_to_nearest(torch.tensor([first + second]),
                              torch.tensor([[0x38, 0x00]], dtype=torch.uint8), torch.tensor(1.0))
    # Midpoints go to the even level, beyond 6 saturates, the sign survives on zero codes.
    codes = [0, 2, 2, 4, 4, 6, 6, 7, 7, 15, 0, 8, 0, 8, 1, 5] + [0] * 16
    assert result.packed.tolist() == [[codes[k] | (codes[k + 1] << 4) for k in range(0, 32, 2)]]
    assert (result.saturated, result.zeroed, result.unscaled) == (2, 3, 1)


def test_relative_error() -> None:
    error = RelativeError()
    error.add(torch.tensor([[3.0, 4.0]]), torch.tensor([[3.0, 0.0]]))
    error.add(torch.tensor([[1.0]]), torch.tensor([[4.0]]))
    assert error.value() == pytest.approx(math.sqrt((16.0 + 9.0) / (9.0 + 16.0)))
    zero = RelativeError()
    zero.add(torch.zeros(1, 2), torch.zeros(1, 2))
    assert zero.value() == 0.0
    zero.add(torch.ones(1, 2), torch.zeros(1, 2))
    assert zero.value() == math.inf


def test_merged_error_is_the_error_of_the_parts_together() -> None:
    """Sums of squares add up: a part with a large norm weighs more than one with a small norm,
    unlike a mean of the parts' ratios."""

    generator = torch.Generator().manual_seed(3)
    references = [torch.randn(64, 32, generator=generator),
                  0.1 * torch.randn(16, 32, generator=generator)]
    values = [references[0] + 0.01 * torch.randn(64, 32, generator=generator),
              references[1].flip(0)]
    parts, whole = [], RelativeError()
    for value, reference in zip(values, references):
        part = RelativeError()
        part.add(value, reference)
        parts.append(part)
    whole.add(torch.cat(values), torch.cat(references))
    merged = RelativeError.merged(parts).value()
    assert merged == pytest.approx(whole.value(), rel=1e-12)
    mean = (parts[0].value() + parts[1].value()) / 2
    assert abs(merged - mean) > 0.5
