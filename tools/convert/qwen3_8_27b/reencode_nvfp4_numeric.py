"""NVFP4 word arithmetic of ``reencode_nvfp4``: divisor, nearest codes, decoded values, errors.

The artifact decodes a weight as ``e2m1(code) * e4m3(scale) / divisor`` with one FP32 divisor per
object; its FP32 decode routes form the step of one E2M1 unit as ``e4m3(scale) * (1.0f /
divisor)``. Every function here uses that step. Packed bytes hold the even K element in the low
nibble.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable

import numpy as np
import torch

from tools.artifact.formats import _E2M1_MAGNITUDES


_E2M1 = torch.tensor(_E2M1_MAGNITUDES, dtype=torch.float32)
# Midpoints between consecutive E2M1 magnitudes; a value on a midpoint rounds to the level with an
# even mantissa: up at 0.75, 1.75 and 3.5, down at 0.25, 1.25, 2.5 and 5.
_MIDPOINTS = torch.tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), dtype=torch.float32)
_TIES_UP = torch.tensor((0.75, 1.75, 3.5), dtype=torch.float32)
_E2M1_MAX = 6.0


def reciprocal_divisor(scale_2: torch.Tensor | float) -> tuple[torch.Tensor, bool]:
    """FP32 divisor for a ModelOpt tensor scale, and whether its FP32 reciprocal is the scale.

    The divisor is ``fl32(1 / scale_2)``; it is exact when ``fl32(1 / divisor) == scale_2``, so
    the FP32 decode routes reproduce the donor's values. When it is not, no other FP32 word has
    that reciprocal or a nearer one. Both divisions are FP32 operations.
    """

    target = np.float32(float(scale_2))
    one = np.float32(1.0)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        divisor = one / target
        back = one / divisor
    if not (np.isfinite(target) and target > 0 and np.isfinite(divisor) and np.isfinite(back)):
        raise ValueError(f"tensor scale {float(target)!r} has no finite positive FP32 divisor")
    return torch.tensor(float(divisor), dtype=torch.float32), bool(back == target)


def divisor_word(divisor: torch.Tensor) -> int:
    return int(divisor.reshape(()).view(torch.int32)) & 0xFFFFFFFF


def code_steps(scales: torch.Tensor, divisor: torch.Tensor) -> torch.Tensor:
    """FP32 value of one E2M1 unit per block as the FP32 decode routes form it: e4m3 * (1/d)."""

    return scales.view(torch.float8_e4m3fn).float() * (
        torch.tensor(1.0, dtype=torch.float32) / divisor.float())


@dataclass(frozen=True, slots=True)
class RoundedCodes:
    """Packed E2M1 codes [N, K/2] of rounded values and what the rounding lost."""

    packed: torch.Tensor
    saturated: int  # values beyond 6 steps, stored as +-6 steps
    zeroed: int  # nonzero values under a nonzero block scale stored as a zero code
    unscaled: int  # nonzero values in blocks whose scale is zero (they would decode to zero)


def round_to_nearest(values: torch.Tensor, scales: torch.Tensor,
                     divisor: torch.Tensor) -> RoundedCodes:
    """Codes of finite FP32 ``values`` [N, K] under given block scales and divisor.

    Each value goes to the nearest E2M1 level of ``value / step`` with ties to even and saturation
    at 6, like ModelOpt and compressed-tensors; a block whose scale is zero gets zero codes.
    """

    rows, columns = values.shape
    step = code_steps(scales, divisor).unsqueeze(2)
    blocks = values.float().reshape(rows, columns // 16, 16)
    scaled = torch.where(step > 0, blocks / torch.where(step > 0, step, 1.0), 0.0)
    magnitude = scaled.abs()
    saturated = int((magnitude > _E2M1_MAX).sum())
    magnitude = magnitude.clamp(max=_E2M1_MAX)
    index = torch.bucketize(magnitude, _MIDPOINTS, right=False) + torch.isin(magnitude, _TIES_UP)
    nonzero = blocks != 0
    zeroed = int((nonzero & (index == 0) & (step > 0)).sum())
    unscaled = int((nonzero & (step == 0)).sum())
    codes = (index | (torch.signbit(scaled).to(index.dtype) << 3)).to(torch.uint8)
    codes = codes.reshape(rows, columns)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return RoundedCodes(packed, saturated, zeroed, unscaled)


def dequantize_words(packed: torch.Tensor, scales: torch.Tensor,
                     divisor: torch.Tensor) -> torch.Tensor:
    """FP32 values [N, K] of packed codes [N, K/2] under block scales [N, K/16] and a divisor."""

    rows = packed.shape[0]
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=2).reshape(rows, -1).long()
    values = torch.where((codes & 0x8) != 0, -_E2M1[codes & 0x7], _E2M1[codes & 0x7])
    return (values.reshape(rows, -1, 16) * code_steps(scales, divisor).unsqueeze(2)).reshape(rows, -1)


def same_codes(left: torch.Tensor, right: torch.Tensor) -> int:
    """Number of equal E2M1 codes in two packed matrices, counting +0 and -0 as equal."""

    count = 0
    for shift in (0, 4):
        a = (left >> shift) & 0x0F
        b = (right >> shift) & 0x0F
        count += int(((a == b) | (((a & 0x7) == 0) & ((b & 0x7) == 0))).sum())
    return count


class RelativeError:
    """Relative RMS error sqrt(sum((x - ref)^2) / sum(ref^2)), accumulated over row chunks."""

    def __init__(self) -> None:
        self.squared_error = 0.0
        self.squared_norm = 0.0

    @classmethod
    def merged(cls, parts: Iterable[RelativeError]) -> RelativeError:
        """The error of the parts' values and references taken together."""

        result = cls()
        for part in parts:
            result.squared_error += part.squared_error
            result.squared_norm += part.squared_norm
        return result

    def add(self, values: torch.Tensor, reference: torch.Tensor) -> None:
        reference = reference.double()
        self.squared_error += float((values.double() - reference).square().sum())
        self.squared_norm += float(reference.square().sum())

    def value(self) -> float:
        """The error; infinite against an all-zero reference that ``values`` do not match."""

        if self.squared_norm > 0.0:
            return math.sqrt(self.squared_error / self.squared_norm)
        return 0.0 if self.squared_error == 0.0 else math.inf


def update_digest(digest, tensor: torch.Tensor) -> None:
    """Feed the stored bytes of ``tensor`` (row-major, little-endian) to a hashlib digest."""

    digest.update(tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy())


def tensor_sha256(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    update_digest(digest, tensor)
    return digest.hexdigest()
