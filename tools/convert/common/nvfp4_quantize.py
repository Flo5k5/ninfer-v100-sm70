"""Round-to-nearest NVFP4 quantizer producing exact artifact words.

The scheme is the one compressed-tensors checkpoints publish for NVFP4 weights:

* one FP32 global divisor per logical weight, ``448 * 6 / amax``, where ``amax``
  is taken over every fused part (gate and up share one divisor);
* one E4M3FN scale per 16-element K block, ``block_amax / 6 * divisor``
  rounded to nearest even and clamped to the finite E4M3FN range, where
  ``block_amax / 6`` is first rounded to BF16 as compressed-tensors computes it
  in the model dtype (this reproduces its published scale words exactly);
* every value divided by its decoded block multiplier ``scale / divisor`` and
  rounded to the nearest E2M1 level, ties to even, saturating at 6.

Codes are packed two per byte with the even K element in the low nibble, the
natural ``[N, K/16]`` scale words are returned unswizzled, and
:func:`tools.artifact.codecs.nvfp4.encode_nvfp4` performs the registered
layout packing. No calibration is involved: this is plain RTN.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Iterable

import torch

from tools.artifact.codecs.nvfp4 import encode_nvfp4


ENCODER_PROFILE = "NVFP4_RTN_AMAX_DIVISOR_BF16_LOCAL_E4M3_RNE_E2M1_RNE_V1"
GROUP = 16
E2M1_MAX = 6.0
E4M3_MAX = 448.0
DEFAULT_CHUNK_ROWS = 2048

_E2M1_LEVELS = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), dtype=torch.float32)
# Midpoints between consecutive E2M1 magnitudes. A value exactly on a midpoint
# rounds to the level with an even mantissa (index 0, 2, 4 or 6).
_E2M1_BOUNDARIES = torch.tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), dtype=torch.float32)
_TIES_TO_UPPER = torch.tensor((0.75, 1.75, 3.5), dtype=torch.float32)


@dataclass(frozen=True, slots=True)
class Nvfp4Words:
    """Exact NVFP4 words for one ``[N, K]`` matrix."""

    packed: torch.Tensor
    scales: torch.Tensor
    divisor: torch.Tensor

    @property
    def divisor_word(self) -> bytes:
        return struct.pack("<f", float(self.divisor))

    def encode(self, shape: tuple[int, int]) -> bytes:
        return encode_nvfp4(self.packed, self.scales, self.divisor_word, shape)


def _require_matrix(values: torch.Tensor, label: str) -> None:
    if values.dim() != 2 or values.shape[1] % GROUP or values.shape[1] % 2:
        raise TypeError(f"{label}: expected a rank-two matrix with K divisible by {GROUP}")
    if not values.dtype.is_floating_point:
        raise TypeError(f"{label}: expected floating-point values")


def tensor_amax(parts: Iterable[torch.Tensor], chunk_rows: int = DEFAULT_CHUNK_ROWS) -> float:
    """Return the finite absolute maximum over every part, in FP32."""

    amax = 0.0
    for part in parts:
        for begin in range(0, part.shape[0], chunk_rows):
            block = part[begin : begin + chunk_rows].float()
            if not bool(torch.isfinite(block).all()):
                raise ValueError("NVFP4 source contains NaN or infinity")
            amax = max(amax, float(block.abs().max()))
    return amax


def global_divisor(amax: float) -> torch.Tensor:
    """FP32 global divisor ``448 * 6 / amax`` shared by every fused part."""

    if not amax > 0.0:
        raise ValueError("NVFP4 global divisor needs a positive amax")
    divisor = torch.tensor(E4M3_MAX * E2M1_MAX, dtype=torch.float32) / torch.tensor(
        amax, dtype=torch.float32
    )
    if not bool(torch.isfinite(divisor)):
        raise ValueError("NVFP4 global divisor is not finite")
    return divisor.reshape(())


def _round_e2m1(magnitude: torch.Tensor) -> torch.Tensor:
    index = torch.bucketize(magnitude, _E2M1_BOUNDARIES, right=False)
    ties = torch.isin(magnitude, _TIES_TO_UPPER)
    return (index + ties.to(index.dtype)).to(torch.uint8)


def _quantize_rows(
    values: torch.Tensor, divisor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = values.shape
    blocks = values.float().reshape(rows, columns // GROUP, GROUP)
    if not bool(torch.isfinite(blocks).all()):
        raise ValueError("NVFP4 source contains NaN or infinity")
    block_amax = blocks.abs().amax(dim=2)
    # compressed-tensors forms block_amax / 6 in the BF16 model dtype before applying the
    # FP32 divisor; rounding it the same way reproduces its published scale words.
    local = (block_amax / E2M1_MAX).to(torch.bfloat16).float()
    scale_value = (local * divisor).clamp_(0.0, E4M3_MAX)
    scale_words = scale_value.to(torch.float8_e4m3fn)
    multiplier = scale_words.float() / divisor
    safe = torch.where(multiplier > 0, multiplier, torch.ones_like(multiplier))
    scaled = blocks / safe.unsqueeze(2)
    scaled = torch.where((multiplier > 0).unsqueeze(2), scaled, torch.zeros_like(scaled))
    magnitude = _round_e2m1(scaled.abs().clamp_(max=E2M1_MAX))
    sign = torch.signbit(blocks).to(torch.uint8) << 3
    codes = (magnitude | sign).reshape(rows, columns)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.contiguous(), scale_words.view(torch.uint8).contiguous()


def quantize_rows(
    values: torch.Tensor,
    divisor: torch.Tensor,
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[N, K]`` values with a given global divisor.

    Returns packed ``uint8 [N, K/2]`` codes and natural ``uint8 [N, K/16]`` E4M3FN
    scale words. Rows are independent once the divisor is fixed, so callers that
    fuse several parts compute the divisor over all of them and quantize each part
    separately.
    """

    _require_matrix(values, "NVFP4 quantize")
    divisor = divisor.to(torch.float32).reshape(())
    if not bool(torch.isfinite(divisor)) or not float(divisor) > 0.0:
        raise ValueError("NVFP4 global divisor must be finite and positive")
    packed, scales = [], []
    for begin in range(0, values.shape[0], chunk_rows):
        part_packed, part_scales = _quantize_rows(values[begin : begin + chunk_rows], divisor)
        packed.append(part_packed)
        scales.append(part_scales)
    return torch.cat(packed), torch.cat(scales)


def quantize_fused(
    parts: list[torch.Tensor],
    *,
    divisor: torch.Tensor | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> Nvfp4Words:
    """Quantize row-concatenated parts that share one global divisor."""

    if not parts:
        raise ValueError("NVFP4 quantize needs at least one part")
    for part in parts:
        _require_matrix(part, "NVFP4 quantize")
    if len({part.shape[1] for part in parts}) != 1:
        raise ValueError("fused NVFP4 parts must share K")
    if divisor is None:
        divisor = global_divisor(tensor_amax(parts, chunk_rows))
    packed, scales = [], []
    for part in parts:
        part_packed, part_scales = quantize_rows(part, divisor, chunk_rows=chunk_rows)
        packed.append(part_packed)
        scales.append(part_scales)
    return Nvfp4Words(
        packed=packed[0] if len(packed) == 1 else torch.cat(packed),
        scales=scales[0] if len(scales) == 1 else torch.cat(scales),
        divisor=divisor.to(torch.float32).reshape(()),
    )


def dequantize(
    packed: torch.Tensor,
    scales: torch.Tensor,
    divisor: torch.Tensor,
) -> torch.Tensor:
    """Decode natural NVFP4 words to FP32 ``[N, K]`` values."""

    if packed.dtype != torch.uint8 or scales.dtype != torch.uint8 or packed.dim() != 2:
        raise TypeError("NVFP4 dequantize expects uint8 packed codes and scale words")
    rows, half = packed.shape
    columns = 2 * half
    if scales.shape != (rows, columns // GROUP):
        raise ValueError("NVFP4 scale words do not match the packed code geometry")
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=2).reshape(rows, columns)
    magnitude = _E2M1_LEVELS[(codes & 0x7).long()]
    values = torch.where((codes & 0x8) != 0, -magnitude, magnitude)
    multiplier = scales.view(torch.float8_e4m3fn).float() / divisor.to(torch.float32).reshape(())
    return (values.reshape(rows, columns // GROUP, GROUP) * multiplier.unsqueeze(2)).reshape(
        rows, columns
    )
