"""Encode one ``reencode_nvfp4`` target and check it against the base and ``--weights``.

The payload of a target is its donor matrices' words in the object's row order (or, for a rounded
target, codes rounded from ``--weights`` under the donor's scales), under the divisor of the
donor's tensor scale. Every parameter's rows are compared with the base object's decoded values
(the dequantized FP8 values for a conversion), and the object with ``--weights`` when given; a
parameter or an object whose relative RMS error exceeds the limit is refused, and so are
``--weights`` rows that are not finite.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Iterator

import torch

from tools.artifact.codecs.fp8_row import dequantize_fp8_row_scaled
from tools.artifact.codecs.nvfp4 import decode_nvfp4_words, encode_nvfp4
from tools.artifact.reader import Artifact
from tools.convert.qwen3_8_27b.graft_single_source import SourceCheckpoint, select_rows
from tools.convert.qwen3_8_27b.reencode_nvfp4_numeric import (
    RelativeError,
    dequantize_words,
    divisor_word,
    round_to_nearest,
    same_codes,
    tensor_sha256,
    update_digest,
)
from tools.convert.qwen3_8_27b.reencode_nvfp4_plan import INPUT_GLOBAL_SCALE, ReencodeError, Target


DEFAULT_MAX_ERROR = 0.3
ROW_CHUNK = 4096


class Encoder:
    def __init__(self, base: Artifact, weights: SourceCheckpoint | None,
                 max_error: float = DEFAULT_MAX_ERROR) -> None:
        self.base = base
        self.weights = weights
        self.max_error = max_error

    def payload(self, target: Target) -> tuple[bytes, dict]:
        """Encoded object payload and its report entry."""

        started = time.perf_counter()
        packed, scales = [], []
        donor_sha256 = {}
        for item in target.donors:
            codes_name, scales_name, scale_name = item.tensors
            codes = target.donor.get(codes_name)
            block_scales = target.donor.get(scales_name).view(torch.uint8)
            packed.append(select_rows(codes, item.ranges))
            scales.append(select_rows(block_scales, item.ranges))
            read = [(codes_name, codes), (scales_name, block_scales),
                    (scale_name, target.donor.get(scale_name))]
            if target.input_divisor is not None:
                name = item.module + INPUT_GLOBAL_SCALE
                read.append((name, target.donor.get(name)))
            for name, tensor in read:
                donor_sha256[name] = tensor_sha256(tensor)
        report = {
            "object": target.object_id,
            "role": target.role,
            "parameters": list(target.parameters),
            "donor_matrices": [item.module for item in target.donors],
            "donor_layout": target.donors[0].layout,
            "codes": "rounded from --weights" if target.rounded else "donor",
            "shape": list(target.shape),
            "weight_divisor": float(target.divisor),
            "weight_divisor_word": f"0x{divisor_word(target.divisor):08x}",
            "divisor_exact": all(item.exact for item in target.donors),
            "donor_sha256": donor_sha256,
        }
        block_scales = torch.cat(scales)
        words = self._check(target, torch.cat(packed), block_scales, report)
        payload = encode_nvfp4(words, block_scales, target.divisor, target.shape)
        report["payload_sha256"] = hashlib.sha256(payload).hexdigest()
        report["seconds"] = round(time.perf_counter() - started, 1)
        return payload, report

    def _check(self, target: Target, packed: torch.Tensor, scales: torch.Tensor,
               report: dict) -> torch.Tensor:
        """Packed codes of the object, compared with the base object and with --weights."""

        if target.converts:
            base_codes, base_scales, base_divisor = None, None, None
            base_values = dequantize_fp8_row_scaled(
                self.base.read_object(target.object_id), target.shape)
        else:
            base_codes, base_scales, base_divisor = decode_nvfp4_words(
                self.base.read_object(target.object_id), target.shape)
        against_base = [RelativeError() for _ in target.donors]
        against_weights = RelativeError()
        rounded, same, saturated, zeroed = [], 0, 0, 0
        digests: dict = {}
        for leaf, begin, end, weights in self._row_chunks(target, digests):
            codes = packed[begin:end]
            if weights is not None:
                result = round_to_nearest(weights, scales[begin:end], target.divisor)
                if target.rounded and result.unscaled:
                    raise ReencodeError(f"{target.object_id}: {result.unscaled} nonzero --weights "
                                        f"values in rows [{begin}, {end}) have a zero donor block "
                                        "scale and would decode to zero")
                if target.rounded:
                    codes = result.packed
                    rounded.append(codes)
                    saturated += result.saturated
                    zeroed += result.zeroed
                else:
                    same += same_codes(codes, result.packed)
            values = dequantize_words(codes, scales[begin:end], target.divisor)
            if weights is not None:
                against_weights.add(values, weights)
            if target.converts:
                against_base[leaf].add(values, base_values[begin:end])
            else:
                against_base[leaf].add(values, dequantize_words(
                    base_codes[begin:end], base_scales[begin:end], base_divisor))
        by_parameter = {parameter: error.value()
                        for parameter, error in zip(target.parameters, against_base)}
        report["relative_rms_error_vs_base"] = RelativeError.merged(against_base).value()
        report["relative_rms_error_vs_base_by_parameter"] = by_parameter
        # Per parameter, conversions included: --weights rows are read in the object's row
        # order, so only the base object shows rows taken in a wrong order (a relative error near
        # 1.4 where they land), and a whole-object figure would dilute a small misplaced block.
        # Two quantizations of the same weights differ by 0.09 to 0.13 on the real model.
        worst = max(by_parameter, key=lambda parameter: math.inf
                    if math.isnan(by_parameter[parameter]) else by_parameter[parameter])
        self._refuse(target, by_parameter[worst], f"the base object's values of {worst}")
        if self.weights is None:
            return packed
        report["relative_rms_error_vs_weights"] = against_weights.value()
        report["weights_sha256"] = {name: digest.hexdigest() for name, digest in digests.items()}
        if target.rounded:
            report["saturated"] = saturated
            report["zeroed"] = zeroed
        else:
            report["codes_equal_to_rounded_weights"] = same / (target.shape[0] * target.shape[1])
        self._refuse(target, against_weights.value(), "--weights")
        return torch.cat(rounded) if target.rounded else packed

    def _refuse(self, target: Target, error: float, reference: str) -> None:
        if not math.isfinite(error) or error > self.max_error:
            raise ReencodeError(f"{target.object_id}: relative RMS error {error:.3f} against "
                                f"{reference} exceeds --max-error {self.max_error}; does the "
                                "donor quantize these weights, in this row order?")

    def _row_chunks(self, target: Target,
                    digests: dict) -> Iterator[tuple[int, int, int, torch.Tensor | None]]:
        """Row ranges of the object, each within one parameter (its index is given first), with
        their finite --weights rows when given; ``digests`` accumulates, per --weights matrix,
        the SHA-256 of the rows read in that order."""

        row = 0
        for leaf, item in enumerate(target.donors):
            if self.weights is None:
                for begin in range(row, row + item.rows, ROW_CHUNK):
                    yield leaf, begin, min(begin + ROW_CHUNK, row + item.rows), None
                row += item.rows
                continue
            name = item.module + ".weight"
            digest = digests.setdefault(name, hashlib.sha256())
            for first, last in item.ranges or ((0, item.rows),):
                for begin in range(first, last, ROW_CHUNK):
                    end = min(begin + ROW_CHUNK, last)
                    stored = self.weights.rows(name, begin, end)
                    update_digest(digest, stored)
                    values = stored.float()
                    if not bool(torch.isfinite(values).all()):
                        raise ReencodeError(f"{name}: rows [{begin}, {end}) of --weights hold "
                                            "non-finite values")
                    yield leaf, row, row + end - begin, values
                    row += end - begin
