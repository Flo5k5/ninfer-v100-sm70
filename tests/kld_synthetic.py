"""Synthetic llama.cpp logits dumps for the kld tool tests.

`encode` mirrors llama.cpp's perplexity log_softmax() (and NInfer's KldBaseWriter).
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np

from tools.kld.dump import MAGIC, Dump, read_dump, words_per_position

LOGIT_WINDOW = 16.0
CODE_MAXIMUM = 65535.0


def encode(logits: np.ndarray) -> np.ndarray:
    """Encode float logits (rows, vocab) into uint16 position records."""
    logits = np.asarray(logits, dtype=np.float32)
    rows, vocab = logits.shape
    out = np.zeros((rows, words_per_position(vocab)), dtype="<u2")
    for row in range(rows):
        values = logits[row]
        max_logit = np.float32(values.max())
        min_logit = np.float32(max(values.min(), max_logit - np.float32(LOGIT_WINDOW)))
        sum_exp = float(np.sum(np.exp(values - max_logit).astype(np.float64)))
        min_log_prob = np.float32(min_logit - max_logit - np.float32(math.log(sum_exp)))
        scale = np.float32((max_logit - min_logit) / np.float32(CODE_MAXIMUM))
        out[row, :4] = np.array([scale, min_log_prob], dtype="<f4").view("<u2")
        if scale != 0:
            codes = np.rint((values - min_logit) * (np.float32(1.0) / scale))
            out[row, 4:4 + vocab] = np.where(values > min_logit, codes, 0).astype("<u2")
    return out


def write_dump(path: Path | str, context: int, tokens: np.ndarray, logits: np.ndarray) -> Dump:
    """Write a dump for `tokens` (chunks*context) from logits (positions, vocab)."""
    tokens = np.asarray(tokens, dtype="<i4")
    if tokens.size % context:
        raise ValueError("tokens must hold whole chunks")
    chunks = tokens.size // context
    positions, vocab = logits.shape
    if positions != chunks * (context - 1 - context // 2):
        raise ValueError("logits must hold one row per scored position")
    with Path(path).open("wb") as handle:
        handle.write(MAGIC + struct.pack("<Iii", context, vocab, chunks))
        handle.write(tokens.tobytes())
        handle.write(encode(logits).tobytes())
    return read_dump(path)
