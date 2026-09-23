"""Reader for llama.cpp `--kl-divergence-base` logits dumps.

A dump is the file written by `llama-perplexity --kl-divergence-base FILE` or by
`ninfer-perplexity --logits-out FILE`. Both use the llama.cpp layout (little-endian):

    char[8] "_logits_" | u32 context | i32 vocab | i32 chunks | i32 tokens[chunks*context]
    per scored position: f32 scale | f32 min_log_prob | u16 q[vocab] | u16 pad (odd vocab)

with log p(i) = min_log_prob + scale*q[i]. Codes span the 16 nats below the most likely token;
less likely tokens take code 0 (the floor). Chunk c covers tokens [c*context, (c+1)*context) and
scores its last context-1-context/2 targets; position j of chunk c predicts token
c*context + context/2 + 1 + j, after context/2 + 1 + j tokens of history.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"_logits_"
HEADER_FIXED_BYTES = 8 + 3 * 4


class DumpError(RuntimeError):
    """A dump or a result is malformed, or two runs cannot be compared."""


def words_per_position(vocab: int) -> int:
    return 2 * ((vocab + 1) // 2) + 4


@dataclass(frozen=True)
class Dump:
    path: Path
    context: int
    vocab: int
    chunks: int
    tokens: np.ndarray
    stored_chunks: int = 0  # chunks in the file; `chunks` may select a prefix of them

    @property
    def first(self) -> int:
        return self.context // 2

    @property
    def positions_per_chunk(self) -> int:
        return self.context - 1 - self.context // 2

    @property
    def positions(self) -> int:
        return self.chunks * self.positions_per_chunk

    @property
    def words(self) -> int:
        return words_per_position(self.vocab)

    @property
    def data_offset(self) -> int:
        return HEADER_FIXED_BYTES + 4 * self.context * (self.stored_chunks or self.chunks)

    @property
    def expected_bytes(self) -> int:
        stored = (self.stored_chunks or self.chunks) * self.positions_per_chunk
        return self.data_offset + 2 * self.words * stored

    def prefix(self, chunks: int) -> Dump:
        """The first `chunks` windows, as a run with llama-perplexity --chunks would produce."""
        if not 0 < chunks <= self.chunks:
            raise DumpError(f"{self.path}: cannot select {chunks} of {self.chunks} chunks")
        return Dump(path=self.path, context=self.context, vocab=self.vocab, chunks=chunks,
                    tokens=self.tokens[:chunks * self.context],
                    stored_chunks=self.stored_chunks or self.chunks)

    def tokens_sha256(self) -> str:
        return hashlib.sha256(self.tokens.astype("<i4").tobytes()).hexdigest()

    def target_indices(self) -> np.ndarray:
        """Global token index predicted by every scored position, in file order."""
        chunk = np.arange(self.chunks, dtype=np.int64)[:, None] * self.context
        local = self.first + 1 + np.arange(self.positions_per_chunk, dtype=np.int64)[None, :]
        return (chunk + local).reshape(-1)

    def targets(self) -> np.ndarray:
        return self.tokens[self.target_indices()]

    def records(self) -> np.memmap:
        return np.memmap(self.path, dtype="<u2", mode="r", offset=self.data_offset,
                         shape=(self.positions, self.words))


def read_dump(path: Path | str) -> Dump:
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(HEADER_FIXED_BYTES)
        if len(header) != HEADER_FIXED_BYTES or header[:8] != MAGIC:
            raise DumpError(f"{path}: not a llama-perplexity logits dump")
        context, vocab, chunks = struct.unpack("<Iii", header[8:])
        if context < 4 or vocab <= 0 or chunks <= 0:
            raise DumpError(f"{path}: invalid header context={context} vocab={vocab} "
                            f"chunks={chunks}")
        tokens = np.frombuffer(handle.read(4 * context * chunks), dtype="<i4")
        if tokens.size != context * chunks:
            raise DumpError(f"{path}: token block is truncated")
    dump = Dump(path=path, context=context, vocab=vocab, chunks=chunks, tokens=tokens.copy())
    if size != dump.expected_bytes:
        raise DumpError(f"{path}: {size} bytes, the header implies {dump.expected_bytes} "
                        f"({dump.positions} positions of {dump.words} words); truncated or "
                        "not written by a complete run")
    return dump


def decode(records: np.ndarray, vocab: int) -> np.ndarray:
    """Dequantize uint16 position records to float32 log-probabilities, shape (rows, vocab)."""
    header = np.ascontiguousarray(records[:, :4]).view("<f4")
    values = records[:, 4:4 + vocab].astype(np.float32)
    values *= header[:, 0:1]
    values += header[:, 1:2]
    return values
