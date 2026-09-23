"""Compare candidate dumps with a reference dump over identical token ids.

KLD(reference || candidate) per position uses llama.cpp's definition: the sum over tokens whose
reference log-probability exceeds -16 of p_ref * (log p_ref - log p_cand). One difference with
`llama-perplexity --kl-divergence`: there the candidate's log-probabilities come from its
unquantized logits, here from its dump, 16-bit and clamped 16 nats below its top token. On a
Qwen3.5-9B comparison this moved the mean KLD by about 1.2% (0.00015 nat) and top-1 agreement by
0.03 point; percentiles agree within 0.1%. Top-1 agreement compares the argmax of both
distributions; perplexity is exp(mean NLL of the target tokens).

The dump format clamps every log-probability to [max_log_prob - 16, max_log_prob], so the dump
perplexity under-counts very surprising targets (llama.cpp's "PPL(base)" has the same bias). Each
run also prints the unclamped perplexity of its positions: `read_exact_ppl` reads it, with the
number of windows it covers, from a ninfer-perplexity report.json or a llama-perplexity log.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import re
import sys
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import numpy as np

from tools.kld.dump import Dump, DumpError, decode, read_dump

COMPARE_SCHEMA = "ninfer-kld-compare/1"
LOG_PROB_FLOOR = -16.0
PERCENTILES = {"p90": 90.0, "p95": 95.0, "p99": 99.0, "p999": 99.9}


class ExactPpl(NamedTuple):
    """An unclamped perplexity and the number of windows it covers (None: not recorded)."""

    value: float
    chunks: int | None


def read_exact_ppl(source: str) -> ExactPpl:
    """Unclamped perplexity from a number, a ninfer-perplexity report.json or a llama log.

    A bare number carries no window count and is taken as given; report.json records its window
    count under `logits_dump.chunks`, and a llama-perplexity log in its
    "calculating perplexity over N chunks" line.
    """
    try:
        return ExactPpl(float(source), None)
    except ValueError:
        pass
    path = Path(source)
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".json":
        report = json.loads(text)
        if report.get("execution", {}).get("window_plan") != "kld-chunks":
            raise DumpError(f"{path}: not a ninfer-perplexity --logits-out report")
        return ExactPpl(float(report["overall"]["perplexity"]),
                        int(report["logits_dump"]["chunks"]))
    values = re.findall(r"Final estimate: PPL = ([0-9.]+)", text)
    if not values:
        raise DumpError(f"{path}: no 'Final estimate: PPL' line (llama-perplexity log expected)")
    chunks = re.findall(r"calculating perplexity over (\d+) chunks", text)
    if not chunks:
        raise DumpError(f"{path}: no 'calculating perplexity over N chunks' line")
    return ExactPpl(float(values[-1]), int(chunks[-1]))


def check_comparable(reference: Dump, candidate: Dump) -> None:
    if (reference.context, reference.chunks) != (candidate.context, candidate.chunks):
        raise DumpError(
            f"{candidate.path}: context/chunks {candidate.context}/{candidate.chunks} differ from "
            f"the reference {reference.context}/{reference.chunks}")
    if reference.vocab != candidate.vocab:
        raise DumpError(f"{candidate.path}: vocabulary {candidate.vocab} differs from the "
                        f"reference {reference.vocab}")
    mismatch = np.flatnonzero(reference.tokens != candidate.tokens)
    if mismatch.size:
        first = int(mismatch[0])
        raise DumpError(
            f"{candidate.path}: token ids differ from the reference at {mismatch.size} of "
            f"{reference.tokens.size} positions, first at {first} (reference "
            f"{int(reference.tokens[first])}, candidate {int(candidate.tokens[first])}); "
            "the runs did not evaluate the same token stream")


_WORKER: dict[str, Any] = {}


def _init_worker(reference: Dump, candidates: list[Dump]) -> None:
    _WORKER["reference"] = reference
    _WORKER["candidates"] = candidates
    _WORKER["targets"] = reference.targets()
    _WORKER["reference_records"] = reference.records()
    _WORKER["candidate_records"] = [candidate.records() for candidate in candidates]


def _score_range(bounds: tuple[int, int], block_rows: int) -> dict[str, np.ndarray]:
    begin, end = bounds
    reference: Dump = _WORKER["reference"]
    vocab = reference.vocab
    count = end - begin
    candidates = len(_WORKER["candidates"])
    out = {
        "nll_reference": np.empty(count, dtype=np.float64),
        "kld": np.empty((candidates, count), dtype=np.float64),
        "same_top": np.empty((candidates, count), dtype=bool),
        "nll_candidate": np.empty((candidates, count), dtype=np.float64),
    }
    for row in range(begin, end, block_rows):
        stop = min(end, row + block_rows)
        local = slice(row - begin, stop - begin)
        rows = np.arange(stop - row)
        targets = _WORKER["targets"][row:stop]
        log_p = decode(_WORKER["reference_records"][row:stop], vocab)
        top_reference = np.argmax(log_p, axis=1)
        out["nll_reference"][local] = -log_p[rows, targets]
        p = np.exp(log_p)
        p[log_p <= LOG_PROB_FLOOR] = 0.0
        for index, records in enumerate(_WORKER["candidate_records"]):
            log_q = decode(records[row:stop], vocab)
            out["kld"][index, local] = np.sum(p * (log_p - log_q), axis=1, dtype=np.float64)
            out["same_top"][index, local] = np.argmax(log_q, axis=1) == top_reference
            out["nll_candidate"][index, local] = -log_q[rows, targets]
    return out


def _distribution(values: np.ndarray) -> dict[str, float]:
    stats = {
        "mean": float(values.mean()),
        "mean_stderr": float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1
        else 0.0,
        "median": float(np.median(values)),
    }
    for name, percentile in PERCENTILES.items():
        stats[name] = float(np.percentile(values, percentile))
    stats["max"] = float(values.max())
    stats["min"] = float(values.min())
    return stats


def _summary(kld: np.ndarray, same_top: np.ndarray, nll_reference: np.ndarray,
             nll_candidate: np.ndarray) -> dict[str, Any]:
    ppl_reference = math.exp(float(nll_reference.mean()))
    ppl_candidate = math.exp(float(nll_candidate.mean()))
    delta_p = np.exp(-nll_candidate) - np.exp(-nll_reference)
    return {
        "positions": int(kld.size),
        "kld": _distribution(kld),
        "top1_agreement": float(same_top.mean()),
        "ppl_reference": ppl_reference,
        "ppl_candidate": ppl_candidate,
        "ppl_ratio": ppl_candidate / ppl_reference,
        "ppl_delta_percent": 100.0 * (ppl_candidate / ppl_reference - 1.0),
        "delta_p": {"mean": float(delta_p.mean()), "rms": float(np.sqrt(np.mean(delta_p ** 2)))},
    }


def load_segments(path: Path, reference: Dump) -> list[dict[str, Any]]:
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    corpus_tokens = np.fromfile(path.parent / manifest["tokens_file"], dtype="<i4")
    evaluated = reference.tokens.size
    if corpus_tokens.size < evaluated or \
            not np.array_equal(corpus_tokens[:evaluated], reference.tokens):
        raise DumpError(f"{path}: the reference dump did not evaluate the tokens of this corpus")
    segments = manifest["segments"]
    for segment in segments:
        if segment["token_end"] <= segment["token_begin"]:
            raise DumpError(f"{path}: empty segment {segment['id']}")
    return segments


def _check_exact_ppl(exact_ppl: dict[str, ExactPpl], names: set[str], chunks: int) -> None:
    unknown = set(exact_ppl) - names - {"reference"}
    if unknown:
        raise DumpError(f"--exact-ppl names no compared run: {sorted(unknown)}")
    for name, exact in exact_ppl.items():
        if exact.chunks is not None and exact.chunks != chunks:
            raise DumpError(
                f"--exact-ppl {name} covers {exact.chunks} windows but the comparison uses "
                f"{chunks}; use the perplexity of the same windows")


def compare(reference_path: Path, candidates: Sequence[tuple[str, Path]], *,
            segments_path: Path | None = None, workers: int = 0, block_rows: int = 32,
            per_token_dir: Path | None = None,
            exact_ppl: dict[str, ExactPpl] | None = None,
            chunks: int | None = None) -> dict[str, Any]:
    exact_ppl = exact_ppl or {}
    reference = read_dump(reference_path)
    candidate_dumps = [read_dump(path) for _, path in candidates]
    if chunks is not None:
        reference = reference.prefix(chunks)
        candidate_dumps = [dump.prefix(chunks) for dump in candidate_dumps]
    _check_exact_ppl(exact_ppl, {name for name, _ in candidates}, reference.chunks)
    for candidate in candidate_dumps:
        check_comparable(reference, candidate)
    segments = load_segments(segments_path, reference) if segments_path else None

    workers = workers or min(16, os.cpu_count() or 1)
    step = max(block_rows, math.ceil(reference.positions / (workers * 8)))
    ranges = [(begin, min(reference.positions, begin + step))
              for begin in range(0, reference.positions, step)]
    if workers == 1:
        _init_worker(reference, candidate_dumps)
        parts = [_score_range(bounds, block_rows) for bounds in ranges]
    else:
        context = multiprocessing.get_context("fork" if sys.platform.startswith("linux")
                                              else "spawn")
        with context.Pool(workers, initializer=_init_worker,
                          initargs=(reference, candidate_dumps)) as pool:
            parts = pool.starmap(_score_range, [(bounds, block_rows) for bounds in ranges])
    nll_reference = np.concatenate([part["nll_reference"] for part in parts])
    kld = np.concatenate([part["kld"] for part in parts], axis=1)
    same_top = np.concatenate([part["same_top"] for part in parts], axis=1)
    nll_candidate = np.concatenate([part["nll_candidate"] for part in parts], axis=1)

    domain_of_position = None
    if segments is not None:
        target_index = reference.target_indices()
        domain_of_position = np.full(reference.positions, "", dtype=object)
        for segment in segments:
            inside = (target_index >= segment["token_begin"]) & (target_index < segment["token_end"])
            domain_of_position[inside] = segment["domain"]

    results = []
    for index, (name, path) in enumerate(candidates):
        result = {"name": name, "path": str(Path(path).resolve())}
        result.update(_summary(kld[index], same_top[index], nll_reference, nll_candidate[index]))
        result["ppl_candidate_exact"] = exact_ppl[name].value if name in exact_ppl else None
        if domain_of_position is not None:
            result["domains"] = {
                domain: _summary(kld[index][mask], same_top[index][mask], nll_reference[mask],
                                 nll_candidate[index][mask])
                for domain in sorted(set(domain_of_position) - {""})
                for mask in [domain_of_position == domain]
            }
        if per_token_dir is not None:
            per_token_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(per_token_dir / f"{name}.npz", kld=kld[index].astype(np.float32),
                                same_top=same_top[index],
                                nll_reference=nll_reference.astype(np.float32),
                                nll_candidate=nll_candidate[index].astype(np.float32))
        results.append(result)

    return {
        "schema": COMPARE_SCHEMA,
        "reference": {
            "path": str(reference.path.resolve()),
            "context": reference.context,
            "vocab": reference.vocab,
            "chunks": reference.chunks,
            "positions": reference.positions,
            "tokens_sha256": reference.tokens_sha256(),
            "ppl": math.exp(float(nll_reference.mean())),
            "ppl_exact": exact_ppl["reference"].value if "reference" in exact_ppl else None,
        },
        "segments": str(Path(segments_path).resolve()) if segments_path else None,
        "candidates": results,
    }
