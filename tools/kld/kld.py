#!/usr/bin/env python3
"""Compare logits dumps: KL divergence, top-1 agreement and perplexity, plus a quality gate.

A dump is the file written by `llama-perplexity --kl-divergence-base FILE` or by
`ninfer-perplexity --logits-out FILE`. Both use the llama.cpp layout (little-endian):

    char[8] "_logits_" | u32 context | i32 vocab | i32 chunks | i32 tokens[chunks*context]
    per scored position: f32 scale | f32 min_log_prob | u16 q[vocab] | u16 pad (odd vocab)

with log p(i) = min_log_prob + scale*q[i]. Chunk c covers tokens [c*context, (c+1)*context) and
scores its last context-1-context/2 targets; position j of chunk c predicts token
c*context + context/2 + 1 + j.

Subcommands:
    inspect  FILE...                           validate and describe dumps
    compare  --reference REF --candidate [NAME=]FILE...  KLD/top-1/PPL of candidates vs reference
    gate     --results JSON... --prod NAME --candidate NAME --ceiling NAME [--previous NAME]

KLD(reference || candidate) per position follows llama.cpp exactly: the sum over tokens whose
reference log-probability exceeds -16 of p_ref * (log p_ref - log p_cand). Top-1 agreement
compares the argmax of both distributions. Perplexity is exp(mean NLL of the target tokens).

The dump format clamps every log-probability to [max_log_prob - 16, max_log_prob], so the dump
perplexity under-counts very surprising targets (llama.cpp's "PPL(base)" has the same bias). The
unclamped perplexity of the same positions is printed by each run; pass it with
`--exact-ppl NAME=SOURCE` (a number, a ninfer-perplexity report.json, or a llama-perplexity log)
and the gate uses it whenever both compared runs provide it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

MAGIC = b"_logits_"
HEADER_FIXED_BYTES = 8 + 3 * 4
LOG_PROB_FLOOR = -16.0
LOGIT_WINDOW = 16.0
CODE_MAXIMUM = 65535.0
COMPARE_SCHEMA = "ninfer-kld-compare/1"
GATE_SCHEMA = "ninfer-kld-gate/1"
PERCENTILES = {"p90": 90.0, "p95": 95.0, "p99": 99.0, "p999": 99.9}

# Per-stage gate of the quantization protocol; see docs/perplexity.md.
DEFAULT_GATE = {
    "max_stage_mean_kld_increase": 0.005,
    "max_total_mean_kld_increase": 0.015,
    "max_p99_kld_ratio_to_prod": 2.0,
    "max_top1_drop_points": 1.0,
    "max_ppl_increase_percent": 1.0,
}


class DumpError(RuntimeError):
    """A dump is malformed or two dumps cannot be compared."""


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

    def prefix(self, chunks: int) -> "Dump":
        """The first `chunks` windows, like a run with llama-perplexity --chunks."""
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


def encode(logits: np.ndarray) -> np.ndarray:
    """Encode float logits (rows, vocab) like llama.cpp's perplexity log_softmax().

    Used to produce synthetic dumps; mirrors the C++ writers in llama.cpp and NInfer.
    """
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


# ---------------------------------------------------------------------------------------------
# compare


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


def compare(reference_path: Path, candidates: Sequence[tuple[str, Path]], *,
            segments_path: Path | None = None, workers: int = 0, block_rows: int = 32,
            per_token_dir: Path | None = None,
            exact_ppl: dict[str, float] | None = None,
            chunks: int | None = None) -> dict[str, Any]:
    exact_ppl = exact_ppl or {}
    unknown = set(exact_ppl) - {name for name, _ in candidates} - {"reference"}
    if unknown:
        raise DumpError(f"--exact-ppl names no compared run: {sorted(unknown)}")
    reference = read_dump(reference_path)
    candidate_dumps = [read_dump(path) for _, path in candidates]
    if chunks is not None:
        reference = reference.prefix(chunks)
        candidate_dumps = [dump.prefix(chunks) for dump in candidate_dumps]
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
    for index, ((name, path), dump) in enumerate(zip(candidates, candidate_dumps)):
        result = {"name": name, "path": str(Path(path).resolve())}
        result.update(_summary(kld[index], same_top[index], nll_reference, nll_candidate[index]))
        result["ppl_candidate_exact"] = exact_ppl.get(name)
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
            "ppl_exact": exact_ppl.get("reference"),
        },
        "segments": str(Path(segments_path).resolve()) if segments_path else None,
        "candidates": results,
    }


# ---------------------------------------------------------------------------------------------
# gate


def read_exact_ppl(source: str) -> float:
    """Unclamped perplexity from a number, a ninfer-perplexity report.json or a llama log."""
    try:
        return float(source)
    except ValueError:
        pass
    path = Path(source)
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".json":
        report = json.loads(text)
        if report.get("execution", {}).get("window_plan") != "kld-chunks":
            raise DumpError(f"{path}: not a ninfer-perplexity --logits-out report")
        return float(report["overall"]["perplexity"])
    matches = re.findall(r"Final estimate: PPL = ([0-9.]+)", text)
    if not matches:
        raise DumpError(f"{path}: no 'Final estimate: PPL' line (llama-perplexity log expected)")
    return float(matches[-1])


def _find(results: Sequence[dict[str, Any]], name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    found = [(document, candidate) for document in results for candidate in document["candidates"]
             if candidate["name"] == name]
    if len(found) != 1:
        raise DumpError(f"expected exactly one result named {name!r}, found {len(found)}")
    return found[0]


def gate(results: Sequence[dict[str, Any]], *, prod: str, candidate: str, ceiling: str,
         previous: str | None = None, limits: dict[str, float] | None = None) -> dict[str, Any]:
    limits = {**DEFAULT_GATE, **(limits or {})}
    named = {role: _find(results, name) for role, name in
             (("prod", prod), ("candidate", candidate), ("ceiling", ceiling),
              ("previous", previous or prod))}
    references = {(document["reference"]["tokens_sha256"], document["reference"]["path"])
                  for document, _ in named.values()}
    if len(references) != 1:
        raise DumpError("gate inputs were compared against different references or corpora")
    for document, _ in named.values():
        if document.get("schema") != COMPARE_SCHEMA:
            raise DumpError(f"unsupported compare result schema {document.get('schema')!r}")
    cand, prev, base, ceil = (named[role][1] for role in ("candidate", "previous", "prod",
                                                          "ceiling"))

    def check(name: str, value: float, limit: float, passed: bool, detail: str) -> dict[str, Any]:
        return {"name": name, "value": value, "limit": limit, "passed": bool(passed),
                "detail": detail}

    stage_increase = cand["kld"]["mean"] - prev["kld"]["mean"]
    total_increase = cand["kld"]["mean"] - base["kld"]["mean"]
    p99_ratio = cand["kld"]["p99"] / base["kld"]["p99"] if base["kld"]["p99"] > 0 else math.inf
    top1_drop = 100.0 * (prev["top1_agreement"] - cand["top1_agreement"])
    exact = cand.get("ppl_candidate_exact") is not None and \
        prev.get("ppl_candidate_exact") is not None
    ppl_key = "ppl_candidate_exact" if exact else "ppl_candidate"
    ppl_increase = 100.0 * (cand[ppl_key] / prev[ppl_key] - 1.0)
    checks = [
        check("stage_mean_kld_increase", stage_increase, limits["max_stage_mean_kld_increase"],
              stage_increase <= limits["max_stage_mean_kld_increase"],
              f"mean KLD {cand['kld']['mean']:.6f} vs previous {prev['kld']['mean']:.6f} nat"),
        check("total_mean_kld_increase", total_increase, limits["max_total_mean_kld_increase"],
              total_increase <= limits["max_total_mean_kld_increase"],
              f"mean KLD {cand['kld']['mean']:.6f} vs prod {base['kld']['mean']:.6f} nat"),
        check("mean_kld_below_ceiling", cand["kld"]["mean"], ceil["kld"]["mean"],
              cand["kld"]["mean"] < ceil["kld"]["mean"],
              f"mean KLD {cand['kld']['mean']:.6f} vs ceiling {ceil['kld']['mean']:.6f} nat"),
        check("p99_kld_ratio_to_prod", p99_ratio, limits["max_p99_kld_ratio_to_prod"],
              p99_ratio <= limits["max_p99_kld_ratio_to_prod"],
              f"p99 KLD {cand['kld']['p99']:.6f} vs prod {base['kld']['p99']:.6f} nat"),
        check("top1_drop_points", top1_drop, limits["max_top1_drop_points"],
              top1_drop <= limits["max_top1_drop_points"],
              f"top-1 {100 * cand['top1_agreement']:.3f}% vs previous "
              f"{100 * prev['top1_agreement']:.3f}%"),
        check("ppl_increase_percent", ppl_increase, limits["max_ppl_increase_percent"],
              ppl_increase <= limits["max_ppl_increase_percent"],
              f"{'exact' if exact else 'dump'} PPL {cand[ppl_key]:.4f} vs previous "
              f"{prev[ppl_key]:.4f}"),
    ]
    return {
        "schema": GATE_SCHEMA,
        "roles": {"candidate": candidate, "previous": previous or prod, "prod": prod,
                  "ceiling": ceiling},
        "limits": limits,
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
    }


# ---------------------------------------------------------------------------------------------
# CLI


def _candidate_argument(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator:
        path = name
        name = Path(path).name
        name = re.sub(r"\.(kld|bin|logits)$", "", name)
    if not name or not path:
        raise argparse.ArgumentTypeError(f"invalid candidate {value!r}; use NAME=FILE or FILE")
    return name, Path(path)


def _print_compare(document: dict[str, Any]) -> None:
    reference = document["reference"]
    print(f"reference: {reference['path']}")
    exact = reference.get("ppl_exact")
    print(f"  context {reference['context']}, chunks {reference['chunks']}, "
          f"positions {reference['positions']}, vocab {reference['vocab']}, "
          f"dump PPL {reference['ppl']:.4f}"
          + (f", exact PPL {exact:.4f}" if exact is not None else ""))
    header = (f"{'candidate':<24}{'mean KLD':>12}{'median':>11}{'p99':>11}{'max':>10}"
              f"{'top-1 %':>10}{'PPL':>10}{'dPPL %':>9}")
    print(header)
    for item in document["candidates"]:
        kld = item["kld"]
        print(f"{item['name']:<24}{kld['mean']:>12.6f}{kld['median']:>11.6f}{kld['p99']:>11.5f}"
              f"{kld['max']:>10.3f}{100 * item['top1_agreement']:>10.3f}"
              f"{item['ppl_candidate']:>10.4f}{item['ppl_delta_percent']:>9.3f}"
              + (f"  exact PPL {item['ppl_candidate_exact']:.4f}"
                 if item.get("ppl_candidate_exact") is not None else ""))
        for domain, stats in item.get("domains", {}).items():
            kld = stats["kld"]
            print(f"  {domain:<22}{kld['mean']:>12.6f}{kld['median']:>11.6f}{kld['p99']:>11.5f}"
                  f"{kld['max']:>10.3f}{100 * stats['top1_agreement']:>10.3f}"
                  f"{stats['ppl_candidate']:>10.4f}{stats['ppl_delta_percent']:>9.3f}")


def _print_gate(document: dict[str, Any]) -> None:
    roles = document["roles"]
    print(f"gate: {roles['candidate']} (previous {roles['previous']}, prod {roles['prod']}, "
          f"ceiling {roles['ceiling']})")
    for item in document["checks"]:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"  {status}  {item['name']:<26} value {item['value']:.6g}  limit "
              f"{item['limit']:.6g}  ({item['detail']})")
    print("verdict:", "PASS" if document["passed"] else "FAIL")


def _write_json(path: Path | None, document: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser("inspect", help="validate and describe dumps")
    inspect_parser.add_argument("dumps", nargs="+", type=Path)

    compare_parser = commands.add_parser("compare", help="compare candidates to a reference")
    compare_parser.add_argument("--reference", required=True, type=Path)
    compare_parser.add_argument("--candidate", required=True, action="append",
                                type=_candidate_argument, help="NAME=FILE or FILE (repeatable)")
    compare_parser.add_argument("--segments", type=Path,
                                help="corpus segments.json for per-domain results")
    compare_parser.add_argument("--json", type=Path, help="write the machine-readable result")
    compare_parser.add_argument("--per-token-dir", type=Path,
                                help="also save per-position KLD/NLL arrays as NAME.npz")
    compare_parser.add_argument("--workers", type=int, default=0,
                                help="worker processes (default min(16, CPUs))")
    compare_parser.add_argument("--block-rows", type=int, default=32)
    compare_parser.add_argument("--chunks", type=int,
                                help="compare only the first N windows of every dump")
    compare_parser.add_argument("--exact-ppl", action="append", default=[],
                                metavar="NAME=SOURCE",
                                help="unclamped PPL of a run (NAME 'reference' or a candidate): "
                                     "a number, a ninfer report.json or a llama-perplexity log")

    gate_parser = commands.add_parser("gate", help="apply the quantization quality gate")
    gate_parser.add_argument("--results", required=True, nargs="+", type=Path,
                             help="compare JSON files against one reference")
    gate_parser.add_argument("--candidate", required=True)
    gate_parser.add_argument("--prod", required=True)
    gate_parser.add_argument("--ceiling", required=True)
    gate_parser.add_argument("--previous", help="previous stage (default: --prod)")
    gate_parser.add_argument("--json", type=Path, help="write the machine-readable verdict")
    for key, value in DEFAULT_GATE.items():
        gate_parser.add_argument("--" + key.replace("_", "-"), type=float, default=value)

    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            for path in args.dumps:
                dump = read_dump(path)
                print(json.dumps({
                    "path": str(path), "context": dump.context, "vocab": dump.vocab,
                    "chunks": dump.chunks, "positions": dump.positions,
                    "bytes": dump.expected_bytes, "tokens_sha256": dump.tokens_sha256()}))
            return 0
        if args.command == "compare":
            names = [name for name, _ in args.candidate]
            if len(set(names)) != len(names):
                parser.error("candidate names must be unique")
            exact_ppl = {}
            for item in args.exact_ppl:
                name, separator, source = item.partition("=")
                if not separator or not name or not source:
                    parser.error(f"invalid --exact-ppl {item!r}; use NAME=SOURCE")
                exact_ppl[name] = read_exact_ppl(source)
            document = compare(args.reference, args.candidate, segments_path=args.segments,
                               workers=args.workers, block_rows=args.block_rows,
                               per_token_dir=args.per_token_dir, exact_ppl=exact_ppl,
                               chunks=args.chunks)
            _print_compare(document)
            _write_json(args.json, document)
            return 0
        documents = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
        limits = {key: getattr(args, key) for key in DEFAULT_GATE}
        verdict = gate(documents, prod=args.prod, candidate=args.candidate,
                       ceiling=args.ceiling, previous=args.previous, limits=limits)
        _print_gate(verdict)
        _write_json(args.json, verdict)
        return 0 if verdict["passed"] else 1
    except DumpError as error:
        print(f"kld: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
