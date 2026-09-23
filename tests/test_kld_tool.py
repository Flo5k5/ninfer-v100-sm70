from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tools.kld import kld

CONTEXT = 16
CHUNKS = 3
VOCAB = 257
POSITIONS = CHUNKS * (CONTEXT - 1 - CONTEXT // 2)


def _tokens(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, VOCAB, CONTEXT * CHUNKS).astype(np.int32)


def _logits(seed: int, scale: float = 4.0) -> np.ndarray:
    return (np.random.default_rng(seed).normal(0.0, scale, (POSITIONS, VOCAB))).astype(np.float32)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    values = logits.astype(np.float64)
    values -= values.max(axis=1, keepdims=True)
    return values - np.log(np.exp(values).sum(axis=1, keepdims=True))


def _exact(reference_logits: np.ndarray, candidate_logits: np.ndarray, targets: np.ndarray):
    log_p = _log_softmax(reference_logits)
    log_q = _log_softmax(candidate_logits)
    p = np.where(log_p > -16.0, np.exp(log_p), 0.0)
    per_position = np.sum(p * (log_p - log_q), axis=1)
    rows = np.arange(POSITIONS)

    def target_nll(log_probs: np.ndarray) -> np.ndarray:
        # The dump format clamps log-probabilities to the 16-nat window below the maximum.
        floor = np.maximum(log_probs.min(axis=1), log_probs.max(axis=1) - 16.0)
        return -np.maximum(log_probs[rows, targets], floor)

    return {
        "kld": per_position,
        "top1": float(np.mean(np.argmax(log_p, 1) == np.argmax(log_q, 1))),
        "ppl_reference": math.exp(target_nll(log_p).mean()),
        "ppl_candidate": math.exp(target_nll(log_q).mean()),
    }


def test_dump_layout_and_targets(tmp_path) -> None:
    tokens = _tokens()
    dump = kld.write_dump(tmp_path / "a.kld", CONTEXT, tokens, _logits(1))
    assert (dump.context, dump.vocab, dump.chunks, dump.positions) == (CONTEXT, VOCAB, CHUNKS,
                                                                        POSITIONS)
    # Odd vocabularies carry one pad word, as in llama-perplexity.
    assert dump.words == VOCAB + 1 + 4
    assert (tmp_path / "a.kld").stat().st_size == 20 + 4 * tokens.size + 2 * dump.words * POSITIONS
    # Chunk c, position j predicts token c*context + context/2 + 1 + j.
    assert dump.target_indices()[:3].tolist() == [9, 10, 11]
    assert dump.target_indices()[7] == CONTEXT + 9


def test_decoded_log_probabilities_match_log_softmax(tmp_path) -> None:
    logits = _logits(2, scale=8.0)
    dump = kld.write_dump(tmp_path / "a.kld", CONTEXT, _tokens(), logits)
    decoded = kld.decode(np.asarray(dump.records()), VOCAB)
    exact = _log_softmax(logits)
    inside = exact > exact.max(axis=1, keepdims=True) - 16.0
    step = 16.0 / 65535.0
    assert np.all(np.abs(decoded - exact)[inside] <= 0.5 * step + 1e-4)
    # Log-probabilities below the 16-nat window are clamped to the window floor.
    assert np.all(decoded[~inside] <= exact.max(axis=1, keepdims=True).repeat(VOCAB, 1)[~inside]
                  - 16.0 + 1e-4)


def test_identical_dumps_have_zero_divergence(tmp_path) -> None:
    tokens = _tokens()
    logits = _logits(3)
    kld.write_dump(tmp_path / "ref.kld", CONTEXT, tokens, logits)
    kld.write_dump(tmp_path / "same.kld", CONTEXT, tokens, logits)
    result = kld.compare(tmp_path / "ref.kld", [("same", tmp_path / "same.kld")], workers=1)
    candidate = result["candidates"][0]
    assert candidate["kld"]["max"] == 0.0
    assert candidate["top1_agreement"] == 1.0
    assert candidate["ppl_candidate"] == candidate["ppl_reference"]


@pytest.mark.parametrize("workers", [1, 2])
def test_compare_matches_exact_fp64_metrics(tmp_path, workers: int) -> None:
    tokens = _tokens()
    reference_logits = _logits(4)
    candidate_logits = reference_logits + np.random.default_rng(5).normal(
        0.0, 0.3, reference_logits.shape).astype(np.float32)
    reference = kld.write_dump(tmp_path / "ref.kld", CONTEXT, tokens, reference_logits)
    kld.write_dump(tmp_path / "cand.kld", CONTEXT, tokens, candidate_logits)
    result = kld.compare(tmp_path / "ref.kld", [("cand", tmp_path / "cand.kld")],
                         workers=workers, block_rows=4)
    candidate = result["candidates"][0]
    exact = _exact(reference_logits, candidate_logits, reference.targets())
    assert candidate["positions"] == POSITIONS
    assert candidate["kld"]["mean"] == pytest.approx(exact["kld"].mean(), rel=2e-3, abs=2e-4)
    assert candidate["kld"]["p99"] == pytest.approx(np.percentile(exact["kld"], 99), rel=5e-3,
                                                    abs=2e-4)
    assert candidate["top1_agreement"] == pytest.approx(exact["top1"], abs=1.0 / POSITIONS)
    assert candidate["ppl_reference"] == pytest.approx(exact["ppl_reference"], rel=1e-4)
    assert candidate["ppl_candidate"] == pytest.approx(exact["ppl_candidate"], rel=1e-4)
    assert result["reference"]["tokens_sha256"] == reference.tokens_sha256()


def test_prefix_of_a_longer_dump_matches_a_shorter_run(tmp_path) -> None:
    tokens = _tokens()
    reference_logits = _logits(11)
    candidate_logits = reference_logits * 1.1
    per_chunk = POSITIONS // CHUNKS
    kld.write_dump(tmp_path / "ref.kld", CONTEXT, tokens, reference_logits)
    kld.write_dump(tmp_path / "cand.kld", CONTEXT, tokens, candidate_logits)
    kld.write_dump(tmp_path / "short.kld", CONTEXT, tokens[:2 * CONTEXT],
                   candidate_logits[:2 * per_chunk])
    full_prefix = kld.compare(tmp_path / "ref.kld", [("cand", tmp_path / "cand.kld")],
                              workers=1, chunks=2)
    mixed = kld.compare(tmp_path / "ref.kld", [("short", tmp_path / "short.kld")],
                        workers=1, chunks=2)
    assert full_prefix["candidates"][0]["positions"] == 2 * per_chunk
    assert full_prefix["candidates"][0]["kld"] == mixed["candidates"][0]["kld"]
    with pytest.raises(kld.DumpError, match="cannot select 3"):
        kld.compare(tmp_path / "ref.kld", [("short", tmp_path / "short.kld")], workers=1,
                    chunks=3)


def test_token_mismatch_is_rejected(tmp_path) -> None:
    tokens = _tokens()
    other = tokens.copy()
    other[5] = (other[5] + 1) % VOCAB
    kld.write_dump(tmp_path / "ref.kld", CONTEXT, tokens, _logits(6))
    kld.write_dump(tmp_path / "cand.kld", CONTEXT, other, _logits(6))
    with pytest.raises(kld.DumpError, match="first at 5"):
        kld.compare(tmp_path / "ref.kld", [("cand", tmp_path / "cand.kld")], workers=1)


def test_truncated_dump_is_rejected(tmp_path) -> None:
    kld.write_dump(tmp_path / "ref.kld", CONTEXT, _tokens(), _logits(7))
    data = (tmp_path / "ref.kld").read_bytes()
    (tmp_path / "cut.kld").write_bytes(data[:-2])
    with pytest.raises(kld.DumpError, match="truncated"):
        kld.read_dump(tmp_path / "cut.kld")


def test_segments_split_results_by_domain(tmp_path) -> None:
    tokens = _tokens()
    reference_logits = _logits(8)
    kld.write_dump(tmp_path / "ref.kld", CONTEXT, tokens, reference_logits)
    kld.write_dump(tmp_path / "cand.kld", CONTEXT, tokens, reference_logits * 1.1)
    segments = {
        "tokens_file": "tokens.bin",
        "segments": [
            {"id": "a", "domain": "first", "token_begin": 0, "token_end": CONTEXT},
            {"id": "b", "domain": "rest", "token_begin": CONTEXT, "token_end": 10 * CONTEXT},
        ],
    }
    (tmp_path / "segments.json").write_text(json.dumps(segments))
    # The corpus may extend past the evaluated chunks.
    np.concatenate([tokens, tokens[:5]]).astype("<i4").tofile(tmp_path / "tokens.bin")
    result = kld.compare(tmp_path / "ref.kld", [("cand", tmp_path / "cand.kld")],
                         segments_path=tmp_path / "segments.json", workers=1)
    domains = result["candidates"][0]["domains"]
    assert domains["first"]["positions"] == POSITIONS // CHUNKS
    assert domains["rest"]["positions"] == POSITIONS - POSITIONS // CHUNKS

    other = tokens.copy()
    other[3] += 1
    other.astype("<i4").tofile(tmp_path / "tokens.bin")
    with pytest.raises(kld.DumpError, match="did not evaluate the tokens"):
        kld.compare(tmp_path / "ref.kld", [("cand", tmp_path / "cand.kld")],
                    segments_path=tmp_path / "segments.json", workers=1)


def _result(name: str, mean: float, p99: float, top1: float, ppl: float) -> dict:
    return {
        "name": name,
        "kld": {"mean": mean, "p99": p99},
        "top1_agreement": top1,
        "ppl_candidate": ppl,
    }


def _document(*candidates: dict) -> dict:
    return {
        "schema": kld.COMPARE_SCHEMA,
        "reference": {"tokens_sha256": "abc", "path": "/ref.kld"},
        "candidates": list(candidates),
    }


def test_gate_passes_within_limits() -> None:
    document = _document(_result("prod", 0.010, 0.10, 0.950, 6.00),
                         _result("q4km", 0.030, 0.30, 0.930, 6.10),
                         _result("stage1", 0.014, 0.15, 0.945, 6.05))
    verdict = kld.gate([document], prod="prod", candidate="stage1", ceiling="q4km")
    assert verdict["passed"], verdict["checks"]


@pytest.mark.parametrize(
    ("stage", "failed"),
    [
        (_result("stage", 0.016, 0.10, 0.95, 6.0), "stage_mean_kld_increase"),
        (_result("stage", 0.011, 0.25, 0.95, 6.0), "p99_kld_ratio_to_prod"),
        (_result("stage", 0.011, 0.10, 0.935, 6.0), "top1_drop_points"),
        (_result("stage", 0.011, 0.10, 0.95, 6.07), "ppl_increase_percent"),
    ],
)
def test_gate_rejects_each_stage_limit(stage: dict, failed: str) -> None:
    document = _document(_result("prod", 0.010, 0.10, 0.950, 6.00),
                         _result("q4km", 0.030, 0.30, 0.930, 6.10), stage)
    verdict = kld.gate([document], prod="prod", candidate="stage", ceiling="q4km")
    assert not verdict["passed"]
    assert [item["name"] for item in verdict["checks"] if not item["passed"]] == [failed]


def test_gate_accumulates_total_budget_and_ceiling() -> None:
    document = _document(_result("prod", 0.010, 0.10, 0.950, 6.00),
                         _result("q4km", 0.024, 0.30, 0.930, 6.10),
                         _result("stage2", 0.022, 0.15, 0.948, 6.02),
                         _result("stage3", 0.026, 0.18, 0.947, 6.03))
    verdict = kld.gate([document], prod="prod", previous="stage2", candidate="stage3",
                       ceiling="q4km")
    failed = {item["name"] for item in verdict["checks"] if not item["passed"]}
    assert failed == {"total_mean_kld_increase", "mean_kld_below_ceiling"}


def test_gate_rejects_mixed_references() -> None:
    first = _document(_result("prod", 0.01, 0.1, 0.95, 6.0), _result("q4km", 0.03, 0.3, 0.93, 6.1))
    second = _document(_result("stage", 0.011, 0.1, 0.95, 6.0))
    second["reference"] = {"tokens_sha256": "other", "path": "/ref.kld"}
    with pytest.raises(kld.DumpError, match="different references"):
        kld.gate([first, second], prod="prod", candidate="stage", ceiling="q4km")


def test_gate_prefers_exact_perplexity_when_both_runs_have_it() -> None:
    prod = _result("prod", 0.010, 0.10, 0.950, 6.00)
    stage = _result("stage", 0.011, 0.10, 0.950, 6.00)
    prod["ppl_candidate_exact"] = 6.20
    stage["ppl_candidate_exact"] = 6.30
    document = _document(prod, _result("q4km", 0.03, 0.3, 0.93, 6.1), stage)
    verdict = kld.gate([document], prod="prod", candidate="stage", ceiling="q4km")
    check = next(item for item in verdict["checks"] if item["name"] == "ppl_increase_percent")
    assert not check["passed"] and "exact PPL" in check["detail"]


def test_exact_ppl_sources(tmp_path) -> None:
    log = tmp_path / "base.log"
    log.write_text("perplexity: 2.5 seconds per pass\nFinal estimate: PPL = 6.1234 +/- 0.05\n")
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"execution": {"window_plan": "kld-chunks"},
                                  "overall": {"perplexity": 6.5}}))
    assert kld.read_exact_ppl("7.25") == 7.25
    assert kld.read_exact_ppl(str(log)) == pytest.approx(6.1234)
    assert kld.read_exact_ppl(str(report)) == 6.5
    report.write_text(json.dumps({"execution": {"window_plan": "sliding"},
                                  "overall": {"perplexity": 6.5}}))
    with pytest.raises(kld.DumpError, match="--logits-out"):
        kld.read_exact_ppl(str(report))


def test_cli_compare_and_gate_exit_codes(tmp_path) -> None:
    tokens = _tokens()
    logits = _logits(9)
    kld.write_dump(tmp_path / "ref.kld", CONTEXT, tokens, logits)
    noise = np.random.default_rng(10)
    kld.write_dump(tmp_path / "prod.kld", CONTEXT, tokens,
                   logits + noise.normal(0.0, 0.01, logits.shape).astype(np.float32))
    kld.write_dump(tmp_path / "q4.kld", CONTEXT, tokens,
                   logits + noise.normal(0.0, 0.5, logits.shape).astype(np.float32))
    kld.write_dump(tmp_path / "stage.kld", CONTEXT, tokens,
                   logits + noise.normal(0.0, 0.01, logits.shape).astype(np.float32))
    assert kld.main(["compare", "--reference", str(tmp_path / "ref.kld"), "--workers", "1",
                     "--candidate", f"prod={tmp_path / 'prod.kld'}",
                     "--candidate", f"q4km={tmp_path / 'q4.kld'}",
                     "--candidate", f"stage={tmp_path / 'stage.kld'}",
                     "--json", str(tmp_path / "compare.json")]) == 0
    assert kld.main(["gate", "--results", str(tmp_path / "compare.json"), "--prod", "prod",
                     "--ceiling", "q4km", "--candidate", "stage",
                     "--json", str(tmp_path / "gate.json")]) == 0
    assert json.loads((tmp_path / "gate.json").read_text())["passed"]
    assert kld.main(["gate", "--results", str(tmp_path / "compare.json"), "--prod", "prod",
                     "--ceiling", "stage", "--candidate", "q4km"]) == 1
