from __future__ import annotations

import json
import math

import numpy as np
import pytest

from kld_synthetic import write_dump
from tools.kld import kld
from tools.kld.compare import COMPARE_SCHEMA, compare
from tools.kld.dump import DumpError, decode, read_dump
from tools.kld.exact_ppl import ExactPpl, read_exact_ppl
from tools.kld.gate import gate

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
    dump = write_dump(tmp_path / "a.kld", CONTEXT, tokens, _logits(1))
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
    dump = write_dump(tmp_path / "a.kld", CONTEXT, _tokens(), logits)
    decoded = decode(np.asarray(dump.records()), VOCAB)
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
    write_dump(tmp_path / "ref.kld", CONTEXT, tokens, logits)
    write_dump(tmp_path / "same.kld", CONTEXT, tokens, logits)
    result = compare(tmp_path / "ref.kld", [("same", tmp_path / "same.kld")], workers=1)
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
    reference = write_dump(tmp_path / "ref.kld", CONTEXT, tokens, reference_logits)
    write_dump(tmp_path / "cand.kld", CONTEXT, tokens, candidate_logits)
    result = compare(tmp_path / "ref.kld", [("cand", tmp_path / "cand.kld")],
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
    write_dump(tmp_path / "ref.kld", CONTEXT, tokens, reference_logits)
    write_dump(tmp_path / "cand.kld", CONTEXT, tokens, candidate_logits)
    write_dump(tmp_path / "short.kld", CONTEXT, tokens[:2 * CONTEXT],
                   candidate_logits[:2 * per_chunk])
    full_prefix = compare(tmp_path / "ref.kld", [("cand", tmp_path / "cand.kld")],
                              workers=1, chunks=2)
    mixed = compare(tmp_path / "ref.kld", [("short", tmp_path / "short.kld")],
                        workers=1, chunks=2)
    assert full_prefix["candidates"][0]["positions"] == 2 * per_chunk
    assert full_prefix["candidates"][0]["kld"] == mixed["candidates"][0]["kld"]
    with pytest.raises(DumpError, match="cannot select 3"):
        compare(tmp_path / "ref.kld", [("short", tmp_path / "short.kld")], workers=1,
                    chunks=3)


def test_token_mismatch_is_rejected(tmp_path) -> None:
    tokens = _tokens()
    other = tokens.copy()
    other[5] = (other[5] + 1) % VOCAB
    write_dump(tmp_path / "ref.kld", CONTEXT, tokens, _logits(6))
    write_dump(tmp_path / "cand.kld", CONTEXT, other, _logits(6))
    with pytest.raises(DumpError, match="first at 5"):
        compare(tmp_path / "ref.kld", [("cand", tmp_path / "cand.kld")], workers=1)


def test_truncated_dump_is_rejected(tmp_path) -> None:
    write_dump(tmp_path / "ref.kld", CONTEXT, _tokens(), _logits(7))
    data = (tmp_path / "ref.kld").read_bytes()
    (tmp_path / "cut.kld").write_bytes(data[:-2])
    with pytest.raises(DumpError, match="truncated"):
        read_dump(tmp_path / "cut.kld")


def test_segments_split_results_by_domain(tmp_path) -> None:
    tokens = _tokens()
    reference_logits = _logits(8)
    write_dump(tmp_path / "ref.kld", CONTEXT, tokens, reference_logits)
    write_dump(tmp_path / "cand.kld", CONTEXT, tokens, reference_logits * 1.1)
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
    result = compare(tmp_path / "ref.kld", [("cand", tmp_path / "cand.kld")],
                         segments_path=tmp_path / "segments.json", workers=1)
    domains = result["candidates"][0]["domains"]
    assert domains["first"]["positions"] == POSITIONS // CHUNKS
    assert domains["rest"]["positions"] == POSITIONS - POSITIONS // CHUNKS

    other = tokens.copy()
    other[3] += 1
    other.astype("<i4").tofile(tmp_path / "tokens.bin")
    with pytest.raises(DumpError, match="did not evaluate the tokens"):
        compare(tmp_path / "ref.kld", [("cand", tmp_path / "cand.kld")],
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
        "schema": COMPARE_SCHEMA,
        "reference": {"tokens_sha256": "abc", "path": "/ref.kld"},
        "candidates": list(candidates),
    }


def test_gate_passes_within_limits() -> None:
    document = _document(_result("prod", 0.010, 0.10, 0.950, 6.00),
                         _result("q4km", 0.030, 0.30, 0.930, 6.10),
                         _result("stage1", 0.014, 0.15, 0.945, 6.05))
    verdict = gate([document], prod="prod", candidate="stage1", ceiling="q4km")
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
    verdict = gate([document], prod="prod", candidate="stage", ceiling="q4km")
    assert not verdict["passed"]
    assert [item["name"] for item in verdict["checks"] if not item["passed"]] == [failed]


def test_gate_accumulates_total_budget_and_ceiling() -> None:
    document = _document(_result("prod", 0.010, 0.10, 0.950, 6.00),
                         _result("q4km", 0.024, 0.30, 0.930, 6.10),
                         _result("stage2", 0.022, 0.15, 0.948, 6.02),
                         _result("stage3", 0.026, 0.18, 0.947, 6.03))
    verdict = gate([document], prod="prod", previous="stage2", candidate="stage3",
                       ceiling="q4km")
    failed = {item["name"] for item in verdict["checks"] if not item["passed"]}
    assert failed == {"total_mean_kld_increase", "mean_kld_below_ceiling"}


def test_gate_rejects_mixed_references() -> None:
    first = _document(_result("prod", 0.01, 0.1, 0.95, 6.0), _result("q4km", 0.03, 0.3, 0.93, 6.1))
    second = _document(_result("stage", 0.011, 0.1, 0.95, 6.0))
    second["reference"] = {"tokens_sha256": "other", "path": "/ref.kld"}
    with pytest.raises(DumpError, match="different references"):
        gate([first, second], prod="prod", candidate="stage", ceiling="q4km")


def test_gate_prefers_exact_perplexity_when_both_runs_have_it() -> None:
    prod = _result("prod", 0.010, 0.10, 0.950, 6.00)
    stage = _result("stage", 0.011, 0.10, 0.950, 6.00)
    prod["ppl_candidate_exact"] = 6.20
    stage["ppl_candidate_exact"] = 6.30
    document = _document(prod, _result("q4km", 0.03, 0.3, 0.93, 6.1), stage)
    verdict = gate([document], prod="prod", candidate="stage", ceiling="q4km")
    check = next(item for item in verdict["checks"] if item["name"] == "ppl_increase_percent")
    assert not check["passed"] and "exact PPL" in check["detail"]


def _llama_log(path, chunks: int = 3, context: int = 4096) -> None:
    path.write_text(f"perplexity: calculating perplexity over {chunks} chunks, n_ctx={context}, "
                    "batch_size=2048, n_seq=1\n[1]4.1000,[2]4.0500,[3]4.0200,\n"
                    "Final estimate: PPL = 4.0100 +/- 0.05\n")


def _report(path, perplexity=6.5, chunks=8, context=4096) -> None:
    path.write_text(json.dumps({"execution": {"window_plan": "kld-chunks"},
                                "overall": {"perplexity": perplexity},
                                "logits_dump": {"chunks": chunks, "context_tokens": context}}))


def test_exact_ppl_sources_record_their_windows_and_context(tmp_path) -> None:
    log = tmp_path / "run.log"
    _llama_log(log)
    report = tmp_path / "report.json"
    _report(report)
    assert read_exact_ppl("7.25@8") == ExactPpl(7.25, 8, None)
    assert read_exact_ppl(str(log)) == ExactPpl(pytest.approx(4.01), 3, 4096)
    assert read_exact_ppl(f"{log}@2") == ExactPpl(pytest.approx(4.05), 2, 4096)
    assert read_exact_ppl(f"{log}@3") == ExactPpl(pytest.approx(4.02), 3, 4096)  # the last window
    assert read_exact_ppl(str(report)) == ExactPpl(6.5, 8, 4096)


@pytest.mark.parametrize(("source", "message"), [
    ("7.25", "bare number does not record the windows"),
    ("nan@2", "must be a finite number, got nan"),
    ("inf@2", "must be a finite number, got inf"),
    ("0@2", "must be positive, got 0.0"),
    ("-1.5@2", "must be positive, got -1.5"),
    ("7.25@0", "N must be a positive integer, got 0"),
])
def test_exact_ppl_rejects_unusable_numbers(source: str, message: str) -> None:
    with pytest.raises(DumpError, match=message):
        read_exact_ppl(source)


def test_exact_ppl_rejects_unusable_files(tmp_path) -> None:
    log = tmp_path / "run.log"
    _llama_log(log)
    with pytest.raises(DumpError, match="the run has 3 windows, not 4"):
        read_exact_ppl(f"{log}@4")
    report = tmp_path / "report.json"
    _report(report)
    with pytest.raises(DumpError, match="@N applies to numbers and llama-perplexity logs only"):
        read_exact_ppl(f"{report}@2")
    _report(report, perplexity=None)  # a NaN perplexity is written as null
    with pytest.raises(DumpError, match="overall.perplexity must be a finite number, got None"):
        read_exact_ppl(str(report))
    report.write_text(json.dumps({"execution": {"window_plan": "kld-chunks"},
                                  "overall": {"perplexity": 6.5}}))
    with pytest.raises(DumpError, match="lacks overall or logits_dump"):
        read_exact_ppl(str(report))
    report.write_text(json.dumps({"execution": {"window_plan": "sliding"}}))
    with pytest.raises(DumpError, match="--logits-out"):
        read_exact_ppl(str(report))
    report.write_text("{not json")
    with pytest.raises(DumpError, match="not valid JSON"):
        read_exact_ppl(str(report))
    log.write_text("Final estimate: PPL = 4.0100 +/- 0.05\n")
    with pytest.raises(DumpError, match="n_ctx=C"):
        read_exact_ppl(str(log))


def test_exact_ppl_paths_may_contain_at_signs(tmp_path) -> None:
    folder = tmp_path / "runs@2x"
    folder.mkdir()
    _llama_log(folder / "run.log")
    assert read_exact_ppl(str(folder / "run.log")).chunks == 3
    assert read_exact_ppl(f"{folder / 'run.log'}@2").chunks == 2
    literal = tmp_path / "odd.log@2"  # an existing path wins over the @N suffix
    _llama_log(literal)
    assert read_exact_ppl(str(literal)) == ExactPpl(pytest.approx(4.01), 3, 4096)
    with pytest.raises(FileNotFoundError):
        read_exact_ppl(str(tmp_path / "run.log@abc"))


def test_exact_ppl_of_other_windows_is_rejected(tmp_path) -> None:
    tokens = _tokens()
    logits = _logits(12)
    write_dump(tmp_path / "ref.kld", CONTEXT, tokens, logits)
    write_dump(tmp_path / "cand.kld", CONTEXT, tokens, logits * 1.01)
    candidates = [("cand", tmp_path / "cand.kld")]
    same = compare(tmp_path / "ref.kld", candidates, workers=1,
                   exact_ppl={"cand": ExactPpl(5.0, CHUNKS, CONTEXT),
                              "reference": ExactPpl(4.9, CHUNKS, None)})
    assert same["candidates"][0]["ppl_candidate_exact"] == 5.0
    assert same["reference"]["ppl_exact"] == 4.9
    # A full-run perplexity attached to a --chunks prefix would shift the gate's PPL check.
    with pytest.raises(DumpError, match="covers 3 windows but the comparison uses 2"):
        compare(tmp_path / "ref.kld", candidates, workers=1, chunks=2,
                exact_ppl={"cand": ExactPpl(5.0, CHUNKS, CONTEXT)})
    with pytest.raises(DumpError, match="context of 2048 tokens, the comparison uses 16"):
        compare(tmp_path / "ref.kld", candidates, workers=1,
                exact_ppl={"cand": ExactPpl(5.0, CHUNKS, 2048)})
    with pytest.raises(DumpError, match="names no compared run"):
        compare(tmp_path / "ref.kld", candidates, workers=1,
                exact_ppl={"other": ExactPpl(5.0, CHUNKS, None)})


def test_cli_compare_and_gate_exit_codes(tmp_path) -> None:
    tokens = _tokens()
    logits = _logits(9)
    write_dump(tmp_path / "ref.kld", CONTEXT, tokens, logits)
    noise = np.random.default_rng(10)
    write_dump(tmp_path / "prod.kld", CONTEXT, tokens,
                   logits + noise.normal(0.0, 0.01, logits.shape).astype(np.float32))
    write_dump(tmp_path / "q4.kld", CONTEXT, tokens,
                   logits + noise.normal(0.0, 0.5, logits.shape).astype(np.float32))
    write_dump(tmp_path / "stage.kld", CONTEXT, tokens,
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


def _gate_args(path) -> list[str]:
    return ["gate", "--results", str(path), "--prod", "prod", "--ceiling", "q4",
            "--candidate", "stage"]


def _stored(tmp_path, *candidates: dict):
    path = tmp_path / "compare.json"
    path.write_text(json.dumps(_document(*candidates)))
    return path


def test_cli_reports_unusable_input_with_exit_code_2(tmp_path, capsys) -> None:
    tokens = _tokens()
    logits = _logits(13)
    write_dump(tmp_path / "ref.kld", CONTEXT, tokens, logits)

    def rejected(arguments: list[str], message: str) -> None:
        assert kld.main(arguments) == 2
        assert message in capsys.readouterr().err

    rejected(["compare", "--reference", str(tmp_path / "missing.kld"), "--workers", "1",
              "--candidate", f"c={tmp_path / 'ref.kld'}"], "No such file")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    rejected(_gate_args(broken), "broken.json: not valid JSON")
    verdict = tmp_path / "gate.json"
    verdict.write_text(json.dumps({"schema": "ninfer-kld-gate/1", "checks": [], "passed": True}))
    rejected(_gate_args(verdict), "is not a compare result")
    no_reference = tmp_path / "no_reference.json"
    no_reference.write_text(json.dumps({"schema": COMPARE_SCHEMA, "candidates": []}))
    rejected(_gate_args(no_reference), "lacks its reference")
    no_candidates = tmp_path / "no_candidates.json"
    no_candidates.write_text(json.dumps({"schema": COMPARE_SCHEMA,
                                         "reference": {"tokens_sha256": "a", "path": "/r"}}))
    rejected(_gate_args(no_candidates), "lacks its list of candidates")
    for stage, message in (
            ({**_result("stage", 0.01, 0.1, 0.95, 6.0), "top1_agreement": None},
             "top1_agreement must be a finite number, got None"),
            ({**_result("stage", 0.01, 0.1, 0.95, 6.0), "kld": 0.01},
             "kld must be an object with mean and p99"),
            ({**_result("stage", 0.01, 0.1, 0.95, 6.0), "ppl_candidate_exact": 0},
             "ppl_candidate_exact must be positive"),
            ({**_result("stage", 0.01, 0.1, 0.95, 6.0), "ppl_candidate": float("nan")},
             "ppl_candidate must be a finite number")):
        rejected(_gate_args(_stored(tmp_path, _result("prod", 0.01, 0.1, 0.95, 6.0),
                                    _result("q4", 0.03, 0.3, 0.93, 6.1), stage)), message)
    for source, message in (("0", "bare number"), ("0@3", "must be positive"),
                            ("nan@3", "must be a finite number")):
        rejected(["compare", "--reference", str(tmp_path / "ref.kld"), "--workers", "1",
                  "--candidate", f"c={tmp_path / 'ref.kld'}", "--exact-ppl", f"c={source}"],
                 message)
    stored = _stored(tmp_path, _result("prod", 0.01, 0.1, 0.95, 6.0),
                     _result("q4", 0.03, 0.3, 0.93, 6.1), _result("stage", 0.01, 0.1, 0.95, 6.0))
    rejected(_gate_args(stored) + ["--max-top1-drop-points", "nan"],
             "limit max_top1_drop_points must be a finite number")


def test_cli_reports_internal_errors_with_exit_code_3(tmp_path, capsys, monkeypatch) -> None:
    def broken_compare(*args, **kwargs):
        raise ZeroDivisionError("a bug")

    monkeypatch.setattr(kld, "compare", broken_compare)
    write_dump(tmp_path / "ref.kld", CONTEXT, _tokens(), _logits(14))
    assert kld.main(["compare", "--reference", str(tmp_path / "ref.kld"), "--workers", "1",
                     "--candidate", f"c={tmp_path / 'ref.kld'}"]) == 3
    error = capsys.readouterr().err
    assert "ZeroDivisionError: a bug" in error and "internal error" in error
