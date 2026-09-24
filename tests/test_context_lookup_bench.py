from __future__ import annotations

import json
import math

import pytest

from tools.bench.run_context_lookup import (
    Pooled,
    join_request_log,
    paired_intervals,
    paired_prompts,
    paired_ratios,
    request_seed,
)


def record(label: str, index: int, *, tokens: int, milliseconds: float, accepted: int,
           seed: int | None = None, workload: str = "edits") -> dict:
    result = {"label": label, "workload": workload, "index": index, "temperature": 1.0,
              "prompt_n": 900, "predicted_n": tokens, "predicted_ms": milliseconds,
              "draft_n": 4 * tokens, "draft_n_accepted": accepted}
    if seed is not None:
        result["seed"] = seed
    return result


def logged(entry: dict, rounds: int) -> dict:
    return {**entry, "logged": {"speculative": {"rounds": rounds, "fallback_steps": 0}}}


def log_entry(source: dict, *, rounds: int, seed: int = 12345, temperature: float = 1.0) -> dict:
    return {
        "event": "request_done",
        "request": {"sampling": {"temperature": temperature, "seed": seed}},
        "result": {"prompt_tokens": source["prompt_n"] + 64, "prefix_cache_hit_tokens": 64,
                   "completion_tokens": source["predicted_n"]},
        "speculative": {"rounds": rounds, "fallback_steps": 1,
                        "drafted_tokens": source["draft_n"],
                        "accepted_tokens": source["draft_n_accepted"],
                        "mtp_round_seconds": 0.5,
                        "lookup": {"rounds": 0, "entry_rounds": 0, "accepted_tokens": 0,
                                   "drafted_tokens": 0, "round_seconds": 0.0}},
    }


def write_log(tmp_path, entries: list[dict]):
    path = tmp_path / "requests.jsonl"
    warmup = {"event": "request_done", "request": {"sampling": {"temperature": 0.0, "seed": 1}},
              "result": {"prompt_tokens": 10, "prefix_cache_hit_tokens": 0,
                         "completion_tokens": 16},
              "speculative": {"drafted_tokens": 40, "accepted_tokens": 12}}
    path.write_text("\n".join(json.dumps(entry)
                              for entry in [{"event": "server_start"}, warmup, *entries]))
    return path


def test_decode_rates_exclude_the_prefill_token_and_use_logged_rounds() -> None:
    # 101 tokens: the first comes from prefill, 100 are decoded in 40 rounds over 1000 ms.
    pooled = Pooled.of([logged(record("a", 0, tokens=101, milliseconds=1000.0, accepted=65),
                               rounds=40)])
    assert pooled.tokens == 100
    assert pooled.rounds == 40
    assert pooled.tokens_per_second == pytest.approx(100.0)
    assert pooled.milliseconds_per_round == pytest.approx(25.0)
    assert pooled.tokens_per_round == pytest.approx(2.5)
    # Without the request log the rounds are unknown rather than estimated.
    unjoined = Pooled.of([record("a", 0, tokens=101, milliseconds=1000.0, accepted=65)])
    assert unjoined.rounds is None and math.isnan(unjoined.milliseconds_per_round)


def test_request_log_joins_on_the_seed_when_counts_coincide(tmp_path) -> None:
    first = record("a", 0, tokens=200, milliseconds=2000.0, accepted=120, seed=11)
    second = record("a", 1, tokens=200, milliseconds=2100.0, accepted=120, seed=22)
    log = write_log(tmp_path, [log_entry(second, rounds=71, seed=22),
                               log_entry(first, rounds=70, seed=11)])
    join_request_log([first, second], log)
    assert first["logged"]["speculative"]["rounds"] == 70
    assert second["logged"]["speculative"]["rounds"] == 71


def test_request_log_join_refuses_ambiguous_or_missing_entries(tmp_path) -> None:
    first = record("a", 0, tokens=200, milliseconds=2000.0, accepted=120)
    twin = record("a", 1, tokens=200, milliseconds=2100.0, accepted=120)
    log = write_log(tmp_path, [log_entry(first, rounds=70), log_entry(twin, rounds=71)])
    with pytest.raises(SystemExit, match="cannot join"):
        join_request_log([first, twin], log)

    lone = record("a", 2, tokens=300, milliseconds=2000.0, accepted=200)
    with pytest.raises(SystemExit, match="0 request-log entries"):
        join_request_log([lone], write_log(tmp_path, [log_entry(first, rounds=70)]))

    # Counts that match but a different seed name a different request.
    seeded = record("a", 3, tokens=200, milliseconds=2000.0, accepted=120, seed=5)
    with pytest.raises(SystemExit, match="cannot join"):
        join_request_log([seeded], write_log(tmp_path, [log_entry(seeded, rounds=70, seed=6)]))


def test_paired_ratios_and_intervals() -> None:
    baseline = [logged(record("base", i, tokens=401, milliseconds=4000.0 + 100 * i,
                              accepted=250), rounds=150) for i in range(6)]
    identical = [dict(entry, label="same") for entry in baseline]
    prompts = [[(base, same)] for base, same in zip(baseline, identical)]
    ratios = paired_ratios(prompts)
    assert ratios == pytest.approx((1.0, 1.0, 1.0))
    assert all(low == pytest.approx(1.0) and high == pytest.approx(1.0)
               for low, high in paired_intervals(prompts, 200))

    # Every request decodes the same tokens in 10% less time, in the same rounds.
    faster = [dict(entry, label="fast", predicted_ms=entry["predicted_ms"] / 1.1)
              for entry in baseline]
    prompts = [[(base, fast)] for base, fast in zip(baseline, faster)]
    tok_s, ms_round, tok_round = paired_ratios(prompts)
    assert tok_s == pytest.approx(1.1)
    assert ms_round == pytest.approx(1 / 1.1)
    assert tok_round == pytest.approx(1.0)
    (low, high), _, _ = paired_intervals(prompts, 200)
    assert low == pytest.approx(1.1) and high == pytest.approx(1.1)

    # Mixed gains: the interval brackets the pooled estimate.
    mixed = [dict(entry, label="mixed", predicted_ms=entry["predicted_ms"] * (0.8 + 0.08 * i))
             for i, entry in enumerate(baseline)]
    prompts = [[(base, other)] for base, other in zip(baseline, mixed)]
    estimate = paired_ratios(prompts)[0]
    low, high = paired_intervals(prompts, 2000)[0]
    assert low < estimate < high


def test_replicate_runs_pair_by_workload_and_index() -> None:
    sampled = {
        label: [record(label, index, tokens=100, milliseconds=1000.0, accepted=50,
                       workload=workload)
                for workload in ("edits", "prose") for index in range(3)]
        for label in ("fixed-a", "adaptive-a", "fixed-b", "adaptive-b")
    }
    sampled["adaptive-b"] = [entry for entry in sampled["adaptive-b"] if entry["index"] != 2]
    prompts = paired_prompts(sampled, ["fixed-a", "fixed-b"], ["adaptive-a", "adaptive-b"],
                             "edits")
    assert [len(prompt) for prompt in prompts] == [2, 2, 1]
    for prompt in prompts:
        for baseline, candidate in prompt:
            assert baseline["workload"] == candidate["workload"] == "edits"
            assert baseline["index"] == candidate["index"]
            assert (baseline["label"], candidate["label"]) in {("fixed-a", "adaptive-a"),
                                                               ("fixed-b", "adaptive-b")}


def test_request_seeds_pair_policies_and_fit_the_server_seed() -> None:
    seeds = {request_seed(0, workload, index)
             for workload in ("rewrite", "edits") for index in range(9)}
    assert len(seeds) == 18
    assert all(0 <= seed < 2**63 for seed in seeds)
    assert request_seed(0, "edits", 3) == request_seed(0, "edits", 3)
    assert request_seed(1, "edits", 3) != request_seed(0, "edits", 3)
