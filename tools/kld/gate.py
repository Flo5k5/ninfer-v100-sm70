"""Per-stage quality gate over `compare` results.

Each conversion stage is checked against the previous stage, against production and against a
known-acceptable ceiling (llama.cpp Q4_K_M). The limits are those of docs/perplexity.md; every
limit can be overridden.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from tools.kld.compare import COMPARE_SCHEMA
from tools.kld.dump import DumpError, finite_number

GATE_SCHEMA = "ninfer-kld-gate/1"

DEFAULT_GATE = {
    "max_stage_mean_kld_increase": 0.005,
    "max_total_mean_kld_increase": 0.015,
    "max_p99_kld_ratio_to_prod": 2.0,
    "max_top1_drop_points": 1.0,
    "max_ppl_increase_percent": 1.0,
}


def _validate_candidate(candidate: Any, where: str) -> None:
    if not isinstance(candidate, dict) or not isinstance(candidate.get("name"), str):
        raise DumpError(f"{where} has a candidate without a name")
    label = f"{where} candidate {candidate['name']!r}"
    kld = candidate.get("kld")
    if not isinstance(kld, dict):
        raise DumpError(f"{label}: kld must be an object with mean and p99, got {kld!r}")
    finite_number(kld.get("mean"), f"{label}: kld.mean")
    finite_number(kld.get("p99"), f"{label}: kld.p99")
    top1 = finite_number(candidate.get("top1_agreement"), f"{label}: top1_agreement")
    if not 0.0 <= top1 <= 1.0:
        raise DumpError(f"{label}: top1_agreement must be in [0, 1], got {top1!r}")
    finite_number(candidate.get("ppl_candidate"), f"{label}: ppl_candidate", positive=True)
    if candidate.get("ppl_candidate_exact") is not None:
        finite_number(candidate["ppl_candidate_exact"], f"{label}: ppl_candidate_exact",
                      positive=True)


def validate_results(results: Sequence[Any]) -> None:
    """Reject anything but well-formed compare results before reading them."""
    for index, document in enumerate(results):
        where = f"results[{index}]"
        if not isinstance(document, dict) or document.get("schema") != COMPARE_SCHEMA:
            schema = document.get("schema") if isinstance(document, dict) else None
            raise DumpError(f"{where} is not a compare result (schema {schema!r}, "
                            f"expected {COMPARE_SCHEMA!r})")
        reference = document.get("reference")
        if not isinstance(reference, dict) or not isinstance(reference.get("tokens_sha256"), str) \
                or not isinstance(reference.get("path"), str):
            raise DumpError(f"{where} lacks its reference (tokens_sha256 and path)")
        candidates = document.get("candidates")
        if not isinstance(candidates, list):
            raise DumpError(f"{where} lacks its list of candidates")
        for candidate in candidates:
            _validate_candidate(candidate, where)


def _find(results: Sequence[dict[str, Any]], name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    found = [(document, candidate) for document in results for candidate in document["candidates"]
             if candidate["name"] == name]
    if len(found) != 1:
        raise DumpError(f"expected exactly one result named {name!r}, found {len(found)}")
    return found[0]


def gate(results: Sequence[dict[str, Any]], *, prod: str, candidate: str, ceiling: str,
         previous: str | None = None, limits: dict[str, float] | None = None) -> dict[str, Any]:
    validate_results(results)
    unknown = set(limits or {}) - set(DEFAULT_GATE)
    if unknown:
        raise DumpError(f"unknown gate limits: {sorted(unknown)}")
    limits = {key: finite_number(value, f"limit {key}")
              for key, value in {**DEFAULT_GATE, **(limits or {})}.items()}
    named = {role: _find(results, name) for role, name in
             (("prod", prod), ("candidate", candidate), ("ceiling", ceiling),
              ("previous", previous or prod))}
    references = {(document["reference"]["tokens_sha256"], document["reference"]["path"])
                  for document, _ in named.values()}
    if len(references) != 1:
        raise DumpError("gate inputs were compared against different references or corpora")
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
