"""Per-stage quality gate over `compare` results.

Each conversion stage is checked against the previous stage, against production and against a
known-acceptable ceiling (llama.cpp Q4_K_M). The limits are those of docs/perplexity.md; every
limit can be overridden.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from tools.kld.compare import COMPARE_SCHEMA
from tools.kld.dump import DumpError

GATE_SCHEMA = "ninfer-kld-gate/1"

DEFAULT_GATE = {
    "max_stage_mean_kld_increase": 0.005,
    "max_total_mean_kld_increase": 0.015,
    "max_p99_kld_ratio_to_prod": 2.0,
    "max_top1_drop_points": 1.0,
    "max_ppl_increase_percent": 1.0,
}

_RESULT_KEYS = ("name", "kld", "top1_agreement", "ppl_candidate")


def validate_results(results: Sequence[Any]) -> None:
    """Reject anything but compare results before reading them."""
    for index, document in enumerate(results):
        if not isinstance(document, dict) or document.get("schema") != COMPARE_SCHEMA:
            schema = document.get("schema") if isinstance(document, dict) else None
            raise DumpError(f"results[{index}] is not a compare result (schema {schema!r}, "
                            f"expected {COMPARE_SCHEMA!r})")
        reference = document.get("reference")
        candidates = document.get("candidates")
        if not isinstance(reference, dict) or not isinstance(candidates, list) or \
                not {"tokens_sha256", "path"} <= reference.keys():
            raise DumpError(f"results[{index}] lacks its reference or candidates")
        for candidate in candidates:
            missing = [key for key in _RESULT_KEYS
                       if not isinstance(candidate, dict) or key not in candidate]
            if missing:
                raise DumpError(f"results[{index}] has a candidate without {missing}")


def _find(results: Sequence[dict[str, Any]], name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    found = [(document, candidate) for document in results for candidate in document["candidates"]
             if candidate["name"] == name]
    if len(found) != 1:
        raise DumpError(f"expected exactly one result named {name!r}, found {len(found)}")
    return found[0]


def gate(results: Sequence[dict[str, Any]], *, prod: str, candidate: str, ceiling: str,
         previous: str | None = None, limits: dict[str, float] | None = None) -> dict[str, Any]:
    validate_results(results)
    limits = {**DEFAULT_GATE, **(limits or {})}
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
