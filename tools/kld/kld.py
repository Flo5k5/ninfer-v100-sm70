#!/usr/bin/env python3
"""Compare logits dumps: KL divergence, top-1 agreement and perplexity, plus a quality gate.

A dump is the file written by `llama-perplexity --kl-divergence-base FILE` or by
`ninfer-perplexity --logits-out FILE` (layout in tools/kld/dump.py and docs/perplexity.md).

Subcommands:
    inspect  FILE...                           validate and describe dumps
    compare  --reference REF --candidate [NAME=]FILE...  KLD/top-1/PPL of candidates vs reference
    gate     --results JSON... --prod NAME --candidate NAME --ceiling NAME [--previous NAME]

KLD uses llama.cpp's definition, with the candidate read from its 16-bit clamped dump rather than
from unquantized logits (see tools/kld/compare.py). Dumps clamp log-probabilities 16 nats below
the top token, so `--exact-ppl NAME=SOURCE` attaches the unclamped perplexity each run prints: a
number, a ninfer-perplexity report.json or a llama-perplexity log, which must cover the compared
windows. The gate uses it whenever both compared runs provide it.

Exit codes: 0 success (gate: PASS), 1 gate FAIL, 2 unusable input (missing or malformed file,
mismatched runs, results that are not compare results).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

from tools.kld.compare import compare, read_exact_ppl
from tools.kld.dump import DumpError, read_dump
from tools.kld.gate import DEFAULT_GATE, gate


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
    except (DumpError, OSError, ValueError) as error:
        print(f"kld: error: {error}", file=sys.stderr)
        return 2
    except KeyError as error:
        print(f"kld: error: missing field {error} in the input", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
