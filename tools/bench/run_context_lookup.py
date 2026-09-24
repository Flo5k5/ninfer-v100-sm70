#!/usr/bin/env python3
"""Context-lookup A/B workloads for an already running `ninfer-serve`.

Three subcommands:

  workloads  build request bodies from a directory of public Python sources and one long document
  run        send selected workloads to a server and append one JSON record per request
  summarize  pool each label's requests, compare labels request by request, and check greedy
             output equality across labels

`run` only uses the public Chat Completions endpoint. `summarize` joins the server's
`--request-log-jsonl` records, which carry the exact decode round count and the per-request MTP and
context-lookup counters.
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import math
import random
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

WORKLOADS = ("rewrite", "edits", "write", "prose", "summary")
REWRITE_FILES = ("graphlib.py", "filecmp.py", "glob.py", "operator.py", "string.py",
                 "tabnanny.py", "queue.py", "json/decoder.py", "shlex.py")
EDIT_FILES = ("textwrap.py", "heapq.py", "base64.py", "selectors.py", "weakref.py", "pprint.py",
              "calendar.py", "contextlib.py", "tokenize.py")
AGENT_SYSTEM = ("You are an autonomous coding agent working in a repository. Use the tools to "
                "change files. Do not explain your work; call the tools directly.")
# The prose questions alternate French and English on purpose: the measured deployment serves both,
# and lookup must not widen on, or slow down, ordinary text in either language. They stay verbatim
# because the published measurements (docs/v100.md) used them.
PROSE = (
    "Explique en quatre phrases pourquoi le ciel est bleu.",
    "Write a Python function that parses an ISO 8601 date string and returns a datetime, with "
    "error handling.",
    "Donne trois arguments pour et trois contre le télétravail, sous forme de liste.",
    "Explain the difference between a mutex and a semaphore with a short example in C.",
    "Rédige un courriel poli pour décaler une réunion de mardi à jeudi.",
    "What are the main causes of inflation? Answer in one paragraph.",
    "Écris une fonction TypeScript qui déduplique un tableau d'objets par une clé donnée.",
    "Summarize the plot of Romeo and Juliet in five sentences.",
    "Quelles sont les étapes pour créer une migration Entity Framework Core ? Réponds en liste "
    "numérotée.",
)
# The measured summary document is a French report, so its instruction is French too.
SUMMARY_INSTRUCTION = "Résume ce document en cinq phrases."


def _tool(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": list(properties)}}}


STRING = {"type": "string"}
TOOLS = [
    _tool("Read", "Read a file from the local filesystem.", {"file_path": STRING}),
    _tool("Edit", "Replace one exact occurrence of old_string with new_string in a file.",
          {"file_path": STRING, "old_string": STRING, "new_string": STRING}),
    _tool("Write", "Write a file, replacing its whole content.",
          {"file_path": STRING, "content": STRING}),
    _tool("Bash", "Run a shell command.", {"command": STRING, "description": STRING}),
]
TOOLS[-1]["function"]["parameters"]["required"] = ["command"]


def _functions(source: str) -> list[str]:
    """Function names defined exactly once (dunders excluded), in order of first mention. A name
    first mentioned inside a longer one sorts after it, since that mention is the longer name."""
    names = [node.name for node in ast.walk(ast.parse(source))
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    unique = {name for name in names if names.count(name) == 1 and not name.startswith("__")}
    return sorted(unique, key=lambda name: (source.index(name), -len(name)))


def _agent_turn(path: str, instruction: str, source: str) -> list[dict[str, Any]]:
    # The Read result uses `cat -n` line prefixes, as coding agents present files.
    numbered = "\n".join(f"{number:>6}\t{line}"
                         for number, line in enumerate(source.split("\n"), 1))
    return [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_read", "type": "function",
            "function": {"name": "Read", "arguments": json.dumps({"file_path": path})}}]},
        {"role": "tool", "tool_call_id": "call_read", "content": numbered},
    ]


def build_workloads(source_dir: Path, document: Path) -> dict[str, list[dict[str, Any]]]:
    """rewrite: a file in a code block, answered with the whole file after two small edits.
    edits: an agent turn after a Read result, answered with Edit calls copying file lines.
    write: the same turn answered with one Write call holding the whole edited file.
    prose: short questions with nothing to copy. summary: one long document, summarized."""
    workloads: dict[str, list[dict[str, Any]]] = {name: [] for name in WORKLOADS}
    for name in REWRITE_FILES:
        source = (source_dir / name).read_text()
        first = _functions(source)[0]
        workloads["rewrite"].append({"max_tokens": 6000, "messages": [{"role": "user", "content": (
            f"Here is the file `{name}`:\n\n```python\n{source}\n```\n\n"
            f"Make two changes: add the comment line `# Reviewed.` at the very top of the file, "
            f"and rename the function `{first}` to `{first}_impl`, updating every reference. "
            "Reply with the complete updated file in a single code block and nothing else.")}]})
    for name in REWRITE_FILES:
        source = (source_dir / name).read_text()
        first = _functions(source)[0]
        path = f"/repo/lib/{name}"
        workloads["write"].append({"max_tokens": 6000, "tools": TOOLS, "messages": _agent_turn(
            path, f"Rewrite {path}: add the comment line `# Reviewed.` at the very top and rename "
            f"the function `{first}` to `{first}_impl`, updating every reference. Save the result "
            "with a single Write call holding the complete new file content (without the "
            "line-number prefixes).", source)})
    for name in EDIT_FILES:
        source = (source_dir / name).read_text()
        functions = _functions(source)
        step = max(1, len(functions) // 3)
        first, second, third = (functions[min(len(functions) - 1, i * step + step // 2)]
                                for i in range(3))
        path = f"/repo/lib/{name}"
        instruction = (
            f"Make these changes to {path}:\n"
            f"1. Give `{first}` a docstring that states what it returns.\n"
            f"2. Add a type annotation for every parameter of `{second}`.\n"
            f"3. Insert a comment `# NOTE: hot path` on the line before each `return` in "
            f"`{third}`.\n"
            "Use one Edit call per change; old_string must be copied exactly from the file, "
            "without the line-number prefixes, and must include enough lines to be unique.")
        workloads["edits"].append({"max_tokens": 3000, "tools": TOOLS,
                                   "messages": _agent_turn(path, instruction, source)})
    for prompt in PROSE:
        workloads["prose"].append({"max_tokens": 400,
                                   "messages": [{"role": "user", "content": prompt}]})
    text = document.read_text()
    for _ in range(9):
        workloads["summary"].append({"max_tokens": 500, "messages": [{"role": "user", "content": (
            text + "\n\n" + SUMMARY_INSTRUCTION)}]})
    return workloads


def request_seed(base: int, workload: str, index: int) -> int:
    """Sampling seed of one request. Every label uses the same seed for the same request, so the
    runs of two policies draw the same random numbers, and the seed identifies the request in the
    server's request log."""
    digest = hashlib.sha256(f"{base}/{workload}/{index}".encode()).digest()
    return int.from_bytes(digest[:8], "little") >> 1


def _post(url: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(url, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def run(args: argparse.Namespace) -> None:
    workloads = json.loads(Path(args.workloads).read_text())
    url = f"{args.server.rstrip('/')}/v1/chat/completions"
    # One short request first so graph installation and allocator growth are not measured.
    _post(url, {"model": "m", "messages": [{"role": "user", "content": "Say hello."}],
                "max_tokens": 16, "reasoning_effort": "none", "temperature": 0.0})
    with open(args.output, "a", encoding="utf-8") as output:
        for item in args.select.split(","):
            name, _, count = item.partition(":")
            requests = workloads[name][: int(count)] if count else workloads[name]
            for index, request in enumerate(requests):
                seed = request_seed(args.seed, name, index)
                body = dict(request)
                body.update({"model": "m", "reasoning_effort": "none",
                             "temperature": args.temperature, "top_p": 0.95, "top_k": 20,
                             "seed": seed})
                started  = time.time()
                response = _post(url, body)
                choice   = response["choices"][0]
                message  = choice["message"]
                timings  = response.get("timings") or {}
                record = {
                    "label": args.label, "workload": name, "index": index,
                    "temperature": args.temperature, "seed": seed,
                    "wall_s": time.time() - started,
                    "finish_reason": choice.get("finish_reason"),
                    "prompt_n": timings.get("prompt_n", 0),
                    "predicted_n": timings.get("predicted_n", 0),
                    "predicted_ms": timings.get("predicted_ms", 0.0),
                    "draft_n": timings.get("draft_n", 0),
                    "draft_n_accepted": timings.get("draft_n_accepted", 0),
                    "content": message.get("content"),
                    "tool_calls": [{"name": call["function"]["name"],
                                    "arguments": call["function"]["arguments"]}
                                   for call in message.get("tool_calls") or []],
                }
                output.write(json.dumps(record) + "\n")
                output.flush()
                print(f"{args.label} {name}[{index}] {record['predicted_n']} tokens "
                      f"{Pooled.of([record]).tokens_per_second:.1f} tok/s", flush=True)


def decode_tokens(record: dict[str, Any]) -> int:
    """Tokens decoded after the first one, which prefill produces; `predicted_ms` spans them."""
    return max(record["predicted_n"] - 1, 0)


def decode_rounds(record: dict[str, Any]) -> int | None:
    """Decode rounds from the joined request log, or None. The response alone cannot count them:
    a stop inside a verified block discards its tail, so tokens minus accepted drafts undercounts."""
    logged = record.get("logged")
    if logged is None:
        return None
    return logged["speculative"]["rounds"] + logged["speculative"]["fallback_steps"]


@dataclass(frozen=True)
class Pooled:
    """Decoded tokens, decode time and decode rounds summed over requests."""
    tokens: int
    milliseconds: float
    rounds: int | None

    @staticmethod
    def of(records: Sequence[dict[str, Any]]) -> Pooled:
        rounds = [decode_rounds(record) for record in records]
        return Pooled(tokens=sum(decode_tokens(record) for record in records),
                      milliseconds=sum(record["predicted_ms"] for record in records),
                      rounds=None if None in rounds else sum(rounds))

    @property
    def tokens_per_second(self) -> float:
        return 1000 * self.tokens / self.milliseconds if self.milliseconds > 0 else math.nan

    @property
    def milliseconds_per_round(self) -> float:
        return self.milliseconds / self.rounds if self.rounds else math.nan

    @property
    def tokens_per_round(self) -> float:
        return self.tokens / self.rounds if self.rounds else math.nan


def _record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (round(record["temperature"], 6), record.get("seed"), record["prompt_n"],
            record["predicted_n"], record["draft_n"], record["draft_n_accepted"])


def _log_key(done: dict[str, Any], seeded: bool) -> tuple[Any, ...]:
    sampling, result = done["request"]["sampling"], done["result"]
    speculative = done["speculative"]
    uncached = result["prompt_tokens"] - min(result["prefix_cache_hit_tokens"],
                                             result["prompt_tokens"])
    return (round(sampling["temperature"], 6), sampling["seed"] if seeded else None, uncached,
            result["completion_tokens"], speculative["drafted_tokens"],
            speculative["accepted_tokens"])


def join_request_log(records: Sequence[dict[str, Any]], log: Path) -> None:
    """Attaches to each record its `request_done` entry from one server's request log. Entries are
    matched on the request's seed, when `run` recorded one, and on the counts both sides report
    (uncached prompt, completion, drafted and accepted tokens); a record whose key matches no entry
    or several entries, or that shares its key with another record, is refused."""
    done = [entry for entry in map(json.loads, log.read_text().splitlines())
            if entry.get("event") == "request_done"]
    seeded = {"seed" in record for record in records}
    if len(seeded) > 1:
        sys.exit(f"{log}: some records carry a seed and some do not")
    entries: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for entry in done:
        entries[_log_key(entry, seeded == {True})].append(entry)
    keys = collections.Counter(_record_key(record) for record in records)
    refused = []
    for record in records:
        key = _record_key(record)
        matches = entries.get(key, [])
        if len(matches) != 1 or keys[key] != 1:
            refused.append(f"{record['label']} {record['workload']}[{record['index']}]: "
                           f"{len(matches)} request-log entries and {keys[key]} records share "
                           "its key")
            continue
        record["logged"] = matches[0]
    if refused:
        sys.exit(f"{log}: cannot join the request log unambiguously\n  " + "\n  ".join(refused))


def _interval(values: list[float], level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of bootstrap replicates."""
    values = sorted(value for value in values if not math.isnan(value))
    if not values:
        return math.nan, math.nan
    tail = (1 - level) / 2 * (len(values) - 1)
    return values[math.floor(tail)], values[math.ceil(len(values) - 1 - tail)]


def label_interval(records: Sequence[dict[str, Any]], rounds: int) -> tuple[float, float]:
    """Bootstrap interval of one label's pooled tokens per second, resampling its requests."""
    generator = random.Random(1)
    return _interval([Pooled.of([generator.choice(records) for _ in records]).tokens_per_second
                      for _ in range(rounds)])


Pairs = list[tuple[dict[str, Any], dict[str, Any]]]


def paired_ratios(prompts: Sequence[Pairs]) -> tuple[float, float, float]:
    """Candidate over baseline for pooled tokens per second, milliseconds per decode round and
    tokens per decode round. Each prompt holds the (baseline, candidate) request pairs sent for it."""
    pairs = [pair for prompt in prompts for pair in prompt]
    baseline = Pooled.of([pair[0] for pair in pairs])
    candidate = Pooled.of([pair[1] for pair in pairs])
    return (candidate.tokens_per_second / baseline.tokens_per_second,
            candidate.milliseconds_per_round / baseline.milliseconds_per_round,
            candidate.tokens_per_round / baseline.tokens_per_round)


def paired_intervals(prompts: Sequence[Pairs], rounds: int) -> list[tuple[float, float]]:
    """Paired bootstrap intervals of `paired_ratios`. Prompts are resampled with all their pairs,
    so both policies always see the same prompts and replicate runs of a prompt stay together."""
    generator = random.Random(1)
    replicates = [paired_ratios([generator.choice(prompts) for _ in prompts])
                  for _ in range(rounds)]
    return [_interval([replicate[metric] for replicate in replicates]) for metric in range(3)]


def paired_prompts(sampled: dict[str, list[dict[str, Any]]], baselines: Sequence[str],
                   candidates: Sequence[str], workload: str) -> list[Pairs]:
    """Pairs each candidate run's requests with the same requests of its baseline run."""
    prompts: dict[int, Pairs] = collections.defaultdict(list)
    for baseline, candidate in zip(baselines, candidates):
        runs = [{record["index"]: record for record in sampled[label]
                 if record["workload"] == workload} for label in (baseline, candidate)]
        for index in sorted(runs[0].keys() & runs[1].keys()):
            prompts[index].append((runs[0][index], runs[1][index]))
    return [prompts[index] for index in sorted(prompts)]


def _gain(ratio: float) -> str:
    return "n/a" if math.isnan(ratio) else f"{100 * (ratio - 1):+.1f}%"


def _gain_interval(interval: tuple[float, float]) -> str:
    return f"[{_gain(interval[0])}, {_gain(interval[1])}]"


def print_label_table(sampled: dict[str, list[dict[str, Any]]], rounds: int) -> None:
    print("label workload requests tokens tok/s ci95 ms/round tok/round widened% "
          "copied/lookup entry% lookup_ms draft_ms")
    for label in sorted(sampled):
        for workload in WORKLOADS:
            records = [r for r in sampled[label] if r["workload"] == workload]
            if not records:
                continue
            pooled = Pooled.of(records)
            low, high = label_interval(records, rounds)
            line = (f"{label} {workload} {len(records)} {pooled.tokens} "
                    f"{pooled.tokens_per_second:.1f} {low:.1f}-{high:.1f}")
            if pooled.rounds is None:
                print(line + " (no request log)")
                continue
            logged = [r["logged"]["speculative"] for r in records]
            lookups = sum(s["lookup"]["rounds"] for s in logged)
            entries = sum(s["lookup"]["entry_rounds"] for s in logged)
            copied = sum(s["lookup"]["accepted_tokens"] for s in logged)
            lookup_s = sum(s["lookup"]["round_seconds"] for s in logged)
            draft_s = sum(s["mtp_round_seconds"] for s in logged)
            print(f"{line} {pooled.milliseconds_per_round:.2f} {pooled.tokens_per_round:.2f} "
                  f"{100 * lookups / max(pooled.rounds, 1):.1f} {copied / max(lookups, 1):.2f} "
                  f"{100 * entries / max(lookups, 1):.1f} "
                  f"{1000 * lookup_s / max(lookups, 1):.2f} "
                  f"{1000 * draft_s / max(pooled.rounds - lookups, 1):.2f}")


def _arm(pooled: Pooled) -> str:
    if pooled.rounds is None:
        return f"{pooled.tokens_per_second:.1f} (no request log)"
    return (f"{pooled.tokens_per_second:.1f} ({pooled.milliseconds_per_round:.2f} ms x "
            f"{pooled.tokens_per_round:.2f})")


def print_comparison(sampled: dict[str, list[dict[str, Any]]], baselines: Sequence[str],
                     candidates: Sequence[str], rounds: int) -> None:
    print(f"\npaired: {','.join(candidates)} over {','.join(baselines)}; changes with 95% "
          f"intervals from {rounds} prompt resamples; * marks a tok/s change whose interval "
          "excludes zero")
    print("workload prompts pairs baseline_tok/s(ms/round x tok/round) candidate_tok/s(...) "
          "tok/s ci95 ms/round ci95 tok/round ci95")
    for workload in WORKLOADS:
        prompts = paired_prompts(sampled, baselines, candidates, workload)
        if not prompts:
            continue
        pairs = [pair for prompt in prompts for pair in prompt]
        ratios = paired_ratios(prompts)
        intervals = paired_intervals(prompts, rounds)
        significant = " *" if intervals[0][0] > 1 or intervals[0][1] < 1 else ""
        print(f"{workload} {len(prompts)} {len(pairs)} {_arm(Pooled.of([p[0] for p in pairs]))} "
              f"{_arm(Pooled.of([p[1] for p in pairs]))} "
              + " ".join(f"{_gain(ratio)} {_gain_interval(interval)}"
                         for ratio, interval in zip(ratios, intervals))
              + significant)


def print_greedy_comparison(by_label: dict[str, dict[str, list[dict[str, Any]]]],
                            reference: str) -> None:
    outputs: dict[tuple[str, int], dict[str, Any]] = collections.defaultdict(dict)
    for label, runs in by_label.items():
        for record in runs["greedy"]:
            outputs[(record["workload"], record["index"])][label] = (
                record["content"], json.dumps(record["tool_calls"], sort_keys=True))
    compared = differing = 0
    for key, per_label in sorted(outputs.items()):
        if reference not in per_label:
            continue
        for label, value in sorted(per_label.items()):
            if label == reference:
                continue
            compared += 1
            if value != per_label[reference]:
                differing += 1
                print(f"greedy output differs: {label} vs {reference} on {key[0]}[{key[1]}]")
    print(f"greedy comparisons against {reference}: {compared}, differing: {differing}")


def summarize(args: argparse.Namespace) -> None:
    by_label: dict[str, dict[str, list[dict[str, Any]]]] = collections.defaultdict(
        lambda: {"sampled": [], "greedy": []})
    for path in args.records:
        for line in Path(path).read_text().splitlines():
            record = json.loads(line)
            kind = "greedy" if record["temperature"] == 0 else "sampled"
            by_label[record["label"]][kind].append(record)
    for label, runs in by_label.items():
        requests = collections.Counter((r["workload"], r["index"]) for r in runs["sampled"])
        repeated = sorted(request for request, count in requests.items() if count > 1)
        if repeated:
            sys.exit(f"{label}: sampled requests recorded more than once: {repeated}; "
                     "give every run its own label")
    for label, log in args.request_log or []:
        if label not in by_label:
            sys.exit(f"--request-log names unknown label {label}")
        join_request_log(by_label[label]["sampled"], Path(log))

    sampled = {label: runs["sampled"] for label, runs in by_label.items() if runs["sampled"]}
    print_label_table(sampled, args.bootstrap)
    for baseline_list, candidate_list in args.compare or []:
        baselines, candidates = baseline_list.split(","), candidate_list.split(",")
        unknown = [label for label in baselines + candidates if label not in sampled]
        if len(baselines) != len(candidates) or unknown:
            sys.exit(f"--compare needs label lists of equal length and known labels: "
                     f"{baseline_list} {candidate_list}")
        print_comparison(sampled, baselines, candidates, args.bootstrap)
    if any(runs["greedy"] for runs in by_label.values()):
        print()
        print_greedy_comparison(by_label, args.greedy_reference)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("workloads", help="build the workload file")
    build.add_argument("--source-dir", required=True, type=Path,
                       help="Python standard-library source directory, e.g. /usr/lib/python3.10")
    build.add_argument("--document", required=True, type=Path, help="long document to summarize")
    build.add_argument("--output", required=True, type=Path)
    runner = commands.add_parser("run", help="run workloads against one server")
    runner.add_argument("--server", default="http://127.0.0.1:8080")
    runner.add_argument("--workloads", required=True)
    runner.add_argument("--select", default=",".join(WORKLOADS),
                        help="comma list of workload[:count]")
    runner.add_argument("--temperature", type=float, default=1.0)
    runner.add_argument("--seed", type=int, default=0,
                        help="base of the per-request sampling seeds; runs to be compared share it")
    runner.add_argument("--label", required=True)
    runner.add_argument("--output", required=True)
    report = commands.add_parser("summarize", help="pool and compare results per label")
    report.add_argument("records", nargs="+")
    report.add_argument("--request-log", nargs=2, action="append", metavar=("LABEL", "JSONL"),
                        help="join the request log of the server that ran LABEL")
    report.add_argument("--compare", nargs=2, action="append", metavar=("BASELINE", "CANDIDATE"),
                        help="pair CANDIDATE's requests with BASELINE's by workload and index; "
                             "comma lists pair replicate runs in order")
    report.add_argument("--bootstrap", type=int, default=4000, help="bootstrap resamples")
    report.add_argument("--greedy-reference", default="off")
    args = parser.parse_args()
    if args.command == "workloads":
        workloads = build_workloads(args.source_dir, args.document)
        args.output.write_text(json.dumps(workloads))
        for name, requests in workloads.items():
            print(f"{name}: {len(requests)} requests")
    elif args.command == "run":
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
