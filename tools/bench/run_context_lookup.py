#!/usr/bin/env python3
"""Context-lookup A/B workloads for an already running `ninfer-serve`.

Three subcommands:

  workloads  build request bodies from a directory of public Python sources and one long document
  run        send selected workloads to a server and append one JSON record per request
  summarize  pool the records of each label and check greedy output equality across labels

`run` only uses the public Chat Completions endpoint. `summarize` optionally joins the server's
`--request-log-jsonl` records, which carry the per-request MTP and context-lookup counters.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import random
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

WORKLOADS = ("rewrite", "edits", "write", "prose", "summary")
REWRITE_FILES = ("graphlib.py", "filecmp.py", "glob.py", "operator.py", "string.py",
                 "tabnanny.py", "queue.py", "json/decoder.py", "shlex.py")
EDIT_FILES = ("textwrap.py", "heapq.py", "base64.py", "selectors.py", "weakref.py", "pprint.py",
              "calendar.py", "contextlib.py", "tokenize.py")
AGENT_SYSTEM = ("You are an autonomous coding agent working in a repository. Use the tools to "
                "change files. Do not explain your work; call the tools directly.")
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
            text + "\n\nRésume ce document en cinq phrases.")}]})
    return workloads


def _post(url: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(url, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def run(args: argparse.Namespace) -> None:
    workloads = json.loads(Path(args.workloads).read_text())
    url = f"{args.server.rstrip('/')}/v1/chat/completions"
    # One short request first so graph installation and allocator growth are not measured. The
    # request log therefore holds one extra record before each run's measured requests.
    _post(url, {"model": "m", "messages": [{"role": "user", "content": "Say hello."}],
                "max_tokens": 16, "reasoning_effort": "none", "temperature": 0.0})
    with open(args.output, "a", encoding="utf-8") as output:
        for item in args.select.split(","):
            name, _, count = item.partition(":")
            requests = workloads[name][: int(count)] if count else workloads[name]
            for index, request in enumerate(requests):
                body = dict(request)
                body.update({"model": "m", "reasoning_effort": "none",
                             "temperature": args.temperature, "top_p": 0.95, "top_k": 20})
                started  = time.time()
                response = _post(url, body)
                message  = response["choices"][0]["message"]
                timings  = response.get("timings") or {}
                record = {
                    "label": args.label, "workload": name, "index": index,
                    "temperature": args.temperature, "wall_s": time.time() - started,
                    "predicted_n": timings.get("predicted_n", 0),
                    "predicted_ms": timings.get("predicted_ms", 0.0),
                    "draft_n_accepted": timings.get("draft_n_accepted", 0),
                    "content": message.get("content"),
                    "tool_calls": [[call["function"]["name"], call["function"]["arguments"]]
                                   for call in message.get("tool_calls") or []],
                }
                output.write(json.dumps(record) + "\n")
                output.flush()
                steps = max(record["predicted_n"] - record["draft_n_accepted"], 1)
                print(f"{args.label} {name}[{index}] {record['predicted_n']} tokens "
                      f"{record['predicted_ms'] / steps:.1f} ms/step "
                      f"{record['predicted_n'] / steps:.2f} tok/step", flush=True)


def _pooled(records: list[dict[str, Any]]) -> tuple[int, float, int]:
    tokens = sum(r["predicted_n"] for r in records)
    milliseconds = sum(r["predicted_ms"] for r in records)
    steps = sum(max(r["predicted_n"] - r["draft_n_accepted"], 1) for r in records)
    return tokens, milliseconds, steps


def _interval(records: list[dict[str, Any]], rounds: int = 2000) -> tuple[float, float]:
    """Percentile bootstrap over requests of the pooled tokens per second."""
    generator = random.Random(1)
    values = sorted(1000 * t / ms for t, ms, _ in (
        _pooled([generator.choice(records) for _ in records]) for _ in range(rounds)))
    return values[int(0.025 * rounds)], values[int(0.975 * rounds)]


def summarize(args: argparse.Namespace) -> None:
    by_label: dict[str, dict[str, list[dict[str, Any]]]] = collections.defaultdict(
        lambda: {"sampled": [], "greedy": []})
    for path in args.records:
        for line in Path(path).read_text().splitlines():
            record = json.loads(line)
            kind = "greedy" if record["temperature"] == 0 else "sampled"
            by_label[record["label"]][kind].append(record)
    for label, log in args.request_log or []:
        # Request-log order: warmup, sampled requests, then (if greedy ran) warmup, greedy.
        runs = by_label[label]
        done = [json.loads(line) for line in Path(log).read_text().splitlines()
                if '"request_done"' in line]
        order = [None] + runs["sampled"] + ([None] + runs["greedy"] if runs["greedy"] else [])
        if len(done) != len(order):
            sys.exit(f"{label}: {len(done)} request-log records, expected {len(order)}")
        for record, logged in zip(order, done):
            if record is not None:
                record["speculative"] = logged["speculative"]

    print("label workload requests tokens tok/s ci95 ms/step tok/step widened% "
          "accepted/lookup lookup_ms draft_ms")
    for label in sorted(by_label):
        for workload in WORKLOADS:
            records = [r for r in by_label[label]["sampled"] if r["workload"] == workload]
            if not records:
                continue
            tokens, milliseconds, steps = _pooled(records)
            low, high = _interval(records)
            line = (f"{label} {workload} {len(records)} {tokens} {1000 * tokens / milliseconds:.1f} "
                    f"{low:.1f}-{high:.1f} {milliseconds / steps:.2f} {tokens / steps:.2f}")
            logged = [r["speculative"] for r in records if "speculative" in r]
            if logged:
                rounds = sum(s["rounds"] + s["fallback_steps"] for s in logged)
                lookups = sum(s["lookup"]["rounds"] for s in logged)
                accepted = sum(s["lookup"]["accepted_tokens"] for s in logged)
                lookup_s = sum(s["lookup"]["round_seconds"] for s in logged)
                draft_s = sum(s["mtp_round_seconds"] for s in logged)
                line += (f" {100 * lookups / max(rounds, 1):.1f} "
                         f"{accepted / max(lookups, 1):.2f} "
                         f"{1000 * lookup_s / max(lookups, 1):.2f} "
                         f"{1000 * draft_s / max(rounds - lookups, 1):.2f}")
            print(line)

    reference = args.greedy_reference
    outputs: dict[tuple[str, int], dict[str, Any]] = collections.defaultdict(dict)
    for label, runs in by_label.items():
        for record in runs["greedy"]:
            outputs[(record["workload"], record["index"])][label] = (
                record["content"], record["tool_calls"])
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
    runner.add_argument("--label", required=True)
    runner.add_argument("--output", required=True)
    report = commands.add_parser("summarize", help="pool results per label")
    report.add_argument("records", nargs="+")
    report.add_argument("--request-log", nargs=2, action="append", metavar=("LABEL", "JSONL"),
                        help="join a server request log recorded for LABEL")
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
