"""Unclamped perplexities printed by runs, tied to the windows they cover.

Dumps clamp log-probabilities 16 nats below the top token, so a perplexity recomputed from a dump
under-counts very surprising targets. Each run also prints the unclamped perplexity of its own
windows, and a comparison may use it only over the same windows. A source is one of:

    VALUE@N       a number over N windows (context not recorded)
    LOG           a llama-perplexity log: its "Final estimate: PPL" over the N chunks and the
                  n_ctx of its "calculating perplexity over N chunks, n_ctx=C" line
    LOG@N         the running perplexity after window N of that log, for its first N windows
    REPORT.json   a ninfer-perplexity --logits-out report: overall.perplexity over
                  logits_dump.chunks windows of logits_dump.context_tokens

A path that exists is always read as a path, even when it ends with @N.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import NamedTuple

from tools.kld.dump import DumpError, finite_number, positive_integer


class ExactPpl(NamedTuple):
    """An unclamped perplexity over `chunks` windows of `context` tokens (None: not recorded)."""

    value: float
    chunks: int
    context: int | None


_WINDOW_SUFFIX = re.compile(r"(.+)@(\d+)")
_LOG_RUN = re.compile(r"calculating perplexity over (\d+) chunks, n_ctx=(\d+)")
_LOG_FINAL = re.compile(r"Final estimate: PPL = (\S+)")


def _perplexity(text: str, label: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise DumpError(f"{label} is not a number: {text!r}") from None
    return finite_number(value, label, positive=True)


def _from_log(path: Path, window: int | None) -> ExactPpl:
    text = path.read_text(encoding="utf-8", errors="replace")
    runs = _LOG_RUN.findall(text)
    if not runs:
        raise DumpError(f"{path}: no 'calculating perplexity over N chunks, n_ctx=C' line "
                        "(llama-perplexity log expected)")
    chunks, context = int(runs[-1][0]), int(runs[-1][1])
    if window is None:
        values = _LOG_FINAL.findall(text)
        if not values:
            raise DumpError(f"{path}: no 'Final estimate: PPL' line")
        return ExactPpl(_perplexity(values[-1], f"{path}: final perplexity"), chunks, context)
    if window > chunks:
        raise DumpError(f"{path}: the run has {chunks} windows, not {window}")
    match = re.search(rf"\[{window}\]([^,\s]+)", text)
    if not match:
        raise DumpError(f"{path}: no running perplexity for window {window}")
    return ExactPpl(_perplexity(match.group(1), f"{path}: perplexity after window {window}"),
                    window, context)


def _from_report(path: Path) -> ExactPpl:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DumpError(f"{path}: not valid JSON ({error})") from None
    execution = report.get("execution") if isinstance(report, dict) else None
    if not isinstance(execution, dict) or execution.get("window_plan") != "kld-chunks":
        raise DumpError(f"{path}: not a ninfer-perplexity --logits-out report")
    overall, dump = report.get("overall"), report.get("logits_dump")
    if not isinstance(overall, dict) or not isinstance(dump, dict):
        raise DumpError(f"{path}: the report lacks overall or logits_dump")
    return ExactPpl(
        finite_number(overall.get("perplexity"), f"{path}: overall.perplexity", positive=True),
        positive_integer(dump.get("chunks"), f"{path}: logits_dump.chunks"),
        positive_integer(dump.get("context_tokens"), f"{path}: logits_dump.context_tokens"))


def read_exact_ppl(source: str) -> ExactPpl:
    """Read an exact perplexity source (see the module documentation)."""
    match = _WINDOW_SUFFIX.fullmatch(source)
    if match and not Path(source).exists():
        base, window = match.group(1), positive_integer(int(match.group(2)), f"{source}: N")
    else:
        base, window = source, None
    if not Path(base).exists():
        try:
            value = float(base)
        except ValueError:
            pass
        else:
            if window is None:
                raise DumpError(f"{source}: a bare number does not record the windows it covers; "
                                "use VALUE@N")
            return ExactPpl(finite_number(value, f"exact perplexity {source}", positive=True),
                            window, None)
    path = Path(base)
    if path.suffix == ".json":
        if window is not None:
            raise DumpError(f"{source}: @N applies to numbers and llama-perplexity logs only")
        return _from_report(path)
    return _from_log(path, window)


def check_exact_ppl(name: str, exact: ExactPpl, chunks: int, context: int) -> None:
    """Refuse an exact perplexity computed over other windows than the comparison."""
    if exact.chunks != chunks:
        raise DumpError(f"the exact perplexity of {name} covers {exact.chunks} windows but the "
                        f"comparison uses {chunks}; use the perplexity of the same windows "
                        "(LOG@N for a prefix of a llama-perplexity run)")
    if exact.context is not None and exact.context != context:
        raise DumpError(f"the exact perplexity of {name} was computed with a context of "
                        f"{exact.context} tokens, the comparison uses {context}")
