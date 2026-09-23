#!/usr/bin/env python3
"""Build the single-stream text corpus of a KLD comparison and its token segment map.

llama-perplexity evaluates one text file as one token stream, so every compared run (llama.cpp and
ninfer-perplexity --logits-out) scores the same file. This tool concatenates the corpus sources of
a manifest, cutting each source at a line boundary once it reaches its token budget, and writes:

    corpus.txt      UTF-8 text, no trailing newline (llama-perplexity strips one)
    tokens.bin      the token stream of corpus.txt, int32 little-endian
    segments.json   token ranges of every source in the evaluated stream, the evaluated-token hash,
                    and the tokenizer that produced them

Token counts come from llama.cpp's `llama-tokenize` with the reference GGUF, using the options that
match llama-perplexity: no escape processing, no parsing of control tokens, no BOS for the Qwen
vocabulary. ninfer-perplexity --logits-reference and `kld.py compare` then verify that every run
evaluated exactly these tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

SEPARATOR = "\n\n"
# Chat-template and tool markers tokenize as special tokens in the NInfer tokenizer but as plain
# text in llama-perplexity (control tokens are not parsed there), so they must not occur.
SPECIAL_MARKER = re.compile(r"<\|[^<>|\s]{1,48}\|>|</?(think|tool_call|tool_response)>")

Tokenizer = Callable[[str], list[int]]


def llama_tokenizer(binary: Path, model: Path, scratch: Path) -> Tokenizer:
    def tokenize(text: str) -> list[int]:
        scratch.write_bytes(text.encode("utf-8"))
        completed = subprocess.run(
            [str(binary), "-m", str(model), "-f", str(scratch), "--ids", "--log-disable",
             "--no-escape", "--no-parse-special", "--no-bos"],
            check=True, capture_output=True, text=True)
        line = completed.stdout.strip().splitlines()[-1]
        return json.loads(line)

    return tokenize


def line_boundaries(text: str) -> list[int]:
    """Character offsets just after every newline, plus the end of the text."""
    offsets = [match.end() for match in re.finditer("\n", text)]
    if not offsets or offsets[-1] != len(text):
        offsets.append(len(text))
    return offsets


def cut_at_budget(text: str, budget: int, tokenize: Tokenizer) -> str:
    """Shortest line-boundary prefix (trailing whitespace removed) with >= budget tokens."""
    candidates = line_boundaries(text)
    if len(tokenize(text.rstrip())) < budget:
        raise ValueError(f"source has fewer than {budget} tokens")
    low, high = 0, len(candidates) - 1
    while low < high:
        middle = (low + high) // 2
        if len(tokenize(text[:candidates[middle]].rstrip())) >= budget:
            high = middle
        else:
            low = middle + 1
    return text[:candidates[low]].rstrip()


def check_text(text: str, label: str) -> None:
    if "\x00" in text:
        raise ValueError(f"{label}: NUL bytes are not allowed")
    marker = SPECIAL_MARKER.search(text)
    if marker:
        raise ValueError(f"{label}: contains the special-token marker {marker.group(0)!r}")


def build(manifest_path: Path, tokenize: Tokenizer) -> tuple[str, list[int], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    context = int(manifest["context"])
    chunks = int(manifest["chunks"])
    evaluated = context * chunks
    sources = manifest["sources"]
    if sources[-1]["tokens"] != "rest" or any(s["tokens"] == "rest" for s in sources[:-1]):
        raise ValueError("only the last source may take the remaining token budget")

    parts: list[tuple[dict[str, Any], str]] = []
    assigned = 0
    for index, source in enumerate(sources):
        path = (manifest_path.parent / source["path"]).resolve()
        # Leading whitespace would merge with the separator into different tokens.
        text = path.read_text(encoding="utf-8").lstrip()
        check_text(text, str(path))
        last = index == len(sources) - 1
        # The last source fills the evaluated windows; the slack keeps the final window complete
        # even if a boundary token merges differently in the joined stream.
        budget = evaluated - assigned + int(manifest.get("slack_tokens", 64)) if last \
            else int(source["tokens"])
        if budget <= 0:
            raise ValueError("the fixed budgets already exceed chunks*context tokens")
        segment = cut_at_budget(text, budget, tokenize)
        parts.append((source, segment))
        assigned += len(tokenize(segment + ("" if last else SEPARATOR)))

    corpus = SEPARATOR.join(segment for _, segment in parts)
    tokens = tokenize(corpus)
    if len(tokens) < evaluated:
        raise ValueError(f"corpus has {len(tokens)} tokens, fewer than chunks*context={evaluated}")

    segments = []
    offset = 0
    for index, (source, segment) in enumerate(parts):
        end = offset + len(segment) + (len(SEPARATOR) if index < len(parts) - 1 else 0)
        prefix = tokenize(corpus[:end])
        if prefix != tokens[:len(prefix)]:
            raise ValueError(f"segment {source['id']} does not end on a token boundary")
        begin = segments[-1]["token_end"] if segments else 0
        segments.append({"id": source["id"], "domain": source["domain"], "source": source["path"],
                         "characters": len(segment), "token_begin": begin,
                         "token_end": len(prefix)})
        offset = end

    evaluated_bytes = b"".join(token.to_bytes(4, "little", signed=True)
                               for token in tokens[:evaluated])
    metadata = {
        "corpus_id": manifest["corpus_id"],
        "context": context,
        "chunks": chunks,
        "scored_positions": chunks * (context - 1 - context // 2),
        "token_count": len(tokens),
        "evaluated_tokens": {"context": context, "chunks": chunks, "count": evaluated,
                             "sha256": hashlib.sha256(evaluated_bytes).hexdigest()},
        "corpus_sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
        "tokens_file": "tokens.bin",
        "segments": segments,
    }
    return corpus, tokens, metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parents[2]
                        / "eval/corpora/kld-v1/manifest.json")
    parser.add_argument("--llama-tokenize", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path, help="reference GGUF (vocab only)")
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    scratch = args.out / ".tokenize-input.txt"
    try:
        corpus, tokens, metadata = build(args.manifest, llama_tokenizer(args.llama_tokenize, args.model,
                                                                 scratch))
    finally:
        scratch.unlink(missing_ok=True)
    metadata["tokenizer"] = {"llama_tokenize": str(args.llama_tokenize.resolve()),
                             "model": str(args.model.resolve())}
    (args.out / "corpus.txt").write_bytes(corpus.encode("utf-8"))
    (args.out / "tokens.bin").write_bytes(b"".join(token.to_bytes(4, "little", signed=True)
                                                   for token in tokens))
    (args.out / "segments.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                            encoding="utf-8")
    print(f"{args.out / 'corpus.txt'}: {metadata['token_count']} tokens, "
          f"{metadata['chunks']} chunks of {metadata['context']}, "
          f"{metadata['scored_positions']} scored positions")
    for segment in metadata["segments"]:
        print(f"  {segment['id']:<14}{segment['domain']:<20}tokens "
              f"[{segment['token_begin']}, {segment['token_end']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
