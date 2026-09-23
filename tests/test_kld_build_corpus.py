from __future__ import annotations

import json
import re

import pytest

from tools.kld import build_corpus


def _tokenize(text: str) -> list[int]:
    # Words and whitespace runs are separate tokens, like the Qwen pre-tokenizer on plain text.
    return [hash(piece) % 1000 for piece in re.findall(r"\S+|\s+", text)]


def _manifest(tmp_path, sources: list[dict], rounds: int = 1) -> object:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"corpus_id": "test", "context": 8, "chunks": 4,
                                    "rounds": rounds, "slack_tokens": 2, "sources": sources}))
    return manifest


def test_builds_segments_on_line_boundaries(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("\n\nalpha beta gamma\ndelta epsilon zeta\neta theta\n")
    (tmp_path / "b.txt").write_text("one two\nthree four\nfive six\n")
    (tmp_path / "c.txt").write_text("\n".join(f"line {i} of code" for i in range(40)) + "\n")
    manifest = _manifest(tmp_path, [
        {"id": "a", "domain": "prose", "path": "a.txt", "tokens": 8},
        {"id": "b", "domain": "list", "path": "b.txt", "tokens": 5},
        {"id": "c", "domain": "code", "path": "c.txt", "tokens": "rest"},
    ])
    corpus, tokens, metadata = build_corpus.build(manifest, _tokenize)

    # Leading whitespace is dropped and each source is cut at the first line boundary that
    # reaches its budget.
    assert corpus.startswith("alpha beta gamma\ndelta epsilon zeta\n\none two\nthree four\n\nline 0")
    assert not corpus.endswith("\n")
    assert len(tokens) >= 32 and tokens == _tokenize(corpus)
    segments = metadata["segments"]
    assert [segment["id"] for segment in segments] == ["a", "b", "c"]
    assert segments[0]["token_begin"] == 0
    assert all(left["token_end"] == right["token_begin"]
               for left, right in zip(segments, segments[1:]))
    assert segments[-1]["token_end"] == len(tokens)
    assert metadata["evaluated_tokens"]["count"] == 32
    assert metadata["scored_positions"] == 4 * 3


def test_rejects_special_token_markers(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("hello <|im_start|> world\n" * 20)
    manifest = _manifest(tmp_path, [{"id": "a", "domain": "x", "path": "a.txt", "tokens": "rest"}])
    with pytest.raises(ValueError, match="special-token marker"):
        build_corpus.build(manifest, _tokenize)


def test_rejects_a_source_shorter_than_its_budget(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("too short\n")
    (tmp_path / "b.txt").write_text("word " * 100)
    manifest = _manifest(tmp_path, [
        {"id": "a", "domain": "x", "path": "a.txt", "tokens": 50},
        {"id": "b", "domain": "y", "path": "b.txt", "tokens": "rest"},
    ])
    with pytest.raises(ValueError, match="fewer than 50 tokens"):
        build_corpus.build(manifest, _tokenize)


def test_rounds_interleave_every_source(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("".join(f"alpha {i} beta\n" for i in range(20)))
    (tmp_path / "b.txt").write_text("".join(f"one {i} two\n" for i in range(20)))
    (tmp_path / "c.txt").write_text("".join(f"code {i} line\n" for i in range(40)))
    manifest = _manifest(tmp_path, [
        {"id": "a", "domain": "prose", "path": "a.txt", "tokens": 8},
        {"id": "b", "domain": "list", "path": "b.txt", "tokens": 8},
        {"id": "c", "domain": "code", "path": "c.txt", "tokens": "rest"},
    ], rounds=2)
    corpus, tokens, metadata = build_corpus.build(manifest, _tokenize)
    segments = metadata["segments"]
    assert [(s["id"], s["round"]) for s in segments] == [
        ("a", 0), ("b", 0), ("c", 0), ("a", 1), ("b", 1), ("c", 1)]
    # Each round continues its source where the previous round stopped.
    assert corpus.startswith("alpha 0 beta\n\none 0 two\n\ncode 0 line\n\nalpha 1 beta\n\n")
    # The first round fills the first half of the evaluated windows.
    assert segments[2]["token_end"] >= 16 and tokens == _tokenize(corpus)
