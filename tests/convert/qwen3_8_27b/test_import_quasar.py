"""Tests of the QUASAR import wrapper: CLI surface, download skip, and the command it builds."""
import pytest
from pathlib import Path
from tools.convert.qwen3_8_27b import import_quasar


def test_default_repo_is_quasar():
    assert import_quasar.QUASAR_HF_REPO == "QUASAR/Qwen3.8-27B-NVFP4-QAT"


def test_no_download_errors_when_checkpoint_missing(tmp_path, monkeypatch):
    base = tmp_path / "base.ninfer"
    base.write_bytes(b"fake")
    monkeypatch.setattr("sys.argv", [
        "import_quasar", "--base", str(base),
        "--out", str(tmp_path / "out.ninfer"), "--no-download",
    ])
    with pytest.raises(SystemExit):
        import_quasar.main()


def test_runs_rewrite_when_checkpoint_present(tmp_path, monkeypatch):
    checkpoint = tmp_path / "QUASAR-Qwen3.8-27B-NVFP4-QAT"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    base = tmp_path / "base.ninfer"
    base.write_bytes(b"fake")
    calls = []
    monkeypatch.setattr(
        import_quasar.subprocess, "run",
        lambda command, **kw: calls.append(command) or type("R", (), {"returncode": 0})())
    monkeypatch.setattr("sys.argv", [
        "import_quasar", "--base", str(base),
        "--checkpoint", str(checkpoint),
        "--out", str(tmp_path / "out.ninfer"),
    ])
    rc = import_quasar.main()
    assert rc == 0
    assert len(calls) == 1
    command = calls[0]
    assert "rewrite_nvfp4" in " ".join(command)
    assert "--reference" in command
    assert str(checkpoint) in command
