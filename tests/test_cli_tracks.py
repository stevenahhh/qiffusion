from __future__ import annotations

import json
from pathlib import Path

from qiffusion.cli import main


def test_qwen_status_writes_prerequisite_report_when_no_engine(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("QIFFUSION_HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("QIFFUSION_DISABLE_HF", "1")
    monkeypatch.setenv("QIFFUSION_DISABLE_OLLAMA", "1")
    monkeypatch.setenv("QIFFUSION_DISABLE_GGUF", "1")
    output = tmp_path / "qwen.json"

    exit_code = main(["qwen-status", "--out", str(output)])

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    stdout = json.loads(capsys.readouterr().out)
    assert report == stdout
    assert report["backend"] == "qwen_bridge"
    assert report["model_id"] == "Qwen/Qwen3.5-4B"
    assert report["status"] == "prerequisite_missing"
    assert report["coding_capability_claim"] is False


def test_backend_status_writes_diffusion_scaffold_report(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "diffusion.json"

    exit_code = main(["backend-status", "--backend", "diffusion", "--out", str(output)])

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    stdout = json.loads(capsys.readouterr().out)
    assert report == stdout
    assert report["backend"] == "diffusion"
    assert report["status"] == "scaffold_ready"
    assert report["coding_capability_claim"] is False
    assert report["candidate_source"] == "none"
