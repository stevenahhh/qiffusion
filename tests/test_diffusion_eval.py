from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from qiffusion.cli import main
from qiffusion.diffusion_model import TinyDiffusionConfig, TinyDiffusionLM, save_checkpoint


def write_checkpoint(path: Path) -> None:
    torch.manual_seed(1)
    model = TinyDiffusionLM(TinyDiffusionConfig(vocab_size=260, dim=32, layers=1))
    save_checkpoint(path, model, model.config)


def test_diffusion_eval_cli_consumes_checkpoint_without_claiming(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny.pt"
    output = tmp_path / "eval.json"
    write_checkpoint(checkpoint)

    exit_code = main(["diffusion-eval", "--checkpoint", str(checkpoint), "--runs", "1", "--out", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["backend"] == "diffusion"
    assert report["checkpoint_path"] == str(checkpoint)
    assert report["fixtures_status"] == "pass"
    assert report["code_smoke_status"] == "fail"
    assert report["coding_capability_claim"] is False


def test_status_gate_continues_for_tiny_diffusion_eval(tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "tiny.pt"
    output = tmp_path / "eval.json"
    write_checkpoint(checkpoint)
    main(["diffusion-eval", "--checkpoint", str(checkpoint), "--runs", "1", "--out", str(output)])

    exit_code = main(["status", "--report", str(output)])

    decision = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert exit_code == 0
    assert decision["status"] == "continue"
