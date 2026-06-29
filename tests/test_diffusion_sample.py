from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from qiffusion.cli import main
from qiffusion.diffusion_model import TinyDiffusionConfig, TinyDiffusionLM, save_checkpoint
from qiffusion.diffusion_sample import DiffusionSampleConfig, sample_from_checkpoint


def write_checkpoint(path: Path) -> None:
    torch.manual_seed(1)
    model = TinyDiffusionLM(TinyDiffusionConfig(vocab_size=260, dim=32, layers=1))
    save_checkpoint(path, model, model.config)


def test_sample_from_checkpoint_is_deterministic(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny.pt"
    write_checkpoint(checkpoint)
    config = DiffusionSampleConfig(checkpoint_path=checkpoint, prompt="def", steps=4, seed=1)

    first = sample_from_checkpoint(config)
    second = sample_from_checkpoint(config)

    assert first == second
    assert first["checkpoint_path"] == str(checkpoint)
    assert first["generated_text"].startswith("def")
    assert first["coding_capability_claim"] is False


def test_diffusion_sample_cli_writes_json(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny.pt"
    output = tmp_path / "sample.json"
    write_checkpoint(checkpoint)

    exit_code = main(
        [
            "diffusion-sample",
            "--checkpoint",
            str(checkpoint),
            "--prompt",
            "def",
            "--steps",
            "4",
            "--seed",
            "1",
            "--out",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["generated_text"].startswith("def")
    assert report["checkpoint_path"] == str(checkpoint)
