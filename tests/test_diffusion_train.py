from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

pytest.importorskip("torch")

from qiffusion.cli import main


def test_diffusion_train_cli_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny.pt"
    report_path = tmp_path / "train.json"

    exit_code = main(
        [
            "diffusion-train",
            "--steps",
            "2",
            "--seed",
            "1",
            "--max-examples",
            "8",
            "--out",
            str(checkpoint),
            "--report-out",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert checkpoint.is_file()
    assert report["status"] == "trained"
    assert report["checkpoint_path"] == str(checkpoint)
    assert report["steps"] == 2
    assert math.isfinite(report["initial_loss"])
    assert math.isfinite(report["final_loss"])
    assert report["coding_capability_claim"] is False


def test_diffusion_train_cli_rejects_zero_steps(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        main(
            [
                "diffusion-train",
                "--steps",
                "0",
                "--out",
                str(tmp_path / "tiny.pt"),
                "--report-out",
                str(tmp_path / "train.json"),
            ]
        )

    assert raised.value.code == 2
