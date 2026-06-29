from __future__ import annotations

import json
from pathlib import Path

from qiffusion.cli import main


def test_diffusion_eval_stub_writes_non_claiming_report(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "diffusion-eval.json"

    exit_code = main(["diffusion-eval", "--report-out", str(output)])

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    stdout = json.loads(capsys.readouterr().out)
    assert report == stdout
    assert report["backend"] == "diffusion"
    assert report["stage"] == "eval"
    assert report["status"] == "stub"
    assert report["fixtures_status"] == "not_run"
    assert report["code_smoke_status"] == "not_run"
    assert report["candidate_source"] == "none"
    assert report["coding_capability_claim"] is False


def test_diffusion_train_and_sample_stubs_remain_non_claiming(
    tmp_path: Path,
) -> None:
    train_report = tmp_path / "train.json"
    sample_report = tmp_path / "sample.json"

    train_exit = main(["diffusion-train", "--report-out", str(train_report)])
    sample_exit = main(["diffusion-sample", "--report-out", str(sample_report)])

    assert train_exit == 0
    assert sample_exit == 0
    for path in (train_report, sample_report):
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["backend"] == "diffusion"
        assert report["coding_capability_claim"] is False
        assert report["fixtures_status"] == "not_run"
        assert report["code_smoke_status"] == "not_run"

