from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "qiffusion.cli", *args],
        text=True,
        capture_output=True,
        check=False,
    )


def test_track_reports_feed_shared_status_gate(tmp_path: Path) -> None:
    qwen = tmp_path / "qwen.json"
    diffusion = tmp_path / "diffusion.json"

    qwen_result = run_cli("qwen-status", "--out", str(qwen))
    diffusion_result = run_cli("backend-status", "--backend", "diffusion", "--out", str(diffusion))
    qwen_status = run_cli("status", "--report", str(qwen))
    diffusion_status = run_cli("status", "--report", str(diffusion))

    assert qwen_result.returncode == 0
    assert diffusion_result.returncode == 0
    assert qwen_status.returncode == 0
    assert diffusion_status.returncode == 0
    assert json.loads(qwen.read_text(encoding="utf-8"))["backend"] == "qwen_bridge"
    assert json.loads(diffusion.read_text(encoding="utf-8"))["backend"] == "diffusion"
    assert json.loads(qwen_status.stdout)["status"] == "continue"
    assert json.loads(diffusion_status.stdout)["status"] == "continue"


def test_status_rejects_malformed_json_report(tmp_path: Path) -> None:
    report = tmp_path / "bad.json"
    report.write_text("[]", encoding="utf-8")

    result = run_cli("status", "--report", str(report))

    assert result.returncode != 0
    assert "expected JSON object" in result.stderr

