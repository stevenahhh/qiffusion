from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

pytest.importorskip("torch")

from qiffusion.cli import main


def test_qwen_diffusion_train_cli_writes_checkpoint_lineage_and_non_claiming_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_fallback(*_args, **_kwargs):
        pytest.fail("external Qwen/Ollama fallback was called")

    monkeypatch.setattr("qiffusion.qwen_bridge.ollama_has_qwen", fail_fallback)
    monkeypatch.setattr("qiffusion.qwen_eval.qwen_eval", fail_fallback)
    monkeypatch.setattr("qiffusion.qwen_ollama.run_ollama_fixture", fail_fallback)
    manifest = _write_manifest(tmp_path, usage="train_allowed", contamination_status="clean")
    checkpoint = tmp_path / "qwen-tiny.pt"
    report_path = tmp_path / "train.json"

    exit_code = main(
        [
            "qwen-diffusion-train",
            "--manifest",
            str(manifest),
            "--tokenizer",
            "byte",
            "--steps",
            "2",
            "--seed",
            "11",
            "--checkpoint-out",
            str(checkpoint),
            "--report-out",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert checkpoint.is_file()
    assert report["status"] == "trained"
    assert report["data_manifest_id"] == "manifest-smoke"
    assert report["objective"] == "masked_ce"
    assert report["mask_schedule"] == "linear"
    assert math.isfinite(report["initial_loss"])
    assert math.isfinite(report["final_loss"])
    assert report["checkpoint_lineage"]["data_manifest_id"] == "manifest-smoke"
    assert report["checkpoint_lineage"]["parent_checkpoint_id"] is None
    assert report["fallback_used"] is False
    assert report["coding_capability_claim"] is False


def test_qwen_diffusion_train_cli_blocks_benchmark_manifest(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, usage="eval_only", contamination_status="blocked", source="HumanEval.jsonl")
    report_path = tmp_path / "blocked.json"

    exit_code = main(
        [
            "qwen-diffusion-train",
            "--manifest",
            str(manifest),
            "--tokenizer",
            "byte",
            "--steps",
            "1",
            "--checkpoint-out",
            str(tmp_path / "blocked.pt"),
            "--report-out",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert report["status"] == "blocked"
    assert report["blocked_sources"] == ["HumanEval.jsonl"]
    assert report["coding_capability_claim"] is False


def _write_manifest(
    tmp_path: Path,
    *,
    usage: str,
    contamination_status: str,
    source: str = "snippet.py",
) -> Path:
    source_path = tmp_path / source
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "ok",
                "manifest_id": "manifest-smoke",
                "root": str(tmp_path),
                "records": [
                    {
                        "source": source,
                        "source_kind": "local",
                        "name": "snippet",
                        "license": "MIT",
                        "split": "train",
                        "tokenizer": "byte",
                        "token_count": 32,
                        "dedup_hash": "abc123",
                        "usage": usage,
                        "contamination_status": contamination_status,
                        "privacy_policy_notes": "local smoke fixture",
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest
