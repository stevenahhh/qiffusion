from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias

import pytest

pytest.importorskip("torch")

from qiffusion.cli import main
from qiffusion.qwen_diffusion_config import write_default_config
from qiffusion.qwen_diffusion_loop import LoopLedgerEntry, QwenDiffusionLoopConfig, run_qwen_diffusion_loop

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


def test_qwen_diffusion_loop_runs_two_iterations_and_writes_complete_ledger(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    config = tmp_path / "config.json"
    ledger = tmp_path / "loop.jsonl"
    write_default_config(config, base_checkpoint_id="Qwen/Qwen3.5-4B")

    result = run_qwen_diffusion_loop(QwenDiffusionLoopConfig(manifest, config, ledger, max_iterations=2))

    entries = _ledger_entries(ledger)
    assert result["iterations"] == 2
    assert result["capability_claim"] is False
    assert len(entries) == 2
    for entry in entries:
        assert {"train", "sample", "eval", "diagnosis", "next_action", "cleanup"} <= set(entry)
        assert entry["cleanup"]["status"] == "clean"
        assert entry["train"]["fallback_used"] is False
        assert entry["sample"]["fallback_used"] is False
        assert entry["eval"]["fallback_used"] is False


def test_qwen_diffusion_loop_continues_after_failed_coding_gate(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    config = tmp_path / "config.json"
    ledger = tmp_path / "loop.jsonl"
    write_default_config(config, base_checkpoint_id="Qwen/Qwen3.5-4B")

    result = run_qwen_diffusion_loop(QwenDiffusionLoopConfig(manifest, config, ledger, max_iterations=2, force_eval_fail=True))

    entries = _ledger_entries(ledger)
    assert result["status"] == "max_iterations_reached"
    assert entries[0]["diagnosis"]["code_gate_status"] == "fail"
    assert entries[0]["next_action"]["action"] == "train_again"
    assert entries[0]["next_action"]["continue_loop"] is True
    assert entries[1]["next_action"]["action"] == "max_iterations_reached"
    assert entries[1]["next_action"]["capability_claim"] is False


def test_qwen_diffusion_loop_uses_no_external_fallbacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_fallback(*_args: JsonValue, **_kwargs: JsonValue) -> None:
        pytest.fail("external Qwen/Ollama fallback was called")

    monkeypatch.setattr("qiffusion.qwen_bridge.ollama_has_qwen", fail_fallback)
    monkeypatch.setattr("qiffusion.qwen_eval.qwen_eval", fail_fallback)
    monkeypatch.setattr("qiffusion.qwen_ollama.run_ollama_fixture", fail_fallback)
    manifest = _write_manifest(tmp_path)
    config = tmp_path / "config.json"
    ledger = tmp_path / "loop.jsonl"
    write_default_config(config, base_checkpoint_id="Qwen/Qwen3.5-4B")

    result = run_qwen_diffusion_loop(QwenDiffusionLoopConfig(manifest, config, ledger, max_iterations=1))

    entry = _ledger_entries(ledger)[0]
    assert result["capability_claim"] is False
    assert entry["train"]["fallback_used"] is False
    assert entry["sample"]["fallback_used"] is False
    assert entry["eval"]["fallback_used"] is False


def test_qwen_diffusion_loop_cli_writes_ledger_and_forced_failure_is_not_complete(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    config = tmp_path / "config.json"
    ledger = tmp_path / "loop-failure.jsonl"
    write_default_config(config, base_checkpoint_id="Qwen/Qwen3.5-4B")

    exit_code = main(
        [
            "qwen-diffusion-loop",
            "--manifest",
            str(manifest),
            "--config",
            str(config),
            "--force-eval-fail",
            "--max-iterations",
            "1",
            "--ledger-out",
            str(ledger),
        ],
    )

    entry = _ledger_entries(ledger)[0]
    assert exit_code == 0
    assert entry["next_action"]["action"] != "complete"
    assert entry["next_action"]["capability_claim"] is False


def _write_manifest(tmp_path: Path) -> Path:
    source = tmp_path / "snippet.py"
    source.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "ok",
                "manifest_id": "manifest-loop",
                "root": str(tmp_path),
                "records": [
                    {
                        "source": "snippet.py",
                        "source_kind": "local",
                        "name": "snippet",
                        "license": "MIT",
                        "split": "train",
                        "tokenizer": "byte",
                        "token_count": 32,
                        "dedup_hash": "abc123",
                        "usage": "train_allowed",
                        "contamination_status": "clean",
                        "privacy_policy_notes": "local smoke fixture",
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _ledger_entries(path: Path) -> list[LoopLedgerEntry]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
