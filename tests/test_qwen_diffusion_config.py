from __future__ import annotations

from pathlib import Path

import pytest

from qiffusion.qwen_diffusion_config import (
    BenchmarkGateEntry,
    ConfigValidationError,
    ContaminationBlockedError,
    QwenDiffusionConfig,
    ResourceProbe,
    config_from_json,
    default_config,
    write_compatibility_contract,
    write_contamination_probe,
    write_default_config,
)


def test_qwen_diffusion_config_round_trips_json(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"

    write_default_config(config_path, base_checkpoint_id="Qwen/Qwen3.5-4B")
    payload = default_config(base_checkpoint_id="Qwen/Qwen3.5-4B").to_json()
    restored = config_from_json(payload)

    assert isinstance(restored, QwenDiffusionConfig)
    assert restored == default_config(base_checkpoint_id="Qwen/Qwen3.5-4B")
    written = config_path.read_text(encoding="utf-8")
    assert '"base_checkpoint_id": "Qwen/Qwen3.5-4B"' in written
    assert '"humaneval"' in written
    assert '"usage": "eval_only"' in written
    assert '"contamination_status": "blocked"' in written


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("attention_mode", "global"),
        ("objective", "ar_ce"),
        ("sampler_algorithm", "ancestral"),
    ],
)
def test_qwen_diffusion_config_rejects_invalid_modes(field_name: str, bad_value: str) -> None:
    payload = default_config(base_checkpoint_id="Qwen/Qwen3.5-4B").to_json()
    payload[field_name] = bad_value

    with pytest.raises(ConfigValidationError):
        _ = config_from_json(payload)


@pytest.mark.parametrize("field_name", ["block_size", "seed"])
def test_qwen_diffusion_config_rejects_bool_integer_fields(field_name: str) -> None:
    payload = default_config(base_checkpoint_id="Qwen/Qwen3.5-4B").to_json()
    payload[field_name] = True

    with pytest.raises(ConfigValidationError):
        _ = config_from_json(payload)


def test_qwen_diffusion_config_accepts_integer_one_fields() -> None:
    payload = default_config(base_checkpoint_id="Qwen/Qwen3.5-4B").to_json()
    payload["block_size"] = 1
    payload["seed"] = 1

    restored = config_from_json(payload)

    assert restored.block_size == 1
    assert restored.seed == 1


def test_qwen_diffusion_config_rejects_benchmark_training_contamination() -> None:
    config = default_config(base_checkpoint_id="Qwen/Qwen3.5-4B")

    with pytest.raises(ContaminationBlockedError):
        config.ensure_training_sources_allowed(("humaneval",))


def test_no_download_compatibility_contract_records_qwen_fields(tmp_path: Path) -> None:
    contract_path = tmp_path / "compatibility.json"

    write_compatibility_contract(contract_path, base_checkpoint_id="Qwen/Qwen3.5-4B")
    written = contract_path.read_text(encoding="utf-8")

    assert '"base_checkpoint_id": "Qwen/Qwen3.5-4B"' in written
    assert '"expected_checkpoint_family": "Qwen"' in written
    assert '"tokenizer_id": "Qwen/Qwen3.5-4B"' in written
    assert '"downloads_allowed": false' in written
    assert '"weights_downloaded": false' in written
    assert '"attention_mode"' in written
    assert '"status": "unknown"' in written


def test_resource_probe_accepts_expected_status_values() -> None:
    statuses = ("available", "missing", "unknown")
    for status in statuses:
        probe = ResourceProbe(status=status, detail=f"{status} in local probe")

        assert probe.to_json()["status"] == status


def test_resource_probe_rejects_unknown_status_value() -> None:
    payload = default_config(base_checkpoint_id="Qwen/Qwen3.5-4B").to_json()
    payload["resource_probe"] = {"status": "downloaded", "detail": "not a valid offline status"}

    with pytest.raises(ConfigValidationError):
        _ = config_from_json(payload)


def test_benchmark_gate_entry_rejects_invalid_direct_values() -> None:
    with pytest.raises(ConfigValidationError):
        _ = BenchmarkGateEntry("training", "dirty", "bad direct")


def test_benchmark_gate_entry_serializes_valid_direct_values() -> None:
    entry = BenchmarkGateEntry("train_allowed", "clean", "local source")

    assert entry.to_json() == {
        "usage": "train_allowed",
        "contamination_status": "clean",
        "reason": "local source",
    }


def test_contamination_probe_reports_benchmark_rejection(tmp_path: Path) -> None:
    report_path = tmp_path / "contamination.json"

    write_contamination_probe(report_path, training_sources=["humaneval"])
    written = report_path.read_text(encoding="utf-8")

    assert '"status": "blocked"' in written
    assert '"blocked_sources": [' in written
    assert '"humaneval"' in written
    assert '"data_usage": {' in written
    assert '"contamination_status": {' in written
    assert '"humaneval": "eval_only"' in written
    assert '"humaneval": "blocked"' in written
