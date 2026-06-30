from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from qiffusion.cli import main
from qiffusion.qwen_bridge import FixtureResult
from qiffusion.qwen_diffusion_config import JsonObject
from qiffusion.qwen_diffusion_eval import (
    QwenDiffusionEvalConfig,
    ReportValidationError,
    eval_qwen_diffusion_checkpoint,
    validate_report,
)
from qiffusion.qwen_diffusion_model import (
    QwenDenoiserConfig,
    TinyQwenTokenDenoiser,
    save_qwen_denoiser_checkpoint,
)


def write_checkpoint(path: Path) -> None:
    torch.manual_seed(1)
    config = QwenDenoiserConfig.tiny(vocab_size=260, max_length=16)
    save_qwen_denoiser_checkpoint(path, TinyQwenTokenDenoiser(config))


def test_eval_separates_local_code_chat_tool_and_software_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "qwen-tiny.pt"
    sample_out = tmp_path / "sample.json"
    write_checkpoint(checkpoint)

    def passing_smoke(_code: str, _task) -> tuple[bool, str, list[FixtureResult]]:
        return True, "passed", []

    monkeypatch.setattr("qiffusion.qwen_diffusion_eval.run_task_smoke", passing_smoke)

    report = eval_qwen_diffusion_checkpoint(
        QwenDiffusionEvalConfig(checkpoint_path=checkpoint, sample_out=sample_out, runs=1),
    )

    assert report["buckets"]["local_code_smoke"]["status"] == "pass"
    assert report["buckets"]["local_code_smoke"]["capability_claim"] is True
    assert report["buckets"]["chat"]["status"] == "blocked"
    assert report["buckets"]["chat"]["capability_claim"] is False
    assert report["buckets"]["tool_agent"]["status"] == "blocked"
    assert report["buckets"]["software_engineering"]["status"] == "blocked"
    assert report["coding_capability_claim"] is False
    assert report["release_capability_claim"] is False


def test_eval_uses_no_external_qwen_or_ollama_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "qwen-tiny.pt"
    sample_out = tmp_path / "sample.json"
    write_checkpoint(checkpoint)

    def fail_fallback(*_args, **_kwargs):
        pytest.fail("external Qwen/Ollama fallback was called")

    monkeypatch.setattr("qiffusion.qwen_bridge.ollama_has_qwen", fail_fallback)
    monkeypatch.setattr("qiffusion.qwen_eval.qwen_eval", fail_fallback)
    monkeypatch.setattr("qiffusion.qwen_ollama.run_ollama_fixture", fail_fallback)

    report = eval_qwen_diffusion_checkpoint(
        QwenDiffusionEvalConfig(checkpoint_path=checkpoint, sample_out=sample_out, runs=1),
    )

    assert report["fallback_used"] is False
    assert report["samples"][0]["fallback_used"] is False


def test_blocked_non_code_buckets_do_not_imply_release_or_coding_capability(tmp_path: Path) -> None:
    checkpoint = tmp_path / "qwen-tiny.pt"
    sample_out = tmp_path / "sample.json"
    write_checkpoint(checkpoint)

    report = eval_qwen_diffusion_checkpoint(
        QwenDiffusionEvalConfig(checkpoint_path=checkpoint, sample_out=sample_out, runs=1),
    )

    assert report["buckets"]["chat"]["status"] == "blocked"
    assert report["buckets"]["tool_agent"]["status"] == "blocked"
    assert report["buckets"]["software_engineering"]["status"] == "blocked"
    assert report["coding_capability_claim"] is False
    assert report["release_capability_claim"] is False


def test_validate_report_rejects_overclaim_from_local_smoke_only() -> None:
    report: JsonObject = {
        "backend": "qwen_token_diffusion",
        "stage": "eval",
        "fallback_used": False,
        "local_code_capability_claim": True,
        "coding_capability_claim": True,
        "chat_capability_claim": False,
        "tool_agent_capability_claim": False,
        "software_engineering_capability_claim": False,
        "release_capability_claim": False,
        "buckets": {
            "local_code_smoke": {"status": "pass", "capability_claim": True},
            "external_benchmark_readiness": {"status": "not_run", "capability_claim": False},
            "chat": {"status": "blocked", "capability_claim": False},
            "tool_agent": {"status": "blocked", "capability_claim": False},
            "software_engineering": {"status": "blocked", "capability_claim": False},
        },
    }

    with pytest.raises(ReportValidationError) as raised:
        validate_report(report)

    assert "coding_capability_claim" in str(raised.value)


def test_cli_validate_report_writes_overclaim_rejection(tmp_path: Path) -> None:
    report_path = tmp_path / "overclaim-input.json"
    output = tmp_path / "overclaim-failure.json"
    report_path.write_text(
        json.dumps(
            {
                "backend": "qwen_token_diffusion",
                "stage": "eval",
                "fallback_used": False,
                "local_code_capability_claim": True,
                "coding_capability_claim": False,
                "chat_capability_claim": True,
                "tool_agent_capability_claim": False,
                "software_engineering_capability_claim": False,
                "release_capability_claim": False,
                "buckets": {
                    "local_code_smoke": {"status": "pass", "capability_claim": True},
                    "external_benchmark_readiness": {"status": "not_run", "capability_claim": False},
                    "chat": {"status": "blocked", "capability_claim": False},
                    "tool_agent": {"status": "blocked", "capability_claim": False},
                    "software_engineering": {"status": "blocked", "capability_claim": False},
                },
            },
        ),
        encoding="utf-8",
    )

    exit_code = main(["qwen-diffusion-eval", "--validate-report", str(report_path), "--out", str(output)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert payload["status"] == "rejected"
    assert "chat_capability_claim" in payload["error"]
