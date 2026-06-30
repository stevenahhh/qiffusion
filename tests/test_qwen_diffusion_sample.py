from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from torch import Tensor, nn

from qiffusion.diffusion_config import BYTE_TOKEN_OFFSET, MASK_TOKEN_ID
from qiffusion.qwen_diffusion_model import QwenDenoiserConfig, TinyQwenTokenDenoiser
from qiffusion.qwen_diffusion_sample import (
    QwenMaskSampleConfig,
    QwenSamplerSettings,
    SamplerValidationError,
    sample_qwen_tokens,
    write_sample_evidence,
    write_sampler_failure_probe,
)


class FixedLogitDenoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = QwenDenoiserConfig.tiny(vocab_size=260, max_length=8)

    def forward(self, input_ids: Tensor) -> Tensor:
        batch_size, sequence_length = input_ids.shape
        logits = torch.full((batch_size, sequence_length, self.config.vocab_size), -20.0)
        logits[:, :, BYTE_TOKEN_OFFSET + ord("a")] = 1.0
        logits[:, :, BYTE_TOKEN_OFFSET + ord("b")] = 2.0
        logits[:, :, BYTE_TOKEN_OFFSET + ord("c")] = 3.0
        return logits


class MaskPreferredDenoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = QwenDenoiserConfig.tiny(vocab_size=260, max_length=8)

    def forward(self, input_ids: Tensor) -> Tensor:
        batch_size, sequence_length = input_ids.shape
        logits = torch.full((batch_size, sequence_length, self.config.vocab_size), -20.0)
        logits[:, :, MASK_TOKEN_ID] = 100.0
        logits[:, :, BYTE_TOKEN_OFFSET + ord("z")] = 1.0
        return logits


class ExtremeLogitDenoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = QwenDenoiserConfig.tiny(vocab_size=260, max_length=8)

    def forward(self, input_ids: Tensor) -> Tensor:
        batch_size, sequence_length = input_ids.shape
        logits = torch.full((batch_size, sequence_length, self.config.vocab_size), -torch.inf)
        logits[:, :, BYTE_TOKEN_OFFSET + ord("x")] = torch.inf
        logits[:, :, BYTE_TOKEN_OFFSET + ord("y")] = torch.inf
        return logits


class AllInvalidLogitDenoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = QwenDenoiserConfig.tiny(vocab_size=260, max_length=8)

    def forward(self, input_ids: Tensor) -> Tensor:
        batch_size, sequence_length = input_ids.shape
        return torch.full((batch_size, sequence_length, self.config.vocab_size), -torch.inf)


def test_qwen_mask_sampler_is_deterministic_with_seed() -> None:
    torch.manual_seed(1)
    model = TinyQwenTokenDenoiser(QwenDenoiserConfig.tiny(vocab_size=260, max_length=8))
    settings = QwenSamplerSettings(steps=4, seed=11, temperature=0.8, top_k=8)
    config = QwenMaskSampleConfig(prompt="def", settings=settings)

    first = sample_qwen_tokens(model, config)
    second = sample_qwen_tokens(model, config)

    assert first == second
    assert first["prompt"] == "def"
    assert first["fallback_used"] is False
    assert first["early_stop"]["reason"] == "max_steps"


def test_qwen_mask_sampler_never_selects_mask_as_replacement_token() -> None:
    model = MaskPreferredDenoiser()
    settings = QwenSamplerSettings(steps=3, seed=5, temperature=1.0, top_k=3)

    report = sample_qwen_tokens(model, QwenMaskSampleConfig(prompt="", settings=settings))

    assert MASK_TOKEN_ID not in report["generated_token_ids"]
    assert all(entry["token_id"] != MASK_TOKEN_ID for entry in report["history"])
    assert all(MASK_TOKEN_ID not in entry["candidate_token_ids"] for entry in report["history"])
    assert report["early_stop"]["reason"] == "max_steps"


@pytest.mark.parametrize("top_k", [2, 4, 7, 99])
def test_qwen_mask_sampler_excludes_mask_from_all_invalid_top_k_candidates(top_k: int) -> None:
    model = AllInvalidLogitDenoiser()
    settings = QwenSamplerSettings(steps=1, seed=3, temperature=1.0, top_k=top_k)

    report = sample_qwen_tokens(model, QwenMaskSampleConfig(prompt="", settings=settings))

    entry = report["history"][0]
    assert entry["token_id"] != MASK_TOKEN_ID
    assert MASK_TOKEN_ID not in entry["candidate_token_ids"]
    assert math.isfinite(entry["confidence"])
    assert math.isfinite(entry["entropy"])


def test_qwen_mask_sampler_records_finite_metadata_for_extreme_logits() -> None:
    model = ExtremeLogitDenoiser()
    settings = QwenSamplerSettings(steps=1, seed=5, temperature=1.0e-9, top_k=4)

    report = sample_qwen_tokens(model, QwenMaskSampleConfig(prompt="", settings=settings))

    entry = report["history"][0]
    assert math.isfinite(entry["confidence"])
    assert math.isfinite(entry["entropy"])


def test_qwen_mask_sampler_records_history_with_confidence_and_entropy() -> None:
    model = FixedLogitDenoiser()
    settings = QwenSamplerSettings(steps=3, seed=7, temperature=1.0, top_k=2)

    report = sample_qwen_tokens(model, QwenMaskSampleConfig(prompt="", settings=settings))

    assert len(report["history"]) == 3
    assert all(entry["token_id_before"] == MASK_TOKEN_ID for entry in report["history"])
    assert all("confidence" in entry for entry in report["history"])
    assert all("entropy" in entry for entry in report["history"])
    assert report["history"][0]["top_k"] == 2


def test_top_k_temperature_path_locks_candidate_filtering() -> None:
    model = FixedLogitDenoiser()
    settings = QwenSamplerSettings(steps=1, seed=3, temperature=0.5, top_k=1)

    report = sample_qwen_tokens(model, QwenMaskSampleConfig(prompt="", settings=settings))

    assert report["generated_token_ids"] == [BYTE_TOKEN_OFFSET + ord("c")]
    assert report["history"][0]["candidate_token_ids"] == [BYTE_TOKEN_OFFSET + ord("c")]
    assert report["history"][0]["confidence"] == 1.0
    assert report["history"][0]["temperature"] == 0.5


def test_qwen_sampler_uses_no_external_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_fallback(*_args, **_kwargs):
        pytest.fail("external Qwen/Ollama fallback was called")

    monkeypatch.setattr("qiffusion.qwen_bridge.ollama_has_qwen", fail_fallback)
    monkeypatch.setattr("qiffusion.qwen_eval.qwen_eval", fail_fallback)
    monkeypatch.setattr("qiffusion.qwen_ollama.run_ollama_fixture", fail_fallback)
    model = FixedLogitDenoiser()
    settings = QwenSamplerSettings(steps=2, seed=11, temperature=1.0, top_k=3)

    report = sample_qwen_tokens(model, QwenMaskSampleConfig(prompt="", settings=settings))

    assert report["fallback_used"] is False
    assert len(report["generated_token_ids"]) == 2


def test_qwen_sampler_rejects_invalid_settings_at_boundary() -> None:
    with pytest.raises(SamplerValidationError) as raised:
        QwenSamplerSettings(steps=1, seed=1, temperature=1.0, top_k=1, algorithm="unsupported")

    assert "algorithm" in str(raised.value)


def test_sample_evidence_records_history_and_no_fallback(tmp_path: Path) -> None:
    artifact = tmp_path / "sample.json"

    write_sample_evidence(artifact, prompt="def add", steps=4, seed=11)

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert len(payload["history"]) == 4
    assert payload["fallback_used"] is False
    assert "confidence" in payload["history"][0]


def test_sampler_failure_probe_records_rejection(tmp_path: Path) -> None:
    artifact = tmp_path / "failure.json"

    write_sampler_failure_probe(artifact, algorithm="unsupported")

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["status"] == "rejected"
    assert payload["algorithm"] == "unsupported"
    assert payload["fallback_used"] is False
