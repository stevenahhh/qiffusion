from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from qiffusion.qwen_diffusion_config import default_config
from qiffusion.qwen_diffusion_model import (
    QwenDenoiserCheckpointError,
    QwenDenoiserConfig,
    TinyQwenTokenDenoiser,
    load_qwen_denoiser_checkpoint,
    save_qwen_denoiser_checkpoint,
    write_mismatch_checkpoint_probe,
    write_tiny_model_evidence,
)


def test_tiny_qwen_token_denoiser_forward_has_vocab_logits() -> None:
    torch.manual_seed(1)
    config = QwenDenoiserConfig.tiny(vocab_size=64, max_length=8)
    model = TinyQwenTokenDenoiser(config)
    input_ids = torch.zeros((1, 8), dtype=torch.long)

    logits = model(input_ids)

    assert tuple(logits.shape) == (1, 8, 64)
    assert config.qwen_config.tokenizer_id == "Qwen/Qwen3.5-4B"
    assert config.compatibility_metadata()["downloads_allowed"] is False


def test_qwen_denoiser_checkpoint_round_trip_preserves_manifest_and_logits(tmp_path: Path) -> None:
    torch.manual_seed(1)
    config = QwenDenoiserConfig.tiny(vocab_size=64, max_length=8)
    model = TinyQwenTokenDenoiser(config)
    input_ids = torch.arange(0, 8, dtype=torch.long).reshape(1, 8)
    expected = model(input_ids).detach()
    checkpoint = tmp_path / "qwen-tiny.pt"

    save_qwen_denoiser_checkpoint(checkpoint, model)
    loaded_model, manifest = load_qwen_denoiser_checkpoint(checkpoint)
    actual = loaded_model(input_ids).detach()

    assert manifest["architecture"] == "tiny-qwen-token-denoiser"
    assert manifest["model_config"]["vocab_size"] == 64
    assert manifest["qwen_config"]["tokenizer_id"] == "Qwen/Qwen3.5-4B"
    assert manifest["compatibility_metadata"]["weights_downloaded"] is False
    assert torch.allclose(actual, expected)


def test_qwen_denoiser_rejects_tokenizer_config_mismatch(tmp_path: Path) -> None:
    config = QwenDenoiserConfig.tiny(vocab_size=64, max_length=8)
    model = TinyQwenTokenDenoiser(config)
    checkpoint = tmp_path / "qwen-tiny.pt"
    save_qwen_denoiser_checkpoint(checkpoint, model)
    payload = torch.load(checkpoint, map_location="cpu")
    payload["manifest"]["qwen_config"]["tokenizer_id"] = "Qwen/other-tokenizer"
    torch.save(payload, checkpoint)

    with pytest.raises(QwenDenoiserCheckpointError) as raised:
        load_qwen_denoiser_checkpoint(checkpoint)

    assert "tokenizer/config mismatch" in str(raised.value)


def test_qwen_denoiser_uses_no_qwen_or_ollama_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_fallback(*_args, **_kwargs):
        pytest.fail("external Qwen/Ollama fallback was called")

    monkeypatch.setattr("qiffusion.qwen_tokenizer.load_tokenizer", fail_fallback)
    monkeypatch.setattr("qiffusion.qwen_ollama.run_ollama_fixture", fail_fallback)
    config = QwenDenoiserConfig.tiny(vocab_size=64, max_length=8)
    model = TinyQwenTokenDenoiser(config)

    logits = model(torch.zeros((1, 8), dtype=torch.long))

    assert tuple(logits.shape) == (1, 8, 64)


def test_tiny_model_evidence_records_logits_config_and_compatibility(tmp_path: Path) -> None:
    artifact = tmp_path / "model.json"

    write_tiny_model_evidence(artifact, batch_size=1, sequence_length=8, vocab_size=64)

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["logits_shape"] == [1, 8, 64]
    assert payload["model_config"]["vocab_size"] == 64
    assert payload["qwen_config"]["objective"] == "masked_ce"
    assert payload["compatibility_metadata"]["downloads_allowed"] is False
    assert payload["fallback_used"] is False


def test_mismatch_checkpoint_probe_records_tokenizer_config_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "mismatch.json"

    write_mismatch_checkpoint_probe(artifact)

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert payload["failure"] == "tokenizer/config mismatch"
    assert "tokenizer/config mismatch" in payload["error"]
    assert payload["fallback_used"] is False


def test_qwen_denoiser_can_use_explicit_compatibility_config() -> None:
    qwen_config = default_config(base_checkpoint_id="Qwen/Qwen3.5-4B")
    config = QwenDenoiserConfig(vocab_size=64, dim=16, layers=1, max_length=8, qwen_config=qwen_config)

    metadata = config.compatibility_metadata()

    assert metadata["base_checkpoint_id"] == "Qwen/Qwen3.5-4B"
    assert metadata["tokenizer_id"] == "Qwen/Qwen3.5-4B"
