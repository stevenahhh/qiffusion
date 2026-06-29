from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from qiffusion.diffusion_model import TinyDiffusionConfig, TinyDiffusionLM, load_checkpoint, save_checkpoint


def test_tiny_model_forward_pass_has_vocab_logits() -> None:
    torch.manual_seed(1)
    model = TinyDiffusionLM(TinyDiffusionConfig(vocab_size=260, dim=32, layers=1))
    inputs = torch.zeros((1, 8), dtype=torch.long)

    logits = model(inputs)

    assert tuple(logits.shape) == (1, 8, 260)


def test_checkpoint_round_trip_preserves_config_and_logits(tmp_path: Path) -> None:
    torch.manual_seed(1)
    config = TinyDiffusionConfig(vocab_size=260, dim=32, layers=1)
    model = TinyDiffusionLM(config)
    inputs = torch.arange(0, 8, dtype=torch.long).reshape(1, 8)
    expected = model(inputs).detach()
    checkpoint = tmp_path / "tiny.pt"

    save_checkpoint(checkpoint, model, config)
    loaded_model, loaded_config = load_checkpoint(checkpoint)
    actual = loaded_model(inputs).detach()

    assert loaded_config == config
    assert torch.allclose(actual, expected)
