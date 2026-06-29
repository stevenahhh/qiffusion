from __future__ import annotations

from qiffusion.diffusion_config import MASK_TOKEN_ID
from qiffusion.diffusion_objective import mask_probability, mask_tokens


def test_mask_tokens_is_reproducible_for_seed() -> None:
    tokens = (10, 11, 12, 13, 14, 15)

    first = mask_tokens(tokens, seed=7)
    second = mask_tokens(tokens, seed=7)

    assert first == second
    assert any(first.masked_positions)
    assert all(token == MASK_TOKEN_ID for token, masked in zip(first.input_ids, first.masked_positions) if masked)


def test_mask_tokens_labels_cover_masked_positions_only() -> None:
    tokens = (10, 11, 12, 13)

    result = mask_tokens(tokens, seed=3)

    assert result.original_ids == tokens
    assert all((label == -100) == (not masked) for label, masked in zip(result.labels, result.masked_positions))
    assert all(label == original for label, original, masked in zip(result.labels, tokens, result.masked_positions) if masked)


def test_mask_probability_increases_with_timestep() -> None:
    early = mask_probability(timestep=0, total_steps=10)
    late = mask_probability(timestep=9, total_steps=10)

    assert 0.0 < early < late < 1.0
