from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Final, Sequence

from qiffusion.diffusion_config import MASK_TOKEN_ID

IGNORE_LABEL: Final = -100
MIN_MASK_PROBABILITY: Final = 0.15
MAX_MASK_PROBABILITY: Final = 0.75


@dataclass(frozen=True, slots=True)
class DiffusionObjectiveError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class MaskedTokens:
    original_ids: tuple[int, ...]
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    masked_positions: tuple[bool, ...]


def mask_probability(timestep: int, total_steps: int) -> float:
    if total_steps < 1:
        raise DiffusionObjectiveError("total_steps must be at least 1")
    clamped_step = min(max(timestep, 0), total_steps - 1)
    progress = (clamped_step + 1) / total_steps
    return MIN_MASK_PROBABILITY + (MAX_MASK_PROBABILITY - MIN_MASK_PROBABILITY) * progress


def mask_tokens(tokens: Sequence[int], seed: int) -> MaskedTokens:
    original = tuple(tokens)
    if len(original) == 0:
        return MaskedTokens((), (), (), ())

    random = Random(seed)
    probability = mask_probability(timestep=seed % len(original), total_steps=len(original))
    masked_positions = tuple(random.random() < probability for _ in original)
    if not any(masked_positions):
        selected = random.randrange(len(original))
        masked_positions = tuple(index == selected for index in range(len(original)))

    input_ids = tuple(MASK_TOKEN_ID if masked else token for token, masked in zip(original, masked_positions))
    labels = tuple(token if masked else IGNORE_LABEL for token, masked in zip(original, masked_positions))
    return MaskedTokens(
        original_ids=original,
        input_ids=input_ids,
        labels=labels,
        masked_positions=masked_positions,
    )
