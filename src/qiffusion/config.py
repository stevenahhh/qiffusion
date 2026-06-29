from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TrackName = Literal["diffusion", "qwen_bridge"]


@dataclass(frozen=True, slots=True)
class Track:
    name: TrackName
    role: str
    done_when: tuple[str, ...]


TRACKS: tuple[Track, ...] = (
    Track(
        name="diffusion",
        role="Primary target: train and sample a local diffusion LLM.",
        done_when=(
            "executable code smoke passes",
            "chat and coding fixtures pass without Qwen fallback",
            "speed and quality evidence are recorded",
        ),
    ),
    Track(
        name="qwen_bridge",
        role="Bootstrap path: local Qwen 4B baseline, teacher data, and verifier comparison.",
        done_when=(
            "local Qwen engine is discovered or prerequisite_missing is recorded",
            "teacher outputs include provenance",
            "diffusion outputs are compared against Qwen outputs under the same gate",
        ),
    ),
)

CODING_CAPABLE_REQUIREMENTS: tuple[str, ...] = (
    "code_smoke_status == pass",
    "coding_capability_claim == true",
    "fixture_status == pass",
    "candidate source is the evaluated model, not a hard-coded repair",
)

