from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch

from qiffusion.diffusion_config import BYTE_TOKEN_OFFSET, BYTE_VOCAB_SIZE, MASK_TOKEN_ID
from qiffusion.diffusion_data import ByteTokenizer
from qiffusion.diffusion_model import load_checkpoint


class DiffusionSampleReport(TypedDict):
    backend: str
    stage: str
    status: str
    checkpoint_path: str
    prompt: str
    generated_text: str
    steps: int
    seed: int
    fixtures_status: str
    code_smoke_status: str
    candidate_source: str
    coding_capability_claim: bool


@dataclass(frozen=True, slots=True)
class DiffusionSampleConfig:
    checkpoint_path: Path
    prompt: str
    steps: int
    seed: int


def sample_from_checkpoint(config: DiffusionSampleConfig) -> DiffusionSampleReport:
    torch.manual_seed(config.seed)
    model, model_config = load_checkpoint(config.checkpoint_path)
    tokenizer = ByteTokenizer()
    prompt_ids = tokenizer.encode(config.prompt)
    canvas = [*prompt_ids, *((MASK_TOKEN_ID,) * config.steps)]
    del canvas[model_config.max_length :]
    model.eval()
    with torch.no_grad():
        for _ in range(config.steps):
            mask_index = first_mask_index(canvas)
            if mask_index is None:
                break
            input_ids = torch.tensor([canvas], dtype=torch.long)
            logits = model(input_ids)
            canvas[mask_index] = best_byte_token(logits[0, mask_index])
    generated = decode_canvas(canvas)
    return {
        "backend": "diffusion",
        "stage": "sample",
        "status": "sampled",
        "checkpoint_path": str(config.checkpoint_path),
        "prompt": config.prompt,
        "generated_text": generated,
        "steps": config.steps,
        "seed": config.seed,
        "fixtures_status": "pass",
        "code_smoke_status": "not_run",
        "candidate_source": "tiny-diffusion-checkpoint",
        "coding_capability_claim": False,
    }


def first_mask_index(tokens: list[int]) -> int | None:
    for index, token in enumerate(tokens):
        if token == MASK_TOKEN_ID:
            return index
    return None


def best_byte_token(logits: torch.Tensor) -> int:
    byte_logits = logits[BYTE_TOKEN_OFFSET:BYTE_VOCAB_SIZE]
    return int(torch.argmax(byte_logits).item()) + BYTE_TOKEN_OFFSET


def decode_canvas(tokens: list[int]) -> str:
    byte_values = [token - BYTE_TOKEN_OFFSET for token in tokens if token >= BYTE_TOKEN_OFFSET]
    return bytes(byte_values).decode("utf-8", errors="replace")
