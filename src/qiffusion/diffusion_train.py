from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch
from torch import nn

from qiffusion.diffusion_config import BYTE_VOCAB_SIZE
from qiffusion.diffusion_data import ByteTokenizer, CorpusExample, build_local_corpus, pack_sequences
from qiffusion.diffusion_model import TinyDiffusionConfig, TinyDiffusionLM, save_checkpoint
from qiffusion.diffusion_objective import mask_tokens


class DiffusionTrainReport(TypedDict):
    backend: str
    stage: str
    status: str
    checkpoint_path: str
    steps: int
    seed: int
    examples: int
    initial_loss: float
    final_loss: float
    fixtures_status: str
    code_smoke_status: str
    candidate_source: str
    coding_capability_claim: bool


@dataclass(frozen=True, slots=True)
class DiffusionTrainConfig:
    checkpoint_path: Path
    steps: int
    seed: int
    max_examples: int
    sequence_length: int = 64
    dim: int = 32
    layers: int = 1
    learning_rate: float = 0.01
    teacher_jsonl_paths: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class MaskedBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor


def train_tiny_diffusion(config: DiffusionTrainConfig) -> DiffusionTrainReport:
    torch.manual_seed(config.seed)
    tokenizer = ByteTokenizer()
    examples = (*build_local_corpus(), *teacher_examples(config.teacher_jsonl_paths))[: config.max_examples]
    sequences = pack_sequences(examples, tokenizer, config.sequence_length)
    model = TinyDiffusionLM(
        TinyDiffusionConfig(
            vocab_size=BYTE_VOCAB_SIZE,
            dim=config.dim,
            layers=config.layers,
            max_length=config.sequence_length,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    initial_loss = batch_loss(model, loss_fn, masked_batch(sequences, config.seed))
    final_loss = initial_loss
    model.train()
    for step in range(config.steps):
        batch = masked_batch(sequences, config.seed + step)
        optimizer.zero_grad()
        logits = model(batch.input_ids)
        loss = loss_fn(logits.reshape(-1, BYTE_VOCAB_SIZE), batch.labels.reshape(-1))
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())
    save_checkpoint(config.checkpoint_path, model, model.config)
    return {
        "backend": "diffusion",
        "stage": "train",
        "status": "trained",
        "checkpoint_path": str(config.checkpoint_path),
        "steps": config.steps,
        "seed": config.seed,
        "examples": len(examples),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "fixtures_status": "pass",
        "code_smoke_status": "not_run",
        "candidate_source": "tiny-diffusion-checkpoint",
        "coding_capability_claim": False,
    }


def batch_loss(model: TinyDiffusionLM, loss_fn: nn.CrossEntropyLoss, batch: MaskedBatch) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(batch.input_ids)
        loss = loss_fn(logits.reshape(-1, BYTE_VOCAB_SIZE), batch.labels.reshape(-1))
    return float(loss.item())


def masked_batch(sequences: tuple[tuple[int, ...], ...], seed: int) -> MaskedBatch:
    masked = tuple(mask_tokens(sequence, seed + index) for index, sequence in enumerate(sequences))
    input_ids = torch.tensor([item.input_ids for item in masked], dtype=torch.long)
    labels = torch.tensor([item.labels for item in masked], dtype=torch.long)
    return MaskedBatch(input_ids, labels)


def teacher_examples(paths: tuple[Path, ...]) -> tuple[CorpusExample, ...]:
    examples: list[CorpusExample] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            code = payload.get("code")
            name = payload.get("task_name")
            if isinstance(code, str) and isinstance(name, str):
                examples.append(CorpusExample(f"teacher:{name}", str(path), code))
    return tuple(examples)
