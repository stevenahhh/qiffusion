from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch
from torch import Tensor, nn


class TinyDiffusionConfigJson(TypedDict):
    vocab_size: int
    dim: int
    layers: int
    max_length: int


@dataclass(frozen=True, slots=True)
class TinyDiffusionConfig:
    vocab_size: int
    dim: int
    layers: int
    max_length: int = 128

    def to_json(self) -> TinyDiffusionConfigJson:
        return {
            "vocab_size": self.vocab_size,
            "dim": self.dim,
            "layers": self.layers,
            "max_length": self.max_length,
        }


@dataclass(frozen=True, slots=True)
class DiffusionCheckpointError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


class TinyDiffusionLM(nn.Module):
    def __init__(self, config: TinyDiffusionConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.position_embedding = nn.Embedding(config.max_length, config.dim)
        self.denoiser = nn.GRU(
            input_size=config.dim,
            hidden_size=config.dim,
            num_layers=config.layers,
            batch_first=True,
        )
        self.output = nn.Linear(config.dim, config.vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        batch_size, sequence_length = input_ids.shape
        positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0).expand(batch_size, sequence_length)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        denoised, _ = self.denoiser(hidden)
        return self.output(denoised)


def config_from_json(payload: TinyDiffusionConfigJson) -> TinyDiffusionConfig:
    return TinyDiffusionConfig(
        vocab_size=payload["vocab_size"],
        dim=payload["dim"],
        layers=payload["layers"],
        max_length=payload["max_length"],
    )


def save_checkpoint(path: Path, model: TinyDiffusionLM, config: TinyDiffusionConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config.to_json(), "state_dict": model.state_dict()}, path)


def load_checkpoint(path: Path) -> tuple[TinyDiffusionLM, TinyDiffusionConfig]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise DiffusionCheckpointError(f"{path}: expected checkpoint mapping")
    config_payload = payload["config"]
    if not isinstance(config_payload, dict):
        raise DiffusionCheckpointError(f"{path}: expected config mapping")
    config = config_from_json(
        {
            "vocab_size": int(config_payload["vocab_size"]),
            "dim": int(config_payload["dim"]),
            "layers": int(config_payload["layers"]),
            "max_length": int(config_payload["max_length"]),
        }
    )
    model = TinyDiffusionLM(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config
