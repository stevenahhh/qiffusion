from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict

import torch
from torch import Tensor, nn

from qiffusion.qwen_diffusion_config import JsonObject, JsonValue, QwenDiffusionConfig, config_from_json, default_config


SCHEMA_VERSION: Final = 1
ARCHITECTURE: Final = "tiny-qwen-token-denoiser"


class QwenDenoiserConfigJson(TypedDict):
    vocab_size: int
    dim: int
    layers: int
    max_length: int
    tokenizer_id: str


class QwenDenoiserManifest(TypedDict):
    schema_version: int
    architecture: str
    model_config: QwenDenoiserConfigJson
    qwen_config: dict[str, JsonValue]
    compatibility_metadata: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class QwenDenoiserCheckpointError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class QwenDenoiserConfig:
    vocab_size: int
    dim: int
    layers: int
    max_length: int
    qwen_config: QwenDiffusionConfig

    @classmethod
    def tiny(cls, vocab_size: int, max_length: int) -> QwenDenoiserConfig:
        return cls(
            vocab_size=vocab_size,
            dim=16,
            layers=1,
            max_length=max_length,
            qwen_config=default_config(base_checkpoint_id="Qwen/Qwen3.5-4B"),
        )

    def to_json(self) -> QwenDenoiserConfigJson:
        return {
            "vocab_size": self.vocab_size,
            "dim": self.dim,
            "layers": self.layers,
            "max_length": self.max_length,
            "tokenizer_id": self.qwen_config.tokenizer_id,
        }

    def compatibility_metadata(self) -> dict[str, JsonValue]:
        return dict(self.qwen_config.compatibility_contract)


class TinyQwenTokenDenoiser(nn.Module):
    def __init__(self, config: QwenDenoiserConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.position_embedding = nn.Embedding(config.max_length, config.dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.dim,
            nhead=4,
            dim_feedforward=config.dim * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.denoiser = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.output = nn.Linear(config.dim, config.vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        batch_size, sequence_length = input_ids.shape
        positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0).expand(batch_size, sequence_length)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        return self.output(self.denoiser(hidden))


def save_qwen_denoiser_checkpoint(path: Path, model: TinyQwenTokenDenoiser) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"manifest": _manifest(model.config), "state_dict": model.state_dict()}, path)


def load_qwen_denoiser_checkpoint(path: Path) -> tuple[TinyQwenTokenDenoiser, QwenDenoiserManifest]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise QwenDenoiserCheckpointError(f"{path}: expected checkpoint mapping")
    manifest = _manifest_from_json(_payload_mapping(payload, "manifest"))
    state_dict = payload["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise QwenDenoiserCheckpointError(f"{path}: expected state_dict mapping")
    model = TinyQwenTokenDenoiser(_config_from_manifest(manifest))
    model.load_state_dict(state_dict)
    model.eval()
    return model, manifest


def write_tiny_model_evidence(path: Path, batch_size: int, sequence_length: int, vocab_size: int) -> None:
    torch.manual_seed(1)
    config = QwenDenoiserConfig.tiny(vocab_size=vocab_size, max_length=sequence_length)
    model = TinyQwenTokenDenoiser(config)
    input_ids = torch.zeros((batch_size, sequence_length), dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids)
    payload: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "architecture": ARCHITECTURE,
        "logits_shape": list(logits.shape),
        "model_config": {
            "vocab_size": config.vocab_size,
            "dim": config.dim,
            "layers": config.layers,
            "max_length": config.max_length,
            "tokenizer_id": config.qwen_config.tokenizer_id,
        },
        "qwen_config": config.qwen_config.to_json(),
        "compatibility_metadata": config.compatibility_metadata(),
        "fallback_used": False,
    }
    _write_json(path, payload)


def write_mismatch_checkpoint_probe(path: Path) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint = Path(temp_dir) / "mismatch.pt"
        config = QwenDenoiserConfig.tiny(vocab_size=64, max_length=8)
        save_qwen_denoiser_checkpoint(checkpoint, TinyQwenTokenDenoiser(config))
        payload = torch.load(checkpoint, map_location="cpu")
        payload["manifest"]["qwen_config"]["tokenizer_id"] = "Qwen/mismatched-tokenizer"
        torch.save(payload, checkpoint)
        try:
            _ = load_qwen_denoiser_checkpoint(checkpoint)
        except QwenDenoiserCheckpointError as exc:
            _write_json(
                path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "blocked",
                    "failure": "tokenizer/config mismatch",
                    "error": str(exc),
                    "fallback_used": False,
                },
            )
            return
    raise QwenDenoiserCheckpointError("mismatch probe did not detect tokenizer/config mismatch")


def _manifest(config: QwenDenoiserConfig) -> QwenDenoiserManifest:
    return {
        "schema_version": SCHEMA_VERSION,
        "architecture": ARCHITECTURE,
        "model_config": config.to_json(),
        "qwen_config": config.qwen_config.to_json(),
        "compatibility_metadata": config.compatibility_metadata(),
    }


def _manifest_from_json(payload: Mapping[str, JsonValue]) -> QwenDenoiserManifest:
    qwen_config = _payload_mapping(payload, "qwen_config")
    compatibility = _payload_mapping(payload, "compatibility_metadata")
    model_config = _payload_mapping(payload, "model_config")
    return {
        "schema_version": _integer(payload, "schema_version"),
        "architecture": _string(payload, "architecture"),
        "model_config": {
            "vocab_size": _integer(model_config, "vocab_size"),
            "dim": _integer(model_config, "dim"),
            "layers": _integer(model_config, "layers"),
            "max_length": _integer(model_config, "max_length"),
            "tokenizer_id": _string(model_config, "tokenizer_id"),
        },
        "qwen_config": dict(qwen_config),
        "compatibility_metadata": dict(compatibility),
    }


def _config_from_manifest(manifest: QwenDenoiserManifest) -> QwenDenoiserConfig:
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["architecture"] != ARCHITECTURE:
        raise QwenDenoiserCheckpointError("unsupported qwen denoiser manifest")
    qwen_config = config_from_json(manifest["qwen_config"])
    model_config = manifest["model_config"]
    compatibility = manifest["compatibility_metadata"]
    expected_tokenizer_id = model_config["tokenizer_id"]
    if expected_tokenizer_id != qwen_config.tokenizer_id or compatibility["tokenizer_id"] != qwen_config.tokenizer_id:
        raise QwenDenoiserCheckpointError("tokenizer/config mismatch in qwen denoiser checkpoint")
    return QwenDenoiserConfig(
        vocab_size=model_config["vocab_size"],
        dim=model_config["dim"],
        layers=model_config["layers"],
        max_length=model_config["max_length"],
        qwen_config=qwen_config,
    )


def _payload_mapping(payload: Mapping[str, JsonValue], field: str) -> Mapping[str, JsonValue]:
    value = payload[field]
    if not isinstance(value, dict):
        raise QwenDenoiserCheckpointError(f"{field}: expected object")
    return value


def _string(payload: Mapping[str, JsonValue], field: str) -> str:
    value = payload[field]
    if not isinstance(value, str):
        raise QwenDenoiserCheckpointError(f"{field}: expected string")
    return value


def _integer(payload: Mapping[str, JsonValue], field: str) -> int:
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise QwenDenoiserCheckpointError(f"{field}: expected integer")
    return value


def _write_json(path: Path, payload: Mapping[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
