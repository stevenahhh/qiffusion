from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import nn

from qiffusion.diffusion_config import BYTE_VOCAB_SIZE, EOS_TOKEN_ID
from qiffusion.diffusion_data import CorpusExample, SYNTHETIC_EXAMPLES
from qiffusion.diffusion_objective import mask_tokens
from qiffusion.qwen_diffusion_config import JsonObject, JsonValue, default_compatibility_contract, default_config
from qiffusion.qwen_diffusion_model import ARCHITECTURE, SCHEMA_VERSION, QwenDenoiserConfig, TinyQwenTokenDenoiser
from qiffusion.qwen_tokenizer import QwenTokenizer, Tokenizer, TokenizerUnavailableError, load_tokenizer


IGNORE_LABEL = -100


@dataclass(frozen=True, slots=True)
class QwenDiffusionTrainConfig:
    manifest_path: Path
    checkpoint_path: Path
    tokenizer_id: str
    steps: int
    seed: int
    sequence_length: int = 64
    dim: int = 16
    layers: int = 1
    learning_rate: float = 0.01


@dataclass(frozen=True, slots=True)
class TrainingManifest:
    manifest_id: str
    root: Path
    records: tuple[Mapping[str, JsonValue], ...]


@dataclass(frozen=True, slots=True)
class MaskedBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor


@dataclass(frozen=True, slots=True)
class TrainingDataBlockedError(Exception):
    manifest_path: Path
    blocked_sources: tuple[str, ...]
    reason: str

    def __str__(self) -> str:
        return f"{self.manifest_path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class TrainingManifestError(Exception):
    manifest_path: Path
    reason: str

    def __str__(self) -> str:
        return f"{self.manifest_path}: {self.reason}"


def train_qwen_diffusion(config: QwenDiffusionTrainConfig) -> JsonObject:
    torch.manual_seed(config.seed)
    tokenizer = load_tokenizer(config.tokenizer_id)
    manifest = load_training_manifest(config.manifest_path)
    examples = training_examples(manifest, config.manifest_path, config.tokenizer_id)
    sequences = pack_training_sequences(examples, tokenizer, config.tokenizer_id, config.sequence_length)
    lineage: JsonObject = {
        "parent_checkpoint_id": None,
        "data_manifest_id": manifest.manifest_id,
        "objective": "masked_ce",
        "mask_schedule": "linear",
        "tokenizer_id": config.tokenizer_id,
        "steps": config.steps,
        "seed": config.seed,
    }
    base_qwen_config = default_config(base_checkpoint_id="Qwen/Qwen3.5-4B")
    qwen_config = replace(
        base_qwen_config,
        tokenizer_id=config.tokenizer_id,
        data_manifest_id=manifest.manifest_id,
        checkpoint_lineage=lineage,
        compatibility_contract=default_compatibility_contract(base_qwen_config.base_checkpoint_id, config.tokenizer_id, base_qwen_config.resource_probe),
    )
    model_config = QwenDenoiserConfig(
        vocab_size=_vocab_size(config.tokenizer_id, tokenizer),
        dim=config.dim,
        layers=config.layers,
        max_length=config.sequence_length,
        qwen_config=qwen_config,
    )
    model = TinyQwenTokenDenoiser(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    initial_loss = batch_loss(model, loss_fn, masked_batch(sequences, config.seed))
    final_loss = initial_loss
    model.train()
    for step in range(config.steps):
        batch = masked_batch(sequences, config.seed + step)
        optimizer.zero_grad()
        logits = model(batch.input_ids)
        loss = loss_fn(logits.reshape(-1, model_config.vocab_size), batch.labels.reshape(-1))
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())
    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "manifest": {
                "schema_version": SCHEMA_VERSION,
                "architecture": ARCHITECTURE,
                "model_config": model.config.to_json(),
                "qwen_config": model.config.qwen_config.to_json(),
                "compatibility_metadata": model.config.compatibility_metadata(),
            },
            "state_dict": model.state_dict(),
            "checkpoint_lineage": lineage,
            "train_loss": {"initial_loss": initial_loss, "final_loss": final_loss},
        },
        config.checkpoint_path,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": "qwen_token_diffusion",
        "stage": "train",
        "status": "trained",
        "checkpoint_path": str(config.checkpoint_path),
        "data_manifest_id": manifest.manifest_id,
        "objective": "masked_ce",
        "mask_schedule": "linear",
        "steps": config.steps,
        "seed": config.seed,
        "examples": len(examples),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "checkpoint_lineage": lineage,
        "fallback_used": False,
        "fixtures_status": "pass",
        "code_smoke_status": "not_run",
        "coding_capability_claim": False,
    }


def blocked_report(error: TrainingDataBlockedError) -> JsonObject:
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": "qwen_token_diffusion",
        "stage": "train",
        "status": "blocked",
        "manifest_path": str(error.manifest_path),
        "blocked_sources": list(error.blocked_sources),
        "reason": error.reason,
        "fallback_used": False,
        "coding_capability_claim": False,
    }


def load_training_manifest(path: Path) -> TrainingManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TrainingManifestError(path, "expected manifest object")
    records = payload.get("records")
    if not isinstance(records, list):
        raise TrainingManifestError(path, "expected records list")
    root_value = payload.get("root")
    root = Path(root_value) if isinstance(root_value, str) else path.parent
    manifest_id = payload.get("manifest_id")
    if not isinstance(manifest_id, str):
        manifest_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return TrainingManifest(
        manifest_id=manifest_id,
        root=root,
        records=tuple(_record(path, record) for record in records),
    )


def training_examples(manifest: TrainingManifest, manifest_path: Path, tokenizer_id: str) -> tuple[CorpusExample, ...]:
    blocked = tuple(record["source"] for record in manifest.records if _blocks_training(record, tokenizer_id))
    if len(blocked) > 0:
        raise TrainingDataBlockedError(manifest_path, tuple(str(source) for source in blocked), "manifest contains non-training or contaminated sources")
    examples = tuple(_example_from_record(manifest, manifest_path, record) for record in manifest.records)
    if len(examples) == 0:
        raise TrainingDataBlockedError(manifest_path, (), "manifest has no trainable examples")
    return examples


def pack_training_sequences(
    examples: Sequence[CorpusExample],
    tokenizer: Tokenizer,
    tokenizer_id: str,
    sequence_length: int,
) -> tuple[tuple[int, ...], ...]:
    eos_id = _eos_id(tokenizer_id, tokenizer)
    packed: list[tuple[int, ...]] = []
    for example in examples:
        encoded = tokenizer.encode(example.text)
        sequence = (*encoded, eos_id) if eos_id is not None else encoded
        trimmed = sequence[:sequence_length]
        padding = (0,) * (sequence_length - len(trimmed))
        packed.append((*trimmed, *padding))
    return tuple(packed)


def batch_loss(model: TinyQwenTokenDenoiser, loss_fn: nn.CrossEntropyLoss, batch: MaskedBatch) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(batch.input_ids)
        loss = loss_fn(logits.reshape(-1, model.config.vocab_size), batch.labels.reshape(-1))
    return float(loss.item())


def masked_batch(sequences: tuple[tuple[int, ...], ...], seed: int) -> MaskedBatch:
    masked = tuple(mask_tokens(sequence, seed + index) for index, sequence in enumerate(sequences))
    input_ids = torch.tensor([item.input_ids for item in masked], dtype=torch.long)
    labels = torch.tensor([item.labels for item in masked], dtype=torch.long)
    return MaskedBatch(input_ids, labels)


def _record(manifest_path: Path, payload: JsonValue) -> Mapping[str, JsonValue]:
    if not isinstance(payload, dict):
        raise TrainingManifestError(manifest_path, "expected record object")
    return payload


def _blocks_training(record: Mapping[str, JsonValue], tokenizer_id: str) -> bool:
    source = _string(record, "source")
    usage, source_kind = _string(record, "usage"), _string(record, "source_kind")
    contamination = _string(record, "contamination_status")
    split = _string(record, "split")
    tokenizer = _string(record, "tokenizer")
    return (
        usage != "train_allowed"
        or source_kind != "local"
        or contamination != "clean"
        or split != "train"
        or tokenizer != tokenizer_id
        or _is_benchmark_source(source)
    )


def _example_from_record(
    manifest: TrainingManifest,
    manifest_path: Path,
    record: Mapping[str, JsonValue],
) -> CorpusExample:
    source = _string(record, "source")
    name = _string(record, "name")
    if source == "synthetic":
        return _synthetic_example(manifest_path, name)
    path = Path(source)
    resolved = path if path.is_absolute() else manifest.root / path
    if not resolved.is_file():
        raise TrainingManifestError(manifest_path, f"{source}: source file is missing")
    return CorpusExample(name=name, source=str(resolved), text=resolved.read_text(encoding="utf-8"))


def _synthetic_example(manifest_path: Path, name: str) -> CorpusExample:
    for example in SYNTHETIC_EXAMPLES:
        if example.name == name:
            return example
    raise TrainingManifestError(manifest_path, f"{name}: synthetic example is missing")


def _vocab_size(tokenizer_id: str, tokenizer: Tokenizer) -> int:
    normalized = tokenizer_id.strip().lower()
    if normalized == "byte":
        return BYTE_VOCAB_SIZE
    if isinstance(tokenizer, QwenTokenizer):
        return tokenizer.config.vocab_size
    raise TokenizerUnavailableError(tokenizer_id, "tokenizer did not expose a vocab size")


def _eos_id(tokenizer_id: str, tokenizer: Tokenizer) -> int | None:
    normalized = tokenizer_id.strip().lower()
    if normalized == "byte":
        return EOS_TOKEN_ID
    if isinstance(tokenizer, QwenTokenizer):
        return tokenizer.config.eos_id
    return None


def _string(payload: Mapping[str, JsonValue], field: str) -> str:
    value = payload[field]
    if not isinstance(value, str):
        raise TrainingManifestError(Path("<manifest>"), f"{field}: expected string")
    return value


def _is_benchmark_source(source: str) -> bool:
    lowered = source.lower()
    return any(marker in lowered for marker in ("humaneval", "mbpp", "evalplus", "livecodebench", "bigcodebench", "swe-bench", "swebench"))
