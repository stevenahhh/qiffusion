from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias, TypeVar

AttentionMode: TypeAlias = Literal["causal", "annealed", "bidirectional"]
Objective: TypeAlias = Literal["masked_ce"]
SamplerAlgorithm: TypeAlias = Literal["p2", "confidence_greedy"]
MaskSchedule: TypeAlias = Literal["linear", "cosine"]
ResourceStatus: TypeAlias = Literal["available", "missing", "unknown"]
DataUsage: TypeAlias = Literal["train_allowed", "eval_only", "unknown_blocked"]
ContaminationStatus: TypeAlias = Literal["clean", "suspect", "blocked"]
JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
EnumValue = TypeVar("EnumValue", bound=str)

SCHEMA_VERSION: Final = 1
DEFAULT_TOKENIZER_ID: Final = "Qwen/Qwen3.5-4B"
CONFIG_FIELDS: Final = (
    "base_checkpoint_id", "tokenizer_id", "attention_mode", "objective", "sampler_algorithm",
    "mask_schedule", "block_size", "seed", "data_manifest_id", "checkpoint_lineage",
)
TENSOR_MAPPING_NOTES: Final = (
    "record Qwen tensor-name mapping before weight conversion",
    "do not download or inspect checkpoint weights in this contract",
)
BENCHMARK_DATASETS: Final = ("humaneval", "mbpp", "evalplus", "livecodebench", "bigcodebench", "swe-bench")


@dataclass(frozen=True, slots=True)
class ConfigValidationError(Exception):
    field: str
    value: str
    expected: str

    def __str__(self) -> str:
        return f"{self.field}={self.value!r} is invalid; expected {self.expected}"


@dataclass(frozen=True, slots=True)
class ContaminationBlockedError(Exception):
    blocked_sources: tuple[str, ...]

    def __str__(self) -> str:
        return f"training sources blocked by benchmark gate: {', '.join(self.blocked_sources)}"


@dataclass(frozen=True, slots=True)
class ResourceProbe:
    status: str
    detail: str

    def __post_init__(self) -> None:
        _ = _parse_resource_status(self.status)

    def to_json(self) -> JsonObject:
        return {"status": self.status, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class BenchmarkGateEntry:
    usage: str
    contamination_status: str
    reason: str

    def __post_init__(self) -> None:
        _ = _parse_data_usage(self.usage)
        _ = _parse_contamination_status(self.contamination_status)

    def to_json(self) -> JsonObject:
        return {"usage": self.usage, "contamination_status": self.contamination_status, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class QwenDiffusionConfig:
    base_checkpoint_id: str
    tokenizer_id: str
    attention_mode: AttentionMode
    objective: Objective
    sampler_algorithm: SamplerAlgorithm
    mask_schedule: MaskSchedule
    block_size: int
    seed: int
    data_manifest_id: str
    checkpoint_lineage: Mapping[str, JsonValue]
    compatibility_contract: Mapping[str, JsonValue]
    resource_probe: ResourceProbe
    benchmark_gate: Mapping[str, BenchmarkGateEntry]

    def __post_init__(self) -> None:
        _ = _parse_attention_mode(self.attention_mode)
        _ = _parse_objective(self.objective)
        _ = _parse_sampler_algorithm(self.sampler_algorithm)
        _ = _parse_mask_schedule(self.mask_schedule)
        if self.block_size <= 0:
            raise ConfigValidationError("block_size", str(self.block_size), "positive integer")

    def to_json(self) -> JsonObject:
        return {
            "schema_version": SCHEMA_VERSION,
            "base_checkpoint_id": self.base_checkpoint_id,
            "tokenizer_id": self.tokenizer_id,
            "attention_mode": self.attention_mode,
            "objective": self.objective,
            "sampler_algorithm": self.sampler_algorithm,
            "mask_schedule": self.mask_schedule,
            "block_size": self.block_size,
            "seed": self.seed,
            "data_manifest_id": self.data_manifest_id,
            "checkpoint_lineage": dict(self.checkpoint_lineage),
            "compatibility_contract": dict(self.compatibility_contract),
            "resource_probe": self.resource_probe.to_json(),
            "benchmark_gate": {key: entry.to_json() for key, entry in sorted(self.benchmark_gate.items())},
        }

    def ensure_training_sources_allowed(self, sources: tuple[str, ...]) -> None:
        blocked = tuple(source for source in sources if self.training_blocked(source))
        if len(blocked) > 0:
            raise ContaminationBlockedError(blocked)

    def training_blocked(self, source: str) -> bool:
        entry = self.benchmark_gate.get(source.lower())
        if entry is None:
            return True
        return entry.usage != "train_allowed" or entry.contamination_status != "clean"


def default_config(base_checkpoint_id: str) -> QwenDiffusionConfig:
    probe = ResourceProbe(status="unknown", detail="offline no-download probe not executed")
    return QwenDiffusionConfig(
        base_checkpoint_id=base_checkpoint_id,
        tokenizer_id=DEFAULT_TOKENIZER_ID,
        attention_mode="bidirectional",
        objective="masked_ce",
        sampler_algorithm="p2",
        mask_schedule="linear",
        block_size=128,
        seed=1,
        data_manifest_id="unbound-local-manifest",
        checkpoint_lineage={
            "base_checkpoint_id": base_checkpoint_id,
            "parent_checkpoint_id": None,
            "conversion_recipe": "qwen-ar-to-masked-diffusion-planned",
        },
        compatibility_contract=default_compatibility_contract(base_checkpoint_id, DEFAULT_TOKENIZER_ID, probe),
        resource_probe=probe,
        benchmark_gate=default_benchmark_gate(),
    )


def default_compatibility_contract(
    base_checkpoint_id: str,
    tokenizer_id: str,
    resource_probe: ResourceProbe,
) -> JsonObject:
    return {
        "schema_version": SCHEMA_VERSION,
        "base_checkpoint_id": base_checkpoint_id,
        "expected_checkpoint_family": base_checkpoint_id.split("/", maxsplit=1)[0],
        "tokenizer_id": tokenizer_id,
        "config_fields": list(CONFIG_FIELDS),
        "tensor_name_mapping_notes": list(TENSOR_MAPPING_NOTES),
        "resource_probe": resource_probe.to_json(),
        "downloads_allowed": False,
        "weights_downloaded": False,
    }


def default_benchmark_gate() -> Mapping[str, BenchmarkGateEntry]:
    return {dataset: BenchmarkGateEntry("eval_only", "blocked", "benchmark dataset is reserved for evaluation and claims") for dataset in BENCHMARK_DATASETS}


def config_from_json(payload: Mapping[str, JsonValue]) -> QwenDiffusionConfig:
    return QwenDiffusionConfig(
        base_checkpoint_id=_string(payload, "base_checkpoint_id"),
        tokenizer_id=_string(payload, "tokenizer_id"),
        attention_mode=_parse_attention_mode(_string(payload, "attention_mode")),
        objective=_parse_objective(_string(payload, "objective")),
        sampler_algorithm=_parse_sampler_algorithm(_string(payload, "sampler_algorithm")),
        mask_schedule=_parse_mask_schedule(_string(payload, "mask_schedule")),
        block_size=_integer(payload, "block_size"),
        seed=_integer(payload, "seed"),
        data_manifest_id=_string(payload, "data_manifest_id"),
        checkpoint_lineage=_mapping(payload, "checkpoint_lineage"),
        compatibility_contract=_mapping(payload, "compatibility_contract"),
        resource_probe=_probe_from_json(_mapping(payload, "resource_probe")),
        benchmark_gate=_benchmark_gate_from_json(_mapping(payload, "benchmark_gate")),
    )


def write_default_config(path: Path, base_checkpoint_id: str) -> None:
    _write_json(path, default_config(base_checkpoint_id).to_json())


def write_compatibility_contract(path: Path, base_checkpoint_id: str) -> None:
    probe = ResourceProbe(status="unknown", detail="offline no-download probe not executed")
    _write_json(path, default_compatibility_contract(base_checkpoint_id, DEFAULT_TOKENIZER_ID, probe))


def write_contamination_probe(path: Path, training_sources: list[str]) -> None:
    config = default_config(DEFAULT_TOKENIZER_ID)
    blocked = tuple(source for source in training_sources if config.training_blocked(source))
    usage: JsonObject = {
        source: (
            "unknown_blocked"
            if config.benchmark_gate.get(source.lower()) is None
            else config.benchmark_gate[source.lower()].usage
        )
        for source in training_sources
    }
    contamination: JsonObject = {
        source: (
            "blocked"
            if config.benchmark_gate.get(source.lower()) is None
            else config.benchmark_gate[source.lower()].contamination_status
        )
        for source in training_sources
    }
    _write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked" if len(blocked) > 0 else "clean",
            "training_sources": list(training_sources),
            "blocked_sources": list(blocked),
            "data_usage": usage,
            "contamination_status": contamination,
        },
    )


def _probe_from_json(payload: Mapping[str, JsonValue]) -> ResourceProbe:
    return ResourceProbe(_parse_resource_status(_string(payload, "status")), _string(payload, "detail"))


def _benchmark_gate_from_json(payload: Mapping[str, JsonValue]) -> Mapping[str, BenchmarkGateEntry]:
    entries: dict[str, BenchmarkGateEntry] = {}
    for source, raw_entry in payload.items():
        if not isinstance(raw_entry, dict):
            raise ConfigValidationError(source, str(raw_entry), "benchmark gate object")
        entries[source] = BenchmarkGateEntry(
            _parse_data_usage(_string(raw_entry, "usage")),
            _parse_contamination_status(_string(raw_entry, "contamination_status")),
            _string(raw_entry, "reason"),
        )
    return entries


def _parse_attention_mode(value: str) -> AttentionMode:
    return _enum("attention_mode", value, ("causal", "annealed", "bidirectional"), "causal, annealed, or bidirectional")


def _parse_objective(value: str) -> Objective:
    return _enum("objective", value, ("masked_ce",), "masked_ce")


def _parse_sampler_algorithm(value: str) -> SamplerAlgorithm:
    return _enum("sampler_algorithm", value, ("p2", "confidence_greedy"), "p2 or confidence_greedy")


def _parse_mask_schedule(value: str) -> MaskSchedule:
    return _enum("mask_schedule", value, ("linear", "cosine"), "linear or cosine")


def _parse_resource_status(value: str) -> ResourceStatus:
    return _enum("resource_probe.status", value, ("available", "missing", "unknown"), "available, missing, or unknown")


def _parse_data_usage(value: str) -> DataUsage:
    return _enum("benchmark_gate.usage", value, ("train_allowed", "eval_only", "unknown_blocked"), "train_allowed, eval_only, or unknown_blocked")


def _parse_contamination_status(value: str) -> ContaminationStatus:
    return _enum("benchmark_gate.contamination_status", value, ("clean", "suspect", "blocked"), "clean, suspect, or blocked")


def _enum(field: str, value: str, allowed: tuple[EnumValue, ...], expected: str) -> EnumValue:
    for candidate in allowed:
        if value == candidate:
            return candidate
    raise ConfigValidationError(field, value, expected)


def _string(payload: Mapping[str, JsonValue], field: str) -> str:
    value = payload[field]
    if not isinstance(value, str):
        raise ConfigValidationError(field, str(value), "string")
    return value


def _integer(payload: Mapping[str, JsonValue], field: str) -> int:
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValidationError(field, str(value), "integer")
    return value


def _mapping(payload: Mapping[str, JsonValue], field: str) -> Mapping[str, JsonValue]:
    value = payload[field]
    if not isinstance(value, dict):
        raise ConfigValidationError(field, str(value), "object")
    return value


def _write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
