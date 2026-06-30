from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
REQUIRED_QWEN_BUCKETS: Final = (
    "local_code_smoke",
    "external_benchmark_readiness",
    "chat",
    "tool_agent",
    "software_engineering",
)
REQUIRED_QWEN_CLAIMS: Final = (
    "coding_capability_claim",
    "chat_capability_claim",
    "tool_agent_capability_claim",
    "software_engineering_capability_claim",
    "release_capability_claim",
)
REQUIRED_LINEAGE_FIELDS: Final = (
    "checkpoint_id",
    "base_checkpoint_id",
    "data_manifest_id",
    "objective",
    "mask_schedule",
    "tokenizer_id",
)


@dataclass(frozen=True, slots=True)
class GateDecision:
    status: str
    reason: str


@dataclass(frozen=True, slots=True)
class DecisionReportError(Exception):
    path: Path
    reason: str

    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"


def load_json(path: Path) -> JsonObject:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise DecisionReportError(path, "expected JSON object")
    return data


def decide_coding_capable(report: JsonObject) -> GateDecision:
    best = _mapping(report.get("best"))
    candidate = best if best is not None else report
    if _is_qwen_diffusion_report(candidate):
        return _decide_qwen_release(candidate)
    if candidate.get("fixtures_status") != "pass":
        return GateDecision("continue", "fixtures are not passing")
    if candidate.get("code_smoke_status") != "pass":
        return GateDecision("continue", "code smoke is not passing")
    if candidate.get("coding_capability_claim") is not True:
        return GateDecision("continue", "coding capability claim is not true")
    return GateDecision("promote", "shared coding-capable gate passed")


def decide_from_file(path: Path) -> GateDecision:
    return decide_coding_capable(load_json(path))


def _decide_qwen_release(report: JsonObject) -> GateDecision:
    if report.get("fallback_used") is not False:
        return GateDecision("blocked", "hidden fallback was used")
    contamination = _mapping(report.get("benchmark_contamination"))
    if contamination is None:
        return GateDecision("blocked", "benchmark contamination evidence is missing")
    if contamination.get("status") != "clean" or not _has_text(contamination.get("evidence")):
        return GateDecision("blocked", "benchmark contamination is not clean")
    lineage = _mapping(report.get("checkpoint_lineage"))
    if lineage is None:
        return GateDecision("blocked", "checkpoint lineage is missing")
    missing_lineage = tuple(field for field in REQUIRED_LINEAGE_FIELDS if not _has_text(lineage.get(field)))
    if len(missing_lineage) > 0:
        return GateDecision("blocked", f"checkpoint lineage is incomplete: {', '.join(missing_lineage)}")
    buckets = _mapping(report.get("buckets"))
    if buckets is None:
        return GateDecision("continue", "qwen eval buckets are missing")
    for name in REQUIRED_QWEN_BUCKETS:
        bucket = _mapping(buckets.get(name))
        if bucket is None:
            return GateDecision("continue", f"{name} bucket is missing")
        if bucket.get("status") != "pass":
            return GateDecision("continue", f"{name} bucket has not passed")
        if bucket.get("capability_claim") is not True:
            return GateDecision("continue", f"{name} bucket capability claim is not true")
        if not _has_text(bucket.get("evidence")):
            return GateDecision("continue", f"{name} bucket evidence is missing")
    for claim in REQUIRED_QWEN_CLAIMS:
        if report.get(claim) is not True:
            return GateDecision("continue", f"{claim} is not true")
    return GateDecision("complete", "qwen diffusion release gate passed")


def _is_qwen_diffusion_report(candidate: JsonValue) -> bool:
    if not isinstance(candidate, dict):
        return False
    return candidate.get("backend") == "qwen_token_diffusion" or isinstance(candidate.get("buckets"), dict)


def _mapping(value: JsonValue) -> JsonObject | None:
    if not isinstance(value, dict):
        return None
    return value


def _has_text(value: JsonValue) -> bool:
    return isinstance(value, str) and value.strip() != ""
