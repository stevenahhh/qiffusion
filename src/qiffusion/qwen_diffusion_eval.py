from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias, TypedDict

from qiffusion.qwen_bridge import FixtureResult
from qiffusion.qwen_diffusion_config import JsonObject, JsonValue
from qiffusion.qwen_diffusion_model import load_qwen_denoiser_checkpoint
from qiffusion.qwen_diffusion_sample import (
    QwenMaskSampleConfig,
    QwenMaskSampleReport,
    QwenSamplerSettings,
    sample_qwen_tokens,
)
from qiffusion.qwen_tasks import CODING_TASKS, run_task_smoke

SCHEMA_VERSION: Final = 1
BucketStatus = Literal["not_run", "blocked", "fail", "pass"]
BucketName = Literal[
    "local_code_smoke",
    "external_benchmark_readiness",
    "chat",
    "tool_agent",
    "software_engineering",
]
REQUIRED_BUCKETS: Final[tuple[BucketName, ...]] = (
    "local_code_smoke",
    "external_benchmark_readiness",
    "chat",
    "tool_agent",
    "software_engineering",
)


class EvalBucket(TypedDict):
    status: BucketStatus
    capability_claim: bool
    evidence: str
    detail: str


class QwenDiffusionEvalReport(TypedDict):
    schema_version: int
    backend: str
    stage: str
    status: str
    checkpoint_path: str
    runs: int
    fixtures_status: str
    code_smoke_status: str
    candidate_source: str
    fallback_used: bool
    local_code_capability_claim: bool
    coding_capability_claim: bool
    chat_capability_claim: bool
    tool_agent_capability_claim: bool
    software_engineering_capability_claim: bool
    release_capability_claim: bool
    buckets: dict[str, EvalBucket]
    samples: list[QwenMaskSampleReport]
    fixture_results: list[FixtureResult]
    smoke_error: str


class MultiSampleReport(TypedDict):
    schema_version: int
    samples: list[QwenMaskSampleReport]


JsonWritePayload: TypeAlias = JsonObject | QwenMaskSampleReport | QwenDiffusionEvalReport | MultiSampleReport
RawEvalReport: TypeAlias = JsonObject | QwenDiffusionEvalReport
RawBuckets: TypeAlias = dict[str, JsonValue] | dict[str, EvalBucket]
RawBucket: TypeAlias = JsonObject | EvalBucket


@dataclass(frozen=True, slots=True)
class QwenDiffusionEvalConfig:
    checkpoint_path: Path
    sample_out: Path
    runs: int
    seed: int = 1
    prompt: str = "def add"
    sample_steps: int = 8


@dataclass(frozen=True, slots=True)
class EvalRunResult:
    samples: list[QwenMaskSampleReport]
    fixture_results: list[FixtureResult]
    smoke_errors: list[str]
    local_code_status: BucketStatus


@dataclass(frozen=True, slots=True)
class ReportValidationError(Exception):
    field: str
    message: str

    def __str__(self) -> str:
        return f"{self.field}: {self.message}"


def eval_qwen_diffusion_checkpoint(config: QwenDiffusionEvalConfig) -> QwenDiffusionEvalReport:
    model, _manifest = load_qwen_denoiser_checkpoint(config.checkpoint_path)
    samples: list[QwenMaskSampleReport] = []
    fixture_results: list[FixtureResult] = []
    smoke_errors: list[str] = []
    for run in range(config.runs):
        sample = sample_qwen_tokens(
            model,
            QwenMaskSampleConfig(
                prompt=config.prompt,
                settings=QwenSamplerSettings(steps=config.sample_steps, seed=config.seed + run),
            ),
        )
        samples.append(sample)
        smoke_ok, message, fixtures = run_task_smoke(sample["generated_text"], CODING_TASKS[0])
        fixture_results.extend(fixtures)
        if not smoke_ok:
            smoke_errors.append(message)
    _write_json(config.sample_out, samples[0] if config.runs == 1 else {"schema_version": SCHEMA_VERSION, "samples": samples})
    local_code_status: BucketStatus = "pass" if len(smoke_errors) == 0 else "fail"
    report = _report(config, EvalRunResult(samples, fixture_results, smoke_errors, local_code_status))
    validate_report(report)
    return report


def validate_report(report: RawEvalReport) -> None:
    buckets = _buckets(report)
    for name in REQUIRED_BUCKETS:
        _ = _bucket_status(buckets, name)
        if _bucket_claim(buckets, name) and _bucket_status(buckets, name) != "pass":
            raise ReportValidationError(name, "bucket claim requires pass status")
    _claim_requires(report, "local_code_capability_claim", buckets, ("local_code_smoke",))
    _claim_requires(
        report,
        "coding_capability_claim",
        buckets,
        ("local_code_smoke", "external_benchmark_readiness", "software_engineering"),
    )
    _claim_requires(report, "chat_capability_claim", buckets, ("chat",))
    _claim_requires(report, "tool_agent_capability_claim", buckets, ("tool_agent",))
    _claim_requires(report, "software_engineering_capability_claim", buckets, ("software_engineering",))
    _claim_requires(report, "release_capability_claim", buckets, REQUIRED_BUCKETS)


def validate_report_file(path: Path, out: Path) -> bool:
    report = _load_json(path)
    try:
        validate_report(report)
    except ReportValidationError as exc:
        _write_json(out, {"schema_version": SCHEMA_VERSION, "status": "rejected", "error": str(exc), "input": str(path)})
        return False
    _write_json(out, {"schema_version": SCHEMA_VERSION, "status": "accepted", "input": str(path)})
    return True


def _report(
    config: QwenDiffusionEvalConfig,
    result: EvalRunResult,
) -> QwenDiffusionEvalReport:
    smoke_error = "; ".join(result.smoke_errors)
    buckets = _initial_buckets(result.local_code_status, smoke_error)
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": "qwen_token_diffusion",
        "stage": "eval",
        "status": "evaluated",
        "checkpoint_path": str(config.checkpoint_path),
        "runs": config.runs,
        "fixtures_status": "pass",
        "code_smoke_status": result.local_code_status,
        "candidate_source": "qwen-token-diffusion-checkpoint",
        "fallback_used": False,
        "local_code_capability_claim": result.local_code_status == "pass",
        "coding_capability_claim": False,
        "chat_capability_claim": False,
        "tool_agent_capability_claim": False,
        "software_engineering_capability_claim": False,
        "release_capability_claim": False,
        "buckets": buckets,
        "samples": result.samples,
        "fixture_results": result.fixture_results,
        "smoke_error": smoke_error,
    }


def _initial_buckets(local_code_status: BucketStatus, smoke_error: str) -> dict[str, EvalBucket]:
    return {
        "local_code_smoke": _bucket(local_code_status, local_code_status == "pass", "local smoke fixture", smoke_error),
        "external_benchmark_readiness": _bucket("not_run", False, "benchmark harness", "benchmark readiness not run in Todo 8"),
        "chat": _bucket("blocked", False, "chat harness", "chat eval harness is not implemented in Todo 8"),
        "tool_agent": _bucket("blocked", False, "tool/agent harness", "tool/agent eval harness is not implemented in Todo 8"),
        "software_engineering": _bucket("blocked", False, "software-engineering harness", "SWE harness is not implemented in Todo 8"),
    }


def _bucket(status: BucketStatus, capability_claim: bool, evidence: str, detail: str) -> EvalBucket:
    return {"status": status, "capability_claim": capability_claim, "evidence": evidence, "detail": detail}


def _claim_requires(
    report: RawEvalReport,
    claim_field: str,
    buckets: RawBuckets,
    required: tuple[BucketName, ...],
) -> None:
    claim = report.get(claim_field)
    if not isinstance(claim, bool):
        raise ReportValidationError(claim_field, "expected boolean")
    fallback_used = report.get("fallback_used")
    if not isinstance(fallback_used, bool):
        raise ReportValidationError("fallback_used", "expected boolean")
    if not claim:
        return
    if fallback_used:
        raise ReportValidationError(claim_field, "fallback_used must be false")
    for name in required:
        if _bucket_status(buckets, name) != "pass" or not _bucket_claim(buckets, name):
            raise ReportValidationError(claim_field, f"{name} bucket has not passed")


def _buckets(report: RawEvalReport) -> RawBuckets:
    raw = report.get("buckets")
    if not isinstance(raw, dict):
        raise ReportValidationError("buckets", "expected object")
    return raw


def _bucket_status(buckets: RawBuckets, name: BucketName) -> BucketStatus:
    bucket = _bucket_payload(buckets, name)
    status = bucket.get("status")
    for candidate in ("not_run", "blocked", "fail", "pass"):
        if status == candidate:
            return candidate
    raise ReportValidationError(name, "status must be not_run, blocked, fail, or pass")


def _bucket_claim(buckets: RawBuckets, name: BucketName) -> bool:
    bucket = _bucket_payload(buckets, name)
    claim = bucket.get("capability_claim")
    if not isinstance(claim, bool):
        raise ReportValidationError(name, "capability_claim must be boolean")
    return claim


def _bucket_payload(buckets: RawBuckets, name: BucketName) -> RawBucket:
    bucket = buckets.get(name)
    if not isinstance(bucket, dict):
        raise ReportValidationError(name, "bucket is required")
    return bucket


def _load_json(path: Path) -> JsonObject:
    payload: JsonValue = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ReportValidationError(str(path), "expected JSON object")
    return payload


def _write_json(path: Path, payload: JsonWritePayload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
