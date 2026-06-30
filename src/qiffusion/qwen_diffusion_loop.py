from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias, TypedDict, assert_never

from qiffusion.qwen_diffusion_config import JsonObject, JsonValue, config_from_json
from qiffusion.qwen_diffusion_eval import (
    BucketStatus,
    EvalBucket,
    QwenDiffusionEvalConfig,
    QwenDiffusionEvalReport,
    eval_qwen_diffusion_checkpoint,
)
from qiffusion.qwen_diffusion_model import load_qwen_denoiser_checkpoint
from qiffusion.qwen_diffusion_sample import QwenMaskSampleConfig, QwenMaskSampleReport, QwenSamplerSettings, sample_qwen_tokens
from qiffusion.qwen_diffusion_train import QwenDiffusionTrainConfig, train_qwen_diffusion

SCHEMA_VERSION: Final = 1
LoopStatus: TypeAlias = Literal["complete", "max_iterations_reached", "resource_blocked"]
NextActionName: TypeAlias = Literal["complete", "train_again", "resolve_resource_blocker", "max_iterations_reached"]
REQUIRED_BUCKETS: Final = ("local_code_smoke", "external_benchmark_readiness", "chat", "tool_agent", "software_engineering")


class LoopDiagnosis(TypedDict):
    status: str
    code_gate_status: BucketStatus
    chat_gate_status: BucketStatus
    tool_agent_status: BucketStatus
    software_engineering_status: BucketStatus
    blockers: list[str]


class LoopNextAction(TypedDict):
    action: NextActionName
    reason: str
    continue_loop: bool
    capability_claim: bool


class LoopCleanup(TypedDict):
    status: str
    temp_paths_removed: list[str]
    background_resources: str
    artifacts: list[str]
    notes: str


class LoopLedgerEntry(TypedDict):
    schema_version: int
    iteration: int
    train: JsonObject
    sample: QwenMaskSampleReport
    eval: QwenDiffusionEvalReport
    diagnosis: LoopDiagnosis
    next_action: LoopNextAction
    cleanup: LoopCleanup


class LoopRunReport(TypedDict):
    schema_version: int
    status: LoopStatus
    iterations: int
    ledger_path: str
    capability_claim: bool
    final_next_action: LoopNextAction


@dataclass(frozen=True, slots=True)
class QwenDiffusionLoopConfig:
    manifest_path: Path
    config_path: Path
    ledger_path: Path
    max_iterations: int
    force_eval_fail: bool = False


def run_qwen_diffusion_loop(config: QwenDiffusionLoopConfig) -> LoopRunReport:
    qwen_config = config_from_json(_load_json(config.config_path))
    tokenizer_id = _manifest_tokenizer(config.manifest_path)
    config.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    config.ledger_path.write_text("", encoding="utf-8")
    final_action = _next_action_from_name("max_iterations_reached", "loop did not run")
    final_status: LoopStatus = "max_iterations_reached"
    completed_iterations = 0
    for iteration in range(1, config.max_iterations + 1):
        checkpoint = config.ledger_path.with_name(f"{config.ledger_path.stem}-iteration-{iteration}.pt")
        train = train_qwen_diffusion(
            QwenDiffusionTrainConfig(
                manifest_path=config.manifest_path,
                checkpoint_path=checkpoint,
                tokenizer_id=tokenizer_id,
                steps=2,
                seed=qwen_config.seed + iteration - 1,
                sequence_length=min(qwen_config.block_size, 64),
            ),
        )
        sample = _sample_checkpoint(checkpoint, qwen_config.seed + iteration - 1)
        eval_report = eval_qwen_diffusion_checkpoint(
            QwenDiffusionEvalConfig(
                checkpoint_path=checkpoint,
                sample_out=config.ledger_path.with_name(f"{config.ledger_path.stem}-iteration-{iteration}-eval-sample.json"),
                runs=1,
                seed=qwen_config.seed + iteration - 1,
            ),
        )
        if config.force_eval_fail:
            eval_report = _force_failed_eval(eval_report)
        diagnosis = diagnose(eval_report)
        final_action = plan_next_action(diagnosis, iteration, config.max_iterations)
        entry = LoopLedgerEntry(
            schema_version=SCHEMA_VERSION,
            iteration=iteration,
            train=train,
            sample=sample,
            eval=eval_report,
            diagnosis=diagnosis,
            next_action=final_action,
            cleanup=cleanup_status(checkpoint, config.ledger_path),
        )
        _append_ledger(config.ledger_path, entry)
        completed_iterations = iteration
        if not final_action["continue_loop"]:
            final_status = _status_for_action(final_action["action"])
            break
    return {
        "schema_version": SCHEMA_VERSION,
        "status": final_status,
        "iterations": completed_iterations,
        "ledger_path": str(config.ledger_path),
        "capability_claim": final_action["capability_claim"],
        "final_next_action": final_action,
    }


def diagnose(report: QwenDiffusionEvalReport) -> LoopDiagnosis:
    blockers = [name for name in REQUIRED_BUCKETS if _bucket_status(report, name) != "pass"]
    return {
        "status": "pass" if len(blockers) == 0 else "blocked",
        "code_gate_status": _bucket_status(report, "local_code_smoke"),
        "chat_gate_status": _bucket_status(report, "chat"),
        "tool_agent_status": _bucket_status(report, "tool_agent"),
        "software_engineering_status": _bucket_status(report, "software_engineering"),
        "blockers": blockers,
    }


def plan_next_action(diagnosis: LoopDiagnosis, iteration: int, max_iterations: int) -> LoopNextAction:
    if len(diagnosis["blockers"]) == 0:
        return _next_action_from_name("complete", "all eval buckets passed")
    if _has_resource_blocker(diagnosis):
        return _next_action_from_name("resolve_resource_blocker", ", ".join(diagnosis["blockers"]))
    if iteration >= max_iterations:
        return _next_action_from_name("max_iterations_reached", ", ".join(diagnosis["blockers"]))
    return _next_action_from_name("train_again", ", ".join(diagnosis["blockers"]))


def cleanup_status(checkpoint: Path, ledger_path: Path) -> LoopCleanup:
    return {
        "status": "clean",
        "temp_paths_removed": [],
        "background_resources": "none",
        "artifacts": [str(checkpoint), str(ledger_path)],
        "notes": "no temp dirs, downloads, subprocesses, or background resources created",
    }


def _sample_checkpoint(checkpoint: Path, seed: int) -> QwenMaskSampleReport:
    model, _manifest = load_qwen_denoiser_checkpoint(checkpoint)
    return sample_qwen_tokens(
        model,
        QwenMaskSampleConfig(
            prompt="def add",
            settings=QwenSamplerSettings(steps=4, seed=seed),
        ),
    )


def _force_failed_eval(report: QwenDiffusionEvalReport) -> QwenDiffusionEvalReport:
    buckets: dict[str, EvalBucket] = {
        name: {
            "status": bucket["status"],
            "capability_claim": bucket["capability_claim"],
            "evidence": bucket["evidence"],
            "detail": bucket["detail"],
        }
        for name, bucket in report["buckets"].items()
    }
    local_code = buckets["local_code_smoke"]
    local_code["status"] = "fail"
    local_code["capability_claim"] = False
    local_code["detail"] = "forced eval failure for loop QA"
    updated = report.copy()
    updated["buckets"] = buckets
    updated["code_smoke_status"] = "fail"
    updated["local_code_capability_claim"] = False
    updated["coding_capability_claim"] = False
    updated["release_capability_claim"] = False
    updated["smoke_error"] = "forced eval failure for loop QA"
    return updated


def _next_action_from_name(action: NextActionName, reason: str) -> LoopNextAction:
    match action:
        case "complete":
            return {"action": action, "reason": reason, "continue_loop": False, "capability_claim": True}
        case "train_again":
            return {"action": action, "reason": reason, "continue_loop": True, "capability_claim": False}
        case "resolve_resource_blocker" | "max_iterations_reached":
            return {"action": action, "reason": reason, "continue_loop": False, "capability_claim": False}
        case unreachable:
            assert_never(unreachable)


def _status_for_action(action: NextActionName) -> LoopStatus:
    match action:
        case "complete":
            return "complete"
        case "resolve_resource_blocker":
            return "resource_blocked"
        case "train_again" | "max_iterations_reached":
            return "max_iterations_reached"
        case unreachable:
            assert_never(unreachable)


def _has_resource_blocker(diagnosis: LoopDiagnosis) -> bool:
    return any(blocker.startswith("resource:") for blocker in diagnosis["blockers"])


def _bucket_status(report: QwenDiffusionEvalReport, name: str) -> BucketStatus:
    status = report["buckets"][name]["status"]
    for candidate in ("not_run", "blocked", "fail", "pass"):
        if status == candidate:
            return candidate
    return "blocked"


def _manifest_tokenizer(path: Path) -> str:
    payload = _load_json(path)
    records = payload["records"]
    if not isinstance(records, list):
        return "byte"
    for record in records:
        if isinstance(record, dict):
            tokenizer = record.get("tokenizer")
            if isinstance(tokenizer, str):
                return tokenizer
    return "byte"


def _load_json(path: Path) -> JsonObject:
    payload: JsonValue = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _append_ledger(path: Path, entry: LoopLedgerEntry) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
