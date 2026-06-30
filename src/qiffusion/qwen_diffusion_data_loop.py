from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, assert_never

from qiffusion.diffusion_corpus_manifest import BENCHMARK_MARKERS
from qiffusion.diffusion_data import ByteTokenizer
from qiffusion.diffusion_teacher_data import TASK_TYPES, JsonObject, JsonValue, TaskType, TeacherProvenance, TeacherRecord
from qiffusion.qwen_tasks import CODING_TASKS, run_task_smoke

SCHEMA_VERSION: Final = 1
ExecutionOutcome = Literal["pass", "fail"]


@dataclass(frozen=True, slots=True)
class QwenDataLoopConfig:
    teacher_jsonl_paths: tuple[Path, ...]
    manifest_path: Path
    output_path: Path
    tokenizer_name: str = "byte"


@dataclass(frozen=True, slots=True)
class DataLoopBlockedError(Exception):
    output_path: Path
    blocked_records: tuple[str, ...]
    reason: str

    def __str__(self) -> str:
        return f"{self.output_path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class LocalRecordParts:
    task_name: str
    task_type: TaskType
    execution_outcome: ExecutionOutcome
    content: str
    code: str
    variant: JsonObject


def write_data_loop_manifest(config: QwenDataLoopConfig) -> JsonObject:
    root, excluded_sources = load_manifest_context(config.manifest_path)
    teacher_records = load_teacher_jsonl(config.teacher_jsonl_paths)
    blocked = tuple(record_id(record) for record in teacher_records if is_benchmark_record(record))
    if len(blocked) > 0:
        raise DataLoopBlockedError(config.output_path, blocked, "benchmark-tagged teacher rows are blocked")
    retained = tuple(
        record
        for record in (*teacher_records, *local_repair_records(), *execution_feedback_records())
        if quality_allowed(record)
    )
    data_path = config.output_path.with_name(f"{config.output_path.stem}.filtered.jsonl")
    data_text = "\n".join(json.dumps(record.to_json(), sort_keys=True) for record in retained)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(data_text + ("\n" if data_text != "" else ""), encoding="utf-8")
    payload = training_manifest(config, root, excluded_sources, data_path, data_text, retained)
    config.output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def write_blocked_report(error: DataLoopBlockedError) -> JsonObject:
    payload: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "reason": error.reason,
        "blocked_records": list(error.blocked_records),
        "records": [],
    }
    error.output_path.parent.mkdir(parents=True, exist_ok=True)
    error.output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_teacher_jsonl(paths: tuple[Path, ...]) -> tuple[TeacherRecord, ...]:
    records: list[TeacherRecord] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line == "":
                continue
            record = teacher_record_from_json(json.loads(line))
            if record is not None:
                records.append(record)
    return tuple(records)


def teacher_record_from_json(payload: JsonValue) -> TeacherRecord | None:
    if not isinstance(payload, dict):
        return None
    raw_task_type = payload.get("task_type")
    if raw_task_type not in TASK_TYPES:
        return None
    provenance = provenance_fields(payload)
    if provenance is None:
        return None
    task_name = text_value(payload, "task_name")
    content = text_value(payload, "content")
    code = text_value(payload, "code")
    outcome = text_value(payload, "execution_outcome")
    run = payload.get("run")
    variant = payload.get("variant")
    if task_name == "" or content == "" or outcome == "":
        return None
    return TeacherRecord(
        source_path=text_value(payload, "source_path"),
        source=provenance.source,
        license=provenance.license,
        teacher_model=provenance.teacher_model,
        prompt_hash=provenance.prompt_hash,
        checker_hash=provenance.checker_hash,
        task_type=raw_task_type,
        execution_outcome=outcome,
        policy_notes=provenance.policy_notes,
        task_name=task_name,
        run=run if isinstance(run, int) else 0,
        content=content,
        code=code,
        variant=variant if isinstance(variant, dict) else {},
    )


def quality_allowed(record: TeacherRecord) -> bool:
    match record.task_type:
        case "code" | "repair":
            return record.execution_outcome == "pass" and record.code != ""
        case "tool":
            return record.execution_outcome == "pass" and record.content != ""
        case "execution_result":
            return record.execution_outcome in ("pass", "fail") and record.content != ""
        case "chat":
            return False
        case unreachable:
            assert_never(unreachable)


def load_manifest_context(path: Path) -> tuple[Path, tuple[str, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return path.parent, ()
    root_value = payload.get("root")
    root = Path(root_value) if isinstance(root_value, str) else path.parent
    records = payload.get("records")
    if not isinstance(records, list):
        return root, ()
    return root, tuple(excluded_manifest_sources(records))


def training_manifest(
    config: QwenDataLoopConfig,
    root: Path,
    excluded_sources: tuple[str, ...],
    data_path: Path,
    data_text: str,
    retained: tuple[TeacherRecord, ...],
) -> JsonObject:
    tokenizer = ByteTokenizer()
    relative_data_path = data_path.name
    digest = hashlib.sha256(data_text.encode("utf-8")).hexdigest()
    record: JsonObject = {
        "source": relative_data_path,
        "source_kind": "local",
        "name": "qwen-data-loop:filtered-training",
        "license": "MIT",
        "split": "train",
        "tokenizer": config.tokenizer_name,
        "token_count": len(tokenizer.encode(data_text)),
        "dedup_hash": digest,
        "usage": "train_allowed",
        "contamination_status": "clean",
        "privacy_policy_notes": "filtered teacher traces, local repairs, and execution feedback; raw user chat logs excluded",
    }
    retained_types: list[JsonValue] = list(sorted({record.task_type for record in retained}))
    data_loop: JsonObject = {
        "filtered_training_file": relative_data_path,
        "retained_records": len(retained),
        "retained_task_types": retained_types,
        "excluded_manifest_sources": list(excluded_sources),
        "input_root": str(root),
        "teacher_jsonl_paths": list(str(path) for path in config.teacher_jsonl_paths),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "manifest_id": f"qwen-data-loop-{digest[:16]}",
        "root": str(data_path.parent),
        "source_manifest": str(config.manifest_path),
        "records": [record],
        "data_loop": data_loop,
    }


def excluded_manifest_sources(records: list[JsonValue]) -> list[str]:
    excluded: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        source = text_value(record, "source")
        if source != "" and (
            text_value(record, "usage") != "train_allowed"
            or text_value(record, "split") != "train"
            or text_value(record, "contamination_status") != "clean"
            or is_benchmark_text(source)
        ):
            excluded.append(source)
    return excluded


def local_repair_records() -> tuple[TeacherRecord, ...]:
    before = "def add(a, b):\n    return a - b\n"
    after = "def add(a, b):\n    return a + b\n"
    parts = LocalRecordParts("local-repair:add", "repair", "pass", f"before:\n{before}after:\n{after}", after, {"before": before, "after": after})
    return (local_record(parts),)


def execution_feedback_records() -> tuple[TeacherRecord, ...]:
    pass_code = "def add(a, b):\n    return a + b\n"
    fail_code = "def add(a, b):\n    return a - b\n"
    return (
        execution_record("local-execution:add-pass", "local-smoke:add-pass", pass_code, "pass"),
        execution_record("local-execution:add-fail", "local-smoke:add-fail", fail_code, "fail"),
    )


def execution_record(task_name: str, command: str, code: str, expected: ExecutionOutcome) -> TeacherRecord:
    ok, message, fixtures = run_task_smoke(code, CODING_TASKS[0])
    outcome: ExecutionOutcome = "pass" if ok else "fail"
    output_name = "stdout" if ok else "stderr"
    content = f"command: {command}\nexit_code: {0 if ok else 1}\n{output_name}: {message}"
    fixture_results: list[JsonValue] = [{"name": result["name"], "status": result["status"], "error": result.get("error", "")} for result in fixtures]
    parts = LocalRecordParts(task_name, "execution_result", outcome, content, "", {"command": command, "expected_outcome": expected, "fixture_results": fixture_results})
    return local_record(parts)


def local_record(parts: LocalRecordParts) -> TeacherRecord:
    digest = sha256_text(f"{parts.task_name}:{parts.content}")
    return TeacherRecord(
        source_path="local:qiffusion.qwen_tasks",
        source="qiffusion-local-verified",
        license="MIT",
        teacher_model="local-deterministic-fixture",
        prompt_hash=f"sha256:{digest}",
        checker_hash=f"sha256:{sha256_text('qiffusion.qwen_tasks.run_task_smoke')}",
        task_type=parts.task_type,
        execution_outcome=parts.execution_outcome,
        policy_notes=("local synthetic repair/execution feedback", "verified by qiffusion.qwen_tasks.run_task_smoke"),
        task_name=parts.task_name,
        run=1,
        content=parts.content,
        code=parts.code,
        variant=parts.variant,
    )


def provenance_fields(payload: JsonObject) -> TeacherProvenance | None:
    notes = payload.get("policy_notes")
    policy_notes = tuple(note for note in notes if isinstance(note, str)) if isinstance(notes, list) else ()
    source = text_value(payload, "source")
    license_name = text_value(payload, "license")
    teacher_model = text_value(payload, "teacher_model")
    prompt_hash = text_value(payload, "prompt_hash")
    checker_hash = text_value(payload, "checker_hash")
    if "" in (source, license_name, teacher_model, prompt_hash, checker_hash) or len(policy_notes) == 0:
        return None
    return TeacherProvenance(source, license_name, teacher_model, prompt_hash, checker_hash, policy_notes)


def is_benchmark_record(record: TeacherRecord) -> bool:
    haystack = " ".join((record.source, record.source_path, record.task_name, record.content, record.code, " ".join(record.policy_notes)))
    return is_benchmark_text(haystack)


def is_benchmark_text(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in BENCHMARK_MARKERS)


def record_id(record: TeacherRecord) -> str:
    return f"{record.source}:{record.task_name}"


def text_value(payload: JsonObject, field: str) -> str:
    value = payload.get(field)
    return value if isinstance(value, str) else ""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
