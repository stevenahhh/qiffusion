from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias, assert_never

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
TaskType: TypeAlias = Literal["code", "chat", "tool", "repair", "execution_result"]

SCHEMA_VERSION: Final = 1
REQUIRED_PROVENANCE: Final = ("source", "license", "teacher_model", "prompt_hash", "checker_hash", "policy_notes")
TASK_TYPES: Final[tuple[TaskType, ...]] = ("code", "chat", "tool", "repair", "execution_result")


@dataclass(frozen=True, slots=True)
class TeacherRecord:
    source_path: str
    source: str
    license: str
    teacher_model: str
    prompt_hash: str
    checker_hash: str
    task_type: TaskType
    execution_outcome: str
    policy_notes: tuple[str, ...]
    task_name: str
    run: int
    content: str
    code: str
    variant: JsonObject

    def to_json(self) -> JsonObject:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION
        return payload


@dataclass(frozen=True, slots=True)
class TeacherReportError(Exception):
    path: Path
    message: str = "expected JSON object"
    missing_fields: tuple[str, ...] = ()

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


def load_report(path: Path) -> JsonObject:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise TeacherReportError(path, "expected JSON object")
    return data


def teacher_records_from_report(path: Path) -> tuple[TeacherRecord, ...]:
    report = load_report(path)
    results = report.get("task_results")
    if not isinstance(results, list):
        return ()
    records: list[TeacherRecord] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        task_type = parse_task_type(item)
        if task_type is None:
            continue
        name = item.get("name")
        run = item.get("run")
        if not isinstance(name, str):
            continue
        provenance = parse_provenance(path, report, item)
        content, code, variant = trace_parts(task_type, item)
        if content == "":
            continue
        status = item.get("status")
        status = status if isinstance(status, str) and status != "" else item.get("execution_outcome")
        if (not isinstance(status, str) or status == "") and item.get("task_type") is not None:
            raise TeacherReportError(path, "missing required execution outcome: execution_outcome", ("execution_outcome",))
        records.append(
            TeacherRecord(
                source_path=str(path),
                source=provenance.source,
                license=provenance.license,
                teacher_model=provenance.teacher_model,
                prompt_hash=provenance.prompt_hash,
                checker_hash=provenance.checker_hash,
                task_type=task_type,
                execution_outcome=status if isinstance(status, str) and status != "" else "unknown",
                policy_notes=provenance.policy_notes,
                task_name=name,
                run=run if isinstance(run, int) else 0,
                content=content,
                code=code,
                variant=variant,
            )
        )
    return tuple(records)


def export_teacher_jsonl(report_paths: tuple[Path, ...], output: Path) -> int:
    records: list[TeacherRecord] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for path in report_paths:
            records.extend(teacher_records_from_report(path))
    except TeacherReportError as error:
        output.write_text(json.dumps(blocked_record(error), sort_keys=True) + "\n", encoding="utf-8")
        raise
    lines = [json.dumps(record.to_json(), sort_keys=True) for record in records]
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(records)


@dataclass(frozen=True, slots=True)
class TeacherProvenance:
    source: str
    license: str
    teacher_model: str
    prompt_hash: str
    checker_hash: str
    policy_notes: tuple[str, ...]


def parse_task_type(item: JsonObject) -> TaskType | None:
    raw_task_type = item.get("task_type")
    if raw_task_type is None:
        return "code" if item.get("status") == "pass" and isinstance(item.get("generated_code"), str) else None
    if raw_task_type in TASK_TYPES:
        return raw_task_type
    return None


def parse_provenance(path: Path, report: JsonObject, item: JsonObject) -> TeacherProvenance:
    missing: list[str] = []
    for field in REQUIRED_PROVENANCE:
        if field == "policy_notes":
            if len(policy_notes(report, item)) == 0:
                missing.append(field)
            continue
        if required_text(report, item, field) == "":
            missing.append(field)
    if len(missing) > 0:
        raise TeacherReportError(
            path,
            f"missing required teacher provenance: {', '.join(missing)}",
            tuple(missing),
        )
    return TeacherProvenance(
        source=required_text(report, item, "source"),
        license=required_text(report, item, "license"),
        teacher_model=required_text(report, item, "teacher_model"),
        prompt_hash=required_text(report, item, "prompt_hash"),
        checker_hash=required_text(report, item, "checker_hash"),
        policy_notes=policy_notes(report, item),
    )


def required_text(report: JsonObject, item: JsonObject, field: str) -> str:
    value = item.get(field, report.get(field))
    if field == "source" and value is None:
        value = item.get("candidate_source", report.get("candidate_source"))
    if field == "teacher_model" and value is None:
        value = item.get("model_id", report.get("model_id"))
    return value if isinstance(value, str) else ""


def policy_notes(report: JsonObject, item: JsonObject) -> tuple[str, ...]:
    value = item.get("policy_notes", report.get("policy_notes"))
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list):
        return ()
    notes = tuple(note for note in value if isinstance(note, str))
    return notes if len(notes) == len(value) else ()


def trace_parts(task_type: TaskType, item: JsonObject) -> tuple[str, str, JsonObject]:
    match task_type:
        case "code":
            code = text_field(item, "generated_code", "code")
            return code, code, {"generated_code": code}
        case "chat":
            content = chat_content(item)
            messages = item.get("messages")
            return content, "", {"messages": messages if isinstance(messages, list) else []}
        case "tool":
            tool_name = text_field(item, "tool_name")
            tool_input = text_field(item, "tool_input")
            tool_output = text_field(item, "tool_output")
            content = f"tool {tool_name} input: {tool_input}\noutput: {tool_output}".strip()
            return content, "", {"tool_name": tool_name, "tool_input": tool_input, "tool_output": tool_output}
        case "repair":
            code = text_field(item, "after", "generated_code", "code")
            before = text_field(item, "before")
            content = f"before:\n{before}\nafter:\n{code}" if before != "" else code
            return content, code, {"before": before, "after": code}
        case "execution_result":
            content = execution_content(item)
            return content, "", {
                "command": text_field(item, "command"),
                "exit_code": item.get("exit_code"),
                "stdout": text_field(item, "stdout"),
                "stderr": text_field(item, "stderr", "error"),
            }
        case unreachable:
            assert_never(unreachable)


def text_field(item: JsonObject, *names: str) -> str:
    for name in names:
        value = item.get(name)
        if isinstance(value, str):
            return value
    return ""


def chat_content(item: JsonObject) -> str:
    messages = item.get("messages")
    if isinstance(messages, list):
        lines: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if isinstance(role, str) and isinstance(content, str):
                lines.append(f"{role}: {content}")
        return "\n".join(lines)
    return text_field(item, "response", "content")


def execution_content(item: JsonObject) -> str:
    lines: list[str] = []
    command = text_field(item, "command")
    if command != "":
        lines.append(f"command: {command}")
    exit_code = item.get("exit_code")
    if isinstance(exit_code, int):
        lines.append(f"exit_code: {exit_code}")
    stdout = text_field(item, "stdout")
    if stdout != "":
        lines.append(f"stdout: {stdout}")
    stderr = text_field(item, "stderr", "error")
    if stderr != "":
        lines.append(f"stderr: {stderr}")
    return "\n".join(lines)


def blocked_record(error: TeacherReportError) -> JsonObject:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "source_path": str(error.path),
        "reason": str(error),
        "missing_fields": list(error.missing_fields),
        "policy_notes": ["teacher trace rejected before training export"],
        "task_type": "blocked",
        "execution_outcome": "blocked",
    }
