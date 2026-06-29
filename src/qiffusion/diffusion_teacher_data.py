from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, TypedDict

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


class TeacherRecordJson(TypedDict):
    source_path: str
    task_name: str
    run: int
    code: str


@dataclass(frozen=True, slots=True)
class TeacherRecord:
    source_path: str
    task_name: str
    run: int
    code: str

    def to_json(self) -> TeacherRecordJson:
        return {
            "source_path": self.source_path,
            "task_name": self.task_name,
            "run": self.run,
            "code": self.code,
        }


@dataclass(frozen=True, slots=True)
class TeacherReportError(Exception):
    path: Path

    def __str__(self) -> str:
        return f"{self.path}: expected JSON object"


def load_report(path: Path) -> JsonObject:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise TeacherReportError(path)
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
        if item.get("status") != "pass":
            continue
        code = item.get("generated_code")
        name = item.get("name")
        run = item.get("run")
        if not isinstance(code, str) or not isinstance(name, str):
            continue
        records.append(
            TeacherRecord(
                source_path=str(path),
                task_name=name,
                run=run if isinstance(run, int) else 0,
                code=code,
            )
        )
    return tuple(records)


def export_teacher_jsonl(report_paths: tuple[Path, ...], output: Path) -> int:
    records: list[TeacherRecord] = []
    for path in report_paths:
        records.extend(teacher_records_from_report(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record.to_json(), sort_keys=True) for record in records]
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(records)

