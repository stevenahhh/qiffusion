from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias

import pytest

from qiffusion.cli import main
from qiffusion.qwen_diffusion_data_loop import DataLoopBlockedError, QwenDataLoopConfig, write_data_loop_manifest

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


def test_data_loop_retains_pass_fail_execution_records_with_provenance(tmp_path: Path) -> None:
    teacher_jsonl = tmp_path / "teacher.jsonl"
    manifest = _write_manifest(tmp_path)
    output = tmp_path / "data-loop.json"
    _write_teacher_jsonl(
        teacher_jsonl,
        [
            _teacher_record("teacher-code", "code", "pass", code="def add(a, b):\n    return a + b\n"),
            _teacher_record(
                "teacher-repair",
                "repair",
                "pass",
                code="def add(a, b):\n    return a + b\n",
                content="before:\ndef add(a, b):\n    return a - b\nafter:\ndef add(a, b):\n    return a + b\n",
            ),
            _teacher_record("teacher-exec-pass", "execution_result", "pass", content="command: pytest\nexit_code: 0\nstdout: passed"),
            _teacher_record("teacher-exec-fail", "execution_result", "fail", content="command: pytest\nexit_code: 1\nstderr: failed"),
            _teacher_record("raw-chat", "chat", "pass", content="user: private chat\nassistant: reply"),
        ],
    )

    report = write_data_loop_manifest(QwenDataLoopConfig((teacher_jsonl,), manifest, output))

    data_loop = report["data_loop"]
    assert isinstance(data_loop, dict)
    filtered_file = data_loop["filtered_training_file"]
    assert isinstance(filtered_file, str)
    data_path = tmp_path / filtered_file
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()]
    execution_rows = [row for row in rows if row["task_type"] == "execution_result" and str(row["task_name"]).startswith("teacher-exec")]
    assert [row["execution_outcome"] for row in execution_rows] == ["pass", "fail"]
    assert {row["task_name"] for row in rows}.isdisjoint({"raw-chat"})
    for row in execution_rows:
        assert row["source"] == "local-teacher"
        assert row["license"] == "MIT"
        assert row["teacher_model"] == "Qwen/Qwen3.5-4B"
        assert row["prompt_hash"] == "sha256:prompt"
        assert row["checker_hash"] == "sha256:checker"
        assert row["policy_notes"] == ["synthetic test fixture"]


def test_data_loop_blocks_benchmark_tagged_teacher_rows(tmp_path: Path) -> None:
    teacher_jsonl = tmp_path / "benchmark-teacher.jsonl"
    manifest = _write_manifest(tmp_path)
    output = tmp_path / "data-loop.json"
    _write_teacher_jsonl(
        teacher_jsonl,
        [_teacher_record("humaneval-add", "code", "pass", source="HumanEval", code="def add(a, b):\n    return a + b\n")],
    )

    with pytest.raises(DataLoopBlockedError) as exc_info:
        write_data_loop_manifest(QwenDataLoopConfig((teacher_jsonl,), manifest, output))

    assert exc_info.value.blocked_records == ("HumanEval:humaneval-add",)


def test_data_loop_cli_writes_train_allowed_manifest_pointing_to_filtered_file(tmp_path: Path) -> None:
    teacher_jsonl = tmp_path / "teacher.jsonl"
    manifest = _write_manifest(tmp_path, include_eval=True)
    output = tmp_path / "data-loop.json"
    _write_teacher_jsonl(
        teacher_jsonl,
        [_teacher_record("teacher-code", "code", "pass", code="def add(a, b):\n    return a + b\n")],
    )

    exit_code = main(["qwen-diffusion-data-loop", "--teacher-jsonl", str(teacher_jsonl), "--manifest", str(manifest), "--out", str(output)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    data_path = tmp_path / payload["records"][0]["source"]
    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["records"][0]["usage"] == "train_allowed"
    assert payload["records"][0]["split"] == "train"
    assert payload["records"][0]["contamination_status"] == "clean"
    assert payload["records"][0]["source"] == payload["data_loop"]["filtered_training_file"]
    assert data_path.is_file()
    assert payload["data_loop"]["excluded_manifest_sources"] == ["HumanEval.jsonl"]
    assert {record["usage"] for record in payload["records"]} == {"train_allowed"}


def _write_manifest(tmp_path: Path, *, include_eval: bool = False) -> Path:
    records: list[JsonObject] = [
        {
            "source": "local.py",
            "source_kind": "local",
            "name": "local",
            "license": "MIT",
            "split": "train",
            "tokenizer": "byte",
            "token_count": 4,
            "dedup_hash": "abc",
            "usage": "train_allowed",
            "contamination_status": "clean",
            "privacy_policy_notes": "local fixture",
        }
    ]
    if include_eval:
        records.append(
            {
                "source": "HumanEval.jsonl",
                "source_kind": "local",
                "name": "humaneval",
                "license": "MIT",
                "split": "eval",
                "tokenizer": "byte",
                "token_count": 4,
                "dedup_hash": "def",
                "usage": "eval_only",
                "contamination_status": "blocked",
                "privacy_policy_notes": "benchmark fixture",
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "status": "ok", "root": str(tmp_path), "records": records}), encoding="utf-8")
    return manifest


def _write_teacher_jsonl(path: Path, records: list[JsonObject]) -> None:
    path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")


def _teacher_record(
    task_name: str,
    task_type: str,
    execution_outcome: str,
    *,
    source: str = "local-teacher",
    code: str = "",
    content: str = "",
) -> JsonObject:
    return {
        "schema_version": 1,
        "source_path": "teacher-report.json",
        "source": source,
        "license": "MIT",
        "teacher_model": "Qwen/Qwen3.5-4B",
        "prompt_hash": "sha256:prompt",
        "checker_hash": "sha256:checker",
        "task_type": task_type,
        "execution_outcome": execution_outcome,
        "policy_notes": ["synthetic test fixture"],
        "task_name": task_name,
        "run": 1,
        "content": content if content != "" else code,
        "code": code,
        "variant": {"content": content, "code": code},
    }
