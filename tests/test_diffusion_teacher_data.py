from __future__ import annotations

import json
from pathlib import Path

from qiffusion.cli import main
from qiffusion.diffusion_teacher_data import export_teacher_jsonl, teacher_records_from_report


def test_teacher_records_include_only_passing_generated_code(tmp_path: Path) -> None:
    report = tmp_path / "eval.json"
    report.write_text(
        json.dumps(
            {
                "task_results": [
                    {
                        "name": "add",
                        "run": 1,
                        "status": "pass",
                        "generated_code": "def add(a, b):\n    return a + b\n",
                    },
                    {
                        "name": "broken",
                        "run": 1,
                        "status": "fail",
                        "generated_code": "def broken():\n    raise RuntimeError()\n",
                    },
                    {"name": "missing_code", "run": 1, "status": "pass"},
                ]
            }
        ),
        encoding="utf-8",
    )

    records = teacher_records_from_report(report)

    assert len(records) == 1
    assert records[0].task_name == "add"
    assert records[0].run == 1
    assert records[0].code.startswith("def add")
    assert records[0].source_path == str(report)


def test_export_teacher_jsonl_writes_deterministic_records(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output = tmp_path / "teacher.jsonl"
    first.write_text(
        json.dumps(
            {
                "task_results": [
                    {"name": "add", "run": 2, "status": "pass", "generated_code": "def add():\n    return 1\n"}
                ]
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "task_results": [
                    {"name": "skip", "run": 1, "status": "fail", "generated_code": "def skip():\n    return 0\n"},
                    {
                        "name": "slugify_title",
                        "run": 1,
                        "status": "pass",
                        "generated_code": "def slugify_title(title):\n    return '-'.join(title.split())\n",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    count = export_teacher_jsonl((second, first), output)

    lines = output.read_text(encoding="utf-8").splitlines()
    payloads = [json.loads(line) for line in lines]
    assert count == 2
    assert [payload["task_name"] for payload in payloads] == ["slugify_title", "add"]
    assert payloads[0]["source_path"] == str(second)


def test_diffusion_export_teacher_cli_writes_jsonl(tmp_path: Path) -> None:
    report = tmp_path / "eval.json"
    output = tmp_path / "teacher.jsonl"
    report.write_text(
        json.dumps(
            {
                "task_results": [
                    {
                        "name": "slugify_title",
                        "run": 1,
                        "status": "pass",
                        "generated_code": "def slugify_title(title):\n    return '-'.join(title.split())\n",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["diffusion-export-teacher", "--qwen-report", str(report), "--out", str(output)])

    assert exit_code == 0
    assert "slugify_title" in output.read_text(encoding="utf-8")
