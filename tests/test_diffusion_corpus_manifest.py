from __future__ import annotations

import json
from pathlib import Path

from qiffusion.cli import main
from qiffusion.diffusion_corpus_manifest import (
    ManifestBuildConfig,
    build_manifest,
)


def test_diffusion_corpus_manifest_cli_writes_provenance_records(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "README.md").write_text("# Fixture\n\nlocal corpus text\n", encoding="utf-8")
    output = tmp_path / "manifest.json"

    exit_code = main(["diffusion-corpus", "manifest", "--root", str(root), "--out", str(output)])

    assert exit_code == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    records = manifest["records"]
    assert records
    first = records[0]
    assert first["source"]
    assert first["license"] == "unknown"
    assert first["split"] == "train"
    assert first["tokenizer"] == "byte"
    assert first["token_count"] > 0
    assert first["dedup_hash"]
    assert first["usage"] == "unknown_blocked"
    assert first["contamination_status"] == "clean"
    assert first["privacy_policy_notes"]


def test_diffusion_corpus_manifest_cli_rejects_malformed_teacher_jsonl(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    malformed = tmp_path / "teacher.jsonl"
    malformed.write_text("{bad json\n", encoding="utf-8")
    output = tmp_path / "manifest-failure.json"

    exit_code = main(
        [
            "diffusion-corpus",
            "manifest",
            "--root",
            str(root),
            "--teacher-jsonl",
            str(malformed),
            "--out",
            str(output),
        ]
    )

    assert exit_code == 2
    failure = json.loads(output.read_text(encoding="utf-8"))
    assert failure["status"] == "error"
    assert failure["error"] == "malformed_teacher_jsonl"
    assert failure["path"] == str(malformed)


def test_diffusion_corpus_manifest_includes_teacher_jsonl_records(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    teacher_jsonl = tmp_path / "teacher.jsonl"
    teacher_jsonl.write_text(
        json.dumps({"task_name": "add", "code": "def add(a, b):\n    return a + b\n"}) + "\n",
        encoding="utf-8",
    )

    manifest = build_manifest(ManifestBuildConfig(root=root, teacher_jsonl_paths=(teacher_jsonl,)))

    teacher_records = [record for record in manifest["records"] if record["source_kind"] == "teacher_jsonl"]
    assert len(teacher_records) == 1
    assert teacher_records[0]["name"] == "teacher:add"
    assert teacher_records[0]["usage"] == "unknown_blocked"


def test_diffusion_corpus_manifest_keeps_benchmark_sources_non_trainable(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    benchmark = root / "HumanEval.jsonl"
    root.mkdir()
    benchmark.write_text(
        json.dumps({"task_name": "humaneval_add", "code": "def add(a, b):\n    return a + b\n"}) + "\n",
        encoding="utf-8",
    )

    manifest = build_manifest(ManifestBuildConfig(root=root, teacher_jsonl_paths=(benchmark,)))

    benchmark_records = [record for record in manifest["records"] if record["source"] == str(benchmark)]
    assert len(benchmark_records) == 1
    assert benchmark_records[0]["usage"] == "eval_only"
    assert benchmark_records[0]["contamination_status"] == "blocked"
