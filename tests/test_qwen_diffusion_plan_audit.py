from __future__ import annotations

import json
from pathlib import Path

from qiffusion.cli import main


def test_plan_audit_accepts_equivalent_evidence_without_exact_commit_subject(tmp_path: Path) -> None:
    plan = tmp_path / "plan.md"
    ledger = tmp_path / "ledger.jsonl"
    evidence_root = tmp_path / "evidence"
    out = tmp_path / "audit.json"
    evidence_root.mkdir()
    (evidence_root / "task-1-manifest.json").write_text('{"status": "ok"}\n', encoding="utf-8")
    plan.write_text(
        "\n".join(
            (
                "- [x] 1. Add corpus manifest and provenance surface.",
                "  Commit: Y | `feat(data): add diffusion corpus manifest`",
            ),
        ),
        encoding="utf-8",
    )
    ledger.write_text(
        json.dumps(
            {
                "event": "task-completed",
                "task": "1. Add corpus manifest and provenance surface",
                "artifact": str(evidence_root / "task-1-manifest.json"),
                "commands": ['git commit -m "feat(diffusion): add qwen training loop foundations"'],
                "verdict": "fully-done",
            },
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "qwen-diffusion-plan-audit",
            "--plan",
            str(plan),
            "--ledger",
            str(ledger),
            "--evidence-root",
            str(evidence_root),
            "--out",
            str(out),
        ],
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["status"] == "pass"
    assert payload["todos"][0]["commit_status"] == "accepted_by_ledger"


def test_plan_audit_require_all_checked_fails_for_unchecked_top_level_item(tmp_path: Path) -> None:
    plan = tmp_path / "plan.md"
    ledger = tmp_path / "ledger.jsonl"
    evidence_root = tmp_path / "evidence"
    out = tmp_path / "audit.json"
    evidence_root.mkdir()
    (evidence_root / "task-1.txt").write_text("ok\n", encoding="utf-8")
    plan.write_text("- [x] 1. Done.\n- [ ] F1. Plan compliance audit\n", encoding="utf-8")
    ledger.write_text('{"event": "task-completed", "task": "1. Done.", "artifact": "task-1.txt"}\n', encoding="utf-8")

    exit_code = main(
        [
            "qwen-diffusion-plan-audit",
            "--plan",
            str(plan),
            "--ledger",
            str(ledger),
            "--evidence-root",
            str(evidence_root),
            "--out",
            str(out),
            "--require-all-checked",
        ],
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert payload["status"] == "fail"
    assert payload["unchecked_top_level"] == ["F1. Plan compliance audit"]


def test_fallback_scan_fails_hidden_fallback_evidence(tmp_path: Path) -> None:
    out = tmp_path / "fallback-scan.json"
    (tmp_path / "hidden.json").write_text('{"fallback_used": true}\n', encoding="utf-8")

    exit_code = main(["qwen-diffusion-plan-audit", "--scan-fallback", "--evidence-root", str(tmp_path), "--out", str(out)])

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert payload["status"] == "fail"
    assert payload["findings"][0]["field"] == "fallback_used"


def test_scope_audit_fails_benchmark_train_allowed_evidence(tmp_path: Path) -> None:
    plan = tmp_path / "plan.md"
    out = tmp_path / "scope.json"
    plan.write_text("Qwen-token masked diffusion plan\n", encoding="utf-8")
    (tmp_path / "benchmark-leak.json").write_text(
        '{"source": "HumanEval", "usage": "train_allowed", "fallback_used": false}\n',
        encoding="utf-8",
    )

    exit_code = main(
        [
            "qwen-diffusion-plan-audit",
            "--scope",
            "qwen-diffusion",
            "--plan",
            str(plan),
            "--evidence-root",
            str(tmp_path),
            "--out",
            str(out),
        ],
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert payload["status"] == "fail"
    assert payload["benchmark_train_allowed"][0]["benchmark"] == "humaneval"
