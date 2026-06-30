from __future__ import annotations

import json
from pathlib import Path

from qiffusion.cli import main
from qiffusion.decision import JsonObject, decide_coding_capable


def test_final_gate_continues_when_chat_tool_or_swe_buckets_are_blocked() -> None:
    report = _passing_report()
    report["buckets"] = {
        **_passing_buckets(),
        "chat": _bucket("blocked", False, "chat harness missing"),
        "tool_agent": _bucket("not_run", False, "tool harness not run"),
        "software_engineering": _bucket("blocked", False, "SWE harness missing"),
    }
    report["coding_capability_claim"] = False
    report["chat_capability_claim"] = False
    report["tool_agent_capability_claim"] = False
    report["software_engineering_capability_claim"] = False
    report["release_capability_claim"] = False

    decision = decide_coding_capable(report)

    assert decision.status == "continue"
    assert "chat" in decision.reason


def test_final_gate_completes_only_with_all_required_evidence_and_lineage() -> None:
    decision = decide_coding_capable(_passing_report())

    assert decision.status == "complete"
    assert decision.reason == "qwen diffusion release gate passed"


def test_final_gate_rejects_byte_tokenizer_for_final_completion() -> None:
    report = _passing_report()
    lineage_value = report["checkpoint_lineage"]
    assert isinstance(lineage_value, dict)
    lineage = dict(lineage_value)
    lineage["tokenizer_id"] = "byte"
    report["checkpoint_lineage"] = lineage

    decision = decide_coding_capable(report)

    assert decision.status == "blocked"
    assert decision.reason == "qwen tokenizer evidence is missing"


def test_final_gate_blocks_hidden_fallback_and_benchmark_contamination() -> None:
    fallback_report = _passing_report()
    fallback_report["fallback_used"] = True
    contaminated_report = _passing_report()
    contaminated_report["benchmark_contamination"] = {"status": "blocked", "evidence": "HumanEval in train split"}

    fallback_decision = decide_coding_capable(fallback_report)
    contaminated_decision = decide_coding_capable(contaminated_report)

    assert fallback_decision.status == "blocked"
    assert fallback_decision.reason == "hidden fallback was used"
    assert contaminated_decision.status == "blocked"
    assert "benchmark contamination" in contaminated_decision.reason


def test_final_gate_does_not_complete_from_max_iterations_without_evidence() -> None:
    report = _passing_report()
    report["status"] = "max_iterations_reached"
    report["final_next_action"] = {"action": "max_iterations_reached", "continue_loop": False, "capability_claim": False}
    report["buckets"] = {
        **_passing_buckets(),
        "external_benchmark_readiness": _bucket("not_run", False, ""),
    }
    report["coding_capability_claim"] = False
    report["release_capability_claim"] = False

    decision = decide_coding_capable(report)

    assert decision.status == "continue"
    assert "external_benchmark_readiness" in decision.reason


def test_status_cli_writes_conservative_qwen_gate_decision(tmp_path: Path, capsys) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_passing_report()), encoding="utf-8")

    exit_code = main(["status", "--report", str(report_path)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "complete"


def _passing_report() -> JsonObject:
    return {
        "backend": "qwen_token_diffusion",
        "stage": "final_gate",
        "status": "evaluated",
        "fixtures_status": "pass",
        "code_smoke_status": "pass",
        "fallback_used": False,
        "coding_capability_claim": True,
        "chat_capability_claim": True,
        "tool_agent_capability_claim": True,
        "software_engineering_capability_claim": True,
        "release_capability_claim": True,
        "buckets": _passing_buckets(),
        "benchmark_contamination": {"status": "clean", "evidence": "manifest manifest-release checked"},
        "checkpoint_lineage": {
            "checkpoint_id": "qwen-diffusion-final",
            "base_checkpoint_id": "Qwen/Qwen3.5-4B",
            "data_manifest_id": "manifest-release",
            "parent_checkpoint_id": "qwen-diffusion-parent",
            "objective": "masked_ce",
            "mask_schedule": "linear",
            "tokenizer_id": "Qwen/Qwen3.5-4B",
        },
    }


def _passing_buckets() -> JsonObject:
    return {
        "local_code_smoke": _bucket("pass", True, "local smoke artifact"),
        "external_benchmark_readiness": _bucket("pass", True, "benchmark harness artifact"),
        "chat": _bucket("pass", True, "chat eval artifact"),
        "tool_agent": _bucket("pass", True, "tool eval artifact"),
        "software_engineering": _bucket("pass", True, "SWE harness artifact"),
    }


def _bucket(status: str, capability_claim: bool, evidence: str) -> JsonObject:
    return {
        "status": status,
        "capability_claim": capability_claim,
        "evidence": evidence,
        "detail": evidence,
    }
