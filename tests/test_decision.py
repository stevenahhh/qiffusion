from __future__ import annotations

from qiffusion.decision import decide_coding_capable


def test_decision_continues_when_code_smoke_fails() -> None:
    decision = decide_coding_capable(
        {
            "fixtures_status": "pass",
            "code_smoke_status": "fail",
            "coding_capability_claim": False,
        }
    )

    assert decision.status == "continue"
    assert decision.reason == "code smoke is not passing"


def test_decision_promotes_only_after_shared_gate_passes() -> None:
    decision = decide_coding_capable(
        {
            "fixtures_status": "pass",
            "code_smoke_status": "pass",
            "coding_capability_claim": True,
        }
    )

    assert decision.status == "promote"


def test_decision_accepts_elf_latest_json_shape() -> None:
    decision = decide_coding_capable(
        {
            "status": "pass",
            "best": {
                "fixtures_status": "pass",
                "code_smoke_status": "fail",
                "coding_capability_claim": False,
            },
        }
    )

    assert decision.status == "continue"
    assert decision.reason == "code smoke is not passing"
