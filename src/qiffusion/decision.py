from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class GateDecision:
    status: str
    reason: str


def load_json(path: Path) -> JsonObject:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def decide_coding_capable(report: JsonObject) -> GateDecision:
    candidate = report.get("best") if isinstance(report.get("best"), dict) else report
    if candidate.get("fixtures_status") != "pass":
        return GateDecision("continue", "fixtures are not passing")
    if candidate.get("code_smoke_status") != "pass":
        return GateDecision("continue", "code smoke is not passing")
    if candidate.get("coding_capability_claim") is not True:
        return GateDecision("continue", "coding capability claim is not true")
    return GateDecision("promote", "shared coding-capable gate passed")


def decide_from_file(path: Path) -> GateDecision:
    return decide_coding_capable(load_json(path))
