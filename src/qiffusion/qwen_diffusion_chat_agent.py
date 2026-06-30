from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from qiffusion.qwen_diffusion_chat_agent_schema import (
    SCHEMA_VERSION,
    ChatAgentSchemaError,
    ChatRecord,
    ToolAgentRecord,
    validate_chat_payload,
    validate_tool_agent_payload,
)
from qiffusion.qwen_diffusion_config import JsonObject, JsonValue
from qiffusion.qwen_diffusion_eval import BucketStatus, EvalBucket


class ChatAgentValidationReport(TypedDict):
    schema_version: int
    backend: str
    stage: str
    status: str
    fixtures_status: str
    code_smoke_status: str
    fallback_used: bool
    local_code_capability_claim: bool
    coding_capability_claim: bool
    chat_capability_claim: bool
    tool_agent_capability_claim: bool
    software_engineering_capability_claim: bool
    release_capability_claim: bool
    chat_schema_valid: bool
    tool_agent_schema_valid: bool
    buckets: dict[str, EvalBucket]
    loop_inputs: list[JsonObject]
    notes: list[str]


def chat_agent_report(chat_payload: JsonObject | None, tool_payload: JsonObject | None) -> ChatAgentValidationReport:
    chat = validate_chat_payload(chat_payload) if chat_payload is not None else None
    tool = validate_tool_agent_payload(tool_payload) if tool_payload is not None else None
    chat_valid = chat is not None
    tool_valid = tool is not None
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": "qwen_token_diffusion",
        "stage": "chat_agent_eval_stub",
        "status": "accepted",
        "fixtures_status": "not_run",
        "code_smoke_status": "not_run",
        "fallback_used": False,
        "local_code_capability_claim": False,
        "coding_capability_claim": False,
        "chat_capability_claim": False,
        "tool_agent_capability_claim": False,
        "software_engineering_capability_claim": False,
        "release_capability_claim": False,
        "chat_schema_valid": chat_valid,
        "tool_agent_schema_valid": tool_valid,
        "buckets": _buckets(chat_valid),
        "loop_inputs": _loop_inputs(chat, tool),
        "notes": [
            "chat/tool schemas validated for loop input only",
            "chat and tool-agent harnesses are not implemented; statuses are blockers, not capability success",
            "raw user chat log ingestion is rejected",
        ],
    }


def write_chat_agent_report(chat_path: Path | None, tool_path: Path | None, out: Path) -> ChatAgentValidationReport:
    report = chat_agent_report(_load_optional(chat_path), _load_optional(tool_path))
    _write_json(out, report)
    return report


def write_rejection(error: ChatAgentSchemaError, out: Path) -> JsonObject:
    payload: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "status": "rejected",
        "field": error.field,
        "error": str(error),
        "coding_capability_claim": False,
        "chat_capability_claim": False,
        "tool_agent_capability_claim": False,
    }
    _write_json(out, payload)
    return payload


def _buckets(chat_valid: bool) -> dict[str, EvalBucket]:
    chat_status: BucketStatus = "not_run" if chat_valid else "blocked"
    return {
        "local_code_smoke": _bucket("not_run", "coding smoke was not run by chat/tool schema validation"),
        "external_benchmark_readiness": _bucket("not_run", "benchmark harness was not run by chat/tool schema validation"),
        "chat": _bucket(chat_status, "chat harness is not implemented in Todo 11"),
        "tool_agent": _bucket("blocked", "tool/agent harness is not implemented in Todo 11"),
        "software_engineering": _bucket("blocked", "SWE harness is not implemented in Todo 11"),
    }


def _bucket(status: BucketStatus, detail: str) -> EvalBucket:
    return {"status": status, "capability_claim": False, "evidence": "qwen-diffusion-chat-agent-validate", "detail": detail}


def _loop_inputs(chat: ChatRecord | None, tool: ToolAgentRecord | None) -> list[JsonObject]:
    records: list[JsonObject] = []
    if chat is not None:
        records.append({"record_type": "chat", "task_name": chat.task_name, "expected_outcome": chat.provenance.expected_outcome})
    if tool is not None:
        records.append({"record_type": "tool_agent", "task_name": tool.task_name, "expected_outcome": tool.provenance.expected_outcome, "tool_name": tool.tool_name})
    return records


def _load_optional(path: Path | None) -> JsonObject | None:
    if path is None:
        return None
    payload: JsonValue = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ChatAgentSchemaError(str(path), "expected JSON object")
    return payload


def _write_json(path: Path, payload: JsonObject | ChatAgentValidationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
