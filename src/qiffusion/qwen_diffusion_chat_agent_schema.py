from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from qiffusion.qwen_diffusion_config import JsonObject, JsonValue

SCHEMA_VERSION: Final = 1
SourceKind: TypeAlias = Literal["synthetic", "teacher_trace", "local_fixture"]
ExpectedOutcome: TypeAlias = Literal["pass", "fail"]
ChatRole: TypeAlias = Literal["system", "user", "assistant"]
ToolTraceRole: TypeAlias = Literal["assistant", "tool"]

ALLOWED_SOURCE_KINDS: Final[tuple[SourceKind, ...]] = ("synthetic", "teacher_trace", "local_fixture")
PROMPT_INJECTION_MARKERS: Final = (
    "ignore previous instructions",
    "reveal the system prompt",
    "act as system",
    "developer message",
    "system prompt",
)


@dataclass(frozen=True, slots=True)
class ChatAgentSchemaError(Exception):
    field: str
    message: str

    def __str__(self) -> str:
        return f"{self.field}: {self.message}"


@dataclass(frozen=True, slots=True)
class Provenance:
    task_name: str
    source: str
    source_kind: SourceKind
    license: str
    teacher_model: str
    prompt_hash: str
    policy_notes: tuple[str, ...]
    expected_outcome: ExpectedOutcome


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: ChatRole
    content: str


@dataclass(frozen=True, slots=True)
class ChatRecord:
    provenance: Provenance
    messages: tuple[ChatMessage, ...]

    @property
    def task_name(self) -> str:
        return self.provenance.task_name


@dataclass(frozen=True, slots=True)
class ToolAgentStep:
    role: ToolTraceRole
    content: str
    tool_name: str


@dataclass(frozen=True, slots=True)
class ToolAgentRecord:
    provenance: Provenance
    tool_name: str
    tool_input: JsonObject
    tool_output: str
    agent_trace: tuple[ToolAgentStep, ...]

    @property
    def task_name(self) -> str:
        return self.provenance.task_name


def validate_chat_payload(payload: JsonObject) -> ChatRecord:
    _require_record_type(payload, "chat")
    provenance = _provenance(payload)
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        raise ChatAgentSchemaError("messages", "expected non-empty message list")
    messages = tuple(_chat_message(message, index) for index, message in enumerate(raw_messages))
    if len(messages) < 2:
        raise ChatAgentSchemaError("messages", "expected multi-turn chat messages")
    if not _has_role(messages, "user") or not _has_role(messages, "assistant"):
        raise ChatAgentSchemaError("messages", "expected at least one user and assistant message")
    return ChatRecord(provenance, messages)


def validate_tool_agent_payload(payload: JsonObject) -> ToolAgentRecord:
    _require_record_type(payload, "tool_agent")
    provenance = _provenance(payload)
    tool_name = _text(payload, "tool_name")
    _validate_tool_name(tool_name)
    tool_input = _json_object(payload, "tool_input")
    tool_output = _text(payload, "tool_output")
    _reject_injected("tool_input", tool_input)
    _reject_injected("tool_output", tool_output)
    raw_trace = payload.get("agent_trace")
    if not isinstance(raw_trace, list):
        raise ChatAgentSchemaError("agent_trace", "expected non-empty tool trace")
    trace = tuple(_tool_step(step, index) for index, step in enumerate(raw_trace))
    if len(trace) == 0 or not _has_tool_step(trace):
        raise ChatAgentSchemaError("agent_trace", "expected at least one tool step")
    return ToolAgentRecord(provenance, tool_name, tool_input, tool_output, trace)


def _provenance(payload: JsonObject) -> Provenance:
    return Provenance(
        task_name=_text(payload, "task_name"),
        source=_text(payload, "source"),
        source_kind=_source_kind(payload),
        license=_text(payload, "license"),
        teacher_model=_text(payload, "teacher_model"),
        prompt_hash=_text(payload, "prompt_hash"),
        policy_notes=_policy_notes(payload),
        expected_outcome=_expected_outcome(payload),
    )


def _require_record_type(payload: JsonObject, expected: str) -> None:
    schema_version = payload.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ChatAgentSchemaError("schema_version", "expected schema version 1")
    record_type = payload.get("record_type")
    if record_type != expected:
        raise ChatAgentSchemaError("record_type", f"expected {expected}")


def _source_kind(payload: JsonObject) -> SourceKind:
    value = _text(payload, "source_kind")
    for source_kind in ALLOWED_SOURCE_KINDS:
        if value == source_kind:
            return source_kind
    raise ChatAgentSchemaError("source_kind", "expected synthetic, teacher_trace, or local_fixture; raw user logs are blocked")


def _expected_outcome(payload: JsonObject) -> ExpectedOutcome:
    value = _text(payload, "expected_outcome")
    for candidate in ("pass", "fail"):
        if value == candidate:
            return candidate
    raise ChatAgentSchemaError("expected_outcome", "expected pass or fail")


def _chat_message(payload: JsonValue, index: int) -> ChatMessage:
    if not isinstance(payload, dict):
        raise ChatAgentSchemaError(f"messages[{index}]", "expected object")
    return ChatMessage(_chat_role(payload, index), _text(payload, "content"))


def _chat_role(payload: JsonObject, index: int) -> ChatRole:
    value = _text(payload, "role")
    for role in ("system", "user", "assistant"):
        if value == role:
            return role
    raise ChatAgentSchemaError(f"messages[{index}].role", "expected system, user, or assistant")


def _tool_step(payload: JsonValue, index: int) -> ToolAgentStep:
    if not isinstance(payload, dict):
        raise ChatAgentSchemaError(f"agent_trace[{index}]", "expected object")
    role = _tool_role(payload, index)
    content = _text(payload, "content")
    _reject_injected(f"agent_trace[{index}].content", content)
    tool_name = _optional_text(payload, "tool_name")
    if role == "tool":
        _validate_tool_name(tool_name)
    return ToolAgentStep(role, content, tool_name)


def _tool_role(payload: JsonObject, index: int) -> ToolTraceRole:
    value = _text(payload, "role")
    for role in ("assistant", "tool"):
        if value == role:
            return role
    raise ChatAgentSchemaError(f"agent_trace[{index}].role", "expected assistant or tool")


def _text(payload: JsonObject, field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or value.strip() == "":
        raise ChatAgentSchemaError(field, "expected non-empty string")
    return value


def _optional_text(payload: JsonObject, field: str) -> str:
    value = payload.get(field)
    return value if isinstance(value, str) else ""


def _policy_notes(payload: JsonObject) -> tuple[str, ...]:
    notes = payload.get("policy_notes")
    if not isinstance(notes, list):
        raise ChatAgentSchemaError("policy_notes", "expected non-empty string list")
    parsed = tuple(note for note in notes if isinstance(note, str) and note.strip() != "")
    if len(parsed) == 0 or len(parsed) != len(notes):
        raise ChatAgentSchemaError("policy_notes", "expected non-empty string list")
    return parsed


def _json_object(payload: JsonObject, field: str) -> JsonObject:
    value = payload.get(field)
    if not isinstance(value, dict):
        raise ChatAgentSchemaError(field, "expected object")
    return value


def _validate_tool_name(value: str) -> None:
    if value == "":
        raise ChatAgentSchemaError("tool_name", "expected non-empty tool name")
    for character in value:
        if not (character.islower() or character.isdigit() or character in ("_", "-")):
            raise ChatAgentSchemaError("tool_name", "expected lowercase tool name with digits, hyphen, or underscore")


def _reject_injected(field: str, value: JsonValue) -> None:
    for text in _strings(value):
        lowered = text.lower()
        for marker in PROMPT_INJECTION_MARKERS:
            if marker in lowered:
                raise ChatAgentSchemaError(field, "prompt injection marker rejected")


def _strings(value: JsonValue) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        strings: list[str] = []
        for item in value:
            strings.extend(_strings(item))
        return tuple(strings)
    if isinstance(value, dict):
        strings: list[str] = []
        for item in value.values():
            strings.extend(_strings(item))
        return tuple(strings)
    return ()


def _has_role(messages: tuple[ChatMessage, ...], role: ChatRole) -> bool:
    return any(message.role == role for message in messages)


def _has_tool_step(steps: tuple[ToolAgentStep, ...]) -> bool:
    return any(step.role == "tool" for step in steps)
