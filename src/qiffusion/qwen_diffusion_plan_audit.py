from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

JsonValue: TypeAlias = str | int | float | bool | None | Sequence["JsonValue"] | Mapping[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

CHECKED_TODO_RE: Final = re.compile(r"^- \[[xX]\] (?P<number>\d+)\. (?P<title>.+)$")
UNCHECKED_RE: Final = re.compile(r"^- \[ \] (?P<label>(?:\d+|F\d+)\. .+)$")
FAILURE_NAME_MARKERS: Final = ("failure", "fallback", "blocked", "overclaim", "mismatch", "invalid", "malformed", "rejected")
BENCHMARK_MARKERS: Final = ("humaneval", "mbpp", "evalplus", "livecodebench", "bigcodebench", "swe-bench", "swebench")


@dataclass(frozen=True, slots=True)
class PlanAuditConfig:
    plan: Path
    ledger: Path
    evidence_root: Path
    require_all_checked: bool


@dataclass(frozen=True, slots=True)
class EvidenceAuditConfig:
    evidence_root: Path
    plan: Path | None = None


@dataclass(frozen=True, slots=True)
class CheckedTodo:
    number: str
    title: str


@dataclass(frozen=True, slots=True)
class EvidenceDocument:
    path: Path
    payload: JsonValue


@dataclass(frozen=True, slots=True)
class AuditFinding:
    path: Path
    field: str
    detail: str

    def to_json(self) -> JsonObject:
        payload: JsonObject = {"path": str(self.path), "field": self.field, "detail": self.detail}
        if self.field == "usage":
            payload["benchmark"] = self.detail
        return payload


def add_qwen_diffusion_plan_audit_parser(subcommands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subcommands.add_parser("qwen-diffusion-plan-audit", help="Audit Qwen diffusion plan and evidence.")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-all-checked", action="store_true")
    parser.add_argument("--scan-fallback", action="store_true")
    parser.add_argument("--scope", choices=("qwen-diffusion",))


def run_qwen_diffusion_plan_audit(args: argparse.Namespace) -> int:
    if args.scan_fallback:
        report = fallback_scan(EvidenceAuditConfig(evidence_root=args.evidence_root))
        return _write_report(args.out, report)
    if args.scope == "qwen-diffusion":
        report = scope_audit(EvidenceAuditConfig(evidence_root=args.evidence_root, plan=args.plan))
        return _write_report(args.out, report)
    report = plan_audit(
        PlanAuditConfig(
            plan=args.plan,
            ledger=args.ledger,
            evidence_root=args.evidence_root,
            require_all_checked=args.require_all_checked,
        ),
    )
    return _write_report(args.out, report)


def plan_audit(config: PlanAuditConfig) -> JsonObject:
    plan_text = config.plan.read_text(encoding="utf-8")
    checked = _checked_todos(plan_text)
    ledger = _ledger_records(config.ledger)
    unchecked = _unchecked_top_level(plan_text)
    todos = [_todo_report(todo, ledger, config.evidence_root) for todo in checked]
    failures = [todo for todo in todos if todo["status"] != "pass"]
    if config.require_all_checked and len(unchecked) > 0:
        failures.append({"status": "fail", "reason": "unchecked top-level item remains"})
    status = "pass" if len(failures) == 0 else "fail"
    return {
        "status": status,
        "checked_todo_count": len(checked),
        "unchecked_top_level": unchecked,
        "todos": todos,
    }


def fallback_scan(config: EvidenceAuditConfig) -> JsonObject:
    hidden: list[AuditFinding] = []
    expected: list[AuditFinding] = []
    for document in _evidence_documents(config.evidence_root):
        for finding in _fallback_findings(document):
            if _expected_failure_fixture(document.path):
                expected.append(finding)
            else:
                hidden.append(finding)
    return {
        "status": "pass" if len(hidden) == 0 else "fail",
        "findings": [finding.to_json() for finding in hidden],
        "expected_failure_findings": [finding.to_json() for finding in expected],
    }


def scope_audit(config: EvidenceAuditConfig) -> JsonObject:
    fallback = fallback_scan(config)
    documents = tuple(_evidence_documents(config.evidence_root))
    benchmark_leaks = _benchmark_train_allowed_findings(documents)
    scope_confirmed = _scope_confirmed(config.plan, documents)
    status = "pass" if fallback["status"] == "pass" and len(benchmark_leaks) == 0 and scope_confirmed else "fail"
    return {
        "status": status,
        "scope": "qwen-diffusion",
        "scope_confirmed": scope_confirmed,
        "fallback_findings": fallback["findings"],
        "benchmark_train_allowed": [finding.to_json() for finding in benchmark_leaks],
    }


def _todo_report(todo: CheckedTodo, ledger: tuple[JsonObject, ...], evidence_root: Path) -> JsonObject:
    entries = tuple(record for record in ledger if _ledger_matches_todo(record, todo))
    evidence = _evidence_for_todo(todo.number, entries, evidence_root)
    status = "pass" if len(entries) > 0 and len(evidence) > 0 else "fail"
    return {
        "number": todo.number,
        "title": todo.title,
        "status": status,
        "ledger_entries": len(entries),
        "evidence": [str(path) for path in evidence],
        "commit_status": _commit_status(entries, evidence),
    }


def _checked_todos(plan_text: str) -> tuple[CheckedTodo, ...]:
    todos: list[CheckedTodo] = []
    for line in plan_text.splitlines():
        match = CHECKED_TODO_RE.match(line)
        if match is not None:
            todos.append(CheckedTodo(match.group("number"), match.group("title")))
    return tuple(todos)


def _unchecked_top_level(plan_text: str) -> list[str]:
    unchecked: list[str] = []
    for line in plan_text.splitlines():
        match = UNCHECKED_RE.match(line)
        if match is not None:
            unchecked.append(match.group("label"))
    return unchecked


def _ledger_records(path: Path) -> tuple[JsonObject, ...]:
    records: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip() == "":
            continue
        payload: JsonValue = json.loads(line)
        if isinstance(payload, dict):
            records.append(payload)
    return tuple(records)


def _ledger_matches_todo(record: JsonObject, todo: CheckedTodo) -> bool:
    task = record.get("task")
    return isinstance(task, str) and task.startswith(f"{todo.number}.")


def _evidence_for_todo(number: str, entries: tuple[JsonObject, ...], evidence_root: Path) -> tuple[Path, ...]:
    paths = {path for path in evidence_root.rglob(f"task-{number}-*") if path.is_file()}
    for entry in entries:
        artifact = entry.get("artifact")
        if isinstance(artifact, str):
            paths.update(_existing_artifact_paths(artifact, evidence_root))
    return tuple(sorted(paths))


def _existing_artifact_paths(artifact: str, evidence_root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for raw_token in re.split(r"[;,]", artifact):
        token = raw_token.strip()
        if token == "" or token.startswith("commit "):
            continue
        path = Path(token)
        candidate = path if path.exists() else evidence_root / token
        if candidate.exists() and candidate.is_file():
            paths.append(candidate)
    return tuple(paths)


def _commit_status(entries: tuple[JsonObject, ...], evidence: tuple[Path, ...]) -> str:
    for entry in entries:
        haystack = json.dumps(entry, sort_keys=True)
        if "git commit" in haystack or re.search(r"\b[0-9a-f]{7,40}\b", haystack) is not None:
            return "accepted_by_ledger"
    if len(evidence) > 0:
        return "accepted_by_evidence"
    return "missing"


def _evidence_documents(root: Path) -> Iterable[EvidenceDocument]:
    for path in sorted(root.rglob("*")):
        if path.suffix not in (".json", ".jsonl") or not path.is_file():
            continue
        for payload in _json_payloads(path):
            yield EvidenceDocument(path, payload)


def _json_payloads(path: Path) -> Iterable[JsonValue]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return
    if path.suffix == ".jsonl":
        for line in text.splitlines():
            if line.strip() != "":
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        return
    try:
        yield json.loads(text)
    except json.JSONDecodeError:
        return


def _fallback_findings(document: EvidenceDocument) -> tuple[AuditFinding, ...]:
    findings: list[AuditFinding] = []
    for field, value in _walk(document.payload):
        if field == "fallback_used" and value is True:
            findings.append(AuditFinding(document.path, field, "fallback_used true"))
    return tuple(findings)


def _benchmark_train_allowed_findings(documents: tuple[EvidenceDocument, ...]) -> tuple[AuditFinding, ...]:
    findings: list[AuditFinding] = []
    for document in documents:
        if _expected_failure_fixture(document.path):
            continue
        if isinstance(document.payload, dict) and document.payload.get("usage") == "train_allowed":
            benchmark = _benchmark_in_value(document.payload)
            if benchmark is not None:
                findings.append(AuditFinding(document.path, "usage", benchmark))
        for _field, value in _walk(document.payload):
            if isinstance(value, dict) and value.get("usage") == "train_allowed":
                benchmark = _benchmark_in_value(value)
                if benchmark is not None:
                    findings.append(AuditFinding(document.path, "usage", benchmark))
    return tuple(findings)


def _walk(value: JsonValue) -> Iterable[tuple[str, JsonValue]]:
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key, nested
            yield from _walk(nested)
    if isinstance(value, list):
        for nested in value:
            yield from _walk(nested)


def _benchmark_in_value(value: JsonValue) -> str | None:
    lowered = json.dumps(value, sort_keys=True).lower()
    for marker in BENCHMARK_MARKERS:
        if marker in lowered:
            return marker
    return None


def _scope_confirmed(plan: Path | None, documents: tuple[EvidenceDocument, ...]) -> bool:
    plan_text = "" if plan is None else plan.read_text(encoding="utf-8").lower()
    if "qwen" in plan_text and "diffusion" in plan_text:
        return True
    return any("qwen_token_diffusion" in json.dumps(document.payload).lower() for document in documents)


def _expected_failure_fixture(path: Path) -> bool:
    lowered = path.name.lower()
    return any(marker in lowered for marker in FAILURE_NAME_MARKERS)


def _write_report(path: Path, report: JsonObject) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 2
