from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Final, Literal, TypedDict

from qiffusion.diffusion_data import ByteTokenizer, CorpusExample, build_local_corpus

CorpusUsage = Literal["train_allowed", "eval_only", "unknown_blocked"]
ContaminationStatus = Literal["clean", "suspect", "blocked"]
CorpusSplit = Literal["train", "eval", "unknown"]
SourceKind = Literal["local", "teacher_jsonl", "external"]

BENCHMARK_MARKERS: Final = (
    "humaneval",
    "mbpp",
    "evalplus",
    "livecodebench",
    "bigcodebench",
    "swe-bench",
    "swebench",
)


class ManifestRecordJson(TypedDict):
    source: str
    source_kind: SourceKind
    name: str
    license: str
    split: CorpusSplit
    tokenizer: str
    token_count: int
    dedup_hash: str
    usage: CorpusUsage
    contamination_status: ContaminationStatus
    privacy_policy_notes: str


class ManifestJson(TypedDict):
    schema_version: int
    status: str
    root: str
    records: list[ManifestRecordJson]


class ManifestErrorJson(TypedDict):
    status: str
    error: str
    path: str
    message: str


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    source: str
    source_kind: SourceKind
    name: str
    license: str
    split: CorpusSplit
    tokenizer: str
    token_count: int
    dedup_hash: str
    usage: CorpusUsage
    contamination_status: ContaminationStatus
    privacy_policy_notes: str

    def to_json(self) -> ManifestRecordJson:
        return {
            "source": self.source,
            "source_kind": self.source_kind,
            "name": self.name,
            "license": self.license,
            "split": self.split,
            "tokenizer": self.tokenizer,
            "token_count": self.token_count,
            "dedup_hash": self.dedup_hash,
            "usage": self.usage,
            "contamination_status": self.contamination_status,
            "privacy_policy_notes": self.privacy_policy_notes,
        }


@dataclass(frozen=True, slots=True)
class ManifestBuildConfig:
    root: Path
    teacher_jsonl_paths: tuple[Path, ...] = ()
    tokenizer_name: str = "byte"


@dataclass(frozen=True, slots=True)
class ManifestRecordSource:
    example: CorpusExample
    source_kind: SourceKind
    split: CorpusSplit


@dataclass(frozen=True, slots=True)
class MalformedTeacherJsonlError(Exception):
    path: Path
    line_number: int
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line_number}: {self.detail}"


def build_manifest(config: ManifestBuildConfig) -> ManifestJson:
    tokenizer = ByteTokenizer()
    records = [
        manifest_record(
            ManifestRecordSource(example, "local", "train"),
            tokenizer_name=config.tokenizer_name,
            tokenizer=tokenizer,
        )
        for example in build_local_corpus(config.root)
    ]
    for path in config.teacher_jsonl_paths:
        records.extend(teacher_manifest_records(path, tokenizer, config.tokenizer_name))
    return {
        "schema_version": 1,
        "status": "ok",
        "root": str(config.root),
        "records": [record.to_json() for record in records],
    }


def write_manifest(config: ManifestBuildConfig, output: Path) -> ManifestJson:
    manifest = build_manifest(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def write_manifest_error(error: MalformedTeacherJsonlError, output: Path) -> ManifestErrorJson:
    payload: ManifestErrorJson = {
        "status": "error",
        "error": "malformed_teacher_jsonl",
        "path": str(error.path),
        "message": str(error),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def teacher_manifest_records(path: Path, tokenizer: ByteTokenizer, tokenizer_name: str) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line == "":
            continue
        try:
            payload = json.loads(line)
        except JSONDecodeError as exc:
            raise MalformedTeacherJsonlError(path, index, exc.msg) from exc
        if not isinstance(payload, dict):
            raise MalformedTeacherJsonlError(path, index, "expected JSON object")
        code = payload.get("code")
        name = payload.get("task_name")
        if not isinstance(code, str) or not isinstance(name, str):
            raise MalformedTeacherJsonlError(path, index, "expected string task_name and code")
        records.append(
            manifest_record(
                ManifestRecordSource(CorpusExample(f"teacher:{name}", str(path), code), "teacher_jsonl", "train"),
                tokenizer_name=tokenizer_name,
                tokenizer=tokenizer,
            )
        )
    return records


def manifest_record(
    source: ManifestRecordSource,
    *,
    tokenizer_name: str,
    tokenizer: ByteTokenizer,
) -> ManifestRecord:
    license_name = "unknown"
    usage = usage_for_source(source.example.source, license_name)
    contamination_status = contamination_for_source(source.example.source)
    return ManifestRecord(
        source=source.example.source,
        source_kind=source.source_kind,
        name=source.example.name,
        license=license_name,
        split=source.split,
        tokenizer=tokenizer_name,
        token_count=len(tokenizer.encode(source.example.text)),
        dedup_hash=hashlib.sha256(source.example.text.encode("utf-8")).hexdigest(),
        usage=usage,
        contamination_status=contamination_status,
        privacy_policy_notes="unknown provenance; do not train until license and privacy review pass",
    )


def usage_for_source(source: str, license_name: str) -> CorpusUsage:
    if is_benchmark_source(source):
        return "eval_only"
    if license_name == "unknown":
        return "unknown_blocked"
    return "train_allowed"


def contamination_for_source(source: str) -> ContaminationStatus:
    if is_benchmark_source(source):
        return "blocked"
    return "clean"


def is_benchmark_source(source: str) -> bool:
    lowered = source.lower()
    return any(marker in lowered for marker in BENCHMARK_MARKERS)
