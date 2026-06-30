from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence, TypedDict

from qiffusion.diffusion_data import ByteTokenizer


TokenizerKind = Literal["byte", "qwen"]


class Tokenizer(Protocol):
    def encode(self, text: str) -> tuple[int, ...]: ...

    def decode(self, tokens: Sequence[int]) -> str: ...


class TransformerTokenizer(Protocol):
    pad_token_id: int | None
    eos_token_id: int | None
    vocab_size: int

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(self, tokens: tuple[int, ...], *, skip_special_tokens: bool) -> str: ...


class TokenizerProbe(TypedDict):
    tokenizer_id: str
    status: Literal["available", "unavailable"]
    reason: str


class QwenTokenizerConfigJson(TypedDict):
    tokenizer_id: str
    kind: TokenizerKind
    vocab_size: int
    pad_id: int | None
    eos_id: int | None


@dataclass(frozen=True, slots=True)
class TokenizerUnavailableError(Exception):
    tokenizer_id: str
    reason: str

    def __str__(self) -> str:
        return f"{self.tokenizer_id}: {self.reason}"


@dataclass(frozen=True, slots=True)
class QwenTokenizerConfig:
    tokenizer_id: str
    kind: TokenizerKind
    vocab_size: int
    pad_id: int | None
    eos_id: int | None

    def to_json(self) -> QwenTokenizerConfigJson:
        return {
            "tokenizer_id": self.tokenizer_id,
            "kind": self.kind,
            "vocab_size": self.vocab_size,
            "pad_id": self.pad_id,
            "eos_id": self.eos_id,
        }


@dataclass(frozen=True, slots=True)
class QwenTokenizer:
    tokenizer: TransformerTokenizer
    config: QwenTokenizerConfig

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, tokens: Sequence[int]) -> str:
        return self.tokenizer.decode(tuple(tokens), skip_special_tokens=True)


def load_tokenizer(tokenizer_id: str) -> Tokenizer:
    normalized = tokenizer_id.strip()
    if not normalized:
        raise TokenizerUnavailableError(tokenizer_id, "tokenizer id is empty")
    if normalized.lower() == "byte":
        return ByteTokenizer()

    try:
        transformers = importlib.import_module("transformers")
    except ModuleNotFoundError as exc:
        if exc.name in (None, "transformers"):
            raise TokenizerUnavailableError(normalized, "transformers is not installed") from None
        raise

    local_path = Path(normalized)
    if not local_path.exists():
        raise TokenizerUnavailableError(normalized, "local tokenizer path does not exist")

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(normalized, local_files_only=True)
    except OSError as exc:
        raise TokenizerUnavailableError(normalized, f"local tokenizer is unavailable: {exc}") from None

    config = QwenTokenizerConfig(
        tokenizer_id=normalized,
        kind="qwen",
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_token_id,
        eos_id=tokenizer.eos_token_id,
    )
    return QwenTokenizer(tokenizer=tokenizer, config=config)


def write_unavailable_probe(tokenizer_id: str, path: Path) -> None:
    try:
        load_tokenizer(tokenizer_id)
    except TokenizerUnavailableError as exc:
        payload: TokenizerProbe = {
            "tokenizer_id": exc.tokenizer_id,
            "status": "unavailable",
            "reason": exc.reason,
        }
    else:
        payload = {
            "tokenizer_id": tokenizer_id,
            "status": "available",
            "reason": "",
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
