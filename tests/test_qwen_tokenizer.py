from __future__ import annotations

import json
from pathlib import Path

import pytest

from qiffusion.diffusion_data import ByteTokenizer
from qiffusion.qwen_tokenizer import TokenizerUnavailableError, load_tokenizer, write_unavailable_probe


def test_load_tokenizer_uses_byte_fallback_when_requested() -> None:
    source = "def add(a, b):\n    return a + b\n"

    tokenizer = load_tokenizer("byte")
    tokens = tokenizer.encode(source)

    assert isinstance(tokenizer, ByteTokenizer)
    assert tokenizer.decode(tokens) == source
    assert tokens[0] == tokenizer.byte_offset + ord("d")


def test_load_tokenizer_reports_unavailable_when_transformers_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_transformers(name: str):
        if name == "transformers":
            raise ModuleNotFoundError(name)
        return __import__(name)

    monkeypatch.setattr("importlib.import_module", missing_transformers)

    with pytest.raises(TokenizerUnavailableError) as raised:
        load_tokenizer("Qwen/Qwen3-tokenizer")

    assert raised.value.tokenizer_id == "Qwen/Qwen3-tokenizer"
    assert raised.value.reason == "transformers is not installed"


def test_qwen_tokenizer_config_serializes_without_transformers_dependency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "missing.json"

    def missing_transformers(name: str):
        if name == "transformers":
            raise ModuleNotFoundError(name)
        return __import__(name)

    monkeypatch.setattr("importlib.import_module", missing_transformers)
    write_unavailable_probe("missing-local-qwen-tokenizer", artifact)

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload == {
        "tokenizer_id": "missing-local-qwen-tokenizer",
        "status": "unavailable",
        "reason": "transformers is not installed",
    }


def test_qwen_tokenizer_round_trip_constraints(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    local_tokenizer = tmp_path / "local-qwen"
    local_tokenizer.mkdir()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(tokenizer_id: str, *, local_files_only: bool):
            assert tokenizer_id == str(local_tokenizer)
            assert local_files_only is True
            return FakeTokenizer()

    class FakeTransformers:
        AutoTokenizer = FakeAutoTokenizer

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 2
        vocab_size = 200_000

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return [ord(char) for char in text]

        def decode(self, tokens: tuple[int, ...], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens is True
            return "".join(chr(token) for token in tokens)

    def fake_import(name: str):
        if name == "transformers":
            return FakeTransformers
        return __import__(name)

    monkeypatch.setattr("importlib.import_module", fake_import)
    tokenizer = load_tokenizer(str(local_tokenizer))

    tokens = tokenizer.encode("def add")

    assert tokens == tuple(ord(char) for char in "def add")
    assert tokenizer.decode(tokens) == "def add"
    assert tokenizer.config.tokenizer_id == str(local_tokenizer)
    assert tokenizer.config.vocab_size == 200_000
    assert json.loads(json.dumps(tokenizer.config.to_json())) == {
        "tokenizer_id": str(local_tokenizer),
        "kind": "qwen",
        "vocab_size": 200_000,
        "pad_id": 0,
        "eos_id": 2,
    }


def test_load_tokenizer_rejects_invalid_tokenizer_id() -> None:
    with pytest.raises(TokenizerUnavailableError) as raised:
        load_tokenizer("")

    assert raised.value.reason == "tokenizer id is empty"
