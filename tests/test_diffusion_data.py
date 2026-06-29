from __future__ import annotations

from pathlib import Path

import pytest

from qiffusion.diffusion_data import ByteTokenizer, TokenDecodeError, build_local_corpus, pack_sequences


def test_byte_tokenizer_round_trips_python_source() -> None:
    tokenizer = ByteTokenizer()
    source = "def add(a, b):\n    return a + b\n"

    tokens = tokenizer.encode(source)
    decoded = tokenizer.decode(tokens)

    assert decoded == source
    assert tokens[0] == tokenizer.byte_offset + ord("d")
    assert tokenizer.mask_id not in tokens


def test_byte_tokenizer_rejects_non_byte_tokens() -> None:
    tokenizer = ByteTokenizer()

    with pytest.raises(TokenDecodeError):
        tokenizer.decode((tokenizer.mask_id,))


def test_local_corpus_is_deterministic_and_non_empty(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Local\n\nexample text\n", encoding="utf-8")

    first = build_local_corpus(tmp_path)
    second = build_local_corpus(tmp_path)

    assert first == second
    assert len(first) >= 10
    assert first[0].name == "readme"
    assert any(example.name == "task:add" for example in first)
    assert all(example.text for example in first)


def test_pack_sequences_pads_and_truncates_deterministically() -> None:
    tokenizer = ByteTokenizer()
    examples = build_local_corpus()[:2]

    sequences = pack_sequences(examples, tokenizer, sequence_length=8)

    assert len(sequences) == 2
    assert all(len(sequence) == 8 for sequence in sequences)
