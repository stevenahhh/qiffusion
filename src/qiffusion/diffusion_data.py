from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

from qiffusion.diffusion_config import BYTE_TOKEN_OFFSET, BYTE_VOCAB_SIZE, EOS_TOKEN_ID, MASK_TOKEN_ID, PAD_TOKEN_ID


@dataclass(frozen=True, slots=True)
class TokenDecodeError(Exception):
    token_id: int

    def __str__(self) -> str:
        return f"token {self.token_id} is not a byte token"


@dataclass(frozen=True, slots=True)
class CorpusExample:
    name: str
    source: str
    text: str


@dataclass(frozen=True, slots=True)
class ByteTokenizer:
    pad_id: int = PAD_TOKEN_ID
    mask_id: int = MASK_TOKEN_ID
    eos_id: int = EOS_TOKEN_ID
    byte_offset: int = BYTE_TOKEN_OFFSET
    vocab_size: int = BYTE_VOCAB_SIZE

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(byte + self.byte_offset for byte in text.encode("utf-8"))

    def decode(self, tokens: Sequence[int]) -> str:
        raw = bytearray()
        for token in tokens:
            if token < self.byte_offset or token >= self.vocab_size:
                raise TokenDecodeError(token)
            raw.append(token - self.byte_offset)
        return raw.decode("utf-8")


SYNTHETIC_EXAMPLES: Final = (
    CorpusExample("task:add", "synthetic", "def add(a, b):\n    return a + b\n"),
    CorpusExample("task:count_even", "synthetic", "def count_even(values):\n    return sum(1 for value in values if value % 2 == 0)\n"),
    CorpusExample("task:reverse_words", "synthetic", "def reverse_words(text):\n    return ' '.join(reversed(text.split()))\n"),
    CorpusExample("task:merge_intervals", "synthetic", "def merge_intervals(intervals):\n    return sorted(intervals)\n"),
    CorpusExample("task:classify_temperature", "synthetic", "def classify_temperature(celsius):\n    return 'hot' if celsius > 30 else 'cold'\n"),
    CorpusExample("task:slugify_title", "synthetic", "def slugify_title(title):\n    return '-'.join(title.lower().split())\n"),
    CorpusExample("task:is_palindrome", "synthetic", "def is_palindrome(text):\n    cleaned = ''.join(text.lower().split())\n    return cleaned == cleaned[::-1]\n"),
    CorpusExample("task:unique_preserve_order", "synthetic", "def unique_preserve_order(values):\n    return list(dict.fromkeys(values))\n"),
    CorpusExample("task:chunk_pairs", "synthetic", "def chunk_pairs(values):\n    return [values[index:index + 2] for index in range(0, len(values), 2)]\n"),
)

LOCAL_SOURCE_FILES: Final = (
    "src/qiffusion/qwen_tasks.py",
    "src/qiffusion/qwen_file_tasks.py",
    "src/qiffusion/qwen_repair_tasks.py",
)


def build_local_corpus(root: Path | None = None) -> tuple[CorpusExample, ...]:
    base = Path.cwd() if root is None else root
    examples: list[CorpusExample] = []
    readme = base / "README.md"
    if readme.is_file():
        examples.append(CorpusExample("readme", str(readme), readme.read_text(encoding="utf-8")))
    examples.extend(SYNTHETIC_EXAMPLES)
    for relative in LOCAL_SOURCE_FILES:
        path = base / relative
        if path.is_file():
            examples.append(CorpusExample(relative, str(path), path.read_text(encoding="utf-8")))
    return tuple(examples)


def encode_corpus(examples: Sequence[CorpusExample], tokenizer: ByteTokenizer) -> tuple[tuple[int, ...], ...]:
    return tuple((*tokenizer.encode(example.text), tokenizer.eos_id) for example in examples)


def pack_sequences(
    examples: Sequence[CorpusExample],
    tokenizer: ByteTokenizer,
    sequence_length: int,
) -> tuple[tuple[int, ...], ...]:
    encoded = encode_corpus(examples, tokenizer)
    packed: list[tuple[int, ...]] = []
    for sequence in encoded:
        trimmed = sequence[:sequence_length]
        padding = (tokenizer.pad_id,) * (sequence_length - len(trimmed))
        packed.append((*trimmed, *padding))
    return tuple(packed)
