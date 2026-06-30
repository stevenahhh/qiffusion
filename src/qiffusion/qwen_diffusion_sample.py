from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol, TypedDict, assert_never

import torch
from torch import Tensor

from qiffusion.diffusion_config import BYTE_TOKEN_OFFSET, MASK_TOKEN_ID
from qiffusion.diffusion_data import ByteTokenizer
from qiffusion.qwen_diffusion_config import JsonValue
from qiffusion.qwen_diffusion_model import QwenDenoiserConfig, TinyQwenTokenDenoiser


SCHEMA_VERSION: Final = 1
SamplerAlgorithm = Literal["p2", "confidence_greedy"]


class EarlyStopMetadata(TypedDict):
    stopped: bool
    reason: str
    completed_steps: int


class SamplerHistoryEntry(TypedDict):
    step: int
    token_index: int
    token_id_before: int
    token_id: int
    confidence: float
    entropy: float
    temperature: float
    top_k: int
    candidate_token_ids: list[int]


class QwenMaskSampleReport(TypedDict):
    schema_version: int
    backend: str
    stage: str
    status: str
    prompt: str
    generated_text: str
    generated_token_ids: list[int]
    steps: int
    seed: int
    algorithm: str
    temperature: float
    top_k: int
    fallback_used: bool
    early_stop: EarlyStopMetadata
    history: list[SamplerHistoryEntry]
    coding_capability_claim: bool


class QwenTokenDenoiser(Protocol):
    config: QwenDenoiserConfig

    def eval(self) -> QwenTokenDenoiser: ...

    def __call__(self, input_ids: Tensor) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class SamplerValidationError(Exception):
    field: str
    value: str
    expected: str

    def __str__(self) -> str:
        return f"{self.field}={self.value!r} is invalid; expected {self.expected}"


@dataclass(frozen=True, slots=True)
class QwenSamplerSettings:
    steps: int
    seed: int
    temperature: float = 1.0
    top_k: int = 16
    algorithm: str = "p2"

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise SamplerValidationError("steps", str(self.steps), "positive integer")
        if self.temperature <= 0:
            raise SamplerValidationError("temperature", str(self.temperature), "positive number")
        if self.top_k <= 0:
            raise SamplerValidationError("top_k", str(self.top_k), "positive integer")
        _ = _parse_algorithm(self.algorithm)


@dataclass(frozen=True, slots=True)
class QwenMaskSampleConfig:
    prompt: str
    settings: QwenSamplerSettings


@dataclass(frozen=True, slots=True)
class TokenChoice:
    token_id: int
    confidence: float
    entropy: float
    candidate_token_ids: list[int]


def sample_qwen_tokens(model: QwenTokenDenoiser, config: QwenMaskSampleConfig) -> QwenMaskSampleReport:
    settings = config.settings
    tokenizer = ByteTokenizer()
    prompt_ids = list(tokenizer.encode(config.prompt))
    canvas = [*prompt_ids, *((MASK_TOKEN_ID,) * settings.steps)]
    del canvas[model.config.max_length :]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(settings.seed)
    history: list[SamplerHistoryEntry] = []

    model.eval()
    with torch.no_grad():
        for step in range(1, settings.steps + 1):
            mask_index = _first_mask_index(canvas)
            if mask_index is None:
                break
            input_ids = torch.tensor([canvas], dtype=torch.long)
            logits = model(input_ids)
            choice = _choose_token(logits[0, mask_index], settings, generator)
            history.append(
                {
                    "step": step,
                    "token_index": mask_index,
                    "token_id_before": MASK_TOKEN_ID,
                    "token_id": choice.token_id,
                    "confidence": choice.confidence,
                    "entropy": choice.entropy,
                    "temperature": settings.temperature,
                    "top_k": min(settings.top_k, model.config.vocab_size),
                    "candidate_token_ids": choice.candidate_token_ids,
                },
            )
            canvas[mask_index] = choice.token_id

    generated_token_ids = canvas[len(prompt_ids) :]
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": "qwen_token_diffusion",
        "stage": "sample",
        "status": "sampled",
        "prompt": config.prompt,
        "generated_text": _decode_byte_tokens(canvas),
        "generated_token_ids": generated_token_ids,
        "steps": settings.steps,
        "seed": settings.seed,
        "algorithm": settings.algorithm,
        "temperature": settings.temperature,
        "top_k": settings.top_k,
        "fallback_used": False,
        "early_stop": _early_stop(settings.steps, len(history)),
        "history": history,
        "coding_capability_claim": False,
    }


def write_sample_evidence(path: Path, prompt: str, steps: int, seed: int) -> None:
    torch.manual_seed(1)
    config = QwenDenoiserConfig.tiny(vocab_size=260, max_length=max(len(prompt.encode("utf-8")) + steps, 1))
    model = TinyQwenTokenDenoiser(config)
    settings = QwenSamplerSettings(steps=steps, seed=seed, temperature=1.0, top_k=8)
    _write_json(path, sample_qwen_tokens(model, QwenMaskSampleConfig(prompt=prompt, settings=settings)))


def write_sampler_failure_probe(path: Path, algorithm: str) -> None:
    try:
        _ = QwenSamplerSettings(steps=1, seed=1, algorithm=algorithm)
    except SamplerValidationError as exc:
        _write_json(
            path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "rejected",
                "algorithm": algorithm,
                "field": exc.field,
                "error": str(exc),
                "fallback_used": False,
            },
        )
        return
    _write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "accepted",
            "algorithm": algorithm,
            "fallback_used": False,
        },
    )


def _choose_token(logits: Tensor, settings: QwenSamplerSettings, generator: torch.Generator) -> TokenChoice:
    scaled = torch.nan_to_num(
        logits / settings.temperature,
        nan=-torch.inf,
        posinf=torch.finfo(logits.dtype).max,
        neginf=-torch.inf,
    ).clone()
    scaled[MASK_TOKEN_ID] = -torch.inf
    top_values, top_indices = torch.topk(scaled, k=min(settings.top_k, scaled.numel() - 1))
    probabilities = torch.softmax(top_values, dim=0)
    probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
    probability_total = probabilities.sum()
    if (not bool(torch.isfinite(probability_total).item())) or float(probability_total.item()) <= 0.0:
        probabilities = torch.full_like(probabilities, 1.0 / probabilities.numel())
    else:
        probabilities = probabilities / probability_total
    match _parse_algorithm(settings.algorithm):
        case "p2":
            selected_index = int(torch.multinomial(probabilities, num_samples=1, generator=generator).item())
        case "confidence_greedy":
            selected_index = int(torch.argmax(probabilities).item())
        case unreachable:
            assert_never(unreachable)
    selected_probability = float(probabilities[selected_index].item())
    safe_probabilities = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
    entropy = float(-(probabilities * torch.log(safe_probabilities)).sum().item())
    return TokenChoice(
        token_id=int(top_indices[selected_index].item()),
        confidence=selected_probability,
        entropy=entropy,
        candidate_token_ids=[int(token.item()) for token in top_indices],
    )


def _parse_algorithm(value: str) -> SamplerAlgorithm:
    for algorithm in ("p2", "confidence_greedy"):
        if value == algorithm:
            return algorithm
    raise SamplerValidationError("algorithm", value, "p2 or confidence_greedy")


def _first_mask_index(tokens: list[int]) -> int | None:
    for index, token in enumerate(tokens):
        if token == MASK_TOKEN_ID:
            return index
    return None


def _early_stop(requested_steps: int, completed_steps: int) -> EarlyStopMetadata:
    stopped = completed_steps < requested_steps
    return {"stopped": stopped, "reason": "all_masks_filled" if stopped else "max_steps", "completed_steps": completed_steps}


def _decode_byte_tokens(tokens: list[int]) -> str:
    byte_values = [token - BYTE_TOKEN_OFFSET for token in tokens if token >= BYTE_TOKEN_OFFSET]
    return bytes(byte_values).decode("utf-8", errors="replace")


def _write_json(path: Path, payload: QwenMaskSampleReport | dict[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
