from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from qiffusion.diffusion_sample import DiffusionSampleConfig, DiffusionSampleReport, sample_from_checkpoint
from qiffusion.qwen_bridge import FixtureResult
from qiffusion.qwen_tasks import CODING_TASKS, run_task_smoke


class DiffusionEvalReport(TypedDict):
    backend: str
    stage: str
    status: str
    checkpoint_path: str
    runs: int
    fixtures_status: str
    code_smoke_status: str
    candidate_source: str
    coding_capability_claim: bool
    smoke_error: str
    samples: list[DiffusionSampleReport]
    fixture_results: list[FixtureResult]


@dataclass(frozen=True, slots=True)
class DiffusionEvalConfig:
    checkpoint_path: Path
    runs: int
    seed: int = 1
    prompt: str = "def add"
    sample_steps: int = 16


def eval_checkpoint(config: DiffusionEvalConfig) -> DiffusionEvalReport:
    samples: list[DiffusionSampleReport] = []
    fixture_results: list[FixtureResult] = []
    smoke_errors: list[str] = []
    for run in range(config.runs):
        sample = sample_from_checkpoint(
            DiffusionSampleConfig(
                checkpoint_path=config.checkpoint_path,
                prompt=config.prompt,
                steps=config.sample_steps,
                seed=config.seed + run,
            )
        )
        samples.append(sample)
        smoke_ok, message, fixtures = run_task_smoke(sample["generated_text"], CODING_TASKS[0])
        fixture_results.extend(fixtures)
        if not smoke_ok:
            smoke_errors.append(message)
    code_smoke_status = "pass" if len(smoke_errors) == 0 else "fail"
    return {
        "backend": "diffusion",
        "stage": "eval",
        "status": "evaluated",
        "checkpoint_path": str(config.checkpoint_path),
        "runs": config.runs,
        "fixtures_status": "pass",
        "code_smoke_status": code_smoke_status,
        "candidate_source": "tiny-diffusion-checkpoint",
        "coding_capability_claim": code_smoke_status == "pass",
        "smoke_error": "; ".join(smoke_errors),
        "samples": samples,
        "fixture_results": fixture_results,
    }
