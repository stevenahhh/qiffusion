from __future__ import annotations

from typing import Literal, TypedDict

from qiffusion.diffusion_config import DIFFUSION_BACKEND, DIFFUSION_REPORT_SCHEMA_VERSION, DiffusionStage

DiffusionStatus = Literal["stub"]


class DiffusionStubReport(TypedDict):
    schema_version: int
    backend: str
    stage: DiffusionStage
    status: DiffusionStatus
    fixtures_status: str
    code_smoke_status: str
    candidate_source: str
    coding_capability_claim: bool
    next_step: str


def diffusion_stub_report(stage: DiffusionStage) -> DiffusionStubReport:
    return {
        "schema_version": DIFFUSION_REPORT_SCHEMA_VERSION,
        "backend": DIFFUSION_BACKEND,
        "stage": stage,
        "status": "stub",
        "fixtures_status": "not_run",
        "code_smoke_status": "not_run",
        "candidate_source": "none",
        "coding_capability_claim": False,
        "next_step": f"implement diffusion {stage} in a later wave",
    }

