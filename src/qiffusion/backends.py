from __future__ import annotations

from typing import Literal, TypedDict

BackendName = Literal["diffusion", "qwen_bridge"]


class BackendStatusReport(TypedDict):
    backend: BackendName
    status: str
    fixtures_status: str
    code_smoke_status: str
    candidate_source: str
    coding_capability_claim: bool
    next_step: str


def diffusion_status() -> BackendStatusReport:
    return {
        "backend": "diffusion",
        "status": "scaffold_ready",
        "fixtures_status": "not_run",
        "code_smoke_status": "not_run",
        "candidate_source": "none",
        "coding_capability_claim": False,
        "next_step": "import ELF block diffusion training and sampling loop",
    }
