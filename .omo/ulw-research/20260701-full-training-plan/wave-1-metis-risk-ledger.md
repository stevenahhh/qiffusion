# Wave 1: Feasibility and Risk Ledger

## Critical Risks

- Sonnet 4.6-level performance is not a feasible local-training claim. Treat it as a benchmark target matrix.
- Current qiffusion evidence does not show coding capability; existing loops remain smoke/prototype infrastructure.
- Benchmark contamination can produce false progress unless train/eval separation is enforced at row level and near-duplicate level.
- Data rights are under-specified unless every source has a ledger row covering license, access terms, privacy, model-output provenance, and redistribution.

## High Risks

- Local 8GB VRAM can support prototypes and quantized baselines, not full 4B training or frontier claim validation.
- Mechanical token speed can look high while generated code remains unusable.
- The local Qwen teacher/baseline path was previously resource-limited.
- Loss improvement is not capability improvement.

## Planning Mitigations

- Add an honesty harness before training scale-up.
- Build a compliance/source ledger before data ingestion.
- Keep local smoke and cloud training as separate lanes.
- Freeze capability claims until external benchmark gates pass.
- Add contamination scans before every training run.
- Add a checkpoint registry before expensive experiments.
