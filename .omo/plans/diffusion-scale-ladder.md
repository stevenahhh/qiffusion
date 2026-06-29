# diffusion-scale-ladder - Work Plan

## TL;DR (For humans)
**What you'll get:** A disciplined scale-up path from the tiny diffusion checkpoint to small and mid checkpoints, with clear evidence before any 4B attempt.
**Why this approach:** It prevents wasting compute on a large model before data, objective, sampling, and eval are stable.
**What it will NOT do:** It will not run 4B training by default or claim final capability.
**Effort:** Large
**Risk:** High - scaling introduces compute, checkpoint, and regression risk.
**Decisions I made for you:** Use tiny -> small -> mid -> target-4b-ready profiles; commit/push after each verified profile.

Your next move: execute after `.omo/plans/diffusion-llm-self-training.md` completes.

---

> TL;DR (machine): Scale ladder plan with profile configs, data growth, train/sample/eval runs, cost/speed reports, and commit/push checkpoints.

## Scope
### Must have
- Consume the tiny checkpoint/eval report from `.omo/plans/diffusion-llm-self-training.md`.
- Add named scale profiles: `tiny`, `small`, `mid`, `target_4b_ready`.
- Train/eval `small` and `mid` only when previous profile passes.
- Produce `selected-checkpoint.json` for later coding/chat plans.
- Commit and push after each profile passes its QA.

### Must NOT have
- No default target 4B run.
- No capability claim based only on loss.
- No committed checkpoint binaries.

## Verification strategy
- Test decision: TDD for config/report code; real CLI QA for each profile.
- Evidence: `.omo/ulw-loop/diffusion-scale-ladder/evidence/*.json|*.txt`

## Execution strategy
### Parallel execution waves
Wave 1: profile/config and artifact policy.
Wave 2: data scale and dedupe.
Wave 3: small profile train/sample/eval.
Wave 4: mid profile train/sample/eval and selected checkpoint.
Wave 5: 4B readiness report without running 4B.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | tiny-loop final eval | 3, 4, 6 | 2 |
| 2 | tiny-loop corpus | 3, 4 | 1 |
| 3 | 1, 2 | 4, 6 | none |
| 4 | 3 | 5, 6 | none |
| 5 | 4 | 6 | none |
| 6 | 3, 4, 5 | final | none |

## Todos
- [ ] 1. Add scale profile config and checkpoint policy
  What to do / Must NOT do: Add config/profile files or module entries for `tiny`, `small`, `mid`, `target_4b_ready`; document artifact directories and ignored binary policy. Must not start training here.
  Parallelization: Wave 1 | Blocked by: tiny-loop final eval | Blocks: 3, 4, 6
  References: `.omo/plans/diffusion-llm-self-training.md`, `.gitignore`, `pyproject.toml:9-13`
  Acceptance criteria: profile loader test passes; `target_4b_ready` is disabled by default.
  QA scenarios:
  ```
  Scenario: profiles list includes disabled target 4b
    Tool: powershell
    Invocation: python -m qiffusion.cli diffusion-scale profiles --out .omo/ulw-loop/diffusion-scale-ladder/evidence/task-1-profiles.json
    Expected: JSON includes target_4b_ready and enabled=false
    Evidence: .omo/ulw-loop/diffusion-scale-ladder/evidence/task-1-profiles.json
  ```
  Commit: YES | `feat(diffusion): add scale profiles`

- [ ] 2. Add scaled corpus manifest and dedupe report
  What to do / Must NOT do: Build corpus manifests from local corpus plus teacher JSONL; include counts, hashes, and license/provenance fields. Must not download data.
  Parallelization: Wave 2 | Blocked by: tiny-loop corpus | Blocks: 3, 4
  References: `.omo/plans/diffusion-llm-self-training.md`, `src/qiffusion/qwen_eval.py`
  Acceptance criteria: manifest test passes; duplicate snippets are removed deterministically.
  QA scenarios:
  ```
  Scenario: scaled corpus manifest exists
    Tool: powershell
    Invocation: python -m qiffusion.cli diffusion-corpus manifest --out .omo/ulw-loop/diffusion-scale-ladder/evidence/task-2-corpus.json
    Expected: JSON includes example_count and sha256
    Evidence: .omo/ulw-loop/diffusion-scale-ladder/evidence/task-2-corpus.json
  ```
  Commit: YES | `feat(diffusion): add scale corpus manifest`

- [ ] 3. Train and evaluate small profile
  What to do / Must NOT do: Run small profile train/sample/eval with bounded CPU/GPU settings and write reports. Must not proceed if loss is non-finite or sample/eval command fails.
  Parallelization: Wave 3 | Blocked by: 1, 2 | Blocks: 4, 6
  References: `.omo/plans/diffusion-llm-self-training.md`
  Acceptance criteria: small train/eval reports exist; status gate remains truthful.
  QA scenarios:
  ```
  Scenario: small profile train sample eval
    Tool: powershell
    Invocation: python -m qiffusion.cli diffusion-scale run --profile small --out-dir .omo/ulw-loop/diffusion-scale-ladder/evidence/small
    Expected: train.json, sample.json, eval.json exist
    Evidence: .omo/ulw-loop/diffusion-scale-ladder/evidence/small/eval.json
  ```
  Commit: YES | `feat(diffusion): train small profile`

- [ ] 4. Train and evaluate mid profile
  What to do / Must NOT do: Run mid profile only after small passes. Must capture runtime, memory, and sample quality report.
  Parallelization: Wave 4 | Blocked by: 3 | Blocks: 5, 6
  References: `.omo/ulw-loop/diffusion-scale-ladder/evidence/small/eval.json`
  Acceptance criteria: mid report exists and records runtime/memory; failure is recorded rather than hidden.
  QA scenarios:
  ```
  Scenario: mid profile produces runtime report
    Tool: powershell
    Invocation: python -m qiffusion.cli diffusion-scale run --profile mid --out-dir .omo/ulw-loop/diffusion-scale-ladder/evidence/mid
    Expected: metrics include runtime_seconds and peak_memory_mb
    Evidence: .omo/ulw-loop/diffusion-scale-ladder/evidence/mid/train.json
  ```
  Commit: YES | `feat(diffusion): train mid profile`

- [ ] 5. Write 4B readiness report without running 4B
  What to do / Must NOT do: Estimate data, memory, checkpoint, sampling, and eval prerequisites for 4B. Must not launch 4B.
  Parallelization: Wave 5 | Blocked by: 4 | Blocks: 6
  References: LLaDA, Dream, DiffuCoder, DiffusionGemma public references; local mid metrics
  Acceptance criteria: report states exact missing prerequisites and go/no-go.
  QA scenarios:
  ```
  Scenario: 4b readiness is explicit and non-running
    Tool: powershell
    Invocation: python -m qiffusion.cli diffusion-scale readiness --target 4b --out .omo/ulw-loop/diffusion-scale-ladder/evidence/task-5-4b-readiness.json
    Expected: JSON contains launch_performed=false
    Evidence: .omo/ulw-loop/diffusion-scale-ladder/evidence/task-5-4b-readiness.json
  ```
  Commit: YES | `docs(diffusion): record 4b readiness gate`

- [ ] 6. Select checkpoint for downstream code/chat plans
  What to do / Must NOT do: Choose the best available checkpoint from tiny/small/mid by deterministic policy and write `selected-checkpoint.json`. Must not pick a failed checkpoint.
  Parallelization: Wave 5 | Blocked by: 3, 4, 5 | Blocks: final
  References: all profile reports
  Acceptance criteria: selected checkpoint exists and points to an existing artifact; git status has no binary staged.
  QA scenarios:
  ```
  Scenario: selected checkpoint manifest exists
    Tool: powershell
    Invocation: python -m qiffusion.cli diffusion-scale select --out .omo/ulw-loop/diffusion-scale-ladder/evidence/selected-checkpoint.json
    Expected: JSON selected_checkpoint path exists
    Evidence: .omo/ulw-loop/diffusion-scale-ladder/evidence/selected-checkpoint.json
  ```
  Commit: YES | `chore(diffusion): select scaled checkpoint`

## Final verification wave
- [ ] F1. Profile config audit.
- [ ] F2. Small/mid report audit.
- [ ] F3. 4B readiness audit.
- [ ] F4. Git artifact audit.

## Commit strategy
- Commit/push after profile config, corpus manifest, small run, mid run, readiness report, and selected checkpoint manifest.
- Final commit footer: `Plan: .omo/plans/diffusion-scale-ladder.md`.

## Success criteria
- A selected scaled checkpoint manifest exists for later code/chat plans.
- Target 4B is prepared but not accidentally launched.
