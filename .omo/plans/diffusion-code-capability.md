# diffusion-code-capability - Work Plan

## TL;DR (For humans)
**What you'll get:** The selected diffusion checkpoint will be trained and tested specifically for coding: function generation, file edit, bug repair, and no-Qwen-fallback proof.
**Why this approach:** Existing Qwen bridge already defines executable coding gates, so diffusion should earn the same claim through generated code that actually runs.
**What it will NOT do:** It will not claim chat capability and will not use Qwen/Ollama at inference time.
**Effort:** Large
**Risk:** High - coding capability requires executable correctness, not just plausible text.
**Decisions I made for you:** Use Qwen only as optional teacher data; reuse and extend qwen-style smoke tests; commit/push after data, training, eval, and no-fallback waves.

Your next move: execute after `.omo/plans/diffusion-scale-ladder.md` writes `selected-checkpoint.json`.

---

> TL;DR (machine): Code-specialization plan for diffusion checkpoint with data, training, code-aware sampling, executable eval, and shared-gate promotion only on true pass.

## Scope
### Must have
- Consume `.omo/ulw-loop/diffusion-scale-ladder/evidence/selected-checkpoint.json`.
- Build a code corpus from local snippets, Qwen passing outputs, repair tasks, and synthetic file edits.
- Add code-generation and code-repair diffusion sampling modes.
- Evaluate generated code with existing function/file-edit/repair smoke tests plus at least one multi-file repair task.
- Write `final-code-eval.json` and shared `status --report` output.
- Commit and push after every verified wave.

### Must NOT have
- No Qwen/Ollama inference fallback.
- No `coding_capability_claim=true` unless executable smoke passes.
- No chat-capability claim.

## Verification strategy
- Test decision: TDD for data/eval/sampler behavior.
- Real-surface QA: CLI code-generate, code-repair, diffusion-code-eval, shared gate.
- Evidence: `.omo/ulw-loop/diffusion-code-capability/evidence/*`

## Execution strategy
### Parallel execution waves
Wave 1: code corpus and provenance.
Wave 2: code-specialized training/adapter.
Wave 3: code-aware sampler and repair mode.
Wave 4: executable code gates and no-fallback proof.
Wave 5: final coding report, commit, push.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | selected checkpoint | 2, 5 | none |
| 2 | 1 | 3, 4, 5 | none |
| 3 | 2 | 4, 5 | none |
| 4 | 2, 3 | 5 | none |
| 5 | 1-4 | final | none |

## Todos
- [ ] 1. Build audited code corpus
  What to do / Must NOT do: Add code corpus command from qwen eval artifacts, local task files, synthetic snippets, and repair examples. Must record provenance and skip failing Qwen outputs.
  Parallelization: Wave 1 | Blocked by: selected checkpoint | Blocks: 2, 5
  References: `src/qiffusion/qwen_eval.py`, `src/qiffusion/qwen_tasks.py`, `.omo/ulw-loop/coding-capability-20260629-1/evidence/G008-C003-repeated-repair.json`
  Acceptance criteria: corpus JSONL exists; every item has `source`, `task_type`, `license_or_origin`, `text`.
  QA scenarios:
  ```
  Scenario: code corpus manifest created
    Tool: powershell
    Invocation: python -m qiffusion.cli diffusion-code corpus --out .omo/ulw-loop/diffusion-code-capability/evidence/task-1-code-corpus.jsonl --manifest .omo/ulw-loop/diffusion-code-capability/evidence/task-1-code-corpus-manifest.json
    Expected: manifest count > 0
    Evidence: .omo/ulw-loop/diffusion-code-capability/evidence/task-1-code-corpus-manifest.json
  ```
  Commit: YES | `feat(diffusion): build code corpus`

- [ ] 2. Continue train selected checkpoint on code data
  What to do / Must NOT do: Add `diffusion-code train` consuming selected checkpoint and code corpus. Must save a new code checkpoint and metrics.
  Parallelization: Wave 2 | Blocked by: 1 | Blocks: 3, 4, 5
  References: `.omo/plans/diffusion-scale-ladder.md`, `.omo/plans/diffusion-llm-self-training.md`
  Acceptance criteria: training report exists with finite loss and checkpoint path.
  QA scenarios:
  ```
  Scenario: code continue-training writes checkpoint
    Tool: powershell
    Invocation: python -m qiffusion.cli diffusion-code train --checkpoint-manifest .omo/ulw-loop/diffusion-scale-ladder/evidence/selected-checkpoint.json --corpus .omo/ulw-loop/diffusion-code-capability/evidence/task-1-code-corpus.jsonl --steps 30 --out .omo/ulw-loop/diffusion-code-capability/evidence/task-2-code.pt --report-out .omo/ulw-loop/diffusion-code-capability/evidence/task-2-code-train.json
    Expected: report status trained
    Evidence: .omo/ulw-loop/diffusion-code-capability/evidence/task-2-code-train.json
  ```
  Commit: YES | `feat(diffusion): train code checkpoint`

- [ ] 3. Add code generation and repair sampling modes
  What to do / Must NOT do: Add CLI modes for function prompt, file-edit prompt, and repair prompt. Must consume diffusion checkpoint only.
  Parallelization: Wave 3 | Blocked by: 2 | Blocks: 4, 5
  References: `src/qiffusion/qwen_file_tasks.py`, `src/qiffusion/qwen_repair_tasks.py`
  Acceptance criteria: deterministic samples for fixed seed; report includes prompt type and checkpoint provenance.
  QA scenarios:
  ```
  Scenario: repair sampler emits candidate code
    Tool: powershell
    Invocation: python -m qiffusion.cli diffusion-code sample --checkpoint .omo/ulw-loop/diffusion-code-capability/evidence/task-2-code.pt --task repair --seed 7 --out .omo/ulw-loop/diffusion-code-capability/evidence/task-3-repair-sample.json
    Expected: JSON contains generated_code
    Evidence: .omo/ulw-loop/diffusion-code-capability/evidence/task-3-repair-sample.json
  ```
  Commit: YES | `feat(diffusion): add code repair sampler`

- [ ] 4. Add executable coding gate for diffusion outputs
  What to do / Must NOT do: Add `diffusion-code eval` that runs function, file-edit, repair, and one multi-file repair smoke. Must reuse smoke helpers as evaluators only.
  Parallelization: Wave 4 | Blocked by: 2, 3 | Blocks: 5
  References: `src/qiffusion/qwen_eval.py:93-142`, `src/qiffusion/decision.py:25-33`
  Acceptance criteria: report has `fixtures_status`, `code_smoke_status`, `coding_capability_claim`; status gate promotes only on actual pass.
  QA scenarios:
  ```
  Scenario: coding eval writes gate-compatible report
    Tool: powershell
    Invocation: python -m qiffusion.cli diffusion-code eval --checkpoint .omo/ulw-loop/diffusion-code-capability/evidence/task-2-code.pt --out .omo/ulw-loop/diffusion-code-capability/evidence/task-4-code-eval.json
    Expected: JSON has backend diffusion and task_results
    Evidence: .omo/ulw-loop/diffusion-code-capability/evidence/task-4-code-eval.json
  ```
  Commit: YES | `feat(diffusion): add coding eval gate`

- [ ] 5. Prove no hidden Qwen fallback and publish coding report
  What to do / Must NOT do: Add static and runtime no-fallback checks, run shared gate, write `final-code-eval.json`, commit/push. Must not force claim true.
  Parallelization: Wave 5 | Blocked by: 1-4 | Blocks: final
  References: `src/qiffusion/qwen_ollama.py`, `src/qiffusion/qwen_bridge.py`
  Acceptance criteria: no-fallback report passes; final code eval exists; gate result captured.
  QA scenarios:
  ```
  Scenario: no qwen fallback proof
    Tool: powershell
    Invocation: python -m qiffusion.cli diffusion-code no-fallback --out .omo/ulw-loop/diffusion-code-capability/evidence/final-no-fallback.json
    Expected: JSON verdict pass
    Evidence: .omo/ulw-loop/diffusion-code-capability/evidence/final-no-fallback.json
  ```
  Commit: YES | `feat(diffusion): publish coding capability report`

## Final verification wave
- [ ] F1. Code data provenance audit.
- [ ] F2. Code eval correctness audit.
- [ ] F3. Shared gate audit.
- [ ] F4. No-fallback audit.

## Commit strategy
- Commit/push after corpus, code training, sampler, eval gate, and final report.
- Final commit footer: `Plan: .omo/plans/diffusion-code-capability.md`.

## Success criteria
- `final-code-eval.json` is gate-compatible.
- If `coding_capability_claim=true`, executable code smoke actually passed.
- No diffusion inference path calls Qwen/Ollama.
