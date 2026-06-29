# diffusion-chat-capability - Work Plan

## TL;DR (For humans)
**What you'll get:** A chat-capability track for the diffusion checkpoint: instruction data, chat sampling, multi-turn evaluation, safety checks, and coding regression.
**Why this approach:** Chat capability is different from coding capability and needs its own gate, but it must not break the coding gate.
**What it will NOT do:** It will not claim final release readiness alone.
**Effort:** Large
**Risk:** High - chat quality is easy to overclaim without scripted multi-turn evidence.
**Decisions I made for you:** Use deterministic scripted chat evals first; rerun coding regression after chat training; commit/push after every verified wave.

Your next move: execute after `.omo/plans/diffusion-code-capability.md` has a selected checkpoint/report.

---

> TL;DR (machine): Instruction/chat specialization plan with multi-turn eval, refusal/context gates, coding regression, and no final claim until release gate.

## Scope
### Must have
- Consume selected checkpoint from scale/code stage.
- Build local instruction and multi-turn corpus.
- Add `diffusion-chat sample` and `diffusion-chat eval`.
- Evaluate instruction following, context retention, basic refusal/safety, and coding regression.
- Commit and push after data, train, sampler, eval, and regression waves.

### Must NOT have
- No subjective-only chat pass.
- No final model release claim.
- No weakening coding gate after chat tuning.

## Verification strategy
- Test decision: TDD for chat data/eval parsing.
- Real-surface QA: CLI chat sample/eval plus coding regression.
- Evidence: `.omo/ulw-loop/diffusion-chat-capability/evidence/*`

## Execution strategy
### Parallel execution waves
Wave 1: chat corpus.
Wave 2: chat continue-training.
Wave 3: chat sampler.
Wave 4: scripted chat gate.
Wave 5: coding regression and final chat report.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | code/scale checkpoint | 2 | none |
| 2 | 1 | 3, 4, 5 | none |
| 3 | 2 | 4 | none |
| 4 | 2, 3 | 5 | none |
| 5 | 4 | final | none |

## Todos
- [ ] 1. Build instruction and multi-turn corpus
  What to do / Must NOT do: Create local chat corpus with instruction, Q&A, coding assistant, and refusal examples. Must track provenance and avoid external downloads in tests.
  Parallelization: Wave 1 | Blocked by: code/scale checkpoint | Blocks: 2
  References: `README.md:3-18`, `.omo/plans/diffusion-code-capability.md`
  Acceptance criteria: corpus manifest exists with task categories.
  QA scenarios:
  ```
  Scenario: chat corpus manifest exists
    Tool: powershell
    Invocation: python -m qiffusion.cli diffusion-chat corpus --out .omo/ulw-loop/diffusion-chat-capability/evidence/task-1-chat.jsonl --manifest .omo/ulw-loop/diffusion-chat-capability/evidence/task-1-chat-manifest.json
    Expected: manifest includes instruction and multi_turn counts
    Evidence: .omo/ulw-loop/diffusion-chat-capability/evidence/task-1-chat-manifest.json
  ```
  Commit: YES | `feat(diffusion): build chat corpus`

- [ ] 2. Continue-train chat checkpoint
  What to do / Must NOT do: Train from selected/code checkpoint on chat corpus. Must preserve checkpoint provenance.
  Parallelization: Wave 2 | Blocked by: 1 | Blocks: 3, 4, 5
  References: `.omo/plans/diffusion-scale-ladder.md`, `.omo/plans/diffusion-code-capability.md`
  Acceptance criteria: chat training report exists with finite loss and checkpoint path.
  QA scenarios:
  ```
  Scenario: chat training writes checkpoint
    Tool: powershell
    Invocation: python -m qiffusion.cli diffusion-chat train --corpus .omo/ulw-loop/diffusion-chat-capability/evidence/task-1-chat.jsonl --steps 30 --out .omo/ulw-loop/diffusion-chat-capability/evidence/task-2-chat.pt --report-out .omo/ulw-loop/diffusion-chat-capability/evidence/task-2-chat-train.json
    Expected: report status trained
    Evidence: .omo/ulw-loop/diffusion-chat-capability/evidence/task-2-chat-train.json
  ```
  Commit: YES | `feat(diffusion): train chat checkpoint`

- [ ] 3. Add chat sampling CLI
  What to do / Must NOT do: Add conversation formatting and `diffusion-chat sample`. Must output structured turns and checkpoint provenance.
  Parallelization: Wave 3 | Blocked by: 2 | Blocks: 4
  References: `.omo/plans/diffusion-llm-self-training.md`
  Acceptance criteria: deterministic fixed-seed chat sample report.
  QA scenarios:
  ```
  Scenario: chat sampler writes response
    Tool: powershell
    Invocation: python -m qiffusion.cli diffusion-chat sample --checkpoint .omo/ulw-loop/diffusion-chat-capability/evidence/task-2-chat.pt --message "Explain add(a,b)" --seed 3 --out .omo/ulw-loop/diffusion-chat-capability/evidence/task-3-chat-sample.json
    Expected: JSON includes assistant_message
    Evidence: .omo/ulw-loop/diffusion-chat-capability/evidence/task-3-chat-sample.json
  ```
  Commit: YES | `feat(diffusion): add chat sampler`

- [ ] 4. Add scripted chat eval gate
  What to do / Must NOT do: Evaluate deterministic multi-turn tasks, context retention, refusal boundary, and concise instruction following. Must not pass from one cherry-picked output.
  Parallelization: Wave 4 | Blocked by: 2, 3 | Blocks: 5
  References: `src/qiffusion/decision.py:25-33` for report style but create separate chat fields.
  Acceptance criteria: report includes `chat_fixtures_status`, `multi_turn_status`, `safety_status`, `chat_capability_claim`.
  QA scenarios:
  ```
  Scenario: chat eval writes report
    Tool: powershell
    Invocation: python -m qiffusion.cli diffusion-chat eval --checkpoint .omo/ulw-loop/diffusion-chat-capability/evidence/task-2-chat.pt --out .omo/ulw-loop/diffusion-chat-capability/evidence/task-4-chat-eval.json
    Expected: JSON includes chat_capability_claim
    Evidence: .omo/ulw-loop/diffusion-chat-capability/evidence/task-4-chat-eval.json
  ```
  Commit: YES | `feat(diffusion): add chat eval gate`

- [ ] 5. Rerun coding regression and publish final chat report
  What to do / Must NOT do: Rerun diffusion coding eval against chat-tuned checkpoint or selected code checkpoint; attach result to `final-chat-eval.json`. Must not mark chat pass if coding regression breaks the final target path.
  Parallelization: Wave 5 | Blocked by: 4 | Blocks: final
  References: `.omo/plans/diffusion-code-capability.md`
  Acceptance criteria: final chat report includes coding regression reference and no-fallback proof.
  QA scenarios:
  ```
  Scenario: final chat report exists
    Tool: powershell
    Invocation: python -m qiffusion.cli diffusion-chat finalize --chat-report .omo/ulw-loop/diffusion-chat-capability/evidence/task-4-chat-eval.json --out .omo/ulw-loop/diffusion-chat-capability/evidence/final-chat-eval.json
    Expected: JSON includes chat_capability_claim and coding_regression
    Evidence: .omo/ulw-loop/diffusion-chat-capability/evidence/final-chat-eval.json
  ```
  Commit: YES | `feat(diffusion): publish chat capability report`

## Final verification wave
- [ ] F1. Chat data provenance audit.
- [ ] F2. Multi-turn eval audit.
- [ ] F3. Safety/refusal audit.
- [ ] F4. Coding regression audit.

## Commit strategy
- Commit/push after chat corpus, chat training, sampler, eval gate, and final report.
- Final commit footer: `Plan: .omo/plans/diffusion-chat-capability.md`.

## Success criteria
- `final-chat-eval.json` exists and is reproducible.
- Chat claim is separate from coding claim.
- Coding regression is attached after chat tuning.
