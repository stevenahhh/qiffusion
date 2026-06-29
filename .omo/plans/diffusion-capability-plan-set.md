# diffusion-capability-plan-set - Work Plan

## TL;DR (For humans)
**What you'll get:** A continuous execution chain from the existing tiny diffusion training loop to a final coding-and-chat capable diffusion model release gate.

**Why this approach:** The current plan proves a tiny loop only. This plan set makes the later work executable in order, with evidence handoffs and commit/push checkpoints between stages.

**What it will NOT do:** It will not implement the model now. It will not let a later worker skip gates. It will not let Qwen/Ollama stand in for diffusion inference.

**Effort:** XL
**Risk:** High - it spans scaling, code capability, chat capability, sampling/serving, and final release claims.
**Decisions I made for you:** Execute in this order: `diffusion-llm-self-training` -> `diffusion-scale-ladder` -> `diffusion-code-capability` -> `diffusion-chat-capability` -> `diffusion-final-release-gate`; each stage commits and pushes after verified sub-waves.

Your next move: run `$omo:start-work` against this plan set or start with `.omo/plans/diffusion-llm-self-training.md`; the later plans name their prerequisites explicitly.

---

> TL;DR (machine): Meta-plan linking five executable plans into a continuous diffusion coding/chat capability path with intermediate commit/push checkpoints.

## Scope
### Must have
- Treat `.omo/plans/diffusion-llm-self-training.md` as stage 0.
- Execute later plans only after their prerequisite evidence exists.
- Each stage must produce reports that the next stage consumes.
- Every wave must include a commit/push checkpoint after tests and real CLI QA pass.
- Final release requires the same checkpoint lineage to pass coding and chat gates.

### Must NOT have (guardrails, anti-slop, scope boundaries)
- No direct jump from tiny loop to final claim.
- No hidden Qwen/Ollama fallback in diffusion inference.
- No committed large checkpoints.
- No weakening of existing shared coding gate.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: plan-level validation plus each child plan's TDD/CLI QA.
- Evidence: `.omo/ulw-loop/diffusion-capability-plan-set/evidence/*.json|*.txt`

## Execution strategy
### Parallel execution waves
Wave A: complete `diffusion-llm-self-training`.
Wave B: complete `diffusion-scale-ladder`.
Wave C: complete `diffusion-code-capability`.
Wave D: complete `diffusion-chat-capability`.
Wave E: complete `diffusion-final-release-gate`.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | none | 2 | none |
| 2 | 1 | 3 | none |
| 3 | 2 | 4, 5 | none |
| 4 | 3 | 5 | none |
| 5 | 3, 4 | final | none |

## Todos
> Implementation + Test = ONE todo. Never separate.

- [ ] 1. Execute tiny self-training plan to produce first real checkpoint
  What to do / Must NOT do: Run `.omo/plans/diffusion-llm-self-training.md` end to end. Must produce train/sample/eval/status artifacts and push commits. Must not claim coding/chat capability.
  Parallelization: Wave A | Blocked by: none | Blocks: 2
  References: `.omo/plans/diffusion-llm-self-training.md`, `src/qiffusion/decision.py:25-33`, `README.md:8`
  Acceptance criteria: final evidence contains a tiny checkpoint path, sample JSON, eval JSON, and gate output; `git log -1 --oneline` after stage shows a pushed implementation commit.
  QA scenarios:
  ```
  Scenario: stage zero handoff exists
    Tool: powershell
    Invocation: Test-Path .omo/ulw-loop/diffusion-llm-self-training/evidence/final-eval.json
    Expected: True
    Evidence: .omo/ulw-loop/diffusion-capability-plan-set/evidence/stage-0-handoff.txt

  Scenario: stage zero did not promote falsely
    Tool: powershell
    Invocation: python -m qiffusion.cli status --report .omo/ulw-loop/diffusion-llm-self-training/evidence/final-eval.json > .omo/ulw-loop/diffusion-capability-plan-set/evidence/stage-0-gate.txt
    Expected: continue unless executable code smoke actually passed
    Evidence: .omo/ulw-loop/diffusion-capability-plan-set/evidence/stage-0-gate.txt
  ```
  Commit: YES | `feat(diffusion): complete tiny self-training loop`

- [ ] 2. Execute scale ladder plan through selected checkpoint
  What to do / Must NOT do: Run `.omo/plans/diffusion-scale-ladder.md`. Must produce a selected scaled checkpoint manifest. Must not run target 4B by default.
  Parallelization: Wave B | Blocked by: 1 | Blocks: 3
  References: `.omo/plans/diffusion-scale-ladder.md`
  Acceptance criteria: scale manifest names tiny/small/mid status and selected checkpoint; push succeeds after each verified profile.
  QA scenarios:
  ```
  Scenario: scale manifest exists
    Tool: powershell
    Invocation: Test-Path .omo/ulw-loop/diffusion-scale-ladder/evidence/selected-checkpoint.json
    Expected: True
    Evidence: .omo/ulw-loop/diffusion-capability-plan-set/evidence/stage-1-scale.txt
  ```
  Commit: YES | `feat(diffusion): complete scale ladder checkpoint`

- [ ] 3. Execute coding capability plan
  What to do / Must NOT do: Run `.omo/plans/diffusion-code-capability.md` against the selected checkpoint. Must produce a coding eval report and shared-gate output. Must not use Qwen for inference.
  Parallelization: Wave C | Blocked by: 2 | Blocks: 4, 5
  References: `.omo/plans/diffusion-code-capability.md`, `src/qiffusion/qwen_eval.py`
  Acceptance criteria: coding report either promotes honestly or records exact failing tasks; no-fallback proof exists; commits pushed after data, training, eval waves.
  QA scenarios:
  ```
  Scenario: diffusion coding gate report exists
    Tool: powershell
    Invocation: Test-Path .omo/ulw-loop/diffusion-code-capability/evidence/final-code-eval.json
    Expected: True
    Evidence: .omo/ulw-loop/diffusion-capability-plan-set/evidence/stage-2-code.txt
  ```
  Commit: YES | `feat(diffusion): complete coding capability gate`

- [ ] 4. Execute chat capability plan
  What to do / Must NOT do: Run `.omo/plans/diffusion-chat-capability.md`. Must preserve coding regression result and add chat report. Must not market chat capability from a single sample.
  Parallelization: Wave D | Blocked by: 3 | Blocks: 5
  References: `.omo/plans/diffusion-chat-capability.md`
  Acceptance criteria: chat report passes scripted multi-turn gate or records failures; coding regression rerun is attached.
  QA scenarios:
  ```
  Scenario: diffusion chat gate report exists
    Tool: powershell
    Invocation: Test-Path .omo/ulw-loop/diffusion-chat-capability/evidence/final-chat-eval.json
    Expected: True
    Evidence: .omo/ulw-loop/diffusion-capability-plan-set/evidence/stage-3-chat.txt
  ```
  Commit: YES | `feat(diffusion): complete chat capability gate`

- [ ] 5. Execute final release gate
  What to do / Must NOT do: Run `.omo/plans/diffusion-final-release-gate.md`. Must aggregate code and chat reports from the same checkpoint lineage and publish a release manifest. Must not promote partial capability.
  Parallelization: Wave E | Blocked by: 3, 4 | Blocks: final verification
  References: `.omo/plans/diffusion-final-release-gate.md`, `README.md:8`
  Acceptance criteria: release manifest says `coding_and_chat_capable=true` only when both gates pass; no-fallback proof exists; final commit and push succeed.
  QA scenarios:
  ```
  Scenario: final release manifest exists
    Tool: powershell
    Invocation: Test-Path .omo/ulw-loop/diffusion-final-release-gate/evidence/final-release-manifest.json
    Expected: True
    Evidence: .omo/ulw-loop/diffusion-capability-plan-set/evidence/stage-4-release.txt
  ```
  Commit: YES | `chore(release): publish diffusion capability manifest`

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [ ] F1. Plan chain audit: every plan was executed in dependency order.
- [ ] F2. Commit audit: every stage has clean commit/push evidence.
- [ ] F3. Capability audit: coding/chat/final claims match reports.
- [ ] F4. No-fallback audit: diffusion inference does not import or call Qwen/Ollama.

## Commit strategy
- Commit and push after every completed stage and after each major wave inside child plans.
- Use Conventional Commits.
- Final commit footer: `Plan: .omo/plans/diffusion-capability-plan-set.md`.

## Success criteria
- The current tiny-loop plan and all downstream plans are executable in sequence.
- Intermediate commits are required, not optional.
- Final release gate cannot pass unless both coding and chat capability reports pass from the same diffusion checkpoint lineage.
