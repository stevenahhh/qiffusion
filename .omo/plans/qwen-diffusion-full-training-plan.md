# qwen-diffusion-full-training-plan - Work Plan

## TL;DR (For humans)
<!-- Fill this LAST, after the detailed plan below is written, so it summarizes the REAL plan. -->
<!-- Plain English for a non-engineer: NO file paths, NO todo numbers, NO wave/agent/tool names. -->

**What you'll get:** <fill last - deliverables in human terms, 1-2 sentences>

**Why this approach:** <fill last - the one or two load-bearing decisions and why>

**What it will NOT do:** <fill last - 1-3 plain lines mirroring Must NOT have>

**Effort:** <Quick | Short | Medium | Large | XL>
**Risk:** <Low | Medium | High> - <one-line driver>
**Decisions I made for you:** <fill last - the best-practice defaults you adopted; the user vetoes any here>

Your next move: <fill - e.g. approve, or run a high-accuracy review>. Full execution detail follows below.

---

> TL;DR (machine): <1 line - effort, risk, deliverables>

## Scope
### Must have
### Must NOT have (guardrails, anti-slop, scope boundaries)

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: <TDD | tests-after | none> + framework
- Evidence: .omo/evidence/task-<N>-qwen-diffusion-full-training-plan.<ext>

## Execution strategy
### Parallel execution waves
> Target 5-8 todos per wave. Fewer than 3 (except the final) means you under-split.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->
- [ ] 1. <title>
  What to do / Must NOT do: <...>
  Parallelization: Wave <N> | Blocked by: <...> | Blocks: <...>
  References (executor has NO interview context - be exhaustive): <src/path:lines>
  Acceptance criteria (agent-executable): <exact command or assertion>
  QA scenarios (name the exact tool + invocation): happy + failure, Evidence .omo/evidence/task-1-qwen-diffusion-full-training-plan.<ext>
  Commit: <Y/N> | <type>(<scope>): <summary>

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [ ] F1. Plan compliance audit
- [ ] F2. Code quality review
- [ ] F3. Real manual QA
- [ ] F4. Scope fidelity
- [ ] F5. Sonnet 4.6 target-matrix audit: overall completion is rejected unless the trained diffusion model reaches the recorded Sonnet 4.6-level coding, agentic, long-context, and speed/quality gates.

## Commit strategy

## Success criteria
- The loop can only be marked fully complete when the final trained Qwen-based diffusion model reaches the Claude Sonnet 4.6-level target matrix recorded in `.omo/ulw-research/20260701-full-training-plan/wave-2-librarian-sonnet-target.md`.
- Partial gains, local smoke passes, loss reductions, faster token emission, or isolated benchmark wins are progress only; they must keep the loop in `continue` until the full target matrix passes.
- Any final status report must include evidence for coding, multi-turn chat, tool/agent, SWE-bench-style repo repair, Terminal-Bench-style terminal work, OSWorld/computer-use-adjacent tasks, long-context behavior, safety, and speed/quality Pareto gates.
