# diffusion-final-release-gate - Work Plan

## TL;DR (For humans)
**What you'll get:** The final release gate for a diffusion model that can honestly be called coding-and-chat capable, with manifest, no-fallback proof, docs, and commit/push evidence.
**Why this approach:** Final capability must come from one checkpoint lineage passing both code and chat gates, not from separate experiments or cherry-picked samples.
**What it will NOT do:** It will not promote partial capability or publish large weights into git.
**Effort:** Medium
**Risk:** High - the release claim is only safe if every upstream report is consistent.
**Decisions I made for you:** Require coding and chat pass from the same lineage; require static/runtime no-Qwen fallback proof; commit/push final manifest and docs.

Your next move: execute after `.omo/plans/diffusion-code-capability.md` and `.omo/plans/diffusion-chat-capability.md` pass.

---

> TL;DR (machine): Aggregate gate for final diffusion release manifest requiring same-lineage coding/chat pass, no-fallback proof, docs, and clean git publication.

## Scope
### Must have
- Consume `final-code-eval.json`, `final-chat-eval.json`, selected checkpoint manifest, and no-fallback reports.
- Verify all reports refer to the same checkpoint lineage.
- Add `coding_and_chat_capable` aggregate decision.
- Produce `final-release-manifest.json`.
- Update README/docs with exact usage and limitations.
- Commit and push final release artifacts.

### Must NOT have
- No final claim if either coding or chat gate fails.
- No hidden Qwen/Ollama fallback.
- No binary weight commit.
- No release based on stale reports.

## Verification strategy
- Test decision: TDD for release aggregation and stale-lineage rejection.
- Real-surface QA: CLI release gate over actual report files.
- Evidence: `.omo/ulw-loop/diffusion-final-release-gate/evidence/*`

## Execution strategy
### Parallel execution waves
Wave 1: manifest schema and lineage checks.
Wave 2: release gate CLI and stale-report rejection.
Wave 3: no-fallback proof and docs.
Wave 4: final QA, commit, push.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | code/chat reports | 2, 4 | none |
| 2 | 1 | 4 | 3 |
| 3 | code/chat reports | 4 | 2 |
| 4 | 1, 2, 3 | final | none |

## Todos
- [ ] 1. Add final release manifest schema and lineage validator
  What to do / Must NOT do: Add manifest builder requiring checkpoint lineage, code report, chat report, artifact hashes, and model card fields. Must reject missing or mismatched lineage.
  Parallelization: Wave 1 | Blocked by: code/chat reports | Blocks: 2, 4
  References: `.omo/plans/diffusion-code-capability.md`, `.omo/plans/diffusion-chat-capability.md`, `src/qiffusion/decision.py:25-33`
  Acceptance criteria: tests cover pass, missing report, mismatched checkpoint, stale report.
  QA scenarios:
  ```
  Scenario: manifest rejects mismatched lineage
    Tool: powershell
    Invocation: python -m qiffusion.cli diffusion-release manifest --code-report bad-code.json --chat-report bad-chat.json --out .omo/ulw-loop/diffusion-final-release-gate/evidence/task-1-bad-manifest.json > .omo/ulw-loop/diffusion-final-release-gate/evidence/task-1-bad.txt 2>&1; if ($LASTEXITCODE -eq 0) { exit 1 } else { exit 0 }
    Expected: command fails
    Evidence: .omo/ulw-loop/diffusion-final-release-gate/evidence/task-1-bad.txt
  ```
  Commit: YES | `feat(release): add diffusion manifest validator`

- [ ] 2. Add aggregate coding-and-chat release gate
  What to do / Must NOT do: Add CLI `diffusion-release gate` that sets `coding_and_chat_capable=true` only when coding and chat claims are true and no-fallback passes.
  Parallelization: Wave 2 | Blocked by: 1 | Blocks: 4
  References: `src/qiffusion/decision.py:25-33`, `README.md:8`
  Acceptance criteria: failing coding or chat report keeps release false; passing reports set aggregate true.
  QA scenarios:
  ```
  Scenario: release gate remains false on partial capability
    Tool: powershell
    Invocation: python -m qiffusion.cli diffusion-release gate --code-report .omo/ulw-loop/diffusion-code-capability/evidence/final-code-eval.json --chat-report .omo/ulw-loop/diffusion-chat-capability/evidence/final-chat-eval.json --out .omo/ulw-loop/diffusion-final-release-gate/evidence/task-2-release-gate.json
    Expected: coding_and_chat_capable is true only when both source reports pass
    Evidence: .omo/ulw-loop/diffusion-final-release-gate/evidence/task-2-release-gate.json
  ```
  Commit: YES | `feat(release): add coding chat aggregate gate`

- [ ] 3. Add final no-fallback and artifact hygiene proof
  What to do / Must NOT do: Combine static import scan, runtime monkeypatch/blocked-env check, and git status check. Must prove Qwen/Ollama not used for diffusion inference and weights not staged.
  Parallelization: Wave 3 | Blocked by: code/chat reports | Blocks: 4
  References: `src/qiffusion/qwen_ollama.py`, `src/qiffusion/qwen_bridge.py`, `.gitignore`
  Acceptance criteria: report includes static_scan, runtime_probe, git_artifact_check.
  QA scenarios:
  ```
  Scenario: final no fallback proof exists
    Tool: powershell
    Invocation: python -m qiffusion.cli diffusion-release no-fallback --out .omo/ulw-loop/diffusion-final-release-gate/evidence/task-3-no-fallback.json
    Expected: JSON verdict pass
    Evidence: .omo/ulw-loop/diffusion-final-release-gate/evidence/task-3-no-fallback.json
  ```
  Commit: YES | `test(release): prove no qwen fallback`

- [ ] 4. Publish final release manifest and docs
  What to do / Must NOT do: Generate `final-release-manifest.json`, update README/model docs, run full tests and all gates, commit, push. Must not say final capable unless manifest says true.
  Parallelization: Wave 4 | Blocked by: 1, 2, 3 | Blocks: final
  References: README, all prior evidence paths
  Acceptance criteria: manifest exists; docs match manifest claim; full tests pass; push succeeds.
  QA scenarios:
  ```
  Scenario: final manifest and docs are consistent
    Tool: powershell
    Invocation: python -m qiffusion.cli diffusion-release finalize --gate-report .omo/ulw-loop/diffusion-final-release-gate/evidence/task-2-release-gate.json --no-fallback .omo/ulw-loop/diffusion-final-release-gate/evidence/task-3-no-fallback.json --out .omo/ulw-loop/diffusion-final-release-gate/evidence/final-release-manifest.json
    Expected: manifest claim matches source gate
    Evidence: .omo/ulw-loop/diffusion-final-release-gate/evidence/final-release-manifest.json
  ```
  Commit: YES | `chore(release): publish final diffusion manifest`

## Final verification wave
- [ ] F1. Lineage audit.
- [ ] F2. Aggregate gate audit.
- [ ] F3. No-fallback/artifact hygiene audit.
- [ ] F4. Documentation honesty audit.

## Commit strategy
- Commit/push after manifest validator, aggregate gate, no-fallback proof, final docs/manifest.
- Final commit footer: `Plan: .omo/plans/diffusion-final-release-gate.md`.

## Success criteria
- `final-release-manifest.json` exists.
- `coding_and_chat_capable=true` appears only when both coding and chat source reports pass from the same checkpoint lineage.
- Repo remains free of large binary checkpoints.
