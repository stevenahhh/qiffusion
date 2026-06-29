---
slug: diffusion-final-release-gate
status: drafted
intent: unclear
pending-action: execute after scale, coding, and chat plans pass
approach: Promote only a checkpoint that has code and chat reports, no hidden fallback, reproducible artifacts, and release documentation.
---

# Draft: diffusion-final-release-gate

## Components (topology ledger)
| id | outcome | status | evidence path |
| --- | --- | --- | --- |
| manifest | Final model manifest and provenance | planned | `.omo/plans/diffusion-final-release-gate.md` |
| gates | Coding/chat/serving regression reports | planned | `.omo/plans/diffusion-final-release-gate.md` |
| packaging | Local inference package and usage docs | planned | `.omo/plans/diffusion-final-release-gate.md` |
| release | Commit, push, and release-readiness summary | planned | `.omo/plans/diffusion-final-release-gate.md` |

## Open assumptions (announced defaults)
| assumption | adopted default | rationale | reversible? |
| --- | --- | --- | --- |
| Final claim | require both coding and chat gates pass in same manifest | Prevents partial capability being marketed as final | yes |
| Fallback proof | static and runtime proof that Qwen/Ollama are not used for diffusion inference | Avoids false capability claim | yes |
| Artifacts | code/config committed; large weights external or ignored with manifest path | Keeps repo clean | yes |

## Findings
- Existing shared gate handles coding only; final gate must aggregate multiple reports.
- Current README states no model is capable without executable evidence.

## Decisions
- Final model is a release candidate only when all gates pass from the same checkpoint lineage.

## Scope IN
- Manifest, aggregate release gate, no-fallback proof, docs, final commit/push.

## Scope OUT
- No claim based on cherry-picked samples.

## Open questions
- None.

## Approval gate
status: plan-written
