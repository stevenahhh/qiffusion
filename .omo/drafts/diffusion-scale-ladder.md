---
slug: diffusion-scale-ladder
status: drafted
intent: unclear
pending-action: execute after diffusion-llm-self-training completes
approach: Turn the tiny diffusion loop into repeatable tiny/small/mid scale runs before attempting target 4B.
---

# Draft: diffusion-scale-ladder

## Components (topology ledger)
| id | outcome | status | evidence path |
| --- | --- | --- | --- |
| scale-config | Named scale profiles and artifact directories | planned | `.omo/plans/diffusion-scale-ladder.md` |
| data-scale | Corpus growth and dedupe pipeline | planned | `.omo/plans/diffusion-scale-ladder.md` |
| train-scale | tiny/small/mid training commands | planned | `.omo/plans/diffusion-scale-ladder.md` |
| eval-scale | quality/speed/cost regression reports | planned | `.omo/plans/diffusion-scale-ladder.md` |

## Open assumptions (announced defaults)
| assumption | adopted default | rationale | reversible? |
| --- | --- | --- | --- |
| First ladder | tiny -> small -> mid -> target-4b-ready | Separates algorithm failures from compute failures | yes |
| Commit cadence | Commit after each profile is runnable and verified | User requested intermediate commits | yes |
| 4B | Prepare configs and checks, do not run by default | Avoids accidental huge training cost | yes |

## Findings
- `.omo/plans/diffusion-llm-self-training.md` must produce the first checkpoint and CLI.
- Current repo has no large-model artifact policy beyond ignored `checkpoints/`, `models/`, `.omo/`.

## Decisions
- Scaling is evidence-gated: no next profile without train/sample/eval report from previous profile.

## Scope IN
- Scale profiles, data growth, benchmark reports, resume/checkpoint policy.

## Scope OUT
- No target 4B training unless explicitly invoked after profile readiness gates pass.

## Open questions
- None.

## Approval gate
status: plan-written
