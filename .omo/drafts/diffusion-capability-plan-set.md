---
slug: diffusion-capability-plan-set
status: drafted
intent: unclear
pending-action: execute plan chain only after explicit start-work
approach: Chain the existing tiny diffusion self-training plan into scale, code capability, chat capability, and final release gates. Each plan has internal commit/push checkpoints so ulw-loop can proceed stage by stage without losing state.
---

# Draft: diffusion-capability-plan-set

## Components (topology ledger)
| id | outcome | status | evidence path |
| --- | --- | --- | --- |
| S0 tiny-loop | Real tiny masked diffusion train/sample/eval surface exists | existing | `.omo/plans/diffusion-llm-self-training.md` |
| S1 scale | Tiny loop scales through small/mid checkpoints with measurable quality/speed gates | planned | `.omo/plans/diffusion-scale-ladder.md` |
| S2 coding | Diffusion checkpoint becomes coding-capable through code data, repair tasks, and executable code gates | planned | `.omo/plans/diffusion-code-capability.md` |
| S3 chat | Diffusion checkpoint becomes chat-capable through instruction/chat data and multi-turn gates | planned | `.omo/plans/diffusion-chat-capability.md` |
| S4 release | Final artifact is promoted only when coding and chat gates pass with no Qwen fallback | planned | `.omo/plans/diffusion-final-release-gate.md` |

## Open assumptions (announced defaults)
| assumption | adopted default | rationale | reversible? |
| --- | --- | --- | --- |
| Plan granularity | Keep five linked plan files, not one huge file | Execution can checkpoint, commit, and push after each stage | yes |
| Gate style | Add separate coding/chat/release reports while preserving existing shared coding gate | Prevents "sample exists" from becoming a capability claim | yes |
| Scaling | require evidence at tiny, small, and mid before target 4B | Debuggable path; target 4B is expensive and should not be first | yes |
| Teacher use | Qwen may generate training data/eval baselines, never serve as hidden inference fallback | Uses current Qwen bridge without faking diffusion capability | yes |

## Findings (cited)
- Existing stage zero is `.omo/plans/diffusion-llm-self-training.md`.
- Current shared gate is `src/qiffusion/decision.py:25-33`.
- README states capability claims require executable smoke and reproducible evidence (`README.md:8`).
- Current diffusion surface is scaffold-only (`README.md:48`, `src/qiffusion/backends.py:18-27`).

## Decisions
- The next executable chain is: self-training -> scale ladder -> coding capability -> chat capability -> final release gate.
- Every stage must end with a clean commit and push.
- Every stage must produce machine-readable evidence consumed by the next stage.

## Scope IN
- Plan files under `.omo/plans` and draft ledgers under `.omo/drafts`.
- Intermediate commit/push strategy embedded per stage.
- Final release gate requiring both coding and chat capability.

## Scope OUT
- No product code implementation in this planning turn.
- No claim that current diffusion checkpoint exists or is capable.

## Open questions
- None. The user explicitly asked to make downstream plans continuously executable.

## Approval gate
status: plan-written
The plan set is ready for execution by `$omo:start-work` / `$omo:ulw-loop`.
