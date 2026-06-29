---
slug: diffusion-code-capability
status: drafted
intent: unclear
pending-action: execute after diffusion-scale-ladder produces a selected checkpoint
approach: Build code-specific data, training, sampling, and executable coding gates around the diffusion checkpoint.
---

# Draft: diffusion-code-capability

## Components (topology ledger)
| id | outcome | status | evidence path |
| --- | --- | --- | --- |
| code-data | Code/preference/repair corpus generated and audited | planned | `.omo/plans/diffusion-code-capability.md` |
| code-train | Continue-training or adapter training for code | planned | `.omo/plans/diffusion-code-capability.md` |
| code-sample | Code-aware masked sampling and repair mode | planned | `.omo/plans/diffusion-code-capability.md` |
| code-gate | Existing qwen-style code smoke plus richer file repair gates | planned | `.omo/plans/diffusion-code-capability.md` |

## Open assumptions (announced defaults)
| assumption | adopted default | rationale | reversible? |
| --- | --- | --- | --- |
| Teacher | Qwen generates optional training labels only | Prevents hidden fallback | yes |
| Gate | reuse and extend executable Python smoke first | Existing repo already has this surface | yes |
| Claim | `coding_capability_claim=true` only after diffusion-generated code passes | Preserves shared gate truthfulness | yes |

## Findings
- Existing qwen eval has function/file-edit/repair benchmarks.
- Existing shared gate is strict but only coding-specific today.

## Decisions
- Coding capability is a real gate, not a training-loss threshold.

## Scope IN
- Code corpus, repair tasks, code sampling, eval report, coding gate.

## Scope OUT
- No chat claim and no final release claim.

## Open questions
- None.

## Approval gate
status: plan-written
