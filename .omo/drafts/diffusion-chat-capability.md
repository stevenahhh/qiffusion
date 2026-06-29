---
slug: diffusion-chat-capability
status: drafted
intent: unclear
pending-action: execute after diffusion-code-capability has a selected checkpoint or branch
approach: Add instruction/chat data and multi-turn eval without weakening the coding gate.
---

# Draft: diffusion-chat-capability

## Components (topology ledger)
| id | outcome | status | evidence path |
| --- | --- | --- | --- |
| chat-data | Instruction and multi-turn local corpus | planned | `.omo/plans/diffusion-chat-capability.md` |
| chat-train | Continue-training for instruction following | planned | `.omo/plans/diffusion-chat-capability.md` |
| chat-sample | Conversation wrapper around diffusion sampler | planned | `.omo/plans/diffusion-chat-capability.md` |
| chat-gate | Multi-turn, refusal, context-retention report | planned | `.omo/plans/diffusion-chat-capability.md` |

## Open assumptions (announced defaults)
| assumption | adopted default | rationale | reversible? |
| --- | --- | --- | --- |
| Safety | minimal refusal/safety gate included | Chat-capable claim needs basic boundary checks | yes |
| Multi-turn | use deterministic scripted eval before subjective chat | Agent-executable evidence | yes |
| Coding regression | rerun coding gate after chat training | Avoid chat tuning breaking code | yes |

## Findings
- No chat surface currently exists.
- Existing gate needs a separate chat report to avoid conflating code and chat.

## Decisions
- Chat capability is separate from coding capability and must not overwrite coding evidence.

## Scope IN
- Instruction data, chat CLI, chat eval, coding regression.

## Scope OUT
- No public serving endpoint unless final release plan adds it.

## Open questions
- None.

## Approval gate
status: plan-written
