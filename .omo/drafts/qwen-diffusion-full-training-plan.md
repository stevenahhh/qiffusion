---
slug: qwen-diffusion-full-training-plan
status: awaiting-approval
intent: unclear
pending-action: write .omo/plans/qwen-diffusion-full-training-plan.md
approach: Build a full Qwen/Qwen3.5-4B diffusion training plan in stages: source-ledger and contamination gates, sandboxed eval, Qwen AR-to-diffusion conversion, dataset iteration loop, SFT/preference/RL stages, cloud-scale full training gates, and Sonnet-4.6-style final claim gates.
---

# Draft: qwen-diffusion-full-training-plan

## Components (topology ledger)
<!-- Lock the SHAPE before depth. One row per top-level component that can succeed or fail independently. -->
<!-- id | outcome (one line) | status: active|deferred | evidence path -->
| C1 | Source-ledger and dataset intake gates classify every candidate dataset as allow/review/block before training | active | .omo/ulw-research/20260701-full-training-plan/wave-2-librarian-license-ledger.md |
| C2 | Contamination and sandbox harness prevents false coding/agent capability claims | active | .omo/ulw-research/20260701-full-training-plan/wave-2-explorer-sandbox-contamination.md |
| C3 | Qwen/Qwen3.5-4B AR-to-diffusion adaptation replaces the current tiny smoke-only trainer | active | .omo/ulw-research/20260701-full-training-plan/wave-2-librarian-objective-choice.md |
| C4 | Multi-stage data loop selects, filters, trains, evaluates, and remixes code/chat/agent/preference data | active | .omo/ulw-research/20260701-full-training-plan/wave-1-librarian-coding-data.md; .omo/ulw-research/20260701-full-training-plan/wave-1-librarian-chat-agent-data.md |
| C5 | Benchmark and claim gates map progress to local smoke, code, chat, tool-agent, SWE-bench, Terminal-Bench, OSWorld, and long-context targets | active | .omo/ulw-research/20260701-full-training-plan/wave-2-librarian-sonnet-target.md |
| C6 | Full-training infrastructure separates local prototypes from approved cloud-scale training and checkpoint promotion | active | .omo/ulw-research/20260701-full-training-plan/wave-1-librarian-hyperparams.md; .omo/ulw-research/20260701-full-training-plan/wave-1-metis-risk-ledger.md |

## Open assumptions (announced defaults)
<!-- Intent is UNCLEAR: research resolves ambiguity, defaults are adopted (not asked), and each is surfaced in the plan's human TL;DR for veto. -->
<!-- assumption | adopted default | rationale | reversible? -->
| Base model | Use Qwen/Qwen3.5-4B as the 4B anchor | User explicitly selected it earlier and license/source audit found Apache-2.0 | yes |
| First diffusion objective | Implement DiffuLLaMA-style AR-to-diffusion adaptation first | Lowest-risk path from existing AR Qwen checkpoint; from-scratch diffusion pretraining is much more expensive | yes |
| First sampler | Use LLaDA-style remasking baseline before entropy/adaptive samplers | Simple enough to verify before adding Dream/DiffusionGemma complexity | yes |
| Dataset policy | REVIEW/BLOCK sources cannot enter training manifests | Prevents license/privacy/benchmark leakage from becoming irreversible | yes |
| Benchmark policy | HumanEval/MBPP/EvalPlus/BigCodeBench/LiveCodeBench/SWE-bench/tau-bench/BFCL/WebArena test splits are eval-only | Avoids contaminated capability claims | yes |
| Paid/cloud budget | No paid APIs, cloud GPUs, large gated downloads, or commercial-use gray data without a separate approval gate | Full Sonnet-level training needs owner approval for spend and terms | yes |
| Capability claim | Keep `coding_capability_claim=false` and no Sonnet-level claim until final gates pass | Current repo evidence is smoke-only; overclaiming is the main failure mode | yes |
| Local machine role | Use local 8GB hardware for prototype, schema, quantized baseline, and eval harness only | Not enough for full 4B full-parameter training or Sonnet-level validation | yes |

## Findings (cited - path:lines)
- qiffusion current train/eval surfaces are smoke-first and conservative: `.omo/ulw-research/20260701-full-training-plan/wave-1-explorer-repo-surface.md`.
- Diffusion method default and alternatives: `.omo/ulw-research/20260701-full-training-plan/wave-1-librarian-diffusion-methods.md`, `.omo/ulw-research/20260701-full-training-plan/wave-2-librarian-objective-choice.md`.
- Coding datasets and eval-only separation: `.omo/ulw-research/20260701-full-training-plan/wave-1-librarian-coding-data.md`.
- Chat/agent/preference/safety dataset map: `.omo/ulw-research/20260701-full-training-plan/wave-1-librarian-chat-agent-data.md`.
- License/source-ledger gates: `.omo/ulw-research/20260701-full-training-plan/wave-2-librarian-license-ledger.md`.
- Sonnet 4.6 target matrix: `.omo/ulw-research/20260701-full-training-plan/wave-2-librarian-sonnet-target.md`.
- Feasibility risk ledger: `.omo/ulw-research/20260701-full-training-plan/wave-1-metis-risk-ledger.md`.
- Final synthesis: `.omo/ulw-research/20260701-full-training-plan/SYNTHESIS.md`.

## Decisions (with rationale)
- Plan the full work as a staged loop, not a single training command.
- The first execution wave must harden source-ledger, sandbox, contamination, and checkpoint registry infrastructure before any large training run.
- The first model architecture wave must adapt Qwen/Qwen3.5-4B into diffusion rather than continuing the current tiny smoke model.
- The data loop must repeatedly discover datasets, classify policy status, decontaminate, sample/train, evaluate, error-bucket failures, and update mixture weights.
- Final quality gates must compare against benchmark families and task distributions, not against subjective "feels like Sonnet" judgments.

## Scope IN
- Full training plan for Qwen/Qwen3.5-4B-based diffusion LLM.
- Dataset discovery, source ledger, compliance gates, and contamination gates.
- Coding, multi-turn chat, tool/agent, preference, safety, and long-context training/eval lanes.
- Qwen AR-to-diffusion adaptation path and later sampler/post-training improvements.
- Local prototype gates and cloud-scale full-training gates.
- Benchmark matrix that can eventually justify or reject Sonnet 4.6-level claims.

## Scope OUT (Must NOT have)
- No claim that the current qiffusion model is coding/chat capable.
- No claim that local 8GB hardware can train or validate Sonnet-level performance.
- No training on eval/test benchmark splits.
- No use of REVIEW/BLOCK datasets in training manifests.
- No paid API/cloud/gated dataset download as part of this planning turn.
- No implementation or full training run in this planning turn.

## Open questions
- None blocking the plan. Budget, cloud provider, and gated-data approvals are deferred as explicit execution gates, not planner questions.

## Approval gate
status: awaiting-approval
pending action: fill `.omo/plans/qwen-diffusion-full-training-plan.md` with decision-complete todos, then run the required UNCLEAR-path high-accuracy plan review.
approval brief: approve the staged approach above, or veto a default before the plan todos are written.
<!-- When exploration is exhausted and unknowns are answered, set status: awaiting-approval. -->
<!-- That durable record is the loop guard: on a later turn read it and resume at the gate instead of re-running exploration. -->
