# Expansion Log: Qwen Diffusion Full Training Plan

Session: 20260701-full-training-plan
Tier: HEAVY
Intent route: UNCLEAR

## Phase 0

Core question: how to build a full training/data/eval loop for a Qwen-based diffusion LLM aimed at multi-turn chat, coding, and agentic use, without making unsupported local-training or Sonnet-level claims.

Success criteria:
- every dataset recommendation includes stage, license/access posture, contamination risk, and training/eval role
- full training plan separates local prototype work from cloud-scale full training
- capability claims are blocked until executable coding, chat, tool, agent, safety, and speed/quality gates pass
- Sonnet-level target is expressed as a benchmark matrix

## Wave 1

Spawned:
- diffusion LLM methods and public diffusion/Qwen-family evidence
- coding data for pretraining/SFT/RL/eval
- multi-turn chat/tool/agent/preference data
- qiffusion repo current training/eval surface
- hyperparameters and scaling strategy
- feasibility/contradiction review

Markers gained:
- exact Qwen diffusion adapter training configs
- newer LLaDA/iLLaDA branches
- diffusion serving support
- BD3LM vs MDLM vs Dream ablations
- OpenCoder corpus breakdown
- LiveCodeBench contamination policy
- Qwen/model-output terms
- code-execution sandbox hardening
- near-duplicate semantic leakage checks
- Sonnet 4.6 target matrix

## Wave 2

Spawned:
- exact Sonnet 4.6 target matrix and benchmark gates
- Qwen/model-output/dataset license constraints
- diffusion objective and noise schedule choices
- qiffusion code-execution sandbox and contamination controls

Convergence status: sufficient for approval brief. Remaining details are implementation-time checks, not planning blockers.
