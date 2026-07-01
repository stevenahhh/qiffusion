# Ultraresearch Synthesis: Qwen Diffusion Full Training Plan

Workers: 10. Waves: 2. Sources: web primary sources, HF dataset/model cards, local qiffusion code, local qiffusion plans. Verifications: local repo codegraph inspection plus source audits.

## Executive Summary

The strongest plan is not to keep stretching the current tiny smoke trainer. qiffusion should first build a real data/source ledger, contamination scanner, generated-code sandbox, and Qwen tokenizer/checkpoint training interface. Only after those gates exist should it run full Qwen/Qwen3.5-4B diffusion adaptation.

The first full-training objective should be DiffuLLaMA-style AR-to-diffusion adaptation from Qwen/Qwen3.5-4B, with LLaDA-style masked remasking as the simplest sampler. Dream, DiffusionGemma, BD3LM, and DiffuCoder/coupled-GRPO are later extensions, not the first implementation branch.

Claude Sonnet 4.6-level performance is the required final-complete gate, not a near-term local claim. The learning loop remains open until the trained diffusion model reaches the target matrix: SWE-bench Verified, Terminal-Bench 2.0, OSWorld-Verified, long-context, multi-turn agent, tool-use, coding, safety, and speed/quality Pareto checks. Local 8GB hardware can support prototypes and quantized baselines, not the full claim.

## Findings By Theme

### Current qiffusion State

The repo already has CLI surfaces for Qwen diffusion smoke training/eval/loop, manifest filtering, and conservative gates. The current model path is still tiny and CPU-friendly, with code that reports no coding-capability claim.

### Diffusion Method Default

Default first objective: DiffuLLaMA-style AR-to-diffusion conversion from Qwen/Qwen3.5-4B. The first sampler should be LLaDA-style iterative masked remasking, then entropy/adaptive stopping inspired by Dream and DiffusionGemma.

### Dataset Shortlist

Use these as training candidates after source-ledger checks:

- Code pretraining: The Stack v2 after review, CommitPack permissive rows, OpenCoder/RefineCode-style data.
- Code SFT/repair: Magicoder/OSS-Instruct, OpenCoder SFT, CommitPackFT, CodeFeedback/OpenCodeInterpreter.
- Chat SFT: OASST1/OASST2, filtered UltraChat/OpenHermes after review, selected WildChat only after privacy/provenance review.
- Tool/agent: ToolBench after review, APIGen only if license permits the intended use, Mind2Web train split.
- Preference/alignment: HelpSteer, UltraFeedback after upstream review, code-specific preference data.

Keep eval-only or blocked unless explicitly approved: HumanEval/MBPP/EvalPlus/BigCodeBench/LiveCodeBench/SWE-bench Verified/tau-bench/BFCL/WebArena test splits, LMSYS-Chat-1M until gated terms are cleared, APIGen-MT-5k for commercial use, and Nectar.

### Required Safety And Honesty Gates

Before large-scale code benchmarks or broad teacher-data ingestion, qiffusion needs a process-level generated-code sandbox, near-duplicate contamination scanning, source-ledger gating, immutable checkpoint registry, stable Qwen baseline/teacher path, and benchmark harnesses that record exact splits/versions.

## Sources Ranked

1. Qwen/Qwen3.5-4B: https://huggingface.co/Qwen/Qwen3.5-4B
2. Anthropic Sonnet 4.6 launch/system card: https://www.anthropic.com/news/claude-sonnet-4-6 and https://www.anthropic.com/claude-sonnet-4-6-system-card
3. DiffuLLaMA: https://arxiv.org/abs/2410.17891 and https://github.com/HKUNLP/DiffuLLaMA
4. LLaDA: https://github.com/ML-GSAI/LLaDA and https://ml-gsai.github.io/LLaDA-demo/
5. DiffusionGemma: https://huggingface.co/google/diffusiongemma-26B-A4B-it
6. The Stack v2: https://huggingface.co/datasets/bigcode/the-stack-v2
7. OpenCoder: https://arxiv.org/abs/2411.04905 and https://github.com/OpenCoder-llm/OpenCoder-llm
8. Magicoder: https://arxiv.org/abs/2312.02120 and https://github.com/ise-uiuc/magicoder
9. Tool/agent sources: https://github.com/OpenBMB/ToolBench, https://apigen-mt.github.io/, https://osu-nlp-group.github.io/Mind2Web/
10. Eval targets: https://www.swebench.com/, https://www.tbench.ai/leaderboard/terminal-bench/2.0, https://os-world.github.io/, https://github.com/bigcode-project/bigcodebench

## Contradictions Resolved

- "Use full actual training now" conflicts with the repo state. The current code has only tiny smoke training, so the plan must first build full-training infrastructure.
- "Sonnet 4.6 level" conflicts with local hardware and current capability evidence. The plan keeps it as the required final-complete gate and makes intermediate stages fail closed.
- "Perfect dataset" is not a single corpus. The defensible answer is a repeated dataset discovery, compliance, decontamination, train, eval, error-bucket, and remix loop.

## Expansion Trace

Wave 1 covered diffusion methods, coding data, chat/agent data, repo state, hyperparameters, and feasibility risks. Wave 2 closed target-matrix, license/source-ledger, objective-choice, and sandbox/contamination leads. Remaining leads are implementation-time details, not blockers for the approval brief.
