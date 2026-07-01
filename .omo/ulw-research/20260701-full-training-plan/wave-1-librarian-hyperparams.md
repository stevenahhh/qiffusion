# Wave 1: Hyperparameters and Scale Strategy

## Key Findings

- Qwen/Qwen3.5-4B is the chosen 4B anchor and is available as an Apache-2.0 open-weight model card.
- Qwen-family reports suggest staged pretraining, long-context extension, SFT, offline preference/RL, and GRPO-style online RL rather than a single monolithic run.
- Qwen3-Coder shows high-end code/agent capability depends on large code ratios, execution-driven RL, long context, and many parallel environments.
- DiffuLLaMA and DiffuCoder provide relevant diffusion scale references: adaptation from AR backbones, tens of billions of tokens per stage, bf16 full fine-tuning, sequence packing, warmup/cosine schedules, and code-specific post-training.

## Practical Recipe Defaults

- Base: Qwen/Qwen3.5-4B.
- First diffusion conversion: DiffuLLaMA-style AR-to-diffusion adaptation.
- First sampler: LLaDA-style masked remasking.
- Next sampler: entropy/confidence token selection and adaptive stopping.
- Later coding RL: DiffuCoder-style coupled-GRPO only after the diffusion model passes basic code eval.
- Context: start at 2K-4K for conversion sanity, then 8K/32K scale-up.
- Local 8GB GPU: prototype/eval only; real 4B training requires CUDA runtime and likely cloud GPUs.

## Sources

- Qwen/Qwen3.5-4B: https://huggingface.co/Qwen/Qwen3.5-4B
- Qwen3-Coder: https://qwenlm.github.io/blog/qwen3-coder/
- DiffuLLaMA: https://arxiv.org/abs/2410.17891
- DiffuCoder: https://huggingface.co/apple/DiffuCoder-7B-Base, https://github.com/apple/ml-diffucoder
