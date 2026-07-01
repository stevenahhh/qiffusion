# Wave 2: Diffusion Objective and Sampler Choice

## Recommendation

Implement DiffuLLaMA-style AR-to-diffusion adaptation first, then add LLaDA-style random or low-confidence remasking as the simplest sampler baseline. After that, add entropy-based/adaptive stopping inspired by Dream and DiffusionGemma. Defer BD3LM/block diffusion and DiffuCoder coupled-GRPO until the basic Qwen diffusion backbone passes core eval.

## Method Notes

- MDLM: clean masked diffusion baseline, but less direct for Qwen AR conversion.
- BD3LM: useful for arbitrary-length/blockwise speed tradeoff, but too much machinery for the first 4B conversion.
- LLaDA: simple random mask objective and response-only masking for SFT; good sampler semantics.
- DiffuLLaMA: best bridge from pretrained AR to diffusion, with attention-mask annealing, sequence packing, bf16 full finetune, and `diffusion_steps`/temperature knobs.
- Dream: useful sampler choices and context-adaptive token-level noise rescheduling, but harder as first conversion.
- DiffuCoder: code-specific diffusion post-training path; use after stable backbone.
- DiffusionGemma: best production sampler reference: max denoising steps, entropy bound, adaptive stopping, temperature decay.

## Sources

- MDLM: https://arxiv.org/abs/2406.07524
- BD3LM: https://arxiv.org/abs/2503.09573, https://github.com/kuleshov-group/bd3lms
- LLaDA: https://github.com/ML-GSAI/LLaDA, https://ml-gsai.github.io/LLaDA-demo/
- DiffuLLaMA: https://arxiv.org/abs/2410.17891, https://github.com/HKUNLP/DiffuLLaMA
- Dream: https://github.com/DreamLM/Dream, https://hkunlp.github.io/blog/2025/dream/
- DiffuCoder: https://github.com/apple/ml-diffucoder, https://huggingface.co/apple/DiffuCoder-7B-Base
- DiffusionGemma: https://huggingface.co/google/diffusiongemma-26B-A4B-it
