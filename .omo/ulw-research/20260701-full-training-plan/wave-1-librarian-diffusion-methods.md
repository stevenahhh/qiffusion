# Wave 1: Diffusion LLM Methods

## Key Findings

- LLaDA is the clearest open masked-diffusion-from-scratch reference.
- Dream is the strongest open AR-initialized diffusion LLM direction.
- DiffuLLaMA is the clearest route for converting an existing AR checkpoint into a diffusion LM.
- DiffusionGemma is a real open-weights major-lab reference and provides useful serving knobs: canvas/block length, denoising steps, entropy-bounded denoising, adaptive stopping, and temperature decay.
- Mercury is a speed target, not an open training reference.
- Qwen-family diffusion evidence exists through community dLLM adapters, especially Qwen3-0.6B and Qwen2.5-Coder-0.5B MDLM/BD3LM variants.

## Sources

- LLaDA: https://github.com/ML-GSAI/LLaDA, https://huggingface.co/GSAI-ML/LLaDA-8B-Base, https://ml-gsai.github.io/LLaDA-demo/
- Dream: https://github.com/DreamLM/Dream, https://huggingface.co/Dream-org/Dream-v0-Base-7B, https://arxiv.org/abs/2508.15487
- DiffuLLaMA: https://arxiv.org/abs/2410.17891, https://github.com/HKUNLP/DiffuLLaMA
- DiffusionGemma: https://huggingface.co/google/diffusiongemma-26B-A4B-it
- Mercury: https://www.inceptionlabs.ai/blog/introducing-mercury, https://arxiv.org/abs/2506.17298
- Qwen diffusion adapters: https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1, https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1

## Planning Implications

- Default first objective: AR-to-diffusion conversion from Qwen/Qwen3.5-4B.
- Default first sampler: block/canvas masked denoising with explicit step count.
- Use DiffusionGemma and Mercury as serving targets, not training templates.
