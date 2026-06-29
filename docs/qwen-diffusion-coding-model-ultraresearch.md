# Qwen Diffusion Coding Model Ultraresearch

Date: 2026-06-30
Scope: How to move `qiffusion` from the current tiny diffusion scaffold toward a Qwen-based diffusion model that can eventually support coding, chat, multi-turn, and agent workflows.

## Executive Summary

The current `qiffusion` diffusion path is a useful scaffold, not a capable model. It is a byte-level, tiny GRU denoiser trained on a tiny local corpus, then sampled with greedy one-mask-at-a-time filling. The final recorded eval correctly reports `coding_capability_claim=false` because the generated sample is not executable Python. This is consistent with the repo's own conservative gate and docs.

The best-supported path forward is not to train a diffusion LLM from scratch. The strongest public recipe is AR-to-diffusion adaptation: start from a Qwen-family autoregressive model, preferably a code-specialized Qwen2.5-Coder or instruction-capable Qwen checkpoint, then convert it with masked/discrete diffusion training. DiffuLLaMA gives the general AR-to-diffusion recipe, Open-dLLM shows a Qwen-family implementation path, DiffuCoder/Dream-Coder show the code-specialized direction, and DiffusionGemma/Block Diffusion/Fast-dLLM show where production inference is moving: block or canvas diffusion, confidence/entropy-driven token selection, cache reuse, and early stopping. [S1] [S2] [S3] [S4] [S6] [S11] [S12] [S13]

The data requirement is also much larger than "more examples." A coding/chat/agent model needs a staged mixture: local verified examples, Qwen-teacher SFT JSONL, code instruction data, large code pretraining data, multi-turn chat data, tool/agent trajectories, and execution-verified preference or RL data. Some public data is useful but risky. The Stack v2 requires provenance and license handling; UltraChat is non-commercial; WildChat has privacy risk; CodeFeedback and some Magicoder-style data include proprietary-model-generated content; SWE-bench and similar benchmarks should be held out for evaluation, not training. [S16] [S17] [S18] [S19] [S20] [S21] [S22] [S23] [S24] [S30]

## Current Repo State

Implemented today:

- `src/qiffusion/diffusion_train.py`: tiny masked diffusion training loop over local examples and optional teacher JSONL.
- `src/qiffusion/diffusion_model.py`: `TinyDiffusionLM`, a small GRU denoiser.
- `src/qiffusion/diffusion_objective.py`: simple token masking with a mask probability schedule.
- `src/qiffusion/diffusion_sample.py`: greedy masked-token sampling.
- `src/qiffusion/diffusion_eval.py`: checkpoint sampling plus a narrow code-smoke check.
- `src/qiffusion/diffusion_teacher_data.py`: export of passing Qwen eval outputs into teacher JSONL.
- `src/qiffusion/decision.py`: shared conservative gate requiring fixture pass, code-smoke pass, and explicit capability claim.

Observed blocker:

- The final tiny model sample was not valid Python, so the shared status gate remained `continue`.
- The current code only proves "train -> sample -> eval -> conservative status" works. It does not prove coding capability.

Immediate repo gaps:

- No Qwen tokenizer-backed diffusion training.
- No Qwen transformer denoiser or AR-to-diffusion conversion path.
- No corpus manifest, dedup report, license/provenance ledger, or data stage registry.
- No multi-turn chat corpus, sampler, or eval.
- No agent/tool trajectory schema.
- No broad code benchmark harness beyond local fixture smoke.

## Architecture Direction

### 1. Keep the model Qwen-based

The public evidence points to Qwen2.5-Coder or Qwen2.5-Instruct as the practical seed model for a Qwen-based diffusion system. Qwen2.5-Coder is code-specialized, and Qwen2.5-Instruct or later Qwen instruction checkpoints are better seeds for chat. The project can still keep a Qwen3.5 bridge/baseline if a local engine exists, but the public diffusion adaptation evidence is currently stronger around Qwen2.5-Coder and Qwen-family implementations such as Open-dLLM. [S2] [S3] [S5] [S6]

### 2. Use AR-to-diffusion adaptation before scratch training

DiffuLLaMA shows that existing AR models can be adapted into diffusion models by unifying the AR and diffusion objectives instead of training a diffusion model from zero. The relevant public recipe is:

- relax or remove causal masking;
- optionally anneal from causal attention to bidirectional attention;
- keep the AR shift operation;
- train with discrete masked diffusion cross-entropy;
- use full-parameter fine-tuning for the conversion stage;
- reserve LoRA for later task-specific specialization. [S1]

Open-dLLM is the closest Qwen-family implementation lead. Its Qwen path points toward representation alignment and a Qwen-compatible diffusion generation API rather than a new custom architecture. [S2] [S3] [S4]

### 3. Move from byte-GRU to Qwen tokenizer plus transformer denoiser

The next model family should be a transformer masked denoiser over Qwen tokens, not bytes. The tiny byte setup is useful for deterministic CPU tests, but it wastes capacity on syntax reconstruction and cannot inherit Qwen priors. LLaDA, MDLM, DiffuLLaMA, Dream, and Open-dLLM all use transformer-style masked/discrete denoising rather than a byte GRU. [S1] [S2] [S7] [S8] [S9]

### 4. Use block or canvas diffusion for variable-length chat/code

Pure fixed-canvas diffusion is awkward for chat and code because outputs are variable length. Block Diffusion and DiffusionGemma show the better direction: autoregressive over blocks or prompt/context, diffusion within the generation canvas, parallel token denoising, and cache-aware inference. Fast-dLLM v1/v2 adds practical acceleration ideas: KV cache, parallel decoding, block diffusion, hierarchical caching, and confidence-aware unmasking. [S11] [S12] [S13]

### 5. Improve the sampler before claiming speed

A credible qiffusion sampler should support:

- fixed-step denoising for reproducibility;
- confidence or entropy-based token selection;
- top-k/top-p and temperature controls;
- block size and canvas length controls;
- early stopping when uncertainty is low;
- history output for debugging;
- no hidden Qwen/Ollama autoregressive fallback.

Open-dLLM's public sample path suggests a concrete starting point: iterative `[MASK]` denoising with about 128 steps, `temperature=0.5`, `top_k=200`, and a named algorithm such as `p2`. This should be treated as a starting config, not a universal optimum. [S4]

## Data Strategy

### Safe ladder

1. Local bootstrap data:
   - repo docs, tests, issue-like prompts, local code snippets;
   - generated unit-test failures and repairs;
   - short tool traces created in this repo;
   - lowest legal/privacy risk.

2. Qwen-teacher SFT JSONL:
   - instruction-response pairs generated by a local Qwen teacher;
   - code repair, file edit, debugging, and explanation turns;
   - deterministic verification metadata;
   - provenance fields for model, prompt, seed, checker, and license source.

3. Public code SFT:
   - OpenCoder-style SFT data;
   - Magicoder/OSS-Instruct-style code instruction examples;
   - CodeFeedback and CodeFeedback-Filtered-Instruction if policy permits proprietary-model-generated data. [S18] [S19] [S20]

4. Large code pretraining:
   - The Stack v2 or a smaller curated permissive subset;
   - use only with per-file provenance, license metadata, opt-out handling, and dedup. [S16] [S17]

5. Multi-turn chat:
   - prefer synthetic or permissive data first;
   - UltraChat is useful for research but non-commercial;
   - WildChat needs aggressive privacy filtering and should not be treated as a drop-in corpus. [S21] [S22]

6. Tool and agent trajectories:
   - ToolBench for API/tool-use style;
   - AgentInstruct for broad synthetic agentic instruction data;
   - local qiffusion tool traces for the exact schema the model must learn. [S23] [S24]

7. Verification and RL data:
   - execution outcomes, failing vs passing candidates, repair attempts, unit-test traces;
   - do not train on benchmarks that will be reported as evaluation.

### Dataset policy

Good default:

- local synthetic data;
- local Qwen-teacher outputs;
- locally verified code tasks;
- AgentInstruct-style synthetic data after policy review;
- The Stack v2 only with full provenance controls.

Use with caution:

- CodeFeedback and Magicoder-derived data because some rows are generated by proprietary models;
- WildChat because privacy-sensitive content can remain after filtering;
- ToolBench because APIs and tool environments drift.

Avoid for training:

- SWE-bench, EvalPlus, LiveCodeBench, BigCodeBench, and similar evaluation sets;
- raw user chat logs;
- unlabeled scraped code with unknown licenses;
- any benchmark split used in release claims.

## Training Plan

### Stage A: data and tokenizer

Deliverables:

- `diffusion-corpus manifest` command;
- JSONL schema for code/chat/tool examples;
- Qwen tokenizer adapter;
- data dedup and benchmark-overlap scanner;
- provenance ledger.

Acceptance:

- manifest records source, license, split, tokenizer, token count, and contamination status;
- benchmark data is excluded from training splits;
- local tests still pass.

### Stage B: Qwen denoiser scaffold

Deliverables:

- Qwen checkpoint loader abstraction;
- bidirectional attention or mask-relaxation config;
- masked diffusion CE objective over Qwen tokens;
- right-shift compatibility flag for AR-initialized weights;
- small smoke model path that can run without full 4B training.

Acceptance:

- training dry run writes a checkpoint manifest;
- no Qwen/Ollama fallback is used during diffusion inference;
- generation still reports `coding_capability_claim=false` unless code smoke passes.

### Stage C: AR-to-diffusion adaptation

Deliverables:

- full-finetune recipe for conversion;
- optional annealed-causal-to-bidirectional mode;
- direct-bidirectional pragmatic mode;
- LoRA only for downstream specialization, not the primary conversion.

Acceptance:

- conversion run logs data mix, checkpoint base, objective, mask schedule, and sampler config;
- initial infill and completion tasks improve over the tiny baseline.

### Stage D: code specialization

Deliverables:

- code-heavy continued training;
- execution-feedback data;
- repair/edit/infill tasks;
- sampling controls tuned for syntactic validity.

Acceptance:

- local qiffusion code fixtures pass;
- EvalPlus harness can run;
- reports separate "local pass" from benchmark pass.

### Stage E: chat and agent specialization

Deliverables:

- multi-turn chat schema;
- tool-call trace schema;
- chat sampler and eval report;
- coding regression after chat tuning.

Acceptance:

- chat report exists but does not imply coding pass;
- tool/agent report exists but does not imply chat or coding pass;
- final release can only promote claims backed by each bucket.

## Inference Plan

Short term:

- keep deterministic tiny sampler for tests;
- add a Qwen-token sampler interface;
- expose mask length, steps, temperature, top-k, algorithm, seed, and output history.

Medium term:

- implement parallel token filling by confidence;
- add entropy-based early stopping;
- add block generation for variable-length outputs;
- cache prompt/context states where the architecture allows it.

Long term:

- evaluate Fast-dLLM-style cache and block diffusion;
- use DiffusionGemma-style canvas/block thinking where applicable;
- report tokens/sec only after model quality gates are stable.

## Capability Gate

The release gate must be split by claim type.

### Required for every build

- unit tests and py_compile;
- train/sample/eval smoke;
- no hidden Qwen/Ollama fallback;
- checkpoint lineage manifest;
- clean tracked git status.

### Coding claim

Minimum:

- local fixtures all pass;
- EvalPlus HumanEval+/MBPP+ runs reproducibly;
- no benchmark overlap in training data. [S25] [S26] [S27]

Stronger:

- LiveCodeBench recent-window run for freshness;
- BigCodeBench for practical library-heavy code;
- report pass@k, sampling config, model hash, dataset hash, and harness version. [S28] [S29]

### Software-engineering claim

Required:

- SWE-bench Verified or equivalent repository-editing benchmark;
- patch must be executed in the standard harness;
- do not train on the benchmark split. [S30]

### Chat claim

Required:

- at least two chat-style evaluators or benchmark families;
- bias controls for verbosity and judge artifacts;
- multi-turn context retention tests. [S31]

### Tool or agent claim

Required:

- at least one tool-use benchmark and one interactive agent benchmark;
- hidden or held-out tasks when possible;
- no claim of real-world reliability from public exemplars alone. [S24] [S32]

## Immediate Next Implementation

The next qiffusion loop should not try to "train until coding-capable" with the current tiny GRU. It should build the missing infrastructure that makes such a loop meaningful.

Recommended next commits:

1. Add a dataset manifest and schema:
   - `src/qiffusion/diffusion_corpus_manifest.py`
   - `tests/test_diffusion_corpus_manifest.py`
   - command: `diffusion-corpus manifest`

2. Add teacher trace schema:
   - code, chat, tool, repair, and execution-result record types;
   - provenance fields: source, license, teacher model, prompt hash, checker hash.

3. Add Qwen tokenizer adapter:
   - optional dependency boundary around `transformers`;
   - fallback to byte tokenizer for tiny tests only.

4. Add Qwen-diffusion config skeleton:
   - base checkpoint id;
   - attention mode: `causal`, `annealed`, `bidirectional`;
   - objective: `masked_ce`;
   - sampler: `mask_iterative`, `block_diffusion`.

5. Add expanded eval harness:
   - local fixtures first;
   - EvalPlus command wrapper next;
   - LiveCodeBench/BigCodeBench/SWE-bench as external, optional gates.

6. Only then run the next training-validation loop:
   - train small Qwen-token denoiser smoke;
   - sample;
   - eval;
   - record failure modes;
   - improve data/objective/sampler;
   - repeat.

## Claim Ledger

| Claim | Status | Evidence |
| --- | --- | --- |
| Current qiffusion tiny diffusion model is not coding-capable | Verified | local final eval and repo code paths |
| Qwen AR-to-diffusion adaptation is the best public path | Verified | DiffuLLaMA, Open-dLLM, DiffuCoder |
| Full finetune is the safer conversion default; LoRA is later specialization | Verified | DiffuLLaMA recipe |
| Block/canvas diffusion is needed for practical variable-length generation | Verified | Block Diffusion, DiffusionGemma, Fast-dLLM |
| The data stack must include code, chat, tool, and verification traces | Verified | OpenCoder, CodeFeedback, ToolBench, AgentInstruct, benchmark evidence |
| SWE-bench should be training-excluded if used for a release claim | Verified as policy inference | benchmark purpose and contamination risk |
| A single code smoke pass is enough for "coding-capable" | Refuted | benchmark scope and current gate analysis |

## Sources

- [S1] DiffuLLaMA, Scaling Diffusion Language Models via Adaptation from Autoregressive Models: https://arxiv.org/abs/2410.17891
- [S2] Open-dLLM repository: https://github.com/pengzhangzhi/Open-dLLM
- [S3] Open-dLLM representation alignment tutorial: https://github.com/pengzhangzhi/Open-dLLM/blob/main/docs/representation_alignment.md
- [S4] Open-dLLM sampling example: https://github.com/pengzhangzhi/Open-dLLM/blob/main/sample.py
- [S5] Qwen2.5-Coder technical report: https://arxiv.org/abs/2409.12186
- [S6] DiffuCoder paper and repository: https://arxiv.org/abs/2506.20639, https://github.com/apple/ml-diffucoder
- [S7] LLaDA paper and repository: https://arxiv.org/abs/2502.09992, https://github.com/ML-GSAI/LLaDA
- [S8] Dream repository: https://github.com/DreamLM/Dream
- [S9] MDLM paper and repository: https://arxiv.org/abs/2406.07524, https://github.com/kuleshov-group/mdlm
- [S10] SEDD paper and repository: https://arxiv.org/abs/2310.16834, https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
- [S11] Block Diffusion paper: https://arxiv.org/abs/2503.09573
- [S12] Fast-dLLM repository: https://github.com/NVlabs/Fast-dLLM
- [S13] DiffusionGemma docs and blog: https://ai.google.dev/gemma/docs/diffusiongemma, https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/
- [S14] Gemini Diffusion blog: https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-diffusion/
- [S15] Mercury Coder blog and paper: https://www.inceptionlabs.ai/blog/introducing-mercury, https://arxiv.org/abs/2506.17298
- [S16] The Stack v2 dataset card: https://huggingface.co/datasets/bigcode/the-stack-v2
- [S17] Software Heritage statement on code LLM training: https://www.softwareheritage.org/2023/10/19/swh-statement-on-llm-for-code/
- [S18] OpenCoder repository and paper: https://github.com/OpenCoder-llm/OpenCoder-llm, https://arxiv.org/abs/2411.04905
- [S19] CodeFeedback and CodeFeedback-Filtered-Instruction: https://huggingface.co/datasets/m-a-p/Code-Feedback, https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction
- [S20] Magicoder paper and repository: https://arxiv.org/abs/2312.02120, https://github.com/ise-uiuc/magicoder
- [S21] UltraChat repository and UltraChat 200k card: https://github.com/thunlp/ultrachat, https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
- [S22] WildChat dataset and privacy analysis: https://huggingface.co/datasets/allenai/WildChat, https://arxiv.org/abs/2407.11438
- [S23] AgentInstruct dataset and paper: https://huggingface.co/datasets/microsoft/orca-agentinstruct-1M-v1, https://arxiv.org/abs/2407.03502
- [S24] ToolBench and ToolLLM: https://github.com/OpenBMB/ToolBench, https://arxiv.org/abs/2307.16789
- [S25] HumanEval: https://github.com/openai/human-eval
- [S26] MBPP: https://github.com/google-research/google-research/tree/master/mbpp
- [S27] EvalPlus: https://github.com/evalplus/evalplus
- [S28] LiveCodeBench: https://github.com/LiveCodeBench/LiveCodeBench
- [S29] BigCodeBench: https://github.com/bigcode-project/bigcodebench
- [S30] SWE-bench and SWE-bench Verified: https://github.com/swe-bench/SWE-bench, https://www.swebench.com/
- [S31] MT-Bench and AlpacaEval: https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge, https://github.com/tatsu-lab/alpaca_eval
- [S32] AgentBench, tau-bench, and WebArena: https://github.com/THUDM/AgentBench, https://github.com/sierra-research/tau-bench, https://github.com/web-arena-x/webarena
