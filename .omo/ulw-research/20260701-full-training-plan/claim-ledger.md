# Claim Ledger

| claim | risk | primary/source domains | counter-search | status |
| --- | --- | --- | --- | --- |
| Qwen/Qwen3.5-4B is available as an Apache-2.0 HF model. | high | huggingface.co | searched exact model and license | verified |
| qiffusion current training path is a tiny scaffold, not full Qwen 4B training. | high | local repo source/docs | codegraph + explorer cross-check | verified |
| DiffuLLaMA-style AR-to-diffusion adaptation is the lowest-risk first objective for a Qwen-based diffusion model. | normal | arxiv.org, github.com, local repo constraints | compared alternatives | verified as plan default |
| Sonnet 4.6-level performance must be represented as benchmark gates, not a local-training promise. | high | anthropic.com, swebench.com, tbench.ai, os-world.github.io | searched exact Sonnet 4.6 benchmark claims | verified |
| REVIEW/BLOCK datasets must not enter training manifests. | high | HF cards, repo/source ledgers | searched licenses/access terms | verified as plan default |
| Current qiffusion generated-code smoke is not an OS/process sandbox. | high | local repo source | searched exec/subprocess/tempfile/timeout | verified |
| Near-duplicate benchmark contamination is not currently implemented beyond exact hash/substring checks. | high | local repo source | searched minhash/simhash/similarity | verified |
