---
slug: diffusion-llm-self-training
status: drafted
intent: unclear
pending-action: execute .omo/plans/diffusion-llm-self-training.md only after user starts work
approach: Build the smallest real self-training diffusion LM path first: deterministic byte-token corpus, masked denoising objective, tiny CPU PyTorch model, checkpointed train/sample/eval CLI, then shared gate integration that remains false until coding smoke passes.
---

# Draft: diffusion-llm-self-training

## Components (topology ledger)
| id | outcome | status | evidence path |
| --- | --- | --- | --- |
| C1 data | Deterministic local corpus and byte tokenizer for first self-training loop | active | `.omo/plans/diffusion-llm-self-training.md` |
| C2 objective | Masked diffusion noising/objective with red-green tests | active | `.omo/plans/diffusion-llm-self-training.md` |
| C3 model | Tiny CPU-friendly denoiser and checkpoint format | active | `.omo/plans/diffusion-llm-self-training.md` |
| C4 CLI | `diffusion-train`, `diffusion-sample`, `diffusion-eval` real surfaces | active | `.omo/plans/diffusion-llm-self-training.md` |
| C5 gate | Shared capability gate stays strict and non-claiming until code smoke passes | active | `.omo/plans/diffusion-llm-self-training.md` |
| C6 scale | 4B/Gemma/Dream-style scaling path documented but not attempted in first loop | deferred | `.omo/plans/diffusion-llm-self-training.md` |

## Open assumptions (announced defaults)
| assumption | adopted default | rationale | reversible? |
| --- | --- | --- | --- |
| Training framework | Add PyTorch behind optional train/dev dependency and keep non-training CLI importable without it | Actual neural training needs tensor autograd; optional dependency avoids breaking existing qwen/status commands | yes |
| First tokenizer | Use deterministic byte-level tokenizer with special mask/pad/bos/eos tokens | Avoids external tokenizer/model downloads and makes tests hermetic | yes |
| First corpus | Build from repo-local coding tasks, Qwen generated evidence, README snippets, and tiny synthetic Python examples | Enough to prove self-training mechanics without pretending general capability | yes |
| First model size | Tiny CPU smoke model, not 4B | First goal is a real train/sample/eval loop with isolated failure modes; 4B is impossible to debug before mechanics exist | yes |
| Diffusion family | Masked-token discrete diffusion / block denoising | Matches LLaDA/Dream/DiffusionGemma direction and current code-token target better than continuous embedding diffusion | yes |
| Capability claim | Keep `coding_capability_claim=false` unless the generated Python passes existing qwen-style executable smoke tests | Preserves README gate boundary and avoids false promotion | yes |
| Teacher data | Use Qwen bridge only to export optional local training examples after base loop exists | Prevents hidden Qwen fallback while using Qwen as bootstrap data source | yes |

## Findings (cited - path:lines)
- `README.md:3-8` defines qiffusion as two-track and says no model is coding-capable until executable smoke tests and reproducible evidence pass.
- `README.md:43-48` says the current diffusion surface is only `backend-status --backend diffusion`, not training/sampling/chat/coding-capable.
- `src/qiffusion/backends.py:18-27` hardcodes diffusion status as `scaffold_ready`, `fixtures_status=not_run`, `coding_capability_claim=False`.
- `src/qiffusion/cli.py:24-43` has argparse surfaces for `plan`, `status`, `qwen-status`, `qwen-eval`, and `backend-status`; there is no diffusion train/sample/eval command.
- `src/qiffusion/decision.py:25-33` promotes only when `fixtures_status == pass`, `code_smoke_status == pass`, and `coding_capability_claim is True`.
- `src/qiffusion/qwen_eval.py:54-90` shows the current evidence shape and repeated task-result pattern to mirror for diffusion eval.
- `pyproject.toml:9-13` currently has no runtime dependencies and only `pytest` in dev, so PyTorch must be added deliberately and preferably isolated.
- External baseline: LLaDA (`https://github.com/ML-GSAI/LLaDA`) is a large language diffusion model using masked-token diffusion ideas.
- External baseline: Dream / Dream-Coder (`https://github.com/HKUNLP/Dream`) demonstrates diffusion language models and code-oriented variants as a later scale/reference target.
- External baseline: DiffuCoder (`https://github.com/apple/ml-diffucoder`) is a code-generation masked diffusion reference point.
- External baseline: DiffusionGemma / Google diffusion LLM material should be treated as serving/architecture inspiration, not a dependency, until an official local checkpoint/interface is selected.

## Decisions (with rationale)
- First work plan target is "train a tiny diffusion LM end to end", not "produce a useful 4B model." The latter becomes a later scaling story after train/sample/eval artifacts exist.
- Use local byte-token data and synthetic Python snippets first. This keeps the first loop hermetic and avoids licensing/download blockers.
- Add PyTorch only for the diffusion training path. Existing qwen bridge/status commands must continue to work if PyTorch is absent.
- Integrate with the existing shared gate but keep diffusion claim false until generated samples pass executable Python coding fixtures.
- Store checkpoints and reports under `.omo/ulw-loop/.../evidence` or user-specified output paths; do not commit large checkpoint artifacts.

## Scope IN
- New diffusion modules for config, tokenizer/data, masking objective, tiny denoiser, training, sampling, and eval.
- CLI commands for train/sample/eval/status.
- Tests and real CLI QA evidence proving a checkpoint is trained and consumed.
- Documentation describing the tiny-loop milestone and deferred scaling path.

## Scope OUT (Must NOT have)
- No 4B training in this first diffusion implementation plan.
- No claim that the diffusion model can chat or code unless executable code smoke passes.
- No hidden Qwen fallback in diffusion sample/eval.
- No internet-dependent tests, external dataset downloads, or GPU-only commands.
- No large binary checkpoint committed to git.

## Open questions
- None blocking. User may veto adopted defaults before execution.

## Approval gate
status: plan-written
Plan file written at `.omo/plans/diffusion-llm-self-training.md`. Execution requires an explicit start command such as `$omo:start-work`.
