# diffusion-llm-self-training - Work Plan

## TL;DR (For humans)
**What you'll get:** A real first diffusion-LLM training path inside qiffusion: it builds a tiny local code corpus, trains a small masked-diffusion language model on CPU, saves a checkpoint, samples from it, evaluates the sample, and records evidence through the existing gate.

**Why this approach:** The diffusion side currently only reports `scaffold_ready`, so jumping straight to 4B would hide whether failures come from data, objective, sampling, model size, or evaluation. This plan first proves the whole train -> sample -> eval loop on a tiny model, then leaves a clean scaling path toward larger LLaDA/Dream/DiffusionGemma-style work.

**What it will NOT do:** It will not train a useful 4B model yet. It will not claim the diffusion model can chat or code unless generated Python passes executable smoke tests. It will not use Qwen as a hidden fallback.

**Effort:** Large
**Risk:** High - the first neural training loop adds a new dependency, model artifact format, data path, sampler, and gate wiring.
**Decisions I made for you:** Use optional PyTorch for training; start with deterministic byte-level tokenization; train a tiny CPU model first; build local code data from repo/Qwen evidence/synthetic snippets; keep `coding_capability_claim=false` until code smoke passes.

Your next move: run this with `$omo:start-work` when you want implementation to begin. Full execution detail follows below.

---

> TL;DR (machine): Large/high-risk plan to turn the diffusion scaffold into a real tiny self-training masked diffusion LM with checkpointed train/sample/eval CLI and strict shared-gate evidence.

## Scope
### Must have
- Preserve the shared capability rule from `README.md:8`: no coding-capable claim until executable smoke tests and reproducible evidence pass.
- Keep existing Qwen bridge commands working while diffusion training dependencies are absent or unconfigured.
- Add a real diffusion training path that trains a tiny masked-token denoiser on local data and writes a checkpoint artifact.
- Add deterministic byte-token data preparation so training tests and smoke runs do not require downloads, Hugging Face credentials, GPU, or internet.
- Add diffusion sample/eval commands that consume the checkpoint artifact produced by the train command.
- Wire diffusion eval reports into the existing `status --report` shared gate shape used by `src/qiffusion/decision.py:25-33`.
- Record real CLI evidence under `.omo/ulw-loop/diffusion-llm-self-training/evidence/`.
- Keep generated checkpoints out of git; commit code, docs, tests, and small JSON/text evidence only when appropriate.

### Must NOT have (guardrails, anti-slop, scope boundaries)
- Must not attempt 4B training in this plan.
- Must not call the diffusion model coding-capable unless generated code passes executable code smoke.
- Must not hide Qwen, Ollama, or any autoregressive model behind diffusion sample/eval.
- Must not require GPU for tests or default QA.
- Must not add external dataset downloads to tests.
- Must not commit `.pt`, `.safetensors`, cache, `__pycache__`, or `.pytest_cache` artifacts.
- Must not weaken existing Qwen eval or shared-gate tests.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: TDD with pytest for module behavior; red-first CLI tests for every new command.
- Real-surface QA: run `python -m qiffusion.cli diffusion-train`, `diffusion-sample`, `diffusion-eval`, and `status --report` against generated artifacts.
- Evidence: `.omo/ulw-loop/diffusion-llm-self-training/evidence/task-<N>-<name>.*`

## Execution strategy
### Parallel execution waves
Wave 1 (foundation, no neural training yet):
- Todo 1: Add diffusion config/artifact schema and CLI command stubs that remain non-claiming.
- Todo 2: Add deterministic byte tokenizer and corpus builder.
- Todo 3: Add local training-data exporter from Qwen/task evidence without invoking Qwen.

Wave 2 (learning mechanics):
- Todo 4: Add masked diffusion noising schedule and loss-target construction.
- Todo 5: Add tiny PyTorch denoiser and checkpoint save/load path.
- Todo 6: Add CPU training loop that produces a measurable checkpoint.

Wave 3 (generation and evaluation):
- Todo 7: Add iterative masked sampler from checkpoint.
- Todo 8: Add diffusion eval report and shared-gate integration.
- Todo 9: Add docs and scaling notes for 4B/DiffusionGemma-style future work.

Wave 4 (final proof):
- Todo 10: Run full train -> sample -> eval -> gate CLI QA and commit/push.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | none | 6, 7, 8, 10 | 2, 3 |
| 2 | none | 4, 6, 10 | 1, 3 |
| 3 | none | 6, 9, 10 | 1, 2 |
| 4 | 2 | 5, 6, 8 | none |
| 5 | 1, 4 | 6, 7, 8 | none |
| 6 | 1, 2, 3, 4, 5 | 7, 8, 10 | none |
| 7 | 5, 6 | 8, 10 | none |
| 8 | 1, 6, 7 | 10 | 9 |
| 9 | 3, 8 | 10 | 8 |
| 10 | 1-9 | final verification | none |

## Todos
> Implementation + Test = ONE todo. Never separate.

- [ ] 1. Add diffusion artifact schema and non-claiming CLI stubs
  What to do / Must NOT do: Add `src/qiffusion/diffusion_config.py` and `src/qiffusion/diffusion_reports.py` with typed report builders for training, sampling, and eval. Extend `src/qiffusion/cli.py` with `diffusion-train`, `diffusion-sample`, and `diffusion-eval` parser entries, but initial implementation may return `prerequisite_missing` until later todos fill behavior. Must not claim capability or import PyTorch at module import time.
  Parallelization: Wave 1 | Blocked by: none | Blocks: 6, 7, 8, 10
  References: `src/qiffusion/backends.py:18-27`, `src/qiffusion/cli.py:24-43`, `src/qiffusion/decision.py:25-33`, `README.md:43-48`
  Acceptance criteria: `python -m pytest tests/test_diffusion_cli.py tests/test_decision.py` passes; `python -m qiffusion.cli diffusion-eval --report-out <tmp>` emits JSON with `backend=diffusion`, `fixtures_status=not_run`, `code_smoke_status=not_run`, `coding_capability_claim=false`.
  QA scenarios:
  ```
  Scenario: diffusion eval stub remains non-claiming
    Tool: powershell
    Invocation: python -m qiffusion.cli diffusion-eval --report-out .omo/ulw-loop/diffusion-llm-self-training/evidence/task-1-eval-stub.json
    Expected: JSON contains backend diffusion and coding_capability_claim false
    Evidence: .omo/ulw-loop/diffusion-llm-self-training/evidence/task-1-eval-stub.json

  Scenario: shared gate rejects diffusion stub
    Tool: powershell
    Invocation: python -m qiffusion.cli status --report .omo/ulw-loop/diffusion-llm-self-training/evidence/task-1-eval-stub.json > .omo/ulw-loop/diffusion-llm-self-training/evidence/task-1-gate.txt
    Expected: status is continue
    Evidence: .omo/ulw-loop/diffusion-llm-self-training/evidence/task-1-gate.txt
  ```
  Commit: YES | `feat(diffusion): add training command surface`

- [ ] 2. Add deterministic byte tokenizer and local corpus builder
  What to do / Must NOT do: Add `src/qiffusion/diffusion_data.py` with byte-level encode/decode, special token IDs, sequence packing, and local corpus builders from `README.md`, `src/qiffusion/qwen_tasks.py`, `src/qiffusion/qwen_file_tasks.py`, `src/qiffusion/qwen_repair_tasks.py`, and small synthetic Python examples. Must not download datasets or use external tokenizers.
  Parallelization: Wave 1 | Blocked by: none | Blocks: 4, 6, 10
  References: `pyproject.toml:9-13`, `src/qiffusion/qwen_eval.py:93-115`, `README.md:3-18`
  Acceptance criteria: `python -m pytest tests/test_diffusion_data.py` passes; encode/decode round-trips ASCII Python; corpus builder returns non-empty examples with deterministic ordering.
  QA scenarios:
  ```
  Scenario: byte tokenizer round trip
    Tool: powershell
    Invocation: python -c "from qiffusion.diffusion_data import ByteTokenizer; t=ByteTokenizer(); s='def add(a, b):\\n    return a + b\\n'; assert t.decode(t.encode(s)) == s; print('pass')" > .omo/ulw-loop/diffusion-llm-self-training/evidence/task-2-tokenizer.txt
    Expected: file is exactly pass
    Evidence: .omo/ulw-loop/diffusion-llm-self-training/evidence/task-2-tokenizer.txt

  Scenario: local corpus has no network dependency
    Tool: powershell
    Invocation: python -c "from qiffusion.diffusion_data import build_local_corpus; data=build_local_corpus(); print(len(data)); assert len(data) >= 10" > .omo/ulw-loop/diffusion-llm-self-training/evidence/task-2-corpus.txt
    Expected: count is at least 10
    Evidence: .omo/ulw-loop/diffusion-llm-self-training/evidence/task-2-corpus.txt
  ```
  Commit: YES | `feat(diffusion): add local byte corpus`

- [ ] 3. Add Qwen/task evidence exporter for optional teacher corpus
  What to do / Must NOT do: Add `src/qiffusion/diffusion_teacher_data.py` to read existing qwen eval JSON artifacts and export generated passing code snippets into a JSONL corpus. It must accept paths explicitly and never call Ollama/Qwen itself. Add tests with tiny fixture JSON.
  Parallelization: Wave 1 | Blocked by: none | Blocks: 6, 9, 10
  References: `src/qiffusion/qwen_eval.py:76-90`, `src/qiffusion/qwen_eval.py:118-142`, `.omo/ulw-loop/coding-capability-20260629-1/evidence/G008-C003-repeated-repair.json`
  Acceptance criteria: `python -m pytest tests/test_diffusion_teacher_data.py` passes; exporter ignores failing task results and writes JSONL with source/provenance fields.
  QA scenarios:
  ```
  Scenario: export passing Qwen code evidence to JSONL
    Tool: powershell
    Invocation: python -m qiffusion.cli diffusion-export-teacher --qwen-report .omo/ulw-loop/coding-capability-20260629-1/evidence/G008-C003-repeated-repair.json --out .omo/ulw-loop/diffusion-llm-self-training/evidence/task-3-teacher.jsonl
    Expected: JSONL exists and includes slugify_title or merge_intervals generated code
    Evidence: .omo/ulw-loop/diffusion-llm-self-training/evidence/task-3-teacher.jsonl

  Scenario: exporter does not invoke Qwen
    Tool: powershell
    Invocation: python -c "from pathlib import Path; text=Path('src/qiffusion/diffusion_teacher_data.py').read_text(); print('run_ollama_fixture' not in text and 'ollama' not in text.lower())" > .omo/ulw-loop/diffusion-llm-self-training/evidence/task-3-no-ollama.txt
    Expected: file is exactly True
    Evidence: .omo/ulw-loop/diffusion-llm-self-training/evidence/task-3-no-ollama.txt
  ```
  Commit: YES | `feat(diffusion): export teacher code corpus`

- [ ] 4. Add masked diffusion noising schedule and loss targets
  What to do / Must NOT do: Add `src/qiffusion/diffusion_objective.py` with deterministic mask sampling, timestep/noise-rate schedule, and target mask for cross-entropy over masked tokens only. This module must be pure Python or NumPy-free unless Todo 5 introduces tensor conversion locally.
  Parallelization: Wave 2 | Blocked by: 2 | Blocks: 5, 6, 8
  References: LLaDA masked diffusion baseline `https://github.com/ML-GSAI/LLaDA`; Dream baseline `https://github.com/HKUNLP/Dream`; `src/qiffusion/qwen_tasks.py` for small pure functions/tests style
  Acceptance criteria: `python -m pytest tests/test_diffusion_objective.py` passes; mask schedule is deterministic with a seed; unmasked positions are excluded from loss targets.
  QA scenarios:
  ```
  Scenario: seeded masking is reproducible
    Tool: powershell
    Invocation: python -c "from qiffusion.diffusion_objective import mask_tokens; print(mask_tokens([1,2,3,4], seed=7) == mask_tokens([1,2,3,4], seed=7))" > .omo/ulw-loop/diffusion-llm-self-training/evidence/task-4-repro.txt
    Expected: file is exactly True
    Evidence: .omo/ulw-loop/diffusion-llm-self-training/evidence/task-4-repro.txt

  Scenario: loss targets cover masked positions only
    Tool: powershell
    Invocation: python -c "from qiffusion.diffusion_objective import mask_tokens; r=mask_tokens([10,11,12,13], seed=3); print(all((m == -100) == (not is_masked) for m, is_masked in zip(r.labels, r.masked_positions)))" > .omo/ulw-loop/diffusion-llm-self-training/evidence/task-4-targets.txt
    Expected: file is exactly True
    Evidence: .omo/ulw-loop/diffusion-llm-self-training/evidence/task-4-targets.txt
  ```
  Commit: YES | `feat(diffusion): add masked denoising objective`

- [ ] 5. Add tiny PyTorch denoiser and checkpoint format
  What to do / Must NOT do: Add optional training dependency handling plus `src/qiffusion/diffusion_model.py`. Implement a tiny Transformer or GRU denoiser with embeddings, positional embeddings, and vocab projection. Add `save_checkpoint`/`load_checkpoint` helpers storing config and weights. Existing non-training imports must not crash when PyTorch is absent.
  Parallelization: Wave 2 | Blocked by: 1, 4 | Blocks: 6, 7, 8
  References: `pyproject.toml:9-13`, `src/qiffusion/qwen_ollama.py` for dependency isolation style, DiffuCoder code-generation reference `https://github.com/apple/ml-diffucoder`
  Acceptance criteria: `python -m pytest tests/test_diffusion_model.py` passes on CPU; checkpoint round-trip preserves config and produces same logits for fixed input.
  QA scenarios:
  ```
  Scenario: tiny model forward pass works on CPU
    Tool: powershell
    Invocation: python -c "from qiffusion.diffusion_model import TinyDiffusionLM; import torch; m=TinyDiffusionLM(vocab_size=260, dim=32, layers=1); print(tuple(m(torch.zeros((1,8), dtype=torch.long)).shape))" > .omo/ulw-loop/diffusion-llm-self-training/evidence/task-5-forward.txt
    Expected: shape is (1, 8, 260)
    Evidence: .omo/ulw-loop/diffusion-llm-self-training/evidence/task-5-forward.txt

  Scenario: checkpoint round trip
    Tool: powershell
    Invocation: python -c "from qiffusion.diffusion_model import TinyDiffusionLM, save_checkpoint, load_checkpoint; import torch; p='.omo/ulw-loop/diffusion-llm-self-training/evidence/task-5-checkpoint.pt'; m=TinyDiffusionLM(260,32,1); save_checkpoint(p,m,{'vocab_size':260,'dim':32,'layers':1}); print(load_checkpoint(p)[1]['vocab_size'])" > .omo/ulw-loop/diffusion-llm-self-training/evidence/task-5-checkpoint.txt
    Expected: file contains 260
    Evidence: .omo/ulw-loop/diffusion-llm-self-training/evidence/task-5-checkpoint.txt
  ```
  Commit: YES | `feat(diffusion): add tiny denoiser checkpoint`

- [ ] 6. Add CPU training loop that writes a real checkpoint and metrics
  What to do / Must NOT do: Add `src/qiffusion/diffusion_train.py` and implement `diffusion-train --out <checkpoint> --report-out <json> --steps <N> --seed <N> --max-examples <N>`. It must train on local corpus plus optional teacher JSONL and write metrics including initial/final loss. It must run in under a few minutes for QA defaults. Must not require GPU.
  Parallelization: Wave 2 | Blocked by: 1, 2, 3, 4, 5 | Blocks: 7, 8, 10
  References: `src/qiffusion/cli.py:58-85`, `src/qiffusion/qwen_eval.py:76-90`, `README.md:20-30`
  Acceptance criteria: `python -m pytest tests/test_diffusion_train.py` passes; real CLI train writes non-empty checkpoint and JSON report; final loss is finite and no worse than a loose threshold against initial loss.
  QA scenarios:
  ```
  Scenario: tiny diffusion training writes checkpoint and metrics
    Tool: powershell
    Invocation: python -m qiffusion.cli diffusion-train --steps 20 --seed 1 --max-examples 24 --out .omo/ulw-loop/diffusion-llm-self-training/evidence/task-6-tiny.pt --report-out .omo/ulw-loop/diffusion-llm-self-training/evidence/task-6-train.json
    Expected: checkpoint exists; JSON has status trained and finite final_loss
    Evidence: .omo/ulw-loop/diffusion-llm-self-training/evidence/task-6-train.json

  Scenario: train rejects zero steps
    Tool: powershell
    Invocation: python -m qiffusion.cli diffusion-train --steps 0 --out .omo/ulw-loop/diffusion-llm-self-training/evidence/task-6-bad.pt --report-out .omo/ulw-loop/diffusion-llm-self-training/evidence/task-6-bad.json > .omo/ulw-loop/diffusion-llm-self-training/evidence/task-6-bad.txt 2>&1; if ($LASTEXITCODE -eq 0) { exit 1 } else { exit 0 }
    Expected: command fails and no capability claim is produced
    Evidence: .omo/ulw-loop/diffusion-llm-self-training/evidence/task-6-bad.txt
  ```
  Commit: YES | `feat(diffusion): train tiny masked lm`

- [ ] 7. Add iterative masked sampler from checkpoint
  What to do / Must NOT do: Add `src/qiffusion/diffusion_sample.py` and CLI `diffusion-sample --checkpoint <pt> --prompt <text> --steps <N> --out <json>`. Sampling should use a simple confidence or left-to-right fallback unmasking loop over masked byte tokens. It must consume the diffusion checkpoint and must not call Qwen/Ollama.
  Parallelization: Wave 3 | Blocked by: 5, 6 | Blocks: 8, 10
  References: DiffusionGemma-style block/canvas denoising as future reference; `src/qiffusion/qwen_ollama.py` must not be imported by diffusion sampling; `src/qiffusion/qwen_eval.py:126-142` for report field style
  Acceptance criteria: `python -m pytest tests/test_diffusion_sample.py` passes; sampler produces deterministic output for fixed seed; sample report includes checkpoint path/provenance.
  QA scenarios:
  ```
  Scenario: sample from trained checkpoint
    Tool: powershell
    Invocation: python -m qiffusion.cli diffusion-sample --checkpoint .omo/ulw-loop/diffusion-llm-self-training/evidence/task-6-tiny.pt --prompt "def" --steps 8 --seed 1 --out .omo/ulw-loop/diffusion-llm-self-training/evidence/task-7-sample.json
    Expected: JSON contains generated_text and checkpoint path
    Evidence: .omo/ulw-loop/diffusion-llm-self-training/evidence/task-7-sample.json

  Scenario: sampler does not import Qwen bridge
    Tool: powershell
    Invocation: python -c "from pathlib import Path; text=Path('src/qiffusion/diffusion_sample.py').read_text().lower(); print('qwen' not in text and 'ollama' not in text)" > .omo/ulw-loop/diffusion-llm-self-training/evidence/task-7-no-qwen.txt
    Expected: file is exactly True
    Evidence: .omo/ulw-loop/diffusion-llm-self-training/evidence/task-7-no-qwen.txt
  ```
  Commit: YES | `feat(diffusion): sample from tiny checkpoint`

- [ ] 8. Add diffusion eval and shared gate integration
  What to do / Must NOT do: Add `src/qiffusion/diffusion_eval.py` and CLI `diffusion-eval --checkpoint <pt> --out <json> --runs <N>`. Evaluate both training viability and generated-code smoke. The report may set `fixtures_status=pass` for training/sample mechanics but must set `code_smoke_status=fail` and `coding_capability_claim=false` unless generated Python passes executable smoke. Reuse existing qwen task smoke helpers only as evaluators, not as generators.
  Parallelization: Wave 3 | Blocked by: 1, 6, 7 | Blocks: 10
  References: `src/qiffusion/decision.py:25-33`, `src/qiffusion/qwen_tasks.py`, `src/qiffusion/qwen_eval.py:150-174`, `README.md:8`
  Acceptance criteria: `python -m pytest tests/test_diffusion_eval.py tests/test_decision.py` passes; `status --report <diffusion-eval.json>` returns continue until code smoke passes.
  QA scenarios:
  ```
  Scenario: diffusion eval consumes checkpoint and stays non-claiming
    Tool: powershell
    Invocation: python -m qiffusion.cli diffusion-eval --checkpoint .omo/ulw-loop/diffusion-llm-self-training/evidence/task-6-tiny.pt --runs 1 --out .omo/ulw-loop/diffusion-llm-self-training/evidence/task-8-eval.json
    Expected: JSON contains backend diffusion, checkpoint path, and coding_capability_claim false
    Evidence: .omo/ulw-loop/diffusion-llm-self-training/evidence/task-8-eval.json

  Scenario: shared gate does not promote tiny diffusion checkpoint
    Tool: powershell
    Invocation: python -m qiffusion.cli status --report .omo/ulw-loop/diffusion-llm-self-training/evidence/task-8-eval.json > .omo/ulw-loop/diffusion-llm-self-training/evidence/task-8-gate.txt
    Expected: JSON status is continue
    Evidence: .omo/ulw-loop/diffusion-llm-self-training/evidence/task-8-gate.txt
  ```
  Commit: YES | `feat(diffusion): evaluate tiny checkpoint`

- [ ] 9. Document diffusion training milestone and deferred scaling path
  What to do / Must NOT do: Update `README.md` and optionally `docs/diffusion-training.md` with exact train/sample/eval commands, artifact policy, and a clear section "Not 4B yet." Include future scaling notes for LLaDA/Dream/DiffuCoder/DiffusionGemma-style work: larger tokenizer/corpus, AR initialization or distillation, block/canvas sampling, and coding benchmark promotion. Must not imply current tiny model is useful.
  Parallelization: Wave 3 | Blocked by: 3, 8 | Blocks: 10
  References: `README.md:3-18`, `README.md:43-48`, LLaDA `https://github.com/ML-GSAI/LLaDA`, Dream `https://github.com/HKUNLP/Dream`, DiffuCoder `https://github.com/apple/ml-diffucoder`
  Acceptance criteria: docs include the exact commands; docs say checkpoints are not committed; docs say 4B is deferred.
  QA scenarios:
  ```
  Scenario: documented commands are present
    Tool: powershell
    Invocation: python -c "from pathlib import Path; text=Path('README.md').read_text(); print(all(x in text for x in ['diffusion-train','diffusion-sample','diffusion-eval']))" > .omo/ulw-loop/diffusion-llm-self-training/evidence/task-9-doc-commands.txt
    Expected: file is exactly True
    Evidence: .omo/ulw-loop/diffusion-llm-self-training/evidence/task-9-doc-commands.txt

  Scenario: docs do not overclaim 4B or coding capability
    Tool: powershell
    Invocation: python -c "from pathlib import Path; text=(Path('README.md').read_text() + Path('docs/diffusion-training.md').read_text()).lower(); print('not 4b yet' in text and 'coding_capability_claim=false' in text)" > .omo/ulw-loop/diffusion-llm-self-training/evidence/task-9-no-overclaim.txt
    Expected: file is exactly True
    Evidence: .omo/ulw-loop/diffusion-llm-self-training/evidence/task-9-no-overclaim.txt
  ```
  Commit: YES | `docs(diffusion): document tiny training loop`

- [ ] 10. Run full diffusion train/sample/eval QA, commit, and push
  What to do / Must NOT do: Run all tests, compile changed modules, run no-excuse checker if available, run the full CLI chain, verify artifacts exist, commit logical changes, and push `main`. Must not commit binary checkpoints or caches.
  Parallelization: Wave 4 | Blocked by: 1-9 | Blocks: final verification
  References: `pyproject.toml:15-16`, `README.md:37-39`, `.gitignore`
  Acceptance criteria: `python -m pytest` passes; full CLI chain produces train/sample/eval/status evidence; `git status --short` has no binary checkpoint staged; `git push origin main` succeeds.
  QA scenarios:
  ```
  Scenario: full tiny diffusion surface works end to end
    Tool: powershell
    Invocation: python -m qiffusion.cli diffusion-train --steps 20 --seed 11 --max-examples 24 --out .omo/ulw-loop/diffusion-llm-self-training/evidence/final-tiny.pt --report-out .omo/ulw-loop/diffusion-llm-self-training/evidence/final-train.json; python -m qiffusion.cli diffusion-sample --checkpoint .omo/ulw-loop/diffusion-llm-self-training/evidence/final-tiny.pt --prompt "def" --steps 8 --seed 11 --out .omo/ulw-loop/diffusion-llm-self-training/evidence/final-sample.json; python -m qiffusion.cli diffusion-eval --checkpoint .omo/ulw-loop/diffusion-llm-self-training/evidence/final-tiny.pt --runs 1 --out .omo/ulw-loop/diffusion-llm-self-training/evidence/final-eval.json; python -m qiffusion.cli status --report .omo/ulw-loop/diffusion-llm-self-training/evidence/final-eval.json > .omo/ulw-loop/diffusion-llm-self-training/evidence/final-gate.txt
    Expected: train/sample/eval artifacts exist; final gate status is continue unless code smoke passes
    Evidence: .omo/ulw-loop/diffusion-llm-self-training/evidence/final-gate.txt

  Scenario: binary checkpoint not staged
    Tool: powershell
    Invocation: git status --short > .omo/ulw-loop/diffusion-llm-self-training/evidence/final-git-status.txt
    Expected: status does not include .pt, .safetensors, __pycache__, or .pytest_cache
    Evidence: .omo/ulw-loop/diffusion-llm-self-training/evidence/final-git-status.txt
  ```
  Commit: YES | `feat(diffusion): train and evaluate tiny masked lm`

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [ ] F1. Plan compliance audit: every todo done, every acceptance criterion met, every evidence path exists.
- [ ] F2. Code quality review: no overbroad abstractions, no import-time PyTorch breakage for non-training commands, files below local size ceiling.
- [ ] F3. Real manual QA: rerun full train -> sample -> eval -> status chain and inspect JSON artifacts.
- [ ] F4. Scope fidelity: no 4B overclaim, no hidden Qwen fallback, no binary checkpoints committed.

## Commit strategy
- Commit after each logical wave when tests and that wave's CLI QA pass.
- Use Conventional Commits.
- Push every completed commit to `origin/main` because this repo's workflow requests automatic commit and push after verified work.
- Final implementation commit footer: `Plan: .omo/plans/diffusion-llm-self-training.md`.

## Success criteria
- A tiny diffusion model checkpoint is trained locally from deterministic local data.
- Sampling consumes the trained checkpoint and produces a JSON sample artifact.
- Diffusion eval consumes the checkpoint and produces a shared-gate-compatible report.
- The shared gate remains conservative: it promotes only if executable coding smoke actually passes.
- Full pytest and CLI QA evidence are captured under `.omo/ulw-loop/diffusion-llm-self-training/evidence/`.
