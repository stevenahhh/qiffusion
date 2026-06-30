# qwen-diffusion-training-loop-plan-set - Work Plan

## TL;DR (For humans)
This plan turns the current tiny diffusion scaffold into a real repeated training and validation loop for a Qwen-based diffusion model. It first builds the missing data, tokenizer, model, sampler, and evaluation surfaces, then repeats train, sample, evaluate, diagnose, and improve until the model itself passes conservative coding and chat gates.

The load-bearing decision is to stop improving the current byte-level toy loop as if it could become the final model. The loop must move to Qwen-token masked diffusion, with provenance-controlled data and no hidden autoregressive fallback.

It will not claim coding capability from one lucky sample. It will not train on benchmark splits used for release claims. It will not use Qwen, Ollama, or another autoregressive model as a hidden inference fallback.

**Effort:** XL
**Risk:** High - real coding capability depends on data quality, compute, checkpoint availability, and diffusion adaptation quality.
**Decisions I made for you:** Use Qwen-family AR-to-diffusion adaptation, start with small verified data and teacher traces, add a no-download Qwen checkpoint compatibility contract before model work, keep `transformers` optional/dynamic, gate large corpora behind provenance controls, keep LoRA for later specialization, and split code/chat/agent capability gates.

Your next move: execute this plan through `$start-work`; no further product decision is needed before the first implementation wave. Full execution detail follows below.

---

> TL;DR (machine): XL/high-risk plan set to build Qwen-token diffusion data, model, sampler, eval, and repeat-loop infrastructure until coding/chat gates pass without fallback.

## Scope
### Must have
- A corpus manifest surface with provenance, license, split, tokenizer, token count, dedup, and benchmark contamination fields.
- A teacher-trace schema for code, repair, chat, tool, and execution-result examples.
- A Qwen tokenizer adapter with an optional `transformers` boundary and byte-token fallback only for tiny local tests.
- A Qwen diffusion config surface covering base checkpoint id, attention mode, objective, sampler, mask schedule, checkpoint lineage, Qwen compatibility contract, and resource probe output.
- A Qwen-token denoiser scaffold that can run a small CPU smoke path without downloading or training a full 4B checkpoint.
- A sampler interface for iterative mask diffusion with deterministic steps, seed, temperature/top-k controls, history, and no hidden autoregressive fallback.
- An expanded eval and release-gate surface that separates local code smoke, benchmark harness readiness, chat, tool/agent, and software-engineering claims.
- A loop runner that records train -> sample -> eval -> diagnose -> next-data/objective/sampler action and repeats until the conservative gates pass or a real resource blocker is recorded.

### Must NOT have (guardrails, anti-slop, scope boundaries)
- Must not claim `coding_capability_claim=true` unless the diffusion model itself passes the local code gate and the claim is backed by explicit evidence.
- Must not train on HumanEval, MBPP, EvalPlus, LiveCodeBench, BigCodeBench, SWE-bench, or other benchmark splits used for reported evaluation.
- Must not add hidden Qwen/Ollama/autoregressive inference fallback to make diffusion samples look better.
- Must not download or train a full 4B model as part of a narrow smoke test unless a later task explicitly records resource availability and model-cache location.
- Must not treat `not_run` or `blocked` chat/tool/software-engineering buckets as final capability success; those states can complete an infrastructure task only as recorded blockers.
- Must not replace the existing byte-level tiny loop; keep it as deterministic regression scaffolding.
- Must not ingest raw user logs, unlabeled scraped code with unknown licenses, or privacy-risk chat data without a manifest and policy gate.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: TDD for new behavior and baseline characterization before changing existing CLI/report behavior; `pytest` remains the primary framework.
- Automated verification: targeted tests for each new module, then `python -m pytest`.
- Manual QA surface: CLI/data-artifact QA using `python -m qiffusion.cli ...` commands that write JSON evidence under `.omo/evidence/qwen-diffusion-training-loop-plan-set/`.
- Evidence root: `.omo/evidence/qwen-diffusion-training-loop-plan-set/`.
- Ledger root: `.omo/start-work/ledger.jsonl`.
- All evidence must record exact command, exit status, generated artifact path, and whether the result is a pass, fail, or resource-blocked state.
- Every worker must probe applicable adversarial classes: `dirty_worktree`, `stale_state`, `misleading_success_output`, and any task-specific classes such as malformed input, prompt injection, privacy/licensing risk, long command, or flaky test.
- Start-work bootstrap: before any product todo is dispatched, the orchestrator creates `.omo/boulder.json`, `.omo/start-work/ledger.jsonl`, and `.omo/evidence/qwen-diffusion-training-loop-plan-set/`, then appends a `work-started` ledger record. This is `.omo` state, not a product implementation todo.
- Dependency policy: `transformers` must not become a mandatory dependency. Todo 3 may use dynamic import and may add an optional extra only if tests prove package metadata is needed; existing byte tests must still pass without installing `transformers`.
- No-fallback verification boundary: any diffusion train/sample/eval/loop test that claims no fallback must monkeypatch or spy the Qwen/Ollama bridge and generation entry points so a call to `qwen_eval`, `qwen_status`, bridge generation, or Ollama execution fails the test.
- Data contamination schema values: dataset usage must resolve to `train_allowed`, `eval_only`, or `unknown_blocked`; contamination status must resolve to `clean`, `suspect`, or `blocked`. Training must reject `eval_only`, `unknown_blocked`, `suspect`, and `blocked` inputs unless a later policy task explicitly changes the rule.
- Cleanup policy: generated evidence under `.omo/evidence/qwen-diffusion-training-loop-plan-set/` may remain ignored, but workers must delete temp dirs/processes and must not leave large checkpoints outside `.omo/evidence/`; each todo ledger entry needs a cleanup receipt and tracked `git status --short` summary.

## Execution strategy
### Parallel execution waves
> Target 5-8 todos per wave. Fewer than 3 (except the final) means you under-split.
- Wave 0, bootstrap: orchestrator-only `.omo` state creation before product work.
- Wave 1, foundation: todos 1-4 may run in parallel. Todo 1 owns new `src/qiffusion/cli.py` parser wiring for `diffusion-corpus`; todo 2 may change only the existing `diffusion-export-teacher` path; todos 3-4 use module APIs or `python -c` drivers and must not add new CLI subcommands in wave 1.
- Wave 2, Qwen-token diffusion scaffold: todos 5-8 start only after their listed foundation dependencies pass.
- Wave 3, repeated improvement loop: todos 9-12 turn the scaffold into a conservative train/validate/improve loop and capability ledger.
- Wave 4, final verification: F1-F4 run only after all todos are checked.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | none | 5, 7, 9, 10, 12 | 2, 3, 4 |
| 2 | none | 9, 11 | 1, 3, 4 |
| 3 | none | 5, 6, 7 | 1, 2, 4 |
| 4 | none | 5, 8, 10, 12 | 1, 2, 3 |
| 5 | 3, 4 | 6, 7 | 8 after 4 |
| 6 | 3, 4, 5 | 7, 10 | 8 |
| 7 | 1, 3, 4, 5, 6 | 9, 10 | none |
| 8 | 4, 6 | 10, 12 | 7 after 6 |
| 9 | 1, 2, 7 | 10, 11 | 12 after 8 |
| 10 | 4, 7, 8, 9 | 12 | 11 |
| 11 | 2, 9 | 12 | 10 |
| 12 | 1, 4, 8, 10, 11 | F1-F4 | none |

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->
- [x] 1. Add corpus manifest and provenance surface.
  What to do / Must NOT do: Add a manifest module and CLI path for local corpora, teacher JSONL, and future external corpora. It must record source, license, split, tokenizer, token counts, dedup hash, usage (`train_allowed`, `eval_only`, `unknown_blocked`), contamination status (`clean`, `suspect`, `blocked`), and privacy/policy notes. Do not ingest broad external corpora yet.
  Parallelization: Wave 1 | Blocked by: none | Blocks: 5, 7, 9, 10, 12
  References (executor has NO interview context - be exhaustive): `docs/qwen-diffusion-coding-model-ultraresearch.md:12`, `docs/qwen-diffusion-coding-model-ultraresearch.md:81`, `docs/qwen-diffusion-coding-model-ultraresearch.md:120`, `docs/qwen-diffusion-coding-model-ultraresearch.md:145`, `docs/qwen-diffusion-coding-model-ultraresearch.md:292`, `src/qiffusion/diffusion_data.py:18`, `src/qiffusion/diffusion_data.py:64`, `src/qiffusion/cli.py:26`
  Acceptance criteria (agent-executable): Write a failing test first for manifest output, implement the module and CLI, preserve existing `tests/test_diffusion_data.py` and `tests/test_cli_tracks.py`, then run `python -m pytest tests/test_diffusion_corpus_manifest.py tests/test_diffusion_data.py tests/test_cli_tracks.py` and `python -m pytest`.
  QA scenarios (name the exact tool + invocation): Happy command: `python -m qiffusion.cli diffusion-corpus manifest --root . --out .omo/evidence/qwen-diffusion-training-loop-plan-set/task-1-manifest.json` and assert JSON includes nonempty records, tokenizer, token counts, usage, and contamination status. Failure command: `python -m qiffusion.cli diffusion-corpus manifest --root . --teacher-jsonl .omo/evidence/qwen-diffusion-training-loop-plan-set/task-1-malformed.jsonl --out .omo/evidence/qwen-diffusion-training-loop-plan-set/task-1-manifest-failure.json` and assert nonzero exit or structured error evidence.
  Commit: Y | `feat(data): add diffusion corpus manifest`

- [x] 2. Add teacher trace schema for code, chat, tool, repair, and execution records.
  What to do / Must NOT do: Define structured teacher-trace records with provenance fields for source, license, teacher model, prompt hash, checker hash, task type, execution outcome, and policy notes. Update only the existing `diffusion-export-teacher` boundary needed to emit or validate these records. Do not add a new CLI subcommand and do not rewrite the Qwen evaluator itself unless a failing test requires it.
  Parallelization: Wave 1 | Blocked by: none | Blocks: 9, 11
  References (executor has NO interview context - be exhaustive): `docs/qwen-diffusion-coding-model-ultraresearch.md:91`, `docs/qwen-diffusion-coding-model-ultraresearch.md:111`, `docs/qwen-diffusion-coding-model-ultraresearch.md:303`, `src/qiffusion/diffusion_teacher_data.py:50`, `src/qiffusion/cli.py:70`, `tests/test_diffusion_teacher_data.py`
  Acceptance criteria (agent-executable): Add tests that reject missing provenance and accept code/chat/tool/repair/execution record variants; run `python -m pytest tests/test_diffusion_teacher_data.py` and `python -m pytest`.
  QA scenarios (name the exact tool + invocation): Happy command: `python -m qiffusion.cli diffusion-export-teacher --qwen-report .omo/evidence/qwen-diffusion-training-loop-plan-set/task-2-qwen-report.json --out .omo/evidence/qwen-diffusion-training-loop-plan-set/task-2-teacher.jsonl` using a fixture created by the worker at that exact report path, then validate each line against the new schema. Failure command: `python -m qiffusion.cli diffusion-export-teacher --qwen-report .omo/evidence/qwen-diffusion-training-loop-plan-set/task-2-missing-provenance-report.json --out .omo/evidence/qwen-diffusion-training-loop-plan-set/task-2-teacher-failure.jsonl` and assert missing license/checker hash rejection is captured.
  Commit: Y | `feat(data): add teacher trace schema`

- [x] 3. Add Qwen tokenizer adapter and tokenization boundary.
  What to do / Must NOT do: Add a tokenizer abstraction that can use a local Qwen tokenizer through optional dynamic import of `transformers` when available, while keeping `ByteTokenizer` as the deterministic tiny-test fallback. Do not make `transformers` a mandatory dependency for existing tests; add an optional extra only if the package metadata tests require it.
  Parallelization: Wave 1 | Blocked by: none | Blocks: 5, 6, 7
  References (executor has NO interview context - be exhaustive): `docs/qwen-diffusion-coding-model-ultraresearch.md:59`, `docs/qwen-diffusion-coding-model-ultraresearch.md:145`, `docs/qwen-diffusion-coding-model-ultraresearch.md:307`, `src/qiffusion/diffusion_data.py:25`, `src/qiffusion/diffusion_config.py`, `pyproject.toml`
  Acceptance criteria (agent-executable): Tests cover byte fallback, unavailable-transformers behavior, Qwen-tokenizer config serialization, and round-trip constraints while preserving existing byte training tests; run `python -m pytest tests/test_qwen_tokenizer.py tests/test_diffusion_train.py tests/test_diffusion_data.py` and `python -m pytest`.
  QA scenarios (name the exact tool + invocation): Happy command: `python -c "from pathlib import Path; from qiffusion.qwen_tokenizer import load_tokenizer; out=Path('.omo/evidence/qwen-diffusion-training-loop-plan-set/task-3-tokenizer.txt'); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(str(load_tokenizer('byte').encode('def add')), encoding='utf-8')"` and assert stdout/artifact contains token ids. Failure command: `python -c "from pathlib import Path; from qiffusion.qwen_tokenizer import write_unavailable_probe; write_unavailable_probe('missing-local-qwen-tokenizer', Path('.omo/evidence/qwen-diffusion-training-loop-plan-set/task-3-tokenizer-failure.json'))"` and assert the artifact reports `status: unavailable` without installing `transformers`.
  Commit: Y | `feat(tokenizer): add qwen tokenizer adapter`

- [x] 4. Add Qwen diffusion config, compatibility contract, lineage, and benchmark gate schema.
  What to do / Must NOT do: Add a serializable config for Qwen diffusion runs: base checkpoint id, tokenizer id, attention mode, objective, sampler algorithm, mask schedule, block size, seed, data manifest id, checkpoint lineage, no-download Qwen compatibility contract, and resource probe output. The compatibility contract must record expected checkpoint family, tokenizer id, config fields, tensor-name mapping notes, and resource availability status without downloading weights. Add benchmark-gate metadata that marks benchmark datasets as eval-only. Do not add real training behavior here.
  Parallelization: Wave 1 | Blocked by: none | Blocks: 8, 10, 12
  References (executor has NO interview context - be exhaustive): `docs/qwen-diffusion-coding-model-ultraresearch.md:46`, `docs/qwen-diffusion-coding-model-ultraresearch.md:67`, `docs/qwen-diffusion-coding-model-ultraresearch.md:136`, `docs/qwen-diffusion-coding-model-ultraresearch.md:161`, `docs/qwen-diffusion-coding-model-ultraresearch.md:242`, `docs/qwen-diffusion-coding-model-ultraresearch.md:311`
  Acceptance criteria (agent-executable): Tests serialize/deserialize valid configs, reject benchmark-training contamination, reject unsupported attention/objective/sampler values, write a no-download compatibility contract, and report resource status as `available`, `missing`, or `unknown`; run `python -m pytest tests/test_qwen_diffusion_config.py` and `python -m pytest`.
  QA scenarios (name the exact tool + invocation): Happy command: `python -c "from pathlib import Path; from qiffusion.qwen_diffusion_config import write_default_config, write_compatibility_contract; root=Path('.omo/evidence/qwen-diffusion-training-loop-plan-set'); root.mkdir(parents=True, exist_ok=True); write_default_config(root/'task-4-config.json', base_checkpoint_id='Qwen/Qwen3.5-4B'); write_compatibility_contract(root/'task-4-compatibility.json', base_checkpoint_id='Qwen/Qwen3.5-4B')"` and assert both JSON files round-trip without network. Failure command: `python -c "from pathlib import Path; from qiffusion.qwen_diffusion_config import write_contamination_probe; write_contamination_probe(Path('.omo/evidence/qwen-diffusion-training-loop-plan-set/task-4-contamination-failure.json'), training_sources=['humaneval'])"` and assert the artifact reports rejection.
  Commit: Y | `feat(config): add qwen diffusion config schema`

- [x] 5. Add Qwen-token denoiser scaffold with tiny CPU smoke mode.
  What to do / Must NOT do: Add an internal tiny masked denoiser API over token ids that can run in CPU mode and follows the compatibility metadata from todo 4. It does not need exact Qwen weight-loading yet, but must not drift from the recorded Qwen config/tokenizer contract. Do not download a full checkpoint or remove the existing `TinyDiffusionLM`.
  Parallelization: Wave 2 | Blocked by: 3, 4 | Blocks: 6, 7
  References (executor has NO interview context - be exhaustive): `docs/qwen-diffusion-coding-model-ultraresearch.md:48`, `docs/qwen-diffusion-coding-model-ultraresearch.md:59`, `docs/qwen-diffusion-coding-model-ultraresearch.md:161`, `src/qiffusion/diffusion_model.py:18`, `src/qiffusion/diffusion_model.py:42`, `src/qiffusion/diffusion_model.py:73`
  Acceptance criteria (agent-executable): Tests instantiate the tiny Qwen-token denoiser, run forward logits with deterministic dimensions, save/load a checkpoint manifest, confirm compatibility metadata is present, monkeypatch Qwen/Ollama bridge calls to fail if used, and preserve existing model tests; run `python -m pytest tests/test_qwen_diffusion_model.py tests/test_diffusion_model.py` and `python -m pytest`.
  QA scenarios (name the exact tool + invocation): Happy command: `python -c "from pathlib import Path; from qiffusion.qwen_diffusion_model import write_tiny_model_evidence; write_tiny_model_evidence(Path('.omo/evidence/qwen-diffusion-training-loop-plan-set/task-5-model.json'), batch_size=1, sequence_length=8, vocab_size=64)"` and assert logits/config/compatibility metadata are present. Failure command: `python -c "from pathlib import Path; from qiffusion.qwen_diffusion_model import write_mismatch_checkpoint_probe; write_mismatch_checkpoint_probe(Path('.omo/evidence/qwen-diffusion-training-loop-plan-set/task-5-mismatch-failure.json'))"` and assert the artifact records tokenizer/config mismatch.
  Commit: Y | `feat(model): add qwen token denoiser scaffold`

- [x] 6. Add iterative mask sampler interface with confidence history and no fallback.
  What to do / Must NOT do: Add a Qwen-token sampler interface supporting fixed steps, seed, temperature, top-k, confidence or entropy scores, early stop metadata, and history. It may run against the tiny scaffold from todo 5. It must not call Qwen/Ollama for generated tokens.
  Parallelization: Wave 2 | Blocked by: 3, 5 | Blocks: 7, 10
  References (executor has NO interview context - be exhaustive): `docs/qwen-diffusion-coding-model-ultraresearch.md:63`, `docs/qwen-diffusion-coding-model-ultraresearch.md:67`, `docs/qwen-diffusion-coding-model-ultraresearch.md:221`, `src/qiffusion/diffusion_sample.py:37`, `src/qiffusion/diffusion_sample.py:45`, `src/qiffusion/diffusion_sample.py:54`
  Acceptance criteria (agent-executable): Tests prove deterministic sampling with seed, history entries per step, confidence/entropy metadata, no external fallback calls via monkeypatch/spies, and preservation of existing byte sampler tests; run `python -m pytest tests/test_qwen_diffusion_sample.py tests/test_diffusion_sample.py` and `python -m pytest`.
  QA scenarios (name the exact tool + invocation): Happy command: `python -c "from pathlib import Path; from qiffusion.qwen_diffusion_sample import write_sample_evidence; write_sample_evidence(Path('.omo/evidence/qwen-diffusion-training-loop-plan-set/task-6-sample.json'), prompt='def add', steps=4, seed=11)"` and assert the JSON includes history, confidence or entropy, and `fallback_used: false`. Failure command: `python -c "from pathlib import Path; from qiffusion.qwen_diffusion_sample import write_sampler_failure_probe; write_sampler_failure_probe(Path('.omo/evidence/qwen-diffusion-training-loop-plan-set/task-6-sampler-failure.json'), algorithm='unsupported')"` and assert rejection is captured.
  Commit: Y | `feat(sample): add qwen token mask sampler`

- [x] 7. Add Qwen-token training smoke loop.
  What to do / Must NOT do: Add a small train routine that consumes manifest-filtered local examples, uses masked CE over Qwen-token ids or byte fallback ids, writes checkpoint lineage, and records train loss. Do not attempt a long run or full 4B conversion in this smoke path.
  Parallelization: Wave 2 | Blocked by: 1, 3, 5, 6 | Blocks: 9, 10
  References (executor has NO interview context - be exhaustive): `docs/qwen-diffusion-coding-model-ultraresearch.md:143`, `docs/qwen-diffusion-coding-model-ultraresearch.md:161`, `docs/qwen-diffusion-coding-model-ultraresearch.md:322`, `src/qiffusion/diffusion_train.py:17`, `src/qiffusion/diffusion_train.py:79`, `src/qiffusion/diffusion_objective.py:14`
  Acceptance criteria (agent-executable): Failing-first test covers a tiny train run and verifies checkpoint lineage, data manifest id, objective, mask schedule, non-claiming status, no external fallback calls via monkeypatch/spies, and existing byte train behavior; run `python -m pytest tests/test_qwen_diffusion_train.py tests/test_diffusion_train.py` and `python -m pytest`.
  QA scenarios (name the exact tool + invocation): Happy command: `python -m qiffusion.cli qwen-diffusion-train --manifest .omo/evidence/qwen-diffusion-training-loop-plan-set/task-1-manifest.json --tokenizer byte --steps 2 --seed 11 --checkpoint-out .omo/evidence/qwen-diffusion-training-loop-plan-set/task-7-tiny.pt --report-out .omo/evidence/qwen-diffusion-training-loop-plan-set/task-7-train.json` and assert report includes lineage, objective, mask schedule, and `coding_capability_claim: false`. Failure command: `python -m qiffusion.cli qwen-diffusion-train --manifest .omo/evidence/qwen-diffusion-training-loop-plan-set/task-7-contaminated-manifest.json --tokenizer byte --steps 1 --report-out .omo/evidence/qwen-diffusion-training-loop-plan-set/task-7-train-failure.json` and assert nonzero exit or structured blocked report.
  Commit: Y | `feat(train): add qwen diffusion smoke training`

- [x] 8. Add expanded eval buckets and conservative capability reports.
  What to do / Must NOT do: Add eval report types for local code smoke, external benchmark readiness, chat, tool/agent, and software-engineering buckets. The report must distinguish `not_run`, `blocked`, `fail`, and `pass`. `not_run` or `blocked` is valid infrastructure evidence but never final capability success. Do not set a broad capability claim from local smoke alone.
  Parallelization: Wave 2 | Blocked by: 4, 6 | Blocks: 10, 12
  References (executor has NO interview context - be exhaustive): `docs/qwen-diffusion-coding-model-ultraresearch.md:191`, `docs/qwen-diffusion-coding-model-ultraresearch.md:206`, `docs/qwen-diffusion-coding-model-ultraresearch.md:242`, `src/qiffusion/diffusion_eval.py:12`, `src/qiffusion/diffusion_eval.py:36`, `src/qiffusion/diffusion_eval.py:54`, `tests/test_decision.py`
  Acceptance criteria (agent-executable): Tests assert claim separation for code/chat/tool/software-engineering buckets, no-fallback via monkeypatch/spies, and that a local smoke pass or blocked chat/tool bucket does not imply release capability; run `python -m pytest tests/test_qwen_diffusion_eval.py tests/test_diffusion_eval.py tests/test_decision.py` and `python -m pytest`.
  QA scenarios (name the exact tool + invocation): Happy command: `python -m qiffusion.cli qwen-diffusion-eval --checkpoint .omo/evidence/qwen-diffusion-training-loop-plan-set/task-7-tiny.pt --sample-out .omo/evidence/qwen-diffusion-training-loop-plan-set/task-8-sample.json --out .omo/evidence/qwen-diffusion-training-loop-plan-set/task-8-eval.json --runs 1` and assert separate code/chat/tool/software-engineering bucket statuses. Failure command: `python -m qiffusion.cli qwen-diffusion-eval --validate-report .omo/evidence/qwen-diffusion-training-loop-plan-set/task-8-overclaim-input.json --out .omo/evidence/qwen-diffusion-training-loop-plan-set/task-8-overclaim-failure.json` and assert overclaim rejection.
  Commit: Y | `feat(eval): split qwen diffusion capability gates`

- [x] 9. Add data improvement loop inputs from teacher traces and execution feedback.
  What to do / Must NOT do: Add quality-filtered ingestion from teacher traces, local repair examples, and execution pass/fail records into manifest-approved training splits. Do not ingest benchmark eval splits or raw user chat logs.
  Parallelization: Wave 3 | Blocked by: 1, 2, 7 | Blocks: 10, 11
  References (executor has NO interview context - be exhaustive): `docs/qwen-diffusion-coding-model-ultraresearch.md:85`, `docs/qwen-diffusion-coding-model-ultraresearch.md:91`, `docs/qwen-diffusion-coding-model-ultraresearch.md:116`, `docs/qwen-diffusion-coding-model-ultraresearch.md:303`, `src/qiffusion/diffusion_teacher_data.py:50`, `src/qiffusion/qwen_tasks.py`
  Acceptance criteria (agent-executable): Tests prove pass/fail execution records are retained with provenance, benchmark splits are excluded, and generated manifests point to the filtered training file; run `python -m pytest tests/test_qwen_diffusion_data_loop.py` and `python -m pytest`.
  QA scenarios (name the exact tool + invocation): Happy command: `python -m qiffusion.cli qwen-diffusion-data-loop --teacher-jsonl .omo/evidence/qwen-diffusion-training-loop-plan-set/task-2-teacher.jsonl --manifest .omo/evidence/qwen-diffusion-training-loop-plan-set/task-1-manifest.json --out .omo/evidence/qwen-diffusion-training-loop-plan-set/task-9-data-loop.json` and assert verified pass/fail execution rows are retained with provenance. Failure command: `python -m qiffusion.cli qwen-diffusion-data-loop --teacher-jsonl .omo/evidence/qwen-diffusion-training-loop-plan-set/task-9-benchmark-teacher.jsonl --manifest .omo/evidence/qwen-diffusion-training-loop-plan-set/task-1-manifest.json --out .omo/evidence/qwen-diffusion-training-loop-plan-set/task-9-data-loop-failure.json` and assert benchmark-tagged rows are excluded or blocked.
  Commit: Y | `feat(data): add verified data improvement loop`

- [x] 10. Add automated train-validation-repeat loop runner.
  What to do / Must NOT do: Add a loop runner that executes train -> sample -> eval -> diagnose -> next-action planning for one or more iterations. It must write every iteration to a JSONL ledger and continue until code/chat gates pass, max iterations is reached, or a real resource blocker is recorded. It must not mark the model capable from `max_iterations` alone, and it must report blocked chat/tool/software-engineering buckets as blockers rather than success.
  Parallelization: Wave 3 | Blocked by: 4, 7, 8, 9 | Blocks: 12
  References (executor has NO interview context - be exhaustive): `docs/qwen-diffusion-coding-model-ultraresearch.md:322`, `docs/qwen-diffusion-coding-model-ultraresearch.md:330`, `src/qiffusion/cli.py:47`, `src/qiffusion/cli.py:55`, `src/qiffusion/cli.py:63`, `src/qiffusion/cli.py:92`
  Acceptance criteria (agent-executable): Tests run a two-iteration tiny loop, verify ledger entries include train/sample/eval/diagnosis/next_action/cleanup, verify the loop continues after a failed coding gate, and verify no external fallback calls via monkeypatch/spies; run `python -m pytest tests/test_qwen_diffusion_loop.py` and `python -m pytest`.
  QA scenarios (name the exact tool + invocation): Happy command: `python -m qiffusion.cli qwen-diffusion-loop --manifest .omo/evidence/qwen-diffusion-training-loop-plan-set/task-9-data-loop.json --config .omo/evidence/qwen-diffusion-training-loop-plan-set/task-4-config.json --max-iterations 2 --ledger-out .omo/evidence/qwen-diffusion-training-loop-plan-set/task-10-loop.jsonl` and assert each JSONL line has train/sample/eval/diagnosis/next_action/cleanup. Failure command: `python -m qiffusion.cli qwen-diffusion-loop --manifest .omo/evidence/qwen-diffusion-training-loop-plan-set/task-9-data-loop.json --config .omo/evidence/qwen-diffusion-training-loop-plan-set/task-4-config.json --force-eval-fail --max-iterations 1 --ledger-out .omo/evidence/qwen-diffusion-training-loop-plan-set/task-10-loop-failure.jsonl` and assert `next_action` is not `complete`.
  Commit: Y | `feat(loop): add qwen diffusion train validation loop`

- [x] 11. Add chat and tool-agent schema/eval stubs that feed the same loop.
  What to do / Must NOT do: Add minimal but real schemas for multi-turn chat and tool/agent examples, plus eval stubs that report `not_run` or `blocked` until real datasets/harnesses exist. These states are blockers for final capability, not success. Do not claim chat or agent capability from coding tasks.
  Parallelization: Wave 3 | Blocked by: 2, 9 | Blocks: 12
  References (executor has NO interview context - be exhaustive): `docs/qwen-diffusion-coding-model-ultraresearch.md:106`, `docs/qwen-diffusion-coding-model-ultraresearch.md:111`, `docs/qwen-diffusion-coding-model-ultraresearch.md:206`, `docs/qwen-diffusion-coding-model-ultraresearch.md:276`, `docs/qwen-diffusion-coding-model-ultraresearch.md:284`
  Acceptance criteria (agent-executable): Tests validate chat/tool records, ensure missing harnesses yield `blocked` or `not_run`, and ensure chat/tool pass flags do not affect coding pass flags; run `python -m pytest tests/test_qwen_diffusion_chat_agent.py` and `python -m pytest`.
  QA scenarios (name the exact tool + invocation): Happy command: `python -m qiffusion.cli qwen-diffusion-chat-agent-validate --chat-json .omo/evidence/qwen-diffusion-training-loop-plan-set/task-11-chat.json --tool-json .omo/evidence/qwen-diffusion-training-loop-plan-set/task-11-tool.json --out .omo/evidence/qwen-diffusion-training-loop-plan-set/task-11-chat-agent.json` and assert chat/tool schemas validate while capability status remains `not_run` or `blocked`. Failure command: `python -m qiffusion.cli qwen-diffusion-chat-agent-validate --tool-json .omo/evidence/qwen-diffusion-training-loop-plan-set/task-11-injected-tool.json --out .omo/evidence/qwen-diffusion-training-loop-plan-set/task-11-chat-agent-failure.json` and assert malformed or prompt-injected tool content is rejected.
  Commit: Y | `feat(eval): add chat and agent schema gates`

- [x] 12. Add final conservative release/capability gate and iteration policy.
  What to do / Must NOT do: Add or update the shared gate so final status only becomes coding/chat capable when each required bucket supplies evidence, no benchmark contamination is present, no hidden fallback was used, and checkpoint lineage is complete. The policy must keep returning `continue` while the loop has plausible next actions.
  Parallelization: Wave 3 | Blocked by: 1, 4, 8, 10, 11 | Blocks: F1-F4
  References (executor has NO interview context - be exhaustive): `docs/qwen-diffusion-coding-model-ultraresearch.md:242`, `docs/qwen-diffusion-coding-model-ultraresearch.md:254`, `docs/qwen-diffusion-coding-model-ultraresearch.md:268`, `docs/qwen-diffusion-coding-model-ultraresearch.md:276`, `docs/qwen-diffusion-coding-model-ultraresearch.md:284`, `docs/qwen-diffusion-coding-model-ultraresearch.md:330`, `src/qiffusion/decision.py`, `tests/test_decision.py`
  Acceptance criteria (agent-executable): Tests prove incomplete or blocked buckets keep status `continue` or `blocked`, benchmark contamination blocks capability, fallback usage blocks capability, and all required bucket evidence promotes only the appropriate claims; run `python -m pytest tests/test_decision.py tests/test_qwen_diffusion_loop.py tests/test_qwen_diffusion_eval.py` and `python -m pytest`.
  QA scenarios (name the exact tool + invocation): Happy command: `python -m qiffusion.cli status --report .omo/evidence/qwen-diffusion-training-loop-plan-set/task-12-passing-report.json > .omo/evidence/qwen-diffusion-training-loop-plan-set/task-12-status.txt` and assert it gates only fully evidenced buckets. Failure command: `python -m qiffusion.cli status --report .omo/evidence/qwen-diffusion-training-loop-plan-set/task-12-fallback-report.json > .omo/evidence/qwen-diffusion-training-loop-plan-set/task-12-fallback-status.txt`, where `task-12-fallback-report.json` contains `fallback_used: true` or missing benchmark contamination metadata; assert `task-12-fallback-status.txt` contains `continue` or `blocked`, never `complete`.
  Commit: Y | `feat(gate): add conservative qwen diffusion capability policy`

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [ ] F1. Plan compliance audit
  Verify every todo acceptance criterion has matching evidence under `.omo/evidence/qwen-diffusion-training-loop-plan-set/`, every checked todo has a ledger entry, every requested commit exists, and no unresolved placeholder remains in this plan. Happy command: `python -m qiffusion.cli qwen-diffusion-plan-audit --plan .omo/plans/qwen-diffusion-training-loop-plan-set.md --ledger .omo/start-work/ledger.jsonl --evidence-root .omo/evidence/qwen-diffusion-training-loop-plan-set --out .omo/evidence/qwen-diffusion-training-loop-plan-set/f1-plan-compliance.json`. Failure command: rerun with `--require-all-checked` before all top-level boxes are checked and assert nonzero exit or `status: fail`.
- [ ] F2. Code quality review
  Review final diff for overbroad abstractions, hidden fallback paths, benchmark contamination risks, missing provenance, weak tests, and stale ignored artifacts. Happy command: `python -m pytest && git diff --check && git status --short > .omo/evidence/qwen-diffusion-training-loop-plan-set/f2-git-status.txt`; reviewer writes findings to `.omo/evidence/qwen-diffusion-training-loop-plan-set/f2-code-review.md`. Failure command: `python -m qiffusion.cli qwen-diffusion-plan-audit --scan-fallback --evidence-root .omo/evidence/qwen-diffusion-training-loop-plan-set --out .omo/evidence/qwen-diffusion-training-loop-plan-set/f2-fallback-scan.json` and assert any fallback usage fails the review.
- [ ] F3. Real manual QA
  Run the actual CLI/data-artifact surface: manifest, tokenizer/config driver, tiny train, sample, eval, loop runner, and status gate. Happy command: `python -m qiffusion.cli qwen-diffusion-loop --manifest .omo/evidence/qwen-diffusion-training-loop-plan-set/task-9-data-loop.json --config .omo/evidence/qwen-diffusion-training-loop-plan-set/task-4-config.json --max-iterations 2 --ledger-out .omo/evidence/qwen-diffusion-training-loop-plan-set/f3-loop.jsonl && python -m qiffusion.cli status --report .omo/evidence/qwen-diffusion-training-loop-plan-set/task-12-passing-report.json > .omo/evidence/qwen-diffusion-training-loop-plan-set/f3-status.txt`; reviewer summarizes in `.omo/evidence/qwen-diffusion-training-loop-plan-set/f3-manual-qa.json`. Failure command: rerun the loop with `--force-eval-fail` and assert status does not become `complete`.
- [ ] F4. Scope fidelity
  Confirm the delivered work preserves the Qwen-based diffusion architecture direction, does not claim unsupported capability, and leaves benchmark corpora held out. Happy command: `python -m qiffusion.cli qwen-diffusion-plan-audit --scope qwen-diffusion --plan .omo/plans/qwen-diffusion-training-loop-plan-set.md --evidence-root .omo/evidence/qwen-diffusion-training-loop-plan-set --out .omo/evidence/qwen-diffusion-training-loop-plan-set/f4-scope-fidelity.json`; reviewer writes final scope notes to `.omo/evidence/qwen-diffusion-training-loop-plan-set/f4-scope-fidelity.md`. Failure command: audit an evidence fixture with `fallback_used: true` or `usage: train_allowed` on a benchmark source and assert it fails.

## Commit strategy
- Commit after each verified todo or tightly coupled wave, never before independent verification and ledger evidence.
- Use atomic commit messages from each todo's `Commit` line.
- Push each completed commit to `origin/main` because the user requested automatic commit and push for this repo.
- Do not commit ignored training artifacts, `.pytest_cache`, `__pycache__`, large checkpoints, or local model caches.
- If a worker finds unrelated worktree changes, record the dirty-worktree risk and avoid modifying those files unless they are in the todo scope.

## Success criteria
- The plan has no placeholders, all todos have executable acceptance criteria, and Boulder state points to this plan.
- The first implementation wave can start without further interview.
- Every product change has failing-first or characterization coverage, automated verification, manual CLI/data artifact QA, adversarial QA, cleanup receipt, and ledger entry.
- The train-validation loop keeps iterating until code/chat/tool/software-engineering gates pass or a concrete blocker such as missing compute, unavailable model checkpoint, or policy-prohibited dataset is recorded.
- `coding_capability_claim=true` is only emitted when the Qwen-based diffusion model itself passes the required gates without hidden fallback and with benchmark contamination checks recorded.
