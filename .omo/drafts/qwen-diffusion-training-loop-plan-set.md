---
slug: qwen-diffusion-training-loop-plan-set
status: approved-by-start-work-bootstrap
intent: unclear-bootstrap-resolved
intent-note: "UNCLEAR was the bootstrap routing path; the plan is now written and first wave needs no further user decision."
pending-action: execute .omo/plans/qwen-diffusion-training-loop-plan-set.md via Boulder start-work state
approach: Build Qwen-token diffusion infrastructure first, then execute repeated train/sample/eval/diagnose loops until conservative code/chat/tool gates pass or a concrete resource blocker is recorded.
---

# Draft: qwen-diffusion-training-loop-plan-set

## Components (topology ledger)
<!-- Lock the SHAPE before depth. One row per top-level component that can succeed or fail independently. -->
<!-- id | outcome (one line) | status: active|deferred | evidence path -->
- C1 | Corpus manifest and provenance controls for local, teacher, and later external data | active | docs/qwen-diffusion-coding-model-ultraresearch.md:81-141
- C2 | Teacher trace schema for code, repair, chat, tool, and execution outcomes | active | docs/qwen-diffusion-coding-model-ultraresearch.md:91-118
- C3 | Qwen tokenizer, config, and checkpoint-lineage boundary | active | docs/qwen-diffusion-coding-model-ultraresearch.md:145-189
- C4 | Qwen-token denoiser and iterative mask sampler without fallback | active | docs/qwen-diffusion-coding-model-ultraresearch.md:59-79
- C5 | Code/chat/tool/software-engineering eval buckets and release gate | active | docs/qwen-diffusion-coding-model-ultraresearch.md:242-290
- C6 | Repeated train-validation-improvement loop and evidence ledger | active | docs/qwen-diffusion-coding-model-ultraresearch.md:322-328
- C7 | No-download Qwen checkpoint compatibility and resource contract | active | docs/qwen-diffusion-coding-model-ultraresearch.md:10-11,48-57,161-175

## Open assumptions (announced defaults)
<!-- Intent is UNCLEAR: research resolves ambiguity, defaults are adopted (not asked), and each is surfaced in the plan's human TL;DR for veto. -->
<!-- assumption | adopted default | rationale | reversible? -->
- Base architecture | Qwen-family AR-to-diffusion masked/discrete diffusion | Best-supported path in research; preserves user's diffusion requirement | reversible before model conversion
- Immediate implementation depth | Build tiny CPU-capable Qwen-token scaffold before full 4B conversion | Lets tests and CI verify behavior without requiring a large checkpoint | reversible
- Data order | local verified data -> Qwen teacher traces -> public SFT -> large corpora after provenance | Minimizes legal/privacy risk and preserves benchmark cleanliness | reversible
- Large corpus gate | The Stack v2 and similar corpora require manifest/license/provenance controls first | Research flags provenance and opt-out handling | reversible
- Benchmark policy | HumanEval/MBPP/EvalPlus/LiveCodeBench/BigCodeBench/SWE-bench are held out | Release claims need uncontaminated evaluation | not reversible for reported claims
- Fallback policy | No Qwen/Ollama/autoregressive fallback during diffusion inference | Capability must come from the diffusion checkpoint itself | not reversible for final claims
- LoRA policy | LoRA only after base diffusion behavior is verified | Research says full fine-tune is safer for AR-to-diffusion conversion | reversible
- Qwen compatibility depth | First wave records checkpoint/tokenizer/config/tensor contract and resource probe without downloading weights | Prevents tiny scaffold drift while keeping first loop executable | reversible
- Dependency policy | Use dynamic import and optional extras only; no mandatory `transformers` dependency for core tests | Keeps current byte scaffold and CI lightweight | reversible
- Blocked bucket policy | Chat/tool/software-engineering `blocked` or `not_run` may complete infra tasks but cannot satisfy final capability | Avoids overclaiming | not reversible for final claims

## Findings (cited - path:lines)
- Current diffusion path is a tiny byte-level GRU scaffold, not a capable model: docs/qwen-diffusion-coding-model-ultraresearch.md:8.
- Best-supported path is Qwen-family AR-to-diffusion adaptation with masked/discrete diffusion: docs/qwen-diffusion-coding-model-ultraresearch.md:10.
- Data requirements include local verified examples, teacher SFT, code, chat, tool/agent trajectories, and execution feedback: docs/qwen-diffusion-coding-model-ultraresearch.md:12.
- Immediate repo gaps include no Qwen tokenizer, no Qwen denoiser/conversion path, no manifest/provenance, no multi-turn/tool schema, and no broad harness: docs/qwen-diffusion-coding-model-ultraresearch.md:31-38.
- Current corpus builder is ad hoc and byte-tokenized: src/qiffusion/diffusion_data.py:25-75.
- Current model is TinyDiffusionLM over embeddings, GRU, and linear output: src/qiffusion/diffusion_model.py:42-61.
- Current sampler fills one byte mask greedily per step and reports no coding claim: src/qiffusion/diffusion_sample.py:37-67.
- Current eval uses local smoke only and sets coding claim directly from local smoke status: src/qiffusion/diffusion_eval.py:36-68.
- CLI already exposes train/sample/eval/export surfaces that new commands should match: src/qiffusion/cli.py:26-178.
- Metis plan review found execution gaps before product dispatch: missing Boulder/ledger bootstrap, underspecified no-fallback checks, dependency policy, and missing Qwen compatibility contract.

## Decisions (with rationale)
- The plan starts with infrastructure, not another byte-GRU training attempt, because the research explicitly says the current tiny model cannot become the target model by simply adding examples.
- The first wave is split into manifest, teacher schema, tokenizer boundary, and config/gate schema so workers can implement them independently with minimal conflict.
- Large checkpoint download and full 4B training are deferred behind resource and provenance evidence so the first loop remains executable on CPU.
- The release gate is split by code, software-engineering, chat, and tool/agent claims so one passing local code smoke cannot inflate all capability claims.
- The loop runner must keep returning `continue` while a plausible next action exists; `max_iterations` alone is not success.
- The orchestrator creates `.omo/boulder.json`, `.omo/start-work/ledger.jsonl`, and the evidence root before product work; this is start-work state, not a worker product todo.
- Todo 4 now owns the no-download Qwen compatibility contract so todo 5's tiny model scaffold cannot drift into a second toy model.
- No-fallback checks must be implemented with monkeypatch/spies over Qwen/Ollama bridge and generation entry points rather than grep-only evidence.
- Manifest usage and contamination values are fixed to `train_allowed|eval_only|unknown_blocked` and `clean|suspect|blocked`.

## Scope IN
- `.omo/plans/qwen-diffusion-training-loop-plan-set.md`.
- `.omo/boulder.json` and `.omo/start-work/ledger.jsonl`.
- Product implementation requested by plan todos under `src/qiffusion`, `tests`, and docs when executed by workers.
- CLI/data-artifact QA evidence under `.omo/evidence/qwen-diffusion-training-loop-plan-set/`.
- Qwen compatibility/resource probe artifacts that do not require network or large checkpoint download.

## Scope OUT (Must NOT have)
- No hidden Qwen/Ollama/autoregressive fallback for diffusion generation.
- No final coding-capable claim before the gated evidence exists.
- No benchmark training contamination.
- No raw user chat logs, unknown-license scraped code, or broad corpus ingestion before manifest/policy gates.
- No full 4B training in the tiny smoke path.
- No mandatory `transformers` dependency for the core package/tests.
- No treating blocked chat/tool/software-engineering buckets as final capability success.

## Open questions
- None for the first execution wave. Resource-dependent full checkpoint training can become a blocker later if local compute/model cache is unavailable.

## Approval gate
status: approved-by-start-work-bootstrap
pending-action: execute .omo/plans/qwen-diffusion-training-loop-plan-set.md via Boulder start-work state
approval-source: user invoked $omo:start-work with "ㄱㄱ" after the approval brief
metis-review: first review verdict ITERATE; required fixes folded into plan/draft before execution
<!-- When exploration is exhausted and unknowns are answered, set status: awaiting-approval. -->
<!-- That durable record is the loop guard: on a later turn read it and resume at the gate instead of re-running exploration. -->
