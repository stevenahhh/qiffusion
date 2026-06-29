# qiffusion

qiffusion is a two-track local model project:

1. Build a diffusion LLM path that can eventually stand on its own.
2. Keep a Qwen 4B bridge path for teacher data, code generation baselines, and verification.

The two tracks share one evaluation gate. A model is not called coding-capable until it passes executable code smoke tests and records reproducible evidence.

## Why both tracks

Going both directions is reasonable if the boundaries stay explicit:

- Diffusion track: the product target.
- Qwen bridge track: the bootstrap, teacher, and reference baseline.
- Shared gates: the only place where capability claims are made.

This avoids pretending the diffusion prototype is better than it is, while still using a stronger model to create data and compare progress.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .[dev]
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m qiffusion.cli plan
.\.venv\Scripts\python.exe -m qiffusion.cli qwen-status --out evidence/qwen-status.json
.\.venv\Scripts\python.exe -m qiffusion.cli qwen-eval --out evidence/qwen-eval.json
.\.venv\Scripts\python.exe -m qiffusion.cli backend-status --backend diffusion --out evidence/diffusion-status.json
```

## Initial tracks

- `diffusion`: native diffusion LLM training and sampling loop.
- `qwen_bridge`: Qwen/Qwen3.5-4B local baseline, teacher output generation, and code smoke comparison.

## Publishing Workflow

This GitHub repository is public. Future completed Codex work in this repo should be verified, committed, and pushed automatically to the configured remote. Repository visibility changes remain an owner-level operation and are not part of feature workers.

## Current CLI Surfaces

`qwen-status` checks for a runnable local Qwen/Qwen3.5-4B bridge engine and writes a JSON report. A Hugging Face snapshot must have non-empty metadata and weights, and the local `transformers` runtime must be usable. It may return `prerequisite_missing` without failing; discovery is not a coding-capability claim. GGUF fallback scans the current directory by default; set `QIFFUSION_GGUF_ROOTS` to opt into additional roots.

`qwen-eval` runs independent local Ollama `qwen3.5:4b` coding tasks, including arithmetic, list, string, interval-merging, and small file-editing checks. Each task asks for Python code, prefers Ollama's `stream: false` HTTP API for clean JSON, parses the response, executes the generated code under narrow smoke tests, and only sets `coding_capability_claim` when every task passes.
Use `--runs N` to require the full independent-task suite to pass repeatedly in one report.

`backend-status --backend diffusion` writes the current diffusion scaffold report. The scaffold is selectable through the shared gate but is not a training, sampling, chat, or coding-capable implementation yet.
