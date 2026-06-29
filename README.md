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
```

## Initial tracks

- `diffusion`: native diffusion LLM training and sampling loop.
- `qwen_bridge`: Qwen/Qwen3.5-4B local baseline, teacher output generation, and code smoke comparison.

