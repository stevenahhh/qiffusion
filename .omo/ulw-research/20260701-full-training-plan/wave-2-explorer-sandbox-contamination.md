# Wave 2: Sandbox and Contamination Controls

## Key Findings

- Generated-code smoke execution is currently in-process. It uses AST checks plus restricted builtins and then `exec`s generated code.
- Timeouts exist around model invocation, not generated-code execution.
- Network/file isolation is not enforced for generated code.
- Manifest contamination controls exist and block benchmark-source rows, but near-duplicate control is SHA-256 exact hash only.
- Teacher-data ingestion checks provenance and blocks benchmark-ish text by substring, but does not perform semantic similarity or MinHash scans.

## Direct Repo References

- `src/qiffusion/qwen_tasks.py`
- `src/qiffusion/qwen_repair_tasks.py`
- `src/qiffusion/qwen_file_tasks.py`
- `src/qiffusion/qwen_ollama.py`
- `src/qiffusion/diffusion_corpus_manifest.py`
- `src/qiffusion/diffusion_teacher_data.py`
- `src/qiffusion/qwen_diffusion_train.py`
- `src/qiffusion/qwen_diffusion_data_loop.py`
- `src/qiffusion/qwen_diffusion_plan_audit.py`
- `src/qiffusion/decision.py`

## Required Before Larger Code Benchmarks

- Add process-level sandbox for generated code with timeout, temp workspace, disabled network, restricted file access, and captured stdout/stderr.
- Add exact and near-duplicate benchmark overlap scanning: SHA-256, normalized text, n-gram/MinHash, and optionally embedding similarity.
- Add benchmark runner wrappers that record split, version, source URL, runtime, timeout, and contamination proof.
- Keep local AST smoke as a cheap preflight, not the final claim gate.
