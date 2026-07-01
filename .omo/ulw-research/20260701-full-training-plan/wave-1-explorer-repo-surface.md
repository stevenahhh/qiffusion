# Wave 1: qiffusion Repo Surface

## Key Findings

- The repo has CLI surfaces for `qwen-diffusion-train`, `qwen-diffusion-eval`, `qwen-diffusion-loop`, `qwen-diffusion-data-loop`, and plan/gate audits.
- Current Qwen diffusion training is still a tiny smoke scaffold; it does not train a full Qwen 4B model.
- Current guardrails include local-only tokenizer loading, no-download bridge checks, train/eval split flags, benchmark source blocking, and conservative status gates.
- Chat/tool/software-engineering buckets are present as schema/blocked surfaces, but robust evaluation is not implemented.
- Full training requires CUDA/cloud runtime, Qwen checkpoint adapter/conversion, real tokenizer-backed data pipeline, benchmark harnesses, checkpoint registry, and release gates.

## Direct Repo References

- `src/qiffusion/diffusion_model.py`
- `src/qiffusion/diffusion_train.py`
- `src/qiffusion/diffusion_eval.py`
- `src/qiffusion/qwen_bridge.py`
- `src/qiffusion/qwen_diffusion_config.py`
- `src/qiffusion/qwen_diffusion_train.py`
- `src/qiffusion/qwen_diffusion_eval.py`
- `src/qiffusion/qwen_diffusion_loop.py`
- `src/qiffusion/qwen_diffusion_data_loop.py`
- `src/qiffusion/qwen_diffusion_plan_audit.py`
- `tests/test_qwen_diffusion_train.py`
- `tests/test_qwen_diffusion_eval.py`
- `tests/test_qwen_diffusion_loop.py`
- `tests/test_qwen_diffusion_release_gate.py`

## Planning Implications

- The next work plan should build missing full-training infrastructure instead of claiming the smoke scaffold is enough.
- Existing smoke CLI/tests remain regression gates.
- Capability claims must remain false until external benchmark and executable smoke gates pass.
