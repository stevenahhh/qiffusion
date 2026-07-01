# Wave 1: Coding Data

## Key Findings

- The Stack v2 is the strongest public code pretraining corpus candidate, subject to access/provenance review.
- CommitPack and CommitPackFT are useful for repository evolution and patch-style behavior.
- Magicoder/OSS-Instruct is a compact code SFT layer with documented decontamination.
- OpenCoder is the best public transparent recipe reference.
- Code-Feedback/OpenCodeInterpreter is directly relevant for execution-feedback and repair loops, with generated-output provenance review needed.
- CodeUltraFeedback is useful for code preference tuning after a competent code model exists.
- HumanEval, MBPP, SWE-bench Verified, BigCodeBench, and LiveCodeBench should be eval-only.

## Recommended Training Role

- Pretraining: The Stack v2, OpenCoder/RefineCode-style corpus, CommitPack.
- SFT: Magicoder/OSS-Instruct, OpenCoder SFT, CommitPackFT, Code-Feedback after policy review.
- Repair/execution: Code-Feedback and locally generated sandboxed repair traces.
- Preference/alignment: CodeUltraFeedback and verified code-review/rubric pairs.
- Evaluation only: HumanEval, MBPP, EvalPlus, APPS test, CodeContests validation/test, BigCodeBench, LiveCodeBench, SWE-bench Verified.

## Sources

- The Stack v2: https://huggingface.co/datasets/bigcode/the-stack-v2
- CommitPack: https://huggingface.co/datasets/bigcode/commitpack
- CommitPackFT: https://huggingface.co/datasets/bigcode/commitpackft
- Magicoder: https://arxiv.org/abs/2312.02120, https://github.com/ise-uiuc/magicoder
- OpenCoder: https://arxiv.org/abs/2411.04905, https://github.com/OpenCoder-llm/OpenCoder-llm
- Code-Feedback: https://huggingface.co/datasets/m-a-p/Code-Feedback, https://opencodeinterpreter.github.io/
- SWE-bench Verified: https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified
- BigCodeBench: https://github.com/bigcode-project/bigcodebench
