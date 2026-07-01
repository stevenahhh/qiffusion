# Wave 2: Sonnet 4.6 Target Matrix

## Verified Target Gates

- SWE-bench Verified: 79.6% average over 10 trials; Anthropic also reports a prompt-modified 80.2% run.
- Terminal-Bench 2.0: 59.1% in Anthropic's system-card setup, max effort/no thinking.
- OSWorld-Verified: 72.5% first-attempt success averaged over five runs.
- Long context: 1M-token context window; Anthropic reports MRCR v2 and GraphWalks 256K/1M subset scores.
- Max output: 128K synchronous Messages API max output, and up to 300K output on Message Batches with the beta header.
- Agentic/coding qualitative target: coding, computer use, long-context reasoning, agent planning, knowledge work, design, and realistic coding-agent behavior.
- IFEval: no machine-readable primary-source Sonnet 4.6 score found.

## Sources

- https://www.anthropic.com/news/claude-sonnet-4-6
- https://www.anthropic.com/claude-sonnet-4-6-system-card
- https://platform.claude.com/docs/en/about-claude/models/overview
- https://platform.claude.com/docs/en/build-with-claude/context-windows
- https://www.swebench.com/
- https://www.tbench.ai/leaderboard/terminal-bench/2.0
- https://os-world.github.io/

## Planning Implications

- These numbers become final claim gates, not near-term local training targets.
- Speed claims must include quality gates; raw generated tokens per second alone is not enough.
