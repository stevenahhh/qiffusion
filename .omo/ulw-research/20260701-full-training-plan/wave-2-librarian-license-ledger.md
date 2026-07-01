# Wave 2: License and Source-Ledger Findings

## Conservative Recommendations

### Allow

- Qwen/Qwen3.5-4B: Apache-2.0.
- CommitPack/CommitPackFT: allow only permissive-license rows with provenance retained.
- OpenCoder stage data: MIT where the dataset card states so.
- Magicoder OSS-Instruct/Evol-Instruct: MIT/Apache-2.0 where cards state so.
- CodeFeedback/OpenCodeInterpreter: Apache-2.0 cards/repos, with generated-output provenance retained.
- OASST1/OASST2: Apache-2.0.
- HelpSteer: CC-BY-4.0.

### Review Before Ingestion

- The Stack v2: gated; must follow Software Heritage/INRIA terms, original licenses, attribution, opt-out updates, and provenance policy.
- WildChat: ODC-BY plus privacy/PII caveats.
- UltraChat: MIT-labeled but ChatGPT-generated; review model-output/upstream terms.
- ToolBench: Apache-2.0 plus research/education-only language.
- UltraFeedback: MIT-labeled but composite upstream prompts/model generations.

### Block Unless Policy Changes

- LMSYS-Chat-1M: gated and license agreement required; block until legal/access terms are cleared.
- APIGen-MT-5k: CC-BY-NC-4.0; block for commercial use.
- Nectar: research preview / non-commercial / LLaMA and OpenAI generated-data terms; block.

## Minimum Source Ledger Fields

- source URL
- hosting org/repo
- declared license
- access condition/gate
- commercial restriction flag
- redistribution/attribution condition
- privacy/PII note
- model-output/synthetic-output provenance note
- contamination/opt-out/removal note
- per-sample provenance availability
- last verified date
- qiffusion action: allow/review/block
