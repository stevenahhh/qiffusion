# Wave 1: Chat, Agent, Preference, Safety Data

## Key Findings

- OASST1/OASST2 are strong human assistant/chat sources.
- UltraChat and OpenHermes-style mixtures are useful SFT layers but require contamination and upstream-policy review.
- LMSYS-Chat-1M and WildChat provide realism but need privacy/access review.
- ToolBench, APIGen, APIGen-MT, and Mind2Web are the most relevant tool-use/agent sources.
- tau-bench, BFCL, WebArena, VisualWebArena, TheAgentCompany, and Mind2Web test splits should be eval-only.
- UltraFeedback, HelpSteer, Nectar, Chatbot Arena Conversations, and OASST ranking signals are preference/reward candidates, subject to license review.

## Recommended Training Role

- Base chat SFT: OASST1/OASST2, UltraChat 200k, filtered OpenHermes, carefully filtered ShareGPT-style data.
- Tool/agent SFT: ToolBench, APIGen/APIGen-MT where license permits, Mind2Web train.
- Preference: UltraFeedback/binarized, HelpSteer/HelpSteer3, approved ranking data.
- Safety: ToxicChat and safety-specific datasets as a separate alignment stage.

## Sources

- OASST1: https://huggingface.co/datasets/OpenAssistant/oasst1
- UltraChat: https://github.com/thunlp/UltraChat, https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
- OpenHermes: https://huggingface.co/datasets/teknium/OpenHermes-2.5
- LMSYS-Chat-1M: https://huggingface.co/datasets/lmsys/lmsys-chat-1m
- WildChat: https://allenai.org/open-data#wildchat
- ToolBench: https://github.com/openbmb/toolbench
- APIGen/APIGen-MT: https://apigen-pipeline.github.io/, https://apigen-mt.github.io/
- Mind2Web: https://osu-nlp-group.github.io/Mind2Web/
- UltraFeedback: https://huggingface.co/datasets/openbmb/UltraFeedback
- HelpSteer: https://huggingface.co/datasets/nvidia/HelpSteer
