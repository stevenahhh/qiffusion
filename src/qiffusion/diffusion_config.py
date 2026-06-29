from __future__ import annotations

from typing import Final, Literal

DIFFUSION_BACKEND: Final = "diffusion"
DIFFUSION_REPORT_SCHEMA_VERSION: Final = 1
BYTE_TOKEN_OFFSET: Final = 3
PAD_TOKEN_ID: Final = 0
MASK_TOKEN_ID: Final = 1
EOS_TOKEN_ID: Final = 2
BYTE_VOCAB_SIZE: Final = BYTE_TOKEN_OFFSET + 256

DiffusionStage = Literal["train", "sample", "eval"]

