from __future__ import annotations

import argparse
import json
from pathlib import Path

from qiffusion.qwen_diffusion_chat_agent import ChatAgentSchemaError, write_chat_agent_report, write_rejection


def add_qwen_diffusion_chat_agent_parser(subcommands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    chat_agent = subcommands.add_parser(
        "qwen-diffusion-chat-agent-validate",
        help="Validate chat and tool-agent schema fixtures without claiming capability.",
    )
    chat_agent.add_argument("--chat-json", type=Path)
    chat_agent.add_argument("--tool-json", type=Path)
    chat_agent.add_argument("--out", type=Path, required=True)


def run_qwen_diffusion_chat_agent_validate(chat_path: Path | None, tool_path: Path | None, output_path: Path) -> int:
    try:
        report = write_chat_agent_report(chat_path, tool_path, output_path)
    except ChatAgentSchemaError as error:
        payload = write_rejection(error, output_path)
        print(json.dumps(payload, sort_keys=True))
        return 2
    print(json.dumps({"status": report["status"], "out": str(output_path)}, sort_keys=True))
    return 0
