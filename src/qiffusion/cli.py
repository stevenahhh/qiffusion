from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, TypeAlias

from qiffusion.backends import diffusion_status
from qiffusion.config import CODING_CAPABLE_REQUIREMENTS, TRACKS
from qiffusion.decision import decide_from_file
from qiffusion.qwen_bridge import DEFAULT_OLLAMA_MODEL, qwen_status
from qiffusion.qwen_eval import qwen_eval

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="qiffusion project control surface.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("plan", help="Print the two-track project plan.")

    status = subcommands.add_parser("status", help="Evaluate a loop summary JSON file.")
    status.add_argument("--report", type=Path, required=True)

    qwen = subcommands.add_parser("qwen-status", help="Write Qwen bridge availability status.")
    qwen.add_argument("--out", type=Path, required=True)

    qwen_eval_parser = subcommands.add_parser("qwen-eval", help="Run the Qwen bridge coding fixture.")
    qwen_eval_parser.add_argument("--out", type=Path, required=True)
    qwen_eval_parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)

    backend = subcommands.add_parser("backend-status", help="Write backend scaffold status.")
    backend.add_argument("--backend", choices=("diffusion",), required=True)
    backend.add_argument("--out", type=Path, required=True)
    return parser


def print_plan() -> None:
    payload = {
        "tracks": [
            {"name": track.name, "role": track.role, "done_when": list(track.done_when)}
            for track in TRACKS
        ],
        "coding_capable_requirements": list(CODING_CAPABLE_REQUIREMENTS),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def write_json(path: Path, payload: JsonValue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        print_plan()
        return 0
    if args.command == "status":
        decision = decide_from_file(args.report)
        print(json.dumps({"status": decision.status, "reason": decision.reason}, sort_keys=True))
        return 0
    if args.command == "qwen-status":
        report = qwen_status()
        write_json(args.out, report)
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.command == "qwen-eval":
        report = qwen_eval(args.model)
        write_json(args.out, report)
        print(json.dumps(report, sort_keys=True))
        return 0 if report["status"] == "available" else 2
    if args.command == "backend-status":
        report = diffusion_status()
        write_json(args.out, report)
        print(json.dumps(report, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
