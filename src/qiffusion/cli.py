from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from qiffusion.config import CODING_CAPABLE_REQUIREMENTS, TRACKS
from qiffusion.decision import decide_from_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="qiffusion project control surface.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("plan", help="Print the two-track project plan.")

    status = subcommands.add_parser("status", help="Evaluate a loop summary JSON file.")
    status.add_argument("--report", type=Path, required=True)
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        print_plan()
        return 0
    if args.command == "status":
        decision = decide_from_file(args.report)
        print(json.dumps({"status": decision.status, "reason": decision.reason}, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

