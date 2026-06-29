from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, TypeAlias

from qiffusion.backends import diffusion_status
from qiffusion.config import CODING_CAPABLE_REQUIREMENTS, TRACKS
from qiffusion.decision import decide_from_file
from qiffusion.diffusion_reports import diffusion_stub_report
from qiffusion.diffusion_teacher_data import export_teacher_jsonl
from qiffusion.qwen_bridge import DEFAULT_OLLAMA_MODEL, qwen_status
from qiffusion.qwen_eval import qwen_eval

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


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
    qwen_eval_parser.add_argument("--runs", type=positive_int, default=1)

    backend = subcommands.add_parser("backend-status", help="Write backend scaffold status.")
    backend.add_argument("--backend", choices=("diffusion",), required=True)
    backend.add_argument("--out", type=Path, required=True)

    diffusion_train = subcommands.add_parser("diffusion-train", help="Write a non-claiming diffusion train stub report.")
    diffusion_train.add_argument("--report-out", type=Path, required=True)
    diffusion_train.add_argument("--out", type=Path)
    diffusion_train.add_argument("--steps", type=positive_int, default=20)
    diffusion_train.add_argument("--seed", type=int, default=1)
    diffusion_train.add_argument("--max-examples", type=positive_int, default=24)
    diffusion_train.add_argument("--teacher-jsonl", type=Path, action="append", default=())

    diffusion_sample = subcommands.add_parser("diffusion-sample", help="Write a non-claiming diffusion sample report.")
    diffusion_sample.add_argument("--report-out", type=Path)
    diffusion_sample.add_argument("--checkpoint", type=Path)
    diffusion_sample.add_argument("--prompt", default="")
    diffusion_sample.add_argument("--steps", type=positive_int, default=8)
    diffusion_sample.add_argument("--seed", type=int, default=1)
    diffusion_sample.add_argument("--out", type=Path)

    diffusion_eval = subcommands.add_parser("diffusion-eval", help="Write a non-claiming diffusion eval stub report.")
    diffusion_eval.add_argument("--report-out", type=Path, required=True)

    diffusion_export = subcommands.add_parser("diffusion-export-teacher", help="Export passing Qwen task code to JSONL.")
    diffusion_export.add_argument("--qwen-report", type=Path, action="append", required=True)
    diffusion_export.add_argument("--out", type=Path, required=True)
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
        report = qwen_eval(args.model, args.runs)
        write_json(args.out, report)
        print(json.dumps(report, sort_keys=True))
        return 0 if report["status"] == "available" else 2
    if args.command == "backend-status":
        report = diffusion_status()
        write_json(args.out, report)
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.command == "diffusion-train":
        if args.out is not None:
            from qiffusion.diffusion_train import DiffusionTrainConfig, train_tiny_diffusion

            report = train_tiny_diffusion(
                DiffusionTrainConfig(
                    checkpoint_path=args.out,
                    steps=args.steps,
                    seed=args.seed,
                    max_examples=args.max_examples,
                    teacher_jsonl_paths=tuple(args.teacher_jsonl),
                )
            )
            write_json(args.report_out, report)
            print(json.dumps(report, sort_keys=True))
            return 0
        report = diffusion_stub_report("train")
        write_json(args.report_out, report)
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.command == "diffusion-sample":
        if args.checkpoint is not None and args.out is not None:
            from qiffusion.diffusion_sample import DiffusionSampleConfig, sample_from_checkpoint

            report = sample_from_checkpoint(
                DiffusionSampleConfig(
                    checkpoint_path=args.checkpoint,
                    prompt=args.prompt,
                    steps=args.steps,
                    seed=args.seed,
                )
            )
            write_json(args.out, report)
            print(json.dumps(report, sort_keys=True))
            return 0
        report = diffusion_stub_report("sample")
        write_json(args.report_out, report)
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.command == "diffusion-eval":
        report = diffusion_stub_report("eval")
        write_json(args.report_out, report)
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.command == "diffusion-export-teacher":
        count = export_teacher_jsonl(tuple(args.qwen_report), args.out)
        payload = {"status": "exported", "records": count, "out": str(args.out)}
        print(json.dumps(payload, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
