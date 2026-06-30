from __future__ import annotations

import argparse
import json
from pathlib import Path

from qiffusion.qwen_diffusion_data_loop import (
    DataLoopBlockedError,
    QwenDataLoopConfig,
    write_blocked_report,
    write_data_loop_manifest,
)


def add_qwen_diffusion_data_loop_parser(subcommands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    qwen_data_loop = subcommands.add_parser("qwen-diffusion-data-loop", help="Write filtered Qwen diffusion training data manifest.")
    qwen_data_loop.add_argument("--teacher-jsonl", type=Path, action="append", required=True)
    qwen_data_loop.add_argument("--manifest", type=Path, required=True)
    qwen_data_loop.add_argument("--out", type=Path, required=True)


def run_qwen_diffusion_data_loop(teacher_jsonl_paths: tuple[Path, ...], manifest_path: Path, output_path: Path) -> int:
    try:
        report = write_data_loop_manifest(
            QwenDataLoopConfig(
                teacher_jsonl_paths=teacher_jsonl_paths,
                manifest_path=manifest_path,
                output_path=output_path,
            )
        )
    except DataLoopBlockedError as error:
        blocked = write_blocked_report(error)
        print(json.dumps(blocked, sort_keys=True))
        return 2
    records = report.get("records")
    record_count = len(records) if isinstance(records, list) else 0
    print(json.dumps({"status": report.get("status", ""), "records": record_count, "out": str(output_path)}, sort_keys=True))
    return 0
