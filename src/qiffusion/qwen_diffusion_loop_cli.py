from __future__ import annotations

import argparse
import json
from pathlib import Path

from qiffusion.qwen_diffusion_loop import QwenDiffusionLoopConfig, run_qwen_diffusion_loop


def add_qwen_diffusion_loop_parser(subcommands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    loop = subcommands.add_parser("qwen-diffusion-loop", help="Run repeated Qwen diffusion train/sample/eval iterations.")
    loop.add_argument("--manifest", type=Path, required=True)
    loop.add_argument("--config", type=Path, required=True)
    loop.add_argument("--max-iterations", type=int, required=True)
    loop.add_argument("--ledger-out", type=Path, required=True)
    loop.add_argument("--force-eval-fail", action="store_true")


def run_qwen_diffusion_loop_cli(args: argparse.Namespace) -> int:
    report = run_qwen_diffusion_loop(
        QwenDiffusionLoopConfig(
            manifest_path=args.manifest,
            config_path=args.config,
            ledger_path=args.ledger_out,
            max_iterations=args.max_iterations,
            force_eval_fail=args.force_eval_fail,
        ),
    )
    print(json.dumps(report, sort_keys=True))
    return 0
