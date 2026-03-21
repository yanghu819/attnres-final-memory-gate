from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .presets import PRESETS

ROOT = Path(__file__).resolve().parents[2]


def _build_env(args: argparse.Namespace) -> dict[str, str]:
    preset = PRESETS[args.preset]
    env = os.environ.copy()
    for key, value in preset.env.items():
        env.setdefault(key, value)
    if args.seed is not None:
        env["AUTORESEARCH_SEED"] = str(args.seed)
    if args.reference_lr is not None:
        env["AUTORESEARCH_REFERENCE_LR"] = str(args.reference_lr)
    if args.max_steps is not None:
        env["AUTORESEARCH_MAX_STEPS"] = str(args.max_steps)
    if args.time_budget is not None:
        env["AUTORESEARCH_TIME_BUDGET"] = str(args.time_budget)
    if args.log_path is not None:
        env["AUTORESEARCH_LOG_PATH"] = str(args.log_path)
    if args.scope is not None:
        env["AUTORESEARCH_ATTNRES_REGISTER_SCOPE"] = args.scope
    if args.scale is not None:
        env["AUTORESEARCH_ATTNRES_TOKEN_REGISTER_SCALE"] = str(args.scale)
    if args.cache_dir is not None:
        env["AUTORESEARCH_CACHE_DIR"] = args.cache_dir
    if args.base_url is not None:
        env["AUTORESEARCH_BASE_URL"] = args.base_url
    return env


def train_command(args: argparse.Namespace) -> int:
    env = _build_env(args)
    cmd = [sys.executable, str(ROOT / "train.py")]
    return subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode


def list_presets_command(_: argparse.Namespace) -> int:
    for name, preset in PRESETS.items():
        print(f"{name}: {preset.description}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autoresearch-attnres-project")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list-presets", help="List named experiment presets.")
    list_parser.set_defaults(func=list_presets_command)

    train_parser = sub.add_parser("train", help="Run one experiment preset.")
    train_parser.add_argument("--preset", choices=sorted(PRESETS.keys()), required=True)
    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument("--reference-lr", type=float)
    train_parser.add_argument("--max-steps", type=int)
    train_parser.add_argument("--time-budget", type=float)
    train_parser.add_argument("--log-path")
    train_parser.add_argument("--scope", choices=["all", "attn_only", "mlp_only", "final_only"])
    train_parser.add_argument("--scale", type=float)
    train_parser.add_argument("--cache-dir")
    train_parser.add_argument("--base-url")
    train_parser.set_defaults(func=train_command)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
