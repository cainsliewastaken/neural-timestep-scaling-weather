#!/usr/bin/env python3
"""Launch ``train.py`` from a generated isoflop sweep YAML config."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run train.py using a sweep config YAML.")
    p.add_argument("--config", type=str, required=True, help="Path to generated sweep YAML.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the train.py command without executing it.",
    )
    return p.parse_args()


def overrides_to_cli(overrides: dict) -> list[str]:
    args: list[str] = []
    for key, value in overrides.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, list):
            rendered = f"[{','.join(str(v) for v in value)}]"
        else:
            rendered = str(value)
        args.append(f"{key}={rendered}")
    return args


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    overrides = cfg.get("hydra_overrides")
    if not overrides:
        raise KeyError(f"{config_path} missing hydra_overrides")

    cmd = [sys.executable, str(ROOT / "train.py"), *overrides_to_cli(overrides)]
    print("Command:", " ".join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
