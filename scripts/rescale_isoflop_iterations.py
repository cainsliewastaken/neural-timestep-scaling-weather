#!/usr/bin/env python3
"""Rescale optimizer.max_iterations so every config in a sweep spends the same train FLOPs.

Reads an existing isoflop config directory (e.g. ``scripts_3T_aspect50_flops``) and
writes a copy in which only the iteration count (and the bookkeeping that depends on
it) changes. Model shapes, rollout settings and every other override are kept.

One iteration is one optimizer step (one backward pass over the batch):

    train_flops_per_iteration = 3 * forward_flops * num_rollout_steps * batch_size
    max_iterations            = round(total_train_flops / train_flops_per_iteration)

``total_train_flops`` defaults to each config's ``metadata.expected_total_train_flops``
(the isoflop target of that budget). The residual mismatch is integer rounding only
and is reported as ``train_flop_rel_err``.

``run_tag`` gets ``--tag-suffix`` appended so the new runs do not resume from (or
collide with) checkpoints of the original configs in /expts/<run_name>/<run_tag>.

Example
-------
python scripts/rescale_isoflop_iterations.py scripts_3T_aspect50_flops
    # -> scripts_3T_aspect50_isotrain_flops/
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("config_dirs", nargs="+", help="Isoflop config directories to rescale.")
    p.add_argument(
        "--total-train-flops",
        type=float,
        default=None,
        help="Train FLOP budget per run. Defaults to metadata.expected_total_train_flops.",
    )
    p.add_argument(
        "--dir-suffix",
        default="_isotrain",
        help="Inserted before the trailing '_flops' of each output directory name.",
    )
    p.add_argument("--tag-suffix", default="_isotrain", help="Appended to run_tag.")
    return p.parse_args()


def output_dir_for(config_dir: Path, dir_suffix: str) -> Path:
    name = config_dir.name
    if name.endswith("_flops"):
        return config_dir.with_name(name[: -len("_flops")] + dir_suffix + "_flops")
    return config_dir.with_name(name + dir_suffix)


def rescale_config(cfg: dict, total_train_flops: float | None, tag_suffix: str) -> dict:
    meta = cfg["metadata"]
    hydra = cfg["hydra_overrides"]
    batch_size = hydra["data.batch_size"]
    rollout_steps = hydra.get("train.num_rollout_steps", 1)
    budget = total_train_flops if total_train_flops is not None else meta["expected_total_train_flops"]

    per_iter = 3 * meta["forward_flops"] * rollout_steps * batch_size
    max_iterations = max(1, int(round(budget / per_iter)))
    total = per_iter * max_iterations
    run_tag = cfg["run_tag"] + tag_suffix

    cfg["run_tag"] = run_tag
    hydra["run_tag"] = run_tag
    hydra["optimizer.max_iterations"] = max_iterations
    meta["original_max_iterations"] = meta["max_iterations"]
    meta["max_iterations"] = max_iterations
    meta["train_flops_per_iteration"] = per_iter
    meta["total_train_flops"] = total
    meta["expected_total_train_flops"] = budget
    meta["train_flop_rel_err"] = (total - budget) / budget
    return cfg


def main() -> None:
    args = parse_args()
    for config_dir in map(Path, args.config_dirs):
        if not config_dir.is_absolute():
            config_dir = (Path.cwd() / config_dir).resolve()
        out_dir = output_dir_for(config_dir, args.dir_suffix)
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest_in = config_dir / "isoflop_manifest.csv"
        with manifest_in.open(encoding="utf-8", newline="") as f:
            manifest_rows = list(csv.DictReader(f))

        print(f"== {config_dir.name} -> {out_dir.name}")
        print(f"{'dt_scale':>8} {'embed':>6} {'depth':>5} {'old_iters':>9} {'new_iters':>9} "
              f"{'train_flops':>12} {'budget':>12} {'err':>8}")
        rows = []
        for row in manifest_rows:
            src = ROOT / row["yaml_file"]
            with src.open(encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            cfg = rescale_config(cfg, args.total_train_flops, args.tag_suffix)
            meta = cfg["metadata"]
            dst = out_dir / src.name
            with dst.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)

            row = dict(row)
            row["yaml_file"] = str(dst.relative_to(ROOT))
            row["run_tag"] = cfg["run_tag"]
            row["original_max_iterations"] = meta["original_max_iterations"]
            row["max_iterations"] = meta["max_iterations"]
            row["total_train_flops"] = meta["total_train_flops"]
            row["expected_total_train_flops"] = meta["expected_total_train_flops"]
            row["train_flop_rel_err"] = meta["train_flop_rel_err"]
            rows.append(row)
            print(f"{meta['dt_scale']:>8g} {meta['embed_dim']:>6d} {meta['depth']:>5d} "
                  f"{meta['original_max_iterations']:>9d} {meta['max_iterations']:>9d} "
                  f"{meta['total_train_flops']:>12.4e} {meta['expected_total_train_flops']:>12.4e} "
                  f"{meta['train_flop_rel_err']:>+8.2%}")

        with (out_dir / "isoflop_manifest.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
