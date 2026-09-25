#!/usr/bin/env python3
"""Aggregate isoflop manifests into width-only and aspect summary CSVs.

Forward (inference) 24h cost
----------------------------
  forward_24h_flops = forward_flops * train_final_time_hours / (dt_scale * base_dt_hours)
                    = total_inference_flops   (base_dt_hours == 1)

Training epoch cost
-------------------
Config metadata ``total_train_flops`` equals the cost of *one global batch*
trained with a 24h-equivalent unroll:

  total_train_flops = forward_flops * 3 * batch_size * rollout_steps
                    = forward_24h_flops * 3 * batch_size

A full dataset epoch has ``iters_per_epoch = n_train_samples // batch_size``
optimizer steps (see ``utils/trainer.py``), so:

  train_epoch_flops = total_train_flops * iters_per_epoch
                    ≈ forward_24h_flops * 3 * n_train_samples

``n_train_samples`` defaults from ``--train-hours`` using the same border
formula as the ERA5 loader:

  n_samples = train_hours - (temporal_context_window + num_rollout_steps - 1) * dt_scale
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BATCH_SIZE = 16  # generator default used for all isoflop configs
TRAIN_FACTOR = 3  # fwd + bwd ≈ 3× forward per training step (matches trainer.py)
TEMPORAL_CONTEXT_WINDOW = 1
NUM_ROLLOUT_STEPS = 1  # generator sets train.num_rollout_steps=1
# Default: hourly ERA5 train years 1979–2016 (excl. valid 2017 + test 2018–2022).
DEFAULT_TRAIN_HOURS = int(round(38 * 365.25 * 24))

SUMMARY_FIELDS = [
    "budget_label",
    "target_rollout_flops",
    "dt_scale",
    "embed_dim",
    "depth",
    "forward_24h_flops",
    "train_epoch_flops",
    "iters_per_epoch",
    "n_train_samples",
    "yaml_file",
]


def _read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_yaml_as_row(yaml_path: Path) -> dict:
    with yaml_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    meta = cfg["metadata"]
    return {
        "yaml_file": str(yaml_path.relative_to(ROOT)),
        "dt_scale": meta["dt_scale"],
        "embed_dim": meta["embed_dim"],
        "depth": meta["depth"],
        "forward_flops": meta["forward_flops"],
        "rollout_budget": meta["rollout_budget"],
        "target_rollout_flops": meta["target_rollout_flops"],
        "train_final_time_hours": meta["train_final_time_hours"],
        "total_train_steps": meta["total_train_steps"],
        "rollout_steps": meta["rollout_steps"],
        "total_train_flops": meta["total_train_flops"],
        "total_inference_flops": meta["total_inference_flops"],
        "budget_label": meta.get("budget_label", cfg.get("run_tag", "").split("_")[0]),
    }


def _load_dir(dir_path: Path) -> list[dict]:
    manifest = dir_path / "isoflop_manifest.csv"
    if manifest.is_file():
        rows = _read_manifest(manifest)
        for row in rows:
            row.setdefault("budget_label", dir_path.name)
        return rows

    rows = []
    for yaml_path in sorted(dir_path.glob("swin_dt_scale_*.yaml")):
        rows.append(_read_yaml_as_row(yaml_path))
    return rows


def _n_train_samples(train_hours: int, dt_scale: float) -> int:
    """Match era5hdf5.compute_total_samples for fixed train_hours timeline."""
    margin = (TEMPORAL_CONTEXT_WINDOW + NUM_ROLLOUT_STEPS - 1) * dt_scale
    n = int(train_hours - margin)
    if n <= 0:
        raise ValueError(
            f"n_train_samples non-positive for train_hours={train_hours}, dt_scale={dt_scale}"
        )
    return n


def _verify_and_summarize(
    row: dict,
    budget_label: str,
    *,
    train_hours: int,
    batch_size: int,
) -> dict:
    dt = float(row["dt_scale"])
    forward = float(row["forward_flops"])
    tfinal = float(row["train_final_time_hours"])
    steps = int(float(row["rollout_steps"]))
    reported_infer = float(row["total_inference_flops"])
    reported_train = float(row["total_train_flops"])
    reported_budget = float(row["rollout_budget"])

    expected_infer = forward * tfinal / dt
    expected_train_one_batch = forward * TRAIN_FACTOR * batch_size * steps

    def _rel(a: float, b: float) -> float:
        return abs(a - b) / max(abs(b), 1.0)

    errs = []
    if _rel(reported_budget, expected_infer) > 1e-9:
        errs.append(f"rollout_budget {reported_budget} != forward*T/dt {expected_infer}")
    if _rel(reported_infer, expected_infer) > 1e-9:
        errs.append(f"total_inference_flops {reported_infer} != forward*T/dt {expected_infer}")
    if _rel(reported_train, expected_train_one_batch) > 1e-9:
        errs.append(
            f"total_train_flops {reported_train} != forward*3*batch*steps "
            f"{expected_train_one_batch}"
        )
    if errs:
        raise ValueError(
            f"FLOP check failed for {row.get('yaml_file', budget_label)} dt={dt}: "
            + "; ".join(errs)
        )

    n_samples = _n_train_samples(train_hours, dt)
    iters_per_epoch = n_samples // batch_size
    if iters_per_epoch < 1:
        raise ValueError(
            f"iters_per_epoch < 1 for n_samples={n_samples}, batch_size={batch_size}"
        )
    # Scale the one-batch / 24h-unroll train cost up to a full data epoch.
    train_epoch_flops = reported_train * iters_per_epoch

    return {
        "budget_label": budget_label,
        "target_rollout_flops": float(row["target_rollout_flops"]),
        "dt_scale": dt,
        "embed_dim": int(float(row["embed_dim"])),
        "depth": int(float(row["depth"])),
        "forward_24h_flops": reported_infer,
        "train_epoch_flops": train_epoch_flops,
        "iters_per_epoch": iters_per_epoch,
        "n_train_samples": n_samples,
        "yaml_file": row.get("yaml_file", ""),
    }


def _budget_sort_key(label: str) -> float:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([eE][+-]?\d+|T)?", label)
    if not m:
        return float("inf")
    val = float(m.group(1))
    suffix = (m.group(2) or "").upper()
    if suffix == "T":
        return val * 1e12
    if suffix.startswith("E"):
        return float(f"{m.group(1)}{suffix}")
    return val


def write_summary(rows: list[dict], out_path: Path) -> None:
    rows = sorted(
        rows,
        key=lambda r: (_budget_sort_key(str(r["budget_label"])), float(r["dt_scale"])),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_path}")


def collect(
    pattern: str,
    label_from_dirname,
    *,
    train_hours: int,
    batch_size: int,
) -> list[dict]:
    rows: list[dict] = []
    for dir_path in sorted(ROOT.glob(pattern)):
        if not dir_path.is_dir():
            continue
        budget_label = label_from_dirname(dir_path.name)
        for row in _load_dir(dir_path):
            rows.append(
                _verify_and_summarize(
                    row,
                    budget_label,
                    train_hours=train_hours,
                    batch_size=batch_size,
                )
            )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--width-out",
        type=Path,
        default=ROOT / "isoflop_width_summary.csv",
    )
    p.add_argument(
        "--aspect-out",
        type=Path,
        default=ROOT / "isoflop_aspect_summary.csv",
    )
    p.add_argument(
        "--train-hours",
        type=int,
        default=DEFAULT_TRAIN_HOURS,
        help=(
            "Hourly timestamps available for training (before dt_scale border). "
            f"Default {DEFAULT_TRAIN_HOURS} ≈ 1979–2016 inclusive."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Global batch size used when counting iters_per_epoch.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"train_hours={args.train_hours} batch_size={args.batch_size} "
        f"(iters_per_epoch ≈ {(args.train_hours - 2) // args.batch_size} at dt=1)",
        flush=True,
    )

    width_rows = collect(
        "scripts_*e12_flops",
        label_from_dirname=lambda name: re.fullmatch(
            r"scripts_(.+)_flops", name
        ).group(1),
        train_hours=args.train_hours,
        batch_size=args.batch_size,
    )
    aspect_rows = collect(
        "scripts_*_aspect_flops",
        label_from_dirname=lambda name: re.fullmatch(
            r"scripts_(.+)_aspect_flops", name
        ).group(1),
        train_hours=args.train_hours,
        batch_size=args.batch_size,
    )

    write_summary(width_rows, args.width_out)
    write_summary(aspect_rows, args.aspect_out)

    width_budgets = sorted({r["budget_label"] for r in width_rows}, key=_budget_sort_key)
    aspect_budgets = sorted({r["budget_label"] for r in aspect_rows}, key=_budget_sort_key)
    print(f"Width budgets: {width_budgets}")
    print(f"Aspect budgets: {aspect_budgets}")
    for label in aspect_budgets:
        n = sum(1 for r in aspect_rows if r["budget_label"] == label)
        if n < 7:
            print(
                f"warning: aspect budget {label} has only {n}/7 dt configs so far",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
