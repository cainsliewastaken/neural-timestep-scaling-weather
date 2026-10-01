#!/usr/bin/env python3
"""Generate isoflop sweep YAML configs for ERA5 Swin training.

For each ``dt_scale`` in the sweep, search ``embed_dim`` (model width) so that

    target_rollout_flops ~= forward_flops * train_final_time_hours / (dt_scale * base_dt_hours)

Each training sample is rolled out ``num_rollout_steps = train_final_time_hours /
(dt_scale * base_dt_hours)`` steps and backpropagated through the whole rollout (one
loss, one optimizer step per batch), like ``Loss_Multistep`` in the 1d/2d repos. The
rollout budget above is what ``--target-rollout-flops`` sets (e.g. 3e9 for a 24 h curve
at ``dt_scale=1`` when ``base_dt_hours=1``).

Training length comes from a total training FLOP budget (``--total-train-flops``):

    max_iterations = total_train_flops / (3 * forward_flops * num_rollout_steps * batch_size)

where ``max_iterations`` is the total number of optimizer steps for the run (one per
global batch) and 3 counts forward + backward.

using ``FlopsCalculator`` (same formulas as training). PyTorch ``FlopCounterMode``
and fvcore are optional fallbacks; both miss ``scaled_dot_product_attention``
unless the helper adds those terms. Each generated config stores Hydra overrides
plus Slurm resources for array submission.

The default analytical backend counts on whatever ``--grid-h/--grid-w`` is set
and scales to the full ERA5 grid (720x1440). Analytical FLOPs scale exactly with
area for this windowed Swin, so the reduced count grid is only needed for the
slow pytorch/fvcore backends.

All configs roll out to the same physical horizon. With the constraints above, each
model gets the same inference (rollout) and training FLOP budget across ``dt_scale``.

Example
-------
python scripts/generate_isoflop_configs.py \\
    --budget-label 3e9 \\
    --target-rollout-flops 3e9 \\
    --total-train-flops 1e18 \\
    --train-final-time-hours 24 \\
    --dt-scales 1 2 3 4 6 8 12 \\
    --depth 12

Then submit:

    bash submit_isoflop_sweep.sh -d scripts_6E17_flops
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import yaml

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from flop_count_utils import (  # noqa: E402
    DEFAULT_FLOP_COUNT_H,
    DEFAULT_FLOP_COUNT_W,
    DEFAULT_H,
    DEFAULT_W,
    count_parameters_analytical,
    get_window_size,
    measure_forward_flops_pytorch,
    valid_embed_dim,
)

# ERA5 training years 1979-2016 (valid 2017, test 2018-2022): 38 years incl. 10 leap years.
TRAIN_HOURS_1979_2016 = (38 * 365 + 10) * 24


def _positive_float(value: str) -> float:
    text = value.strip()
    if not text:
        raise argparse.ArgumentTypeError("dt_scales values must be non-empty numbers")
    try:
        parsed = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid dt_scale value {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"dt_scale must be positive, got {parsed}")
    return parsed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate isoflop ERA5 training configs from PyTorch FLOP counts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--budget-label", required=True, help="Folder token, e.g. 6E17 or 350M.")
    p.add_argument(
        "--target-rollout-flops",
        "--target-flops-per-dt",
        dest="target_rollout_flops",
        type=float,
        required=True,
        help=(
            "Rollout FLOP budget matched by sizing: "
            "target ~= forward_flops * train_final_time_hours / (dt_scale * base_dt_hours)."
        ),
    )
    p.add_argument(
        "--train-final-time-hours",
        type=float,
        required=True,
        help=(
            "Physical rollout horizon in hours (same for every dt_scale). "
            "train.num_rollout_steps = train_final_time / (dt_scale * base_dt_hours)."
        ),
    )
    add_rollout_training_args(p)
    p.add_argument(
        "--base-dt-hours",
        type=float,
        default=1.0,
        help="Native ERA5 snapshot interval in hours (1h for latlon_025deg_hdf5_1h).",
    )
    p.add_argument(
        "--dt-scales",
        type=_positive_float,
        nargs="+",
        required=True,
        help="Values for data.dt_scale on the isoflop curve.",
    )
    p.add_argument("--output-dir", type=str, default=None, help="Defaults to scripts_<label>_flops/.")
    p.add_argument("--run-name", type=str, default="era5_isoflop")
    p.add_argument("--sweep-param", type=str, default="embed_dim", choices=("embed_dim", "depth"))
    p.add_argument("--depth", type=int, default=12, help="Fixed depth when sweeping embed_dim.")
    p.add_argument("--embed-dim", type=int, default=768, help="Fixed embed_dim when sweeping depth.")
    p.add_argument(
        "--head-dim",
        type=int,
        default=32,
        help="Attention head width; embed_dim is searched in steps of this value.",
    )
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--sp1", type=int, default=1)
    p.add_argument("--sp2", type=int, default=2)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument(
        "--min",
        dest="lo",
        type=int,
        default=32,
        help="Minimum embed_dim (or depth) in the isoflop search; must be >= head_dim.",
    )
    p.add_argument("--max", dest="hi", type=int, default=2048)
    p.add_argument("--device", type=str, default="cpu",
                   help="Requested device; PyTorch flop counting always uses CPU to avoid OOM.")
    p.add_argument(
        "--flop-backend",
        type=str,
        default="analytical",
        choices=("analytical", "pytorch", "fvcore"),
        help=(
            "How to count forward FLOPs. 'analytical' uses FlopsCalculator "
            "(same as training) and includes windowed attention. "
            "pytorch/fvcore miss SDPA unless corrected in flop_count_utils."
        ),
    )
    p.add_argument("--nodes", type=int, default=2)
    p.add_argument("--gpus-per-node", type=int, default=4)
    p.add_argument("--grid-h", type=int, default=DEFAULT_FLOP_COUNT_H,
                   help="Grid height for PyTorch FLOP counting (scaled to 720 in output).")
    p.add_argument("--grid-w", type=int, default=DEFAULT_FLOP_COUNT_W,
                   help="Grid width for PyTorch FLOP counting (scaled to 1440 in output).")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def add_rollout_training_args(p: argparse.ArgumentParser) -> None:
    """Training-budget and time-stepping options shared with generate_isoflop_configs_aspect."""
    p.add_argument(
        "--total-train-flops",
        type=float,
        required=True,
        help=(
            "Total training FLOP budget per run: optimizer.max_iterations = "
            "total_train_flops / (3 * forward_flops * num_rollout_steps * batch_size)."
        ),
    )
    p.add_argument(
        "--train-hours",
        type=int,
        default=TRAIN_HOURS_1979_2016,
        help="Hourly snapshots in the training years (reporting only: epochs = samples seen / train samples).",
    )
    p.add_argument(
        "--step-method",
        type=str,
        default="euler",
        choices=("euler", "direct"),
        help="model.step_method for TimeStepper.",
    )
    p.add_argument(
        "--tendency-stats",
        type=str,
        default="/registry/stats/tendency_stats_v1.0.h5",
        help="model.tendency_stats (container path) used by the euler step.",
    )
    p.add_argument(
        "--time-unit-hours",
        type=float,
        default=8.0,
        help=(
            "model.time_unit_hours: hours per model time unit (dt = dt_scale * base_dt / unit). "
            "Default: first-order Taylor scale min_c sigma_x,c / sigma_dxdt,c in the model's "
            "normalization (upstream global_std; 1-2 h increments give 6.7-8.1 h, fastest "
            "q200-q300 and v), rounded to 8 h so 24 h = 3 units."
        ),
    )
    p.add_argument(
        "--rollout-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="train.rollout_activation_checkpointing (recompute each rollout step in backward).",
    )


def _round_embed_dim(value: int, head_dim: int) -> int:
    value = max(head_dim, value)
    return (value // head_dim) * head_dim


def _forward_flops_for(
    args: argparse.Namespace,
    sweep_value: int,
    window_size: tuple[int, int],
    cache: dict[tuple[int, int], float],
    *,
    label: str = "",
) -> float:
    embed_dim = sweep_value if args.sweep_param == "embed_dim" else args.embed_dim
    depth = args.depth if args.sweep_param == "embed_dim" else sweep_value
    key = (embed_dim, depth)
    if key in cache:
        if args.verbose:
            print(
                f"  [cache hit] {label} {args.sweep_param}={sweep_value} "
                f"forward_flops(count_grid)={cache[key]:.3e}",
                flush=True,
            )
        return cache[key]

    count_label = label or "search"
    print(
        f"  [flop count] {count_label} {args.sweep_param}={sweep_value} "
        f"(depth={depth}, grid={args.grid_h}x{args.grid_w}, backend={args.flop_backend}) ...",
        flush=True,
    )
    t0 = time.monotonic()

    num_heads = embed_dim // args.head_dim
    flops = measure_forward_flops_pytorch(
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_size=args.patch_size,
        window_size=window_size,
        h=args.grid_h,
        w=args.grid_w,
        device=args.device,
        backend=args.flop_backend,
        verbose=args.verbose,
    )
    cache[key] = flops
    elapsed = time.monotonic() - t0
    print(
        f"  [flop count] done {args.sweep_param}={sweep_value} "
        f"forward_flops(count_grid)={flops:.3e} ({elapsed:.1f}s)",
        flush=True,
    )
    return flops


def _embed_dim_candidates(lo: int, hi: int, head_dim: int) -> list[int]:
    lo = _round_embed_dim(lo, head_dim)
    hi = _round_embed_dim(hi, head_dim)
    if lo > hi:
        return [lo]
    return list(range(lo, hi + 1, head_dim))


def search_sweep_value(
    args: argparse.Namespace,
    target_forward: float,
    window_size: tuple[int, int],
    cache: dict[tuple[int, int], float],
    *,
    label: str = "",
) -> int:
    lo, hi = args.lo, args.hi
    search_label = label or "search"

    def fwd(v: int) -> float:
        if args.sweep_param == "embed_dim":
            v = _round_embed_dim(v, args.head_dim)
            if not valid_embed_dim(v, args.head_dim):
                return float("inf")
        return _forward_flops_for(args, v, window_size, cache, label=label)

    if args.sweep_param == "embed_dim":
        lo = _round_embed_dim(lo, args.head_dim)
        hi = _round_embed_dim(hi, args.head_dim)
        expand_cap = _round_embed_dim(8192, args.head_dim)
        print(
            f"  [search] {search_label} target_forward(count_grid)={target_forward:.3e} "
            f"bracket embed_dim=[{lo}, {hi}] (step={args.head_dim})",
            flush=True,
        )
        while fwd(hi) < target_forward and hi < expand_cap:
            prev_hi = hi
            lo = hi
            hi = _round_embed_dim(min(expand_cap, hi * 2), args.head_dim)
            print(
                f"  [search] {search_label} below target at embed_dim={prev_hi}, "
                f"expanding hi -> {hi}",
                flush=True,
            )
        values = _embed_dim_candidates(lo, hi, args.head_dim)
    else:
        print(
            f"  [search] {search_label} target_forward(count_grid)={target_forward:.3e} "
            f"bracket {args.sweep_param}=[{lo}, {hi}]",
            flush=True,
        )
        expand_cap = 8192
        while fwd(hi) < target_forward and hi < expand_cap:
            prev_hi = hi
            lo = hi
            hi = min(expand_cap, hi * 2)
            print(
                f"  [search] {search_label} below target at {args.sweep_param}={prev_hi}, "
                f"expanding hi -> {hi}",
                flush=True,
            )
        values = list(range(lo, hi + 1))

    i_lo, i_hi = 0, len(values) - 1
    step = 0
    while i_lo < i_hi:
        step += 1
        i_mid = (i_lo + i_hi + 1) // 2
        mid_flops = fwd(values[i_mid])
        if args.verbose:
            print(
                f"  [search] {search_label} step {step}: probe {args.sweep_param}={values[i_mid]} "
                f"forward={mid_flops:.3e} (idx {i_lo}..{i_hi})",
                flush=True,
            )
        if mid_flops <= target_forward:
            i_lo = i_mid
        else:
            i_hi = i_mid - 1

    candidate_indices = {i_lo}
    if i_lo + 1 < len(values):
        candidate_indices.add(i_lo + 1)
    best = min(
        (values[i] for i in candidate_indices),
        key=lambda v: abs(fwd(v) - target_forward),
    )
    print(
        f"  [search] {search_label} chose {args.sweep_param}={best} "
        f"({step} binary-search step(s), {len(cache)} size(s) counted)",
        flush=True,
    )
    return best


def rollout_steps_for_dt(
    train_final_time_hours: float,
    dt_scale: float,
    base_dt_hours: float,
) -> int:
    """Rollout steps per training sample so each dt_scale covers the same physical horizon."""
    if train_final_time_hours <= 0 or dt_scale <= 0 or base_dt_hours <= 0:
        raise ValueError("train_final_time_hours, dt_scale, and base_dt_hours must be positive")
    return max(1, int(round(train_final_time_hours / (dt_scale * base_dt_hours))))


def train_flops_per_iteration(forward_flops: float, rollout_steps: int, batch_size: int) -> float:
    """One optimizer step: batch_size samples, each rolled out rollout_steps (x3 for fwd+bwd)."""
    return 3 * forward_flops * rollout_steps * batch_size


def max_iterations_for_budget(
    total_train_flops: float,
    forward_flops: float,
    rollout_steps: int,
    batch_size: int,
) -> int:
    """Total optimizer steps that spend total_train_flops."""
    per_iter = train_flops_per_iteration(forward_flops, rollout_steps, batch_size)
    return max(1, int(round(total_train_flops / per_iter)))


def train_samples_for_dt(train_hours: int, dt_scale: float, rollout_steps: int) -> int:
    """Training windows per epoch; matches era5hdf5.compute_total_samples (context window 1)."""
    return max(1, int(train_hours - rollout_steps * dt_scale))


def rollout_training_plan(
    args: argparse.Namespace,
    *,
    dt_scale: float,
    forward_flops: float,
) -> dict:
    """Rollout length, optimizer steps and FLOP bookkeeping for one config."""
    rollout_steps = rollout_steps_for_dt(args.train_final_time_hours, dt_scale, args.base_dt_hours)
    max_iterations = max_iterations_for_budget(
        args.total_train_flops, forward_flops, rollout_steps, args.batch_size
    )
    per_iter = train_flops_per_iteration(forward_flops, rollout_steps, args.batch_size)
    n_train_samples = train_samples_for_dt(args.train_hours, dt_scale, rollout_steps)
    return {
        "rollout_steps": rollout_steps,
        "max_iterations": max_iterations,
        "train_flops_per_iteration": per_iter,
        "total_train_flops": per_iter * max_iterations,
        "total_inference_flops": forward_flops * rollout_steps,
        "n_train_samples": n_train_samples,
        "epochs": max_iterations * args.batch_size / n_train_samples,
    }


def rollout_training_overrides(args: argparse.Namespace, plan: dict) -> dict:
    """Hydra overrides for multistep rollout training with the chosen step method."""
    overrides = {
        "optimizer.max_iterations": plan["max_iterations"],
        "train.temporal_context_window": 1,
        "train.num_rollout_steps": plan["rollout_steps"],
        "train.rollout_activation_checkpointing": args.rollout_checkpointing,
        "model.step_method": args.step_method,
        "model.time_unit_hours": args.time_unit_hours,
    }
    if args.step_method == "euler":
        overrides["model.tendency_stats"] = args.tendency_stats
    return overrides


def rollout_budget(
    forward_flops: float,
    dt_scale: float,
    train_final_time_hours: float,
    base_dt_hours: float = 1.0,
) -> float:
    """Isoflop budget: forward_flops * train_final_time_hours / (dt_scale * base_dt)."""
    return forward_flops * train_final_time_hours / (dt_scale * base_dt_hours)


def target_forward_flops(
    target_rollout_flops: float,
    dt_scale: float,
    train_final_time_hours: float,
    grid_scale: float = 1.0,
    base_dt_hours: float = 1.0,
) -> float:
    """Per-step forward FLOPs (count grid) so forward * tfinal / (dt*base_dt) ~= target."""
    return (
        target_rollout_flops
        * dt_scale
        * base_dt_hours
        / train_final_time_hours
        * grid_scale
    )


def constant_total_inference_flops(
    target_rollout_flops: float,
    base_dt_hours: float,
) -> float:
    """Stepwise autoregressive FLOPs (forward * rollout_steps); equals target when base_dt=1."""
    return target_rollout_flops / base_dt_hours


def resolve_reference_at_min_dt(
    args: argparse.Namespace,
    min_dt: float,
    window_size: tuple[int, int],
    grid_scale: float,
    cache: dict[tuple[int, int], float],
) -> tuple[int, int, float, float]:
    """Search model size at min dt and return (embed, depth, forward_flops, train_flops_per_iteration)."""
    target_forward = target_forward_flops(
        args.target_rollout_flops,
        min_dt,
        args.train_final_time_hours,
        grid_scale,
        args.base_dt_hours,
    )
    ref_steps = rollout_steps_for_dt(args.train_final_time_hours, min_dt, args.base_dt_hours)
    print(
        f"reference min_dt_scale={min_dt} rollout_steps={ref_steps} "
        f"target_forward(count_grid)={target_forward:.3e} ...",
        flush=True,
    )
    best = search_sweep_value(
        args,
        target_forward,
        window_size,
        cache,
        label=f"reference min_dt={min_dt}",
    )
    forward_flops_count_grid = _forward_flops_for(
        args,
        best,
        window_size,
        cache,
        label=f"reference min_dt={min_dt} final",
    )
    forward_flops = forward_flops_count_grid / grid_scale
    embed_dim = best if args.sweep_param == "embed_dim" else args.embed_dim
    depth = args.depth if args.sweep_param == "embed_dim" else best
    per_iter_train_flops = train_flops_per_iteration(forward_flops, ref_steps, args.batch_size)
    return embed_dim, depth, forward_flops, per_iter_train_flops


def format_run_tag(
    budget_label: str,
    dt_scale: float,
    embed_dim: int,
    depth: int,
) -> str:
    dt_token = str(dt_scale).replace(".", "p")
    return f"{budget_label}_dt{dt_token}_e{embed_dim}_d{depth}"


def build_config_dict(
    args: argparse.Namespace,
    *,
    dt_scale: float,
    embed_dim: int,
    depth: int,
    train_window_size: tuple[int, int],
    forward_flops: float,
    forward_flops_count_grid: float,
    plan: dict,
    num_parameters: int,
    reference_dt_scale: float,
    train_final_time_hours: float,
    expected_total_train_flops: float,
    expected_total_inference_flops: float,
) -> dict:
    num_heads = embed_dim // args.head_dim
    run_tag = format_run_tag(args.budget_label, dt_scale, embed_dim, depth)
    overrides = {
        "run_name": args.run_name,
        "run_tag": run_tag,
        "data.dt_scale": int(dt_scale) if float(dt_scale).is_integer() else dt_scale,
        "data.batch_size": args.batch_size,
        "model.embed_dim": embed_dim,
        "model.depth": depth,
        "model.num_heads": num_heads,
        "model.patch_size": args.patch_size,
        "model.window_size": list(train_window_size),
        "parallelism.sp1": args.sp1,
        "parallelism.sp2": args.sp2,
        "parallelism.micro_batch_size": args.micro_batch_size,
        "parallelism.use_transformer_engine": True,
        "optimizer": "adamw",
        "optimizer.lr": args.lr,
        "train.clip_grad_norm": 1.0,
        **rollout_training_overrides(args, plan),
        "inference.time_horizon_in_hours": train_final_time_hours,
    }
    return {
        "schema_version": 1,
        "run_name": args.run_name,
        "run_tag": run_tag,
        "metadata": {
            "budget_label": args.budget_label,
            "target_rollout_flops": args.target_rollout_flops,
            "train_final_time_hours": train_final_time_hours,
            "base_dt_hours": args.base_dt_hours,
            "model_dt_hours": dt_scale * args.base_dt_hours,
            "rollout_steps": plan["rollout_steps"],
            "reference_dt_scale": reference_dt_scale,
            "expected_total_train_flops": expected_total_train_flops,
            "expected_total_inference_flops": expected_total_inference_flops,
            "dt_scale": dt_scale,
            "embed_dim": embed_dim,
            "depth": depth,
            "num_heads": num_heads,
            "patch_size": args.patch_size,
            "window_size": list(train_window_size),
            "forward_flops": forward_flops,
            "forward_flops_count_grid": forward_flops_count_grid,
            "rollout_budget": rollout_budget(
                forward_flops, dt_scale, train_final_time_hours, args.base_dt_hours
            ),
            "train_flops_per_iteration": plan["train_flops_per_iteration"],
            "max_iterations": plan["max_iterations"],
            "total_train_flops": plan["total_train_flops"],
            "n_train_samples": plan["n_train_samples"],
            "epochs": plan["epochs"],
            "total_inference_flops": plan["total_inference_flops"],
            "num_parameters": num_parameters,
            "grid": [DEFAULT_H, DEFAULT_W],
            "flop_count_grid": [args.grid_h, args.grid_w],
            "flop_backend": args.flop_backend,
        },
        "slurm": {
            "nodes": args.nodes,
            "gpus_per_node": args.gpus_per_node,
        },
        "hydra_overrides": overrides,
    }


def main() -> None:
    args = parse_args()
    if args.train_final_time_hours <= 0:
        raise ValueError(f"train_final_time_hours must be positive, got {args.train_final_time_hours}")
    if args.base_dt_hours <= 0:
        raise ValueError(f"base_dt_hours must be positive, got {args.base_dt_hours}")
    if args.head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {args.head_dim}")
    if args.sweep_param == "embed_dim" and args.lo < args.head_dim:
        raise ValueError(
            f"--min embed_dim ({args.lo}) must be >= --head-dim ({args.head_dim})"
        )
    if args.hi < args.lo:
        raise ValueError(f"--max ({args.hi}) must be >= --min ({args.lo})")

    if args.total_train_flops <= 0:
        raise ValueError(f"total_train_flops must be positive, got {args.total_train_flops}")

    min_dt = min(args.dt_scales)
    expected_train_flops = args.total_train_flops
    expected_inference_flops = constant_total_inference_flops(
        args.target_rollout_flops,
        args.base_dt_hours,
    )
    output_dir = ROOT / args.output_dir if args.output_dir else ROOT / f"scripts_{args.budget_label}_flops"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Flop counting builds a single-replica model (see make_domain_metadata defaults).
    window_size = get_window_size(
        args.patch_size,
        sp1=1,
        sp2=1,
        img_size=(args.grid_h, args.grid_w),
    )
    train_window_size = get_window_size(
        args.patch_size,
        args.sp1,
        args.sp2,
        img_size=(DEFAULT_H, DEFAULT_W),
    )
    grid_scale = (args.grid_h * args.grid_w) / float(DEFAULT_H * DEFAULT_W)
    cache: dict[tuple[int, int], float] = {}
    rows: list[dict] = []

    requested = args.device.strip().lower()
    if requested in {"gpu", "cuda"}:
        print(
            "note: PyTorch flop counting uses CPU (and a reduced count grid by default) "
            "to avoid GPU OOM; training configs still target the full 720x1440 grid.",
            flush=True,
        )

    ref_embed, ref_depth, ref_forward_flops, ref_per_iter_train_flops = resolve_reference_at_min_dt(
        args, min_dt, window_size, grid_scale, cache
    )
    ref_rollout_steps = rollout_steps_for_dt(args.train_final_time_hours, min_dt, args.base_dt_hours)

    print(
        f"budget={args.budget_label} target_rollout_flops={args.target_rollout_flops:.3e} "
        f"train_final_time_hours={args.train_final_time_hours} base_dt_hours={args.base_dt_hours} "
        f"expected_total_train_flops={expected_train_flops:.3e} "
        f"expected_total_inference_flops={expected_inference_flops:.3e} "
        f"reference min_dt={min_dt} ref_rollout_steps={ref_rollout_steps} ref_embed={ref_embed} "
        f"flop_backend={args.flop_backend} count_grid={args.grid_h}x{args.grid_w} "
        f"grid_scale={grid_scale:.4f} flop_device=cpu",
        flush=True,
    )
    header = (
        f"{'dt_scale':>10} {args.sweep_param:>12} {'forward_flops':>14} "
        f"{'F*T/dt':>14} {'rollout_steps':>14} {'max_iters':>10} {'epochs':>8} "
        f"{'train_flops':>14} {'infer_flops':>14} {'params':>12} {'rel_err':>9}"
    )
    print(header)
    print("-" * len(header))

    for dt_scale in args.dt_scales:
        if dt_scale <= 0:
            raise ValueError(f"dt_scale must be positive, got {dt_scale}")
        target_forward = target_forward_flops(
            args.target_rollout_flops,
            dt_scale,
            args.train_final_time_hours,
            grid_scale,
            args.base_dt_hours,
        )
        rollout_steps = rollout_steps_for_dt(args.train_final_time_hours, dt_scale, args.base_dt_hours)
        print(
            f"searching dt_scale={dt_scale} rollout_steps={rollout_steps} "
            f"target_forward(count_grid)={target_forward:.3e} ...",
            flush=True,
        )
        best = search_sweep_value(
            args, target_forward, window_size, cache, label=f"dt_scale={dt_scale}"
        )
        forward_flops_count_grid = _forward_flops_for(
            args, best, window_size, cache, label=f"dt_scale={dt_scale} final"
        )
        forward_flops = forward_flops_count_grid / grid_scale

        embed_dim = best if args.sweep_param == "embed_dim" else args.embed_dim
        depth = args.depth if args.sweep_param == "embed_dim" else best
        num_heads = embed_dim // args.head_dim

        num_parameters = count_parameters_analytical(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            patch_size=args.patch_size,
            window_size=train_window_size,
            h=DEFAULT_H,
            w=DEFAULT_W,
        )
        plan = rollout_training_plan(args, dt_scale=dt_scale, forward_flops=forward_flops)
        budget = rollout_budget(
            forward_flops, dt_scale, args.train_final_time_hours, args.base_dt_hours
        )

        rel = (budget - args.target_rollout_flops) / args.target_rollout_flops
        print(
            f"{dt_scale:>10g} {best:>12d} {forward_flops:>14.3e} "
            f"{budget:>14.3e} {plan['rollout_steps']:>14d} {plan['max_iterations']:>10d} "
            f"{plan['epochs']:>8.2f} {plan['total_train_flops']:>14.3e} "
            f"{plan['total_inference_flops']:>14.3e} {num_parameters:>12d} {rel:>+8.1%}",
            flush=True,
        )

        cfg = build_config_dict(
            args,
            dt_scale=dt_scale,
            embed_dim=embed_dim,
            depth=depth,
            train_window_size=train_window_size,
            forward_flops=forward_flops,
            forward_flops_count_grid=forward_flops_count_grid,
            plan=plan,
            num_parameters=num_parameters,
            reference_dt_scale=min_dt,
            train_final_time_hours=args.train_final_time_hours,
            expected_total_train_flops=expected_train_flops,
            expected_total_inference_flops=expected_inference_flops,
        )
        dt_token = str(dt_scale).replace(".", "p")
        out_path = output_dir / f"swin_dt_scale_{dt_token}.yaml"
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        rows.append(
            {
                "yaml_file": str(out_path.relative_to(ROOT)),
                "dt_scale": dt_scale,
                "embed_dim": embed_dim,
                "depth": depth,
                "num_heads": num_heads,
                "forward_flops": forward_flops,
                "rollout_budget": budget,
                "target_rollout_flops": args.target_rollout_flops,
                "train_final_time_hours": args.train_final_time_hours,
                "model_dt_hours": dt_scale * args.base_dt_hours,
                "rollout_steps": plan["rollout_steps"],
                "max_iterations": plan["max_iterations"],
                "epochs": plan["epochs"],
                "total_train_flops": plan["total_train_flops"],
                "total_inference_flops": plan["total_inference_flops"],
                "expected_total_train_flops": expected_train_flops,
                "expected_total_inference_flops": expected_inference_flops,
                "reference_dt_scale": min_dt,
                "num_parameters": num_parameters,
                "run_tag": cfg["run_tag"],
            }
        )

    manifest = output_dir / "isoflop_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("-" * len(header))
    print(f"Wrote {len(rows)} configs to {output_dir}")
    print(f"Manifest: {manifest}")
    print(f"Submit with: bash submit_isoflop_sweep.sh -d {output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
