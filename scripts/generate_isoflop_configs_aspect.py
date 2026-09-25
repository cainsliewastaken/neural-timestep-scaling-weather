#!/usr/bin/env python3
"""Generate isoflop sweep YAML configs with a Kaplan aspect-ratio band.

For each ``dt_scale``, search ``(embed_dim, depth)`` so that

    target_rollout_flops ~= forward_flops * train_final_time_hours / (dt_scale * base_dt_hours)

and

    --min-aspect-ratio <= embed_dim / depth <= --max-aspect-ratio

(default 40..100). Kaplan et al. (arXiv:2001.08361) keep width/depth in this
range when scaling; GPT-2 XL is 1600/48 ≈ 33, and models with fewer than 2
layers or extreme ratios fall off their trend.

For each depth in ``[--min-depth, --max-depth]`` (default 2..30), ``embed_dim``
is binary-searched only inside the aspect band. The in-band shape closest to
the FLOP target is selected.

Example
-------
python scripts/generate_isoflop_configs_aspect.py \\
    --budget-label 12T \\
    --target-rollout-flops 12e12 \\
    --train-final-time-hours 24
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
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

# Absolute embed_dim ceiling when expanding the search bracket.
_EMBED_EXPAND_ABS_CAP = 8192
# Max times we may double the upper embed_dim bound.
_MAX_EXPAND_STEPS = 8


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
        description=(
            "Generate isoflop ERA5 configs by jointly searching embed_dim and depth "
            "inside a Kaplan aspect-ratio band (default embed_dim/depth in [40, 100])."
        ),
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
            "optimizer.max_iterations = train_final_time / (dt_scale * base_dt_hours)."
        ),
    )
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
        default=[1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0],
        help="Values for data.dt_scale on the isoflop curve.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Defaults to scripts_<label>_aspect_flops/.",
    )
    p.add_argument("--run-name", type=str, default="era5_isoflop")
    p.add_argument(
        "--min-depth",
        type=int,
        default=2,
        help=(
            "Minimum transformer depth. Kaplan et al. find models with "
            "fewer than 2 layers deviate from the scaling trend."
        ),
    )
    p.add_argument(
        "--max-depth",
        type=int,
        default=30,
        help="Maximum transformer depth in the joint search.",
    )
    p.add_argument(
        "--min-aspect-ratio",
        type=float,
        default=40.0,
        help="Lower bound on embed_dim/depth (Kaplan band).",
    )
    p.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=100.0,
        help="Upper bound on embed_dim/depth (Kaplan band).",
    )
    p.add_argument(
        "--target-aspect-ratio",
        type=float,
        default=None,
        help=(
            "Optional preferred aspect inside the band (used only for "
            "reporting aspect_rel_err). Defaults to the band midpoint."
        ),
    )
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
        help="Minimum embed_dim in the isoflop search; must be >= head_dim.",
    )
    p.add_argument(
        "--max",
        dest="hi",
        type=int,
        default=4096,
        help="Maximum embed_dim (must cover max_aspect * max_depth).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Requested device; PyTorch flop counting always uses CPU to avoid OOM.",
    )
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
    p.add_argument(
        "--grid-h",
        type=int,
        default=DEFAULT_FLOP_COUNT_H,
        help="Grid height for PyTorch FLOP counting (scaled to 720 in output).",
    )
    p.add_argument(
        "--grid-w",
        type=int,
        default=DEFAULT_FLOP_COUNT_W,
        help="Grid width for PyTorch FLOP counting (scaled to 1440 in output).",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _round_embed_dim(value: int, head_dim: int) -> int:
    value = max(head_dim, value)
    return (value // head_dim) * head_dim


def _embed_for_aspect(depth: int, target_aspect: float, head_dim: int) -> int:
    """Nearest valid embed_dim to ``target_aspect * depth``."""
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    raw = target_aspect * depth
    lo = _round_embed_dim(int(raw), head_dim)
    hi = lo + head_dim
    if abs(lo - raw) <= abs(hi - raw):
        return max(head_dim, lo)
    return max(head_dim, hi)


def _aspect_band_embeds(
    depth: int,
    min_aspect: float,
    max_aspect: float,
    head_dim: int,
    lo: int,
    hi: int,
) -> list[int]:
    """Valid embed_dims at ``depth`` with aspect in [min_aspect, max_aspect]."""
    raw_lo = min_aspect * depth
    raw_hi = max_aspect * depth
    e_lo = int(math.ceil(raw_lo / head_dim) * head_dim)
    e_hi = int(math.floor(raw_hi / head_dim) * head_dim)
    e_lo = max(e_lo, _round_embed_dim(lo, head_dim), head_dim)
    e_hi = min(e_hi, _round_embed_dim(hi, head_dim))
    if e_lo > e_hi:
        return []
    return list(range(e_lo, e_hi + 1, head_dim))


def _embed_dim_candidates(lo: int, hi: int, head_dim: int) -> list[int]:
    lo = _round_embed_dim(lo, head_dim)
    hi = _round_embed_dim(hi, head_dim)
    if lo > hi:
        return [lo]
    return list(range(lo, hi + 1, head_dim))


def _forward_flops_for(
    args: argparse.Namespace,
    embed_dim: int,
    depth: int,
    window_size: tuple[int, int],
    cache: dict[tuple[int, int], float],
    *,
    label: str = "",
) -> float:
    key = (embed_dim, depth)
    if key in cache:
        if args.verbose:
            print(
                f"  [cache hit] {label} embed_dim={embed_dim} depth={depth} "
                f"forward_flops(count_grid)={cache[key]:.3e}",
                flush=True,
            )
        return cache[key]

    count_label = label or "search"
    print(
        f"  [flop count] {count_label} embed_dim={embed_dim} depth={depth} "
        f"(grid={args.grid_h}x{args.grid_w}, backend={args.flop_backend}) ...",
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
        f"  [flop count] done embed_dim={embed_dim} depth={depth} "
        f"forward_flops(count_grid)={flops:.3e} ({elapsed:.1f}s)",
        flush=True,
    )
    return flops


@dataclass(frozen=True)
class SizeCandidate:
    embed_dim: int
    depth: int
    forward_flops_count_grid: float

    @property
    def aspect_ratio(self) -> float:
        return self.embed_dim / float(self.depth)


def search_embed_dim_for_depth(
    args: argparse.Namespace,
    depth: int,
    target_forward: float,
    window_size: tuple[int, int],
    cache: dict[tuple[int, int], float],
    *,
    label: str = "",
    hint_embed_dim: int | None = None,
) -> SizeCandidate | None:
    """Binary-search embed_dim at fixed depth. Return None if depth is unreachable.

    Avoids exhaustively scanning widths: only probes O(log N) candidates. An optional
    ``hint_embed_dim`` (e.g. from a nearby depth) shrinks the initial bracket so we
    do not re-count the global max width on every depth.
    """
    search_label = label or f"depth={depth}"
    lo = _round_embed_dim(args.lo, args.head_dim)
    hi = _round_embed_dim(args.hi, args.head_dim)
    expand_cap = _round_embed_dim(_EMBED_EXPAND_ABS_CAP, args.head_dim)

    def fwd(embed_dim: int) -> float:
        embed_dim = _round_embed_dim(embed_dim, args.head_dim)
        if not valid_embed_dim(embed_dim, args.head_dim):
            return float("inf")
        return _forward_flops_for(
            args, embed_dim, depth, window_size, cache, label=search_label
        )

    # Unreachable-low: thinnest model already exceeds the FLOP target.
    lo_flops = fwd(lo)
    if lo_flops > target_forward:
        print(
            f"  [search] {search_label} skip: min embed_dim={lo} already over target "
            f"(forward={lo_flops:.3e} > {target_forward:.3e})",
            flush=True,
        )
        return None

    # Warm-start the upper bound near a previous depth's solution when possible.
    if hint_embed_dim is not None:
        warm_hi = _round_embed_dim(
            max(lo + args.head_dim, min(hi, hint_embed_dim * 2)),
            args.head_dim,
        )
        if warm_hi > lo and fwd(warm_hi) >= target_forward:
            hi = warm_hi

    print(
        f"  [search] {search_label} target_forward(count_grid)={target_forward:.3e} "
        f"binary-search embed_dim=[{lo}, {hi}] (step={args.head_dim})",
        flush=True,
    )

    # Expand upper bound only while the current hi is still under target.
    expand_steps = 0
    while True:
        hi_flops = fwd(hi)
        if hi_flops >= target_forward:
            break
        if hi >= expand_cap or expand_steps >= _MAX_EXPAND_STEPS:
            print(
                f"  [search] {search_label} skip: max embed_dim={hi} still under target "
                f"(forward={hi_flops:.3e} < {target_forward:.3e})",
                flush=True,
            )
            return None
        prev_hi = hi
        next_hi = _round_embed_dim(
            min(expand_cap, max(hi * 2, hi + args.head_dim)),
            args.head_dim,
        )
        if next_hi <= hi:
            print(
                f"  [search] {search_label} skip: max embed_dim={hi} still under target "
                f"(forward={hi_flops:.3e} < {target_forward:.3e})",
                flush=True,
            )
            return None
        # After confirming hi is under target, raise the lower bracket to hi so the
        # binary search does not re-probe already-known-under widths.
        lo = hi
        hi = next_hi
        expand_steps += 1
        print(
            f"  [search] {search_label} below target at embed_dim={prev_hi}, "
            f"expanding hi -> {hi} (expand {expand_steps}/{_MAX_EXPAND_STEPS})",
            flush=True,
        )

    values = _embed_dim_candidates(lo, hi, args.head_dim)
    if not values:
        print(f"  [search] {search_label} skip: empty embed_dim candidate list", flush=True)
        return None

    # Standard binary search for the largest width with flops <= target, then pick
    # the nearer of that width and its successor.
    i_lo, i_hi = 0, len(values) - 1
    step = 0
    max_steps = int(math.ceil(math.log2(max(len(values), 2)))) + 2
    while i_lo < i_hi:
        step += 1
        if step > max_steps:
            print(
                f"  [search] {search_label} abort binary search: exceeded step cap "
                f"{max_steps} (idx {i_lo}..{i_hi})",
                flush=True,
            )
            break
        i_mid = (i_lo + i_hi + 1) // 2
        mid_flops = fwd(values[i_mid])
        if args.verbose:
            print(
                f"  [search] {search_label} step {step}: probe embed_dim={values[i_mid]} "
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
    best_embed = min(
        (values[i] for i in candidate_indices),
        key=lambda v: abs(fwd(v) - target_forward),
    )
    best_flops = fwd(best_embed)
    print(
        f"  [search] {search_label} chose embed_dim={best_embed} "
        f"({step} binary-search step(s), aspect={best_embed / depth:.2f}, "
        f"{len(cache)} size(s) counted)",
        flush=True,
    )
    return SizeCandidate(
        embed_dim=best_embed,
        depth=depth,
        forward_flops_count_grid=best_flops,
    )


def search_best_size(
    args: argparse.Namespace,
    target_forward: float,
    window_size: tuple[int, int],
    cache: dict[tuple[int, int], float],
    *,
    label: str = "",
) -> SizeCandidate | None:
    """Pick the in-band (embed_dim, depth) closest to the FLOP target.

    At each depth, ``embed_dim`` is binary-searched only inside
    ``[min_aspect, max_aspect]``. Extreme 1280/1 shapes are excluded because
    they lie outside the band.

    A fixed aspect (``min_aspect == max_aspect``) keeps every depth. Width is
    the nearest head-multiple of ``aspect * depth``, so a depth is not dropped
    when ``aspect * depth`` is not divisible by ``head_dim``.
    """
    search_label = label or "search"
    embed_cap = _round_embed_dim(
        max(args.hi, int(math.ceil(args.max_aspect_ratio * args.max_depth))),
        args.head_dim,
    )
    embed_cap = min(embed_cap, _round_embed_dim(_EMBED_EXPAND_ABS_CAP, args.head_dim))
    fixed_aspect = args.min_aspect_ratio == args.max_aspect_ratio
    candidates: list[SizeCandidate] = []
    for depth in range(args.min_depth, args.max_depth + 1):
        if fixed_aspect:
            embed = _embed_for_aspect(depth, args.min_aspect_ratio, args.head_dim)
            embed = min(max(embed, _round_embed_dim(args.lo, args.head_dim)), embed_cap)
            values = [embed] if valid_embed_dim(embed, args.head_dim) else []
        else:
            values = _aspect_band_embeds(
                depth,
                args.min_aspect_ratio,
                args.max_aspect_ratio,
                args.head_dim,
                args.lo,
                embed_cap,
            )
        if not values:
            print(
                f"  [band] {search_label} depth={depth} skip: empty embed range "
                f"for aspect [{args.min_aspect_ratio:g}, {args.max_aspect_ratio:g}]",
                flush=True,
            )
            continue

        def fwd(embed_dim: int, depth: int = depth) -> float:
            return _forward_flops_for(
                args,
                embed_dim,
                depth,
                window_size,
                cache,
                label=f"{search_label} depth={depth}",
            )

        i_lo, i_hi = 0, len(values) - 1
        step = 0
        max_steps = int(math.ceil(math.log2(max(len(values), 2)))) + 2
        while i_lo < i_hi:
            step += 1
            if step > max_steps:
                break
            i_mid = (i_lo + i_hi + 1) // 2
            if fwd(values[i_mid]) <= target_forward:
                i_lo = i_mid
            else:
                i_hi = i_mid - 1
        probe = {i_lo}
        if i_lo + 1 < len(values):
            probe.add(i_lo + 1)
        best_embed = min(
            (values[i] for i in probe),
            key=lambda v: abs(fwd(v) - target_forward),
        )
        cand = SizeCandidate(
            embed_dim=best_embed,
            depth=depth,
            forward_flops_count_grid=fwd(best_embed),
        )
        candidates.append(cand)
        print(
            f"  [band] {search_label} depth={depth} embed_dim={cand.embed_dim} "
            f"aspect={cand.aspect_ratio:.2f} forward={cand.forward_flops_count_grid:.3e} "
            f"(band embed {values[0]}..{values[-1]})",
            flush=True,
        )

    if not candidates:
        print(
            f"  [select] {search_label} skip: no in-band (embed_dim, depth) "
            f"for depths [{args.min_depth}, {args.max_depth}] and aspect "
            f"[{args.min_aspect_ratio:g}, {args.max_aspect_ratio:g}]",
            flush=True,
        )
        return None

    best = min(
        candidates,
        key=lambda c: abs(c.forward_flops_count_grid - target_forward),
    )
    print(
        f"  [select] {search_label} chose embed_dim={best.embed_dim} depth={best.depth} "
        f"aspect={best.aspect_ratio:.2f} "
        f"(band [{args.min_aspect_ratio:g}, {args.max_aspect_ratio:g}]) "
        f"from {len(candidates)} depth candidate(s)",
        flush=True,
    )
    return best


def train_steps_for_dt(
    train_final_time_hours: float,
    dt_scale: float,
    base_dt_hours: float,
) -> int:
    """Optimizer steps so each dt_scale covers the same physical time window."""
    if train_final_time_hours <= 0 or dt_scale <= 0 or base_dt_hours <= 0:
        raise ValueError("train_final_time_hours, dt_scale, and base_dt_hours must be positive")
    return max(1, int(round(train_final_time_hours / (dt_scale * base_dt_hours))))


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


def constant_total_train_flops(
    target_rollout_flops: float,
    batch_size: int,
    base_dt_hours: float,
) -> float:
    """Total training FLOPs when rollout budget + fixed physical time hold."""
    return target_rollout_flops * 3 * batch_size / base_dt_hours


def constant_total_inference_flops(
    target_rollout_flops: float,
    base_dt_hours: float,
) -> float:
    """Stepwise autoregressive FLOPs (forward * rollout_steps); equals target when base_dt=1."""
    return target_rollout_flops / base_dt_hours


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
    max_iterations: int,
    num_parameters: int,
    reference_dt_scale: float,
    train_final_time_hours: float,
    total_inference_flops: float,
    expected_total_train_flops: float,
    expected_total_inference_flops: float,
    aspect_ratio: float,
    flop_rel_err: float,
    aspect_rel_err: float,
) -> dict:
    num_heads = embed_dim // args.head_dim
    run_tag = format_run_tag(args.budget_label, dt_scale, embed_dim, depth)
    per_step_train_flops = forward_flops * 3 * args.batch_size
    total_train_flops = per_step_train_flops * max_iterations
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
        "optimizer.max_iterations": max_iterations,
        "train.clip_grad_norm": 1.0,
        "train.num_rollout_steps": 1,
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
            "total_train_steps": max_iterations,
            "rollout_steps": max_iterations,
            "reference_dt_scale": reference_dt_scale,
            "expected_total_train_flops": expected_total_train_flops,
            "expected_total_inference_flops": expected_total_inference_flops,
            "dt_scale": dt_scale,
            "embed_dim": embed_dim,
            "depth": depth,
            "num_heads": num_heads,
            "aspect_ratio": aspect_ratio,
            "min_aspect_ratio": args.min_aspect_ratio,
            "max_aspect_ratio": args.max_aspect_ratio,
            "target_aspect_ratio": args.target_aspect_ratio,
            "aspect_rel_err": aspect_rel_err,
            "flop_rel_err": flop_rel_err,
            "patch_size": args.patch_size,
            "window_size": list(train_window_size),
            "forward_flops": forward_flops,
            "forward_flops_count_grid": forward_flops_count_grid,
            "rollout_budget": rollout_budget(
                forward_flops, dt_scale, train_final_time_hours, args.base_dt_hours
            ),
            "per_step_train_flops": per_step_train_flops,
            "max_iterations": max_iterations,
            "total_train_flops": total_train_flops,
            "total_inference_flops": total_inference_flops,
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
    if args.lo < args.head_dim:
        raise ValueError(
            f"--min embed_dim ({args.lo}) must be >= --head-dim ({args.head_dim})"
        )
    if args.hi < args.lo:
        raise ValueError(f"--max ({args.hi}) must be >= --min ({args.lo})")
    if args.min_depth < 1:
        raise ValueError(f"--min-depth must be >= 1, got {args.min_depth}")
    if args.max_depth < args.min_depth:
        raise ValueError(
            f"--max-depth ({args.max_depth}) must be >= --min-depth ({args.min_depth})"
        )
    if args.min_aspect_ratio <= 0:
        raise ValueError(
            f"--min-aspect-ratio must be positive, got {args.min_aspect_ratio}"
        )
    if args.max_aspect_ratio < args.min_aspect_ratio:
        raise ValueError(
            f"--max-aspect-ratio ({args.max_aspect_ratio}) must be >= "
            f"--min-aspect-ratio ({args.min_aspect_ratio})"
        )
    if args.target_aspect_ratio is None:
        args.target_aspect_ratio = 0.5 * (args.min_aspect_ratio + args.max_aspect_ratio)
    if args.target_aspect_ratio <= 0:
        raise ValueError(
            f"--target-aspect-ratio must be positive, got {args.target_aspect_ratio}"
        )

    min_dt = min(args.dt_scales)
    expected_train_flops = constant_total_train_flops(
        args.target_rollout_flops,
        args.batch_size,
        args.base_dt_hours,
    )
    expected_inference_flops = constant_total_inference_flops(
        args.target_rollout_flops,
        args.base_dt_hours,
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / f"scripts_{args.budget_label}_aspect_flops"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

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

    print(
        f"budget={args.budget_label} target_rollout_flops={args.target_rollout_flops:.3e} "
        f"train_final_time_hours={args.train_final_time_hours} base_dt_hours={args.base_dt_hours} "
        f"expected_total_train_flops={expected_train_flops:.3e} "
        f"expected_total_inference_flops={expected_inference_flops:.3e} "
        f"depth=[{args.min_depth}, {args.max_depth}] "
        f"aspect_band=[{args.min_aspect_ratio:g}, {args.max_aspect_ratio:g}] "
        f"target_aspect_ratio={args.target_aspect_ratio:g} "
        f"flop_backend={args.flop_backend} count_grid={args.grid_h}x{args.grid_w} "
        f"grid_scale={grid_scale:.4f} flop_device=cpu",
        flush=True,
    )
    header = (
        f"{'dt_scale':>10} {'embed_dim':>10} {'depth':>6} {'aspect':>8} "
        f"{'forward_flops':>14} {'F*T/dt':>14} {'rollout_steps':>14} "
        f"{'train_flops':>14} {'infer_flops':>14} {'params':>12} "
        f"{'flop_err':>9} {'asp_err':>9}"
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
        max_iterations = train_steps_for_dt(
            args.train_final_time_hours, dt_scale, args.base_dt_hours
        )
        print(
            f"searching dt_scale={dt_scale} rollout_steps={max_iterations} "
            f"target_forward(count_grid)={target_forward:.3e} ...",
            flush=True,
        )
        best = search_best_size(
            args,
            target_forward,
            window_size,
            cache,
            label=f"dt_scale={dt_scale}",
        )
        if best is None:
            print(
                f"{dt_scale:>10g} {'—':>10} {'—':>6} {'—':>8} "
                f"{'skipped':>14} (budget unreachable at this dt_scale)",
                flush=True,
            )
            continue

        forward_flops_count_grid = best.forward_flops_count_grid
        forward_flops = forward_flops_count_grid / grid_scale
        embed_dim = best.embed_dim
        depth = best.depth
        aspect_ratio = best.aspect_ratio
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
        per_step_train_flops = forward_flops * 3 * args.batch_size
        total_train_flops = per_step_train_flops * max_iterations
        total_inference_flops = forward_flops * max_iterations
        budget = rollout_budget(
            forward_flops, dt_scale, args.train_final_time_hours, args.base_dt_hours
        )
        flop_rel_err = (budget - args.target_rollout_flops) / args.target_rollout_flops
        if args.min_aspect_ratio <= aspect_ratio <= args.max_aspect_ratio:
            aspect_rel_err = 0.0
        elif aspect_ratio < args.min_aspect_ratio:
            aspect_rel_err = (aspect_ratio - args.min_aspect_ratio) / args.min_aspect_ratio
        else:
            aspect_rel_err = (aspect_ratio - args.max_aspect_ratio) / args.max_aspect_ratio

        print(
            f"{dt_scale:>10g} {embed_dim:>10d} {depth:>6d} {aspect_ratio:>8.2f} "
            f"{forward_flops:>14.3e} {budget:>14.3e} {max_iterations:>14d} "
            f"{total_train_flops:>14.3e} {total_inference_flops:>14.3e} "
            f"{num_parameters:>12d} {flop_rel_err:>+8.1%} {aspect_rel_err:>+8.1%}",
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
            max_iterations=max_iterations,
            num_parameters=num_parameters,
            reference_dt_scale=min_dt,
            train_final_time_hours=args.train_final_time_hours,
            total_inference_flops=total_inference_flops,
            expected_total_train_flops=expected_train_flops,
            expected_total_inference_flops=expected_inference_flops,
            aspect_ratio=aspect_ratio,
            flop_rel_err=flop_rel_err,
            aspect_rel_err=aspect_rel_err,
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
                "aspect_ratio": aspect_ratio,
                "min_aspect_ratio": args.min_aspect_ratio,
                "max_aspect_ratio": args.max_aspect_ratio,
                "target_aspect_ratio": args.target_aspect_ratio,
                "aspect_rel_err": aspect_rel_err,
                "flop_rel_err": flop_rel_err,
                "forward_flops": forward_flops,
                "rollout_budget": budget,
                "target_rollout_flops": args.target_rollout_flops,
                "train_final_time_hours": args.train_final_time_hours,
                "model_dt_hours": dt_scale * args.base_dt_hours,
                "total_train_steps": max_iterations,
                "rollout_steps": max_iterations,
                "total_train_flops": total_train_flops,
                "total_inference_flops": total_inference_flops,
                "expected_total_train_flops": expected_train_flops,
                "expected_total_inference_flops": expected_inference_flops,
                "reference_dt_scale": min_dt,
                "num_parameters": num_parameters,
                "run_tag": cfg["run_tag"],
            }
        )

    print("-" * len(header))
    if not rows:
        print(
            f"No configs written to {output_dir}: every dt_scale was unreachable "
            f"for target_rollout_flops={args.target_rollout_flops:.3e}.",
            flush=True,
        )
        return

    manifest = output_dir / "isoflop_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} configs to {output_dir}")
    print(f"Manifest: {manifest}")
    print(f"Submit with: bash submit_isoflop_sweep.sh -d {output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
