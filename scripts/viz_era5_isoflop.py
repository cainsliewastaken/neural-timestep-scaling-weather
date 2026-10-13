#!/usr/bin/env python3
"""ERA5 isoFLOP figures: the 2D-turbulence paper figures, drawn from eval_isoflop.py outputs.

The figure functions are imported from 2d_turbulence/evaluation/viz_2d_isoflop.py, so styling,
fits and layout match the 2D figures exactly. Only the inputs change:

  time            lead time in hours (eval horizon = 72 h; training horizon T_f = 24 h)
  Error Norm      E = mean over the model's own lead times 0 < t_k <= T_f of the IC-mean norm ||e(t_k)||.
                  t=0 (e=0 exactly) is excluded: the 2D code keeps it, which gives a model with fewer
                  steps a free zero (1/3 of the mean for dt=12 h) and biases E toward large dt
                  (--error-metric mean_incl_t0 reproduces the 2D definition); ||x|| is the latitude-weighted RMS over all 71
                  channels in normalized units (--unweighted for the plain grid RMS)
  I(t), R(t)      ||sum of one-step errors|| and ||e|| - I, as in the 2D channel figures
  alpha           E[||eps_k|| / dt], dt in hours
  Tr(dH/du)/N     Hutchinson trace of the Euler tendency (normalized units per hour)
  training FLOPs  iteration * train_flops_per_iteration from the sweep YAML

Run (login node, nersc/pytorch image):
  shifter --image=nersc/pytorch:26.01.01 python scripts/viz_era5_isoflop.py \
      --eval-root $SCRATCH/era5_expts/era5_isoflop_v2 --budgets 3T 6T --output-dir ../plots/isoflop_paper
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TURB_ROOT = ROOT.parents[1] / "2d_turbulence"
sys.path.insert(0, str(TURB_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib.axes import Axes  # noqa: E402

from evaluation import viz_2d_isoflop as V  # noqa: E402

FINAL = 10**9  # "epoch" tag of the end-of-training point in the trajectory series

# The 2D figures label time and step size without units; ERA5 uses hours.
_LABELS = {
    "Time": "Lead time (h)",
    r"Network $\Delta t$": r"Network $\Delta t$ (h)",
    r"Network step size $\Delta t$": r"Network step size $\Delta t$ (h)",
    r"$\Delta t$": r"$\Delta t$ (h)",
    r"Optimal $\Delta t$": r"Optimal $\Delta t$ (h)",
}
_set_xlabel, _set_ylabel = Axes.set_xlabel, Axes.set_ylabel
Axes.set_xlabel = lambda self, label, *a, **k: _set_xlabel(self, _LABELS.get(label, label), *a, **k)
Axes.set_ylabel = lambda self, label, *a, **k: _set_ylabel(self, _LABELS.get(label, label), *a, **k)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-root", type=Path, required=True, help="dir holding <tag>/eval/isoflop_eval.npz")
    p.add_argument("--budgets", nargs="+", default=["3T", "6T"])
    p.add_argument("--output-dir", type=Path, default=ROOT.parent / "plots" / "isoflop_paper")
    p.add_argument("--output-prefix", default="isoflop_curves_era5")
    p.add_argument("--eval-final-time", type=float, default=24.0, help="T_f in hours")
    p.add_argument("--error-curve-final-time", type=float, default=72.0)
    p.add_argument("--unweighted", action="store_true", help="plain grid RMS instead of latitude-weighted")
    p.add_argument("--error-metric", choices=["mean_excl_t0", "mean_incl_t0", "final"], default="mean_excl_t0",
                   help="scalar Error Norm: mean of ||e(t_k)|| over the model's own 0 < t_k <= T_f (default), "
                        "the same including t=0 (2D definition), or ||e(T_f)||")
    return p.parse_args()


def _norm(ms):
    """[..., C] per-channel mean squares -> RMS over channels (the field norm / sqrt(N))."""
    return np.sqrt(np.mean(ms, axis=-1))


def scalar_error(curve, n_tf, metric):
    """curve[k] = ||e(t_k)||, t_0 = 0; n_tf points up to T_f inclusive."""
    if metric == "mean_incl_t0":
        return float(np.mean(curve[:n_tf]))
    if metric == "mean_excl_t0":
        return float(np.mean(curve[1:n_tf]))
    return float(curve[n_tf - 1])


def load_run(path: Path, *, final_time: float, wkey: str, metric: str):
    d = np.load(path, allow_pickle=False)
    dt = float(d["dt_hours"])
    t = np.asarray(d["lead_hours"], dtype=float)
    n_tf = V._eval_points_for_time(dt, final_time)

    e = _norm(d[f"final_e_{wkey}"])        # [ic, t]
    inj = _norm(d[f"final_I_{wkey}"])
    res = _norm(d[f"final_R_{wkey}"])
    eps = _norm(d[f"final_eps_{wkey}"])     # one-step error norm, eps_0 = 0
    rmse_curve, eps_curve, r_curve = e.mean(0), inj.mean(0), res.mean(0)
    trace = np.asarray(d["trace"], dtype=float)

    tag = str(d["run_tag"])
    rollout = V.SeriesData(
        dt=dt,
        source_id=tag,
        time_axis=t,
        rmse_curve=rmse_curve,
        eps_curve=eps_curve,
        r_curve=r_curve,
        scalar_error=scalar_error(rmse_curve, n_tf, metric),
        epsilon_bar=float(np.mean(eps_curve[:n_tf])),
        jacobian_trace_bar=float(np.mean(trace)) if trace.size else float("nan"),
        eps_over_dt_bar=float(np.mean(eps[:, 1:n_tf])) / dt,
        epoch=FINAL,
        train_flops_so_far=float(d["traj_train_flops"][-1]),
        parameter_count=int(d["num_parameters"]),
        npz_path=None,
    )

    traj = []
    te = _norm(d[f"traj_e_{wkey}"])         # [ckpt, traj ic, t]
    n_tf_traj = min(n_tf, te.shape[-1])
    iters = np.asarray(d["traj_iters"])
    for c in range(te.shape[0]):
        curve = te[c].mean(0)
        is_final = c == te.shape[0] - 1
        traj.append(
            V.SeriesData(
                dt=dt,
                source_id=tag,
                time_axis=np.asarray(d["traj_lead_hours"], dtype=float),
                rmse_curve=curve,
                eps_curve=np.full_like(curve, np.nan),
                r_curve=np.full_like(curve, np.nan),
                scalar_error=scalar_error(curve, n_tf_traj, metric),
                epsilon_bar=float("nan"),
                jacobian_trace_bar=float("nan"),
                epoch=FINAL if is_final else int(iters[c]),
                train_flops_so_far=float(d["traj_train_flops"][c]),
                parameter_count=int(d["num_parameters"]),
            )
        )
    info = dict(tag=tag, dt=dt, total_train_flops=float(d["total_train_flops"]),
                n_ics=int(e.shape[0]), embed_dim=int(d["embed_dim"]), depth=int(d["depth"]))
    return rollout, traj, info


def rename(src: Path, dst: Path):
    if src.is_file():
        os.replace(src, dst)
        print(f"  {dst}")


def main():
    a = parse_args()
    out = a.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    wkey = "u" if a.unweighted else "w"
    tag = ("" if a.error_metric == "mean_excl_t0" else f"_{a.error_metric}") + ("_unweighted" if a.unweighted else "")
    prefix = a.output_prefix + tag
    V._plot_common_style()

    curve_order, display, flops_by_curve = [], {}, {}
    series, series_traj, rows = {}, {}, []
    for b in a.budgets:
        paths = sorted(glob.glob(str(a.eval_root / f"{b}_dt*" / "eval" / "isoflop_eval.npz")))
        paths = [p for p in paths if "-cool" not in p]
        if not paths:
            print(f"{b}: no eval files under {a.eval_root}; skipped")
            continue
        entries, traj = [], []
        for p in paths:
            r, tr, info = load_run(Path(p), final_time=a.eval_final_time, wkey=wkey, metric=a.error_metric)
            entries.append(r)
            traj.extend(tr)
            rows.append({**info, "budget": b, "error_norm": r.scalar_error, "mean_I": r.epsilon_bar,
                         "alpha": r.eps_over_dt_bar, "trace_per_dim": r.jacobian_trace_bar,
                         "e_Tf": float(np.interp(a.eval_final_time, r.time_axis, r.rmse_curve)),
                         "I_Tf": float(np.interp(a.eval_final_time, r.time_axis, r.eps_curve)),
                         "params": r.parameter_count})
        entries.sort(key=lambda s: s.dt)
        series[b], series_traj[b] = entries, traj
        # label each budget by the final training FLOPs of its optimal-dt run (as --final-train-compute)
        opt_dt = V._optimal_dt_from_rollout(entries)
        opt = V._find_matching_entry(entries, opt_dt)
        flops = float(opt.train_flops_so_far)
        curve_order.append(b)
        flops_by_curve[b] = flops
        display[b] = V._format_flops_label(flops)
        print(f"{b}: {len(entries)} runs, optimal dt={opt_dt:g} h, E={opt.scalar_error:.4f}, "
              f"final training FLOPs={flops:.3e}")

    if not curve_order:
        sys.exit("no eval files found")
    colors, markers = V._assign_curve_colors_markers(curve_order, flops_by_curve)
    legend_title = r"Final training FLOPs; fit $\log_{10}(E)=A(\log_{10}(\Delta t)-h)^2+k$"

    print("Writing figures:")
    V.plot_two_panel(series, curve_order, display, colors, markers, flops_by_curve, out, prefix,
                     compute_xlabel="Training FLOPs", legend_title=legend_title)
    print(f"  {out / f'{prefix}_2panel.png'}")
    print(f"  {V.plot_error_vs_total_flops(series_traj, curve_order, display, colors, markers, flops_by_curve, out, prefix, x_scale_to_flops=1.0)}")
    p = V.plot_optimal_dt_scaling_at_epoch(series_traj, series, curve_order, display, colors, markers,
                                           flops_by_curve, V._optimal_dt_by_curve_from_rollout(series, curve_order),
                                           out, prefix, epoch=FINAL)
    rename(p, out / f"{prefix}_optimal_dt_scaling_final.png")
    for p in V.plot_one_panel(series, curve_order, display, colors, flops_by_curve, out, prefix,
                              compute_xlabel="Training FLOPs"):
        print(f"  {p}")

    V.plot_four_panel_per_budget(series, curve_order, display, out,
                                 error_curve_final_time=a.error_curve_final_time, budget_title_suffix=" FLOPs")
    V.plot_combined_bars_and_channels(series, curve_order, display, out, eval_final_time=a.eval_final_time,
                                      panel_a="mean_I", budget_title_suffix=" FLOPs")
    V.plot_combined_bars_and_channels(series, curve_order, display, out, eval_final_time=a.eval_final_time,
                                      panel_a="eps_over_dt", budget_title_suffix=" FLOPs")
    suffix = tag
    for b in curve_order:
        stem = V._budget_file_stem(display[b])
        rename(out / f"fno_error_2d_{stem}.png", out / f"era5_error_{stem}{suffix}.png")
        rename(out / f"fno_channels_2d_{stem}.png", out / f"era5_channels_{stem}{suffix}.png")
        rename(out / f"fno_channels_eps_dt_2d_{stem}.png", out / f"era5_channels_eps_dt_{stem}{suffix}.png")

    csv_path = out / f"{prefix}_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["budget"], r["dt"])))
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
