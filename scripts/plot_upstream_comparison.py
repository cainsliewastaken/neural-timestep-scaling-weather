#!/usr/bin/env python3
"""Per-variable RMSE vs lead time: our best-dt model per compute budget against the upstream
compute-optimal one-step Swin at the same budget (Subramanian et al. 2026, Fig. 7).

Ours: latitude-weighted RMSE in physical units per IC, averaged over the eval ICs (2020), from
eval_isoflop.py; the band is +-1 standard error over ICs. Upstream: digitized Fig. 7
(scripts/digitize_upstream_fig7.py), test year 2020.

The best dt of a budget is the one with the lowest Error Norm (viz_era5_isoflop.py default metric,
read from its summary CSV) unless --best is given, e.g. --best 3T=8 6T=6.

    shifter --image=nersc/pytorch:26.01.01 python scripts/plot_upstream_comparison.py \
        --eval-root $SCRATCH/era5_expts/era5_isoflop_v2 --output-dir ../plots/isoflop_paper
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "2d_turbulence"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evaluation import paper_style as P  # noqa: E402
from evaluation.viz_2d_isoflop import CB_BLUE, CB_RED  # noqa: E402

VARS = [("u10m", "u10m (m/s)"), ("q700", "q700 (kg/kg)"), ("t850", "t850 (K)"), ("z500", r"z500 (m$^2$/s$^2$)")]
# our budget -> (upstream budget, colour)
BUDGETS = {"3T": ("3E18", CB_BLUE), "6T": ("6E18", CB_RED)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=ROOT.parent / "plots" / "isoflop_paper")
    p.add_argument("--upstream-json", type=Path, default=ROOT / "scripts" / "data" / "upstream_fig7_rmse.json")
    p.add_argument("--summary-csv", type=Path, default=None,
                   help="viz_era5_isoflop.py summary (default <output-dir>/isoflop_curves_era5_summary.csv)")
    p.add_argument("--best", nargs="*", default=None, help="override, e.g. 3T=8 6T=6")
    p.add_argument("--max-lead", type=float, default=72.0)
    return p.parse_args()


def best_dts(a):
    if a.best:
        return {k: float(v) for k, v in (x.split("=") for x in a.best)}
    path = a.summary_csv or a.output_dir / "isoflop_curves_era5_summary.csv"
    rows = list(csv.DictReader(open(path)))
    out = {}
    for b in BUDGETS:
        rb = [r for r in rows if r["budget"] == b]
        if rb:
            out[b] = float(min(rb, key=lambda r: float(r["error_norm"]))["dt"])
    return out


def our_rmse(eval_root, budget, dt):
    paths = [p for p in glob.glob(str(eval_root / f"{budget}_dt{int(dt)}p0_*" / "eval" / "isoflop_eval.npz"))
             if "-cool" not in p]
    assert len(paths) == 1, f"{budget} dt={dt}: found {paths}"
    d = np.load(paths[0])
    ch, st = list(d["channels"]), d["stds"]
    out = {}
    for v, _ in VARS:
        c = ch.index(v)
        r = np.sqrt(d["final_e_w"][:, :, c]) * st[c]  # [ic, t] per-IC RMSE
        out[v] = (r.mean(0), r.std(0, ddof=1) / np.sqrt(r.shape[0]))
    return np.asarray(d["lead_hours"]), out, str(d["run_tag"]), int(d["num_parameters"]), r.shape[0]


def main():
    a = parse_args()
    out_dir = a.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    up = json.load(open(a.upstream_json))
    best = best_dts(a)
    P.apply()

    ours = {}
    for b, dt in best.items():
        ours[b] = our_rmse(a.eval_root, b, dt)
        print(f"{b}: best dt = {dt:g} h ({ours[b][2]}, {ours[b][3] / 1e6:.1f}M params, {ours[b][4]} ICs)")

    # ---- figure 1: RMSE vs lead time ----
    fig, axs = P.subplots(2, 2, height=4.4)
    for ax, (v, label) in zip(axs.flat, VARS):
        for b, (ub, colour) in BUDGETS.items():
            if b not in ours:
                continue
            lead, rm, tag, npar, _ = ours[b]
            mean, se = rm[v]
            m = lead <= a.max_lead
            ax.plot(lead[m], mean[m], "-o", ms=2.5, color=colour,
                    label=rf"ours {ub}, $\Delta t={best[b]:g}$ h ({npar / 1e6:.1f}M)")
            ax.fill_between(lead[m], (mean - se)[m], (mean + se)[m], color=colour, alpha=0.2, lw=0)
            uh = np.array(sorted(int(h) for h in up["rmse"][ub][v]))
            uv = np.array([up["rmse"][ub][v][str(h)] for h in uh])
            mu = uh <= a.max_lead
            ax.plot(np.r_[0, uh[mu]], np.r_[0, uv[mu]], "--", color=colour,
                    label=rf"upstream {ub}, $\Delta t=6$ h one-step ({up['params'][ub]})")
        ax.set_title(label)
        ax.set_xlabel("Lead time (h)")
        ax.set_ylabel("RMSE")
        ax.set_xlim(0, a.max_lead)
        ax.set_ylim(bottom=0)
        ax.set_xticks(np.arange(0, a.max_lead + 1, 12))
        ax.grid(alpha=0.25)
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    p1 = out_dir / "era5_vs_upstream_rmse.png"
    P.save(fig, p1)
    plt.close(fig)

    # ---- figure 2: ratio ours / upstream at shared lead times ----
    fig, axs = P.subplots(1, 2, height=2.8, sharey=True)
    var_colours = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
    for ax, (b, (ub, _)) in zip(axs, BUDGETS.items()):
        if b not in ours:
            continue
        lead, rm, tag, npar, _ = ours[b]
        for (v, label), c in zip(VARS, var_colours):
            hs = [h for h in lead if 0 < h <= a.max_lead and str(int(h)) in up["rmse"][ub][v]]
            ratio = [rm[v][0][list(lead).index(h)] / up["rmse"][ub][v][str(int(h))] for h in hs]
            ax.plot(hs, ratio, "-o", ms=3, color=c, label=v)
        ax.axhline(1.0, color="0.3", lw=0.8, ls=":")
        ax.set_title(rf"{ub} FLOPs: ours ($\Delta t={best[b]:g}$ h) / upstream")
        ax.set_xlabel("Lead time (h)")
        ax.set_xlim(0, a.max_lead)
        ax.set_xticks(np.arange(0, a.max_lead + 1, 12))
        ax.grid(alpha=0.25)
    axs[0].set_ylabel(r"RMSE ratio ($<1$: ours better)")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncol=4)
    p2 = out_dir / "era5_vs_upstream_rmse_ratio.png"
    P.save(fig, p2)
    plt.close(fig)
    print(f"wrote\n  {p1}\n  {p2}")


if __name__ == "__main__":
    main()
