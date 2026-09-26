#!/usr/bin/env python3
"""Taylor-expansion timescales of the raw ERA5 state, per channel.

    x(t + dt) = x + dt * x' + dt^2 / 2 * x'' + ...

Each term is smaller than the last by roughly dt / tau, with tau a ratio of successive
time derivatives:

    tau_1 = sigma(x) / sigma(x')     Taylor microscale: time for the linear term to move
                                     the state by one standard deviation
    tau_2 = sigma(x') / sigma(x'')   Euler scale: forward Euler's local error relative to
                                     its step is ~ dt / (2 tau_2)

Derivatives use finite differences of the raw state (no climatology removed), centred on
the same time so sigma(x') and sigma(x'') describe the same instants:

    step h = 1 h (frames t0, t0+1, t0+2):  x' ~ (x2 - x0) / 2,  x'' ~ x2 - 2 x1 + x0
    step h = 2 h (frames t0, t0+2, t0+4):  x' ~ (x4 - x0) / 4,  x'' ~ (x4 - 2 x2 + x0) / 4

If the h = 1 h and h = 2 h estimates agree, hourly data resolves the scale. Variance of
x'' is also split by UTC hour of its centre, to spot ERA5 4D-Var window boundaries
(09 / 21 UTC), which would inflate second differences.

Every channel sigma is the root of the mean over points of the per-point temporal variance
(unweighted, as upstream's stats; a cos-lat weighted tau_2 is reported too), so spatial
structure never enters. For the raw state sigma(x) still includes the seasonal and
diurnal cycles. Input is the partials written by
``compute_tendency_stats.py accumulate`` (coarse grid, raw frames at lags 0-4 h).

Outputs <out>/taylor_scales_<tag>.h5 and taylor_scales_summary_<tag>.csv.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_tendency_stats import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR, LAGS_HOURS  # noqa: E402

NEEDED_LAGS = (0, 1, 2, 4)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory holding partials/ from compute_tendency_stats.py.")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Only used for channel names.")
    p.add_argument("--tag", default="v1.0")
    return p.parse_args()


class Moments:
    """Running sums for mean / variance, per channel (pooled) and per point."""

    def __init__(self, shape_point):
        self.n = 0
        self.s1 = np.zeros(shape_point)
        self.s2 = np.zeros(shape_point)

    def add(self, v: np.ndarray) -> None:  # v: [samples, C, H, W]
        self.n += v.shape[0]
        self.s1 += v.sum(axis=0)
        self.s2 += np.square(v).sum(axis=0)

    def var_point(self) -> np.ndarray:
        m = self.s1 / self.n
        return np.maximum(self.s2 / self.n - m**2, 0.0)

    def var_channel(self, w: np.ndarray | None = None) -> np.ndarray:
        """Mean over points of the per-point temporal variance (no spatial variance);
        w [H] optional latitude weights."""
        wt = np.ones(self.s1.shape[1]) if w is None else w
        return np.einsum("chw,h->c", self.var_point(), wt) / (wt.sum() * self.s1.shape[2])


def main() -> None:
    args = parse_args()
    parts = sorted(glob.glob(os.path.join(args.out_dir, "partials", "part_*.npz")))
    assert parts, f"no partials in {args.out_dir}/partials"
    idx = {lag: LAGS_HOURS.index(lag) for lag in NEEDED_LAGS}

    with np.load(parts[0]) as z:
        n_ch, hc, wc = z["coarse"].shape[2:]
        lat_c = z["lat_coarse"]
    shape = (n_ch, hc, wc)
    w = np.cos(np.deg2rad(lat_c))

    state = Moments(shape)  # x at t0 (temporal std per point)
    d1 = {1: Moments(shape), 2: Moments(shape)}  # centred first derivative, step h
    d2 = {1: Moments(shape), 2: Moments(shape)}  # second derivative, step h
    d2_hour_s2 = np.zeros((24, n_ch))  # sum of (x'')^2 by UTC hour of centre, h = 1
    d2_hour_n = np.zeros(24)

    t_start = time.time()
    n_anchor = 0
    for k, path in enumerate(parts):
        with np.load(path) as z:
            coarse = z["coarse"]  # [A, L, C, H, W] raw state
            hours = z["hours"]  # [A, L] hours since 1900-01-01 (UTC)
        x = {lag: coarse[:, idx[lag]].astype(np.float64) for lag in NEEDED_LAGS}
        del coarse
        n_anchor += x[0].shape[0]

        state.add(x[0])
        d1[1].add((x[2] - x[0]) / 2.0)
        d2[1].add(x[2] - 2.0 * x[1] + x[0])
        d1[2].add((x[4] - x[0]) / 4.0)
        d2[2].add((x[4] - 2.0 * x[2] + x[0]) / 4.0)

        centre_hour = hours[:, idx[1]] % 24
        sq = np.square(x[2] - 2.0 * x[1] + x[0]).mean(axis=(2, 3))  # [A, C]
        for a, hr in enumerate(centre_hour):
            d2_hour_s2[hr] += sq[a]
            d2_hour_n[hr] += 1
        print(f"partial {k + 1}/{len(parts)} ({n_anchor} anchors) {time.time() - t_start:.0f}s", flush=True)

    with h5py.File(sorted(glob.glob(os.path.join(args.data_dir, "*.h5")))[0], "r") as f:
        channels = [c.decode("utf-8").strip("\x00") for c in f["channel"][:]]
    assert len(channels) == n_ch

    sig_x = np.sqrt(state.var_channel())
    sig_d1 = {h: np.sqrt(d1[h].var_channel()) for h in (1, 2)}
    sig_d2 = {h: np.sqrt(d2[h].var_channel()) for h in (1, 2)}
    tau1 = {h: sig_x / sig_d1[h] for h in (1, 2)}
    tau2 = {h: sig_d1[h] / sig_d2[h] for h in (1, 2)}
    tau2_w = np.sqrt(d1[1].var_channel(w)) / np.sqrt(d2[1].var_channel(w))
    with np.errstate(divide="ignore", invalid="ignore"):
        tau2_point = np.sqrt(d1[1].var_point()) / np.sqrt(d2[1].var_point())
    hour_rms = np.sqrt(d2_hour_s2 / np.maximum(d2_hour_n, 1)[:, None])  # [24, C]
    hour_ratio = hour_rms / np.median(hour_rms, axis=0, keepdims=True)

    os.makedirs(args.out_dir, exist_ok=True)
    h5_path = os.path.join(args.out_dir, f"taylor_scales_{args.tag}.h5")
    with h5py.File(h5_path, "w") as f:
        f.attrs["tag"] = args.tag
        f.attrs["num_samples"] = np.int64(n_anchor)
        f.attrs["description"] = (
            "Raw-state Taylor scales in hours: tau_1 = sigma(x)/sigma(x'), tau_2 = sigma(x')/sigma(x''), "
            "centred finite differences with step h (1 or 2 h). Channel sigmas average the per-point temporal "
            "variance over points, unweighted (tau_2_area_weighted uses cos-lat). hour_ratio = rms(x'') by UTC hour of centre / median over hours."
        )
        f.create_dataset("channel", data=np.asarray(channels, dtype="S5"))
        f.create_dataset("latitude", data=lat_c)
        for h in (1, 2):
            f.create_dataset(f"tau_1_h{h}", data=tau1[h].astype(np.float32))
            f.create_dataset(f"tau_2_h{h}", data=tau2[h].astype(np.float32))
            f.create_dataset(f"sigma_dxdt_h{h}", data=sig_d1[h].astype(np.float32))
            f.create_dataset(f"sigma_d2xdt2_h{h}", data=sig_d2[h].astype(np.float32))
        f.create_dataset("sigma_x", data=sig_x.astype(np.float32))
        f.create_dataset("tau_2_area_weighted_h1", data=tau2_w.astype(np.float32))
        f.create_dataset("tau_2_point_h1", data=tau2_point.astype(np.float32))
        f.create_dataset("hour_ratio_d2", data=hour_ratio.astype(np.float32))
        f.create_dataset("hour_count", data=d2_hour_n.astype(np.int64))

    csv_path = os.path.join(args.out_dir, f"taylor_scales_summary_{args.tag}.csv")
    with open(csv_path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["channel", "tau_2_h1", "tau_2_h2", "tau_2_area_weighted_h1", "tau_2_point_median_h1",
                     "tau_1_h1", "tau_1_h2", "d2_rms_ratio_max_hour", "d2_rms_ratio_max"])
        for c, name in enumerate(channels):
            wr.writerow([name, tau2[1][c], tau2[2][c], tau2_w[c], np.nanmedian(tau2_point[c]),
                         tau1[1][c], tau1[2][c], int(np.argmax(hour_ratio[:, c])), hour_ratio[:, c].max()])
    print(f"wrote {h5_path}\nwrote {csv_path}", flush=True)

    order = np.argsort(tau2[1])
    print("\nchannel   tau_2[h] (h=1)  (h=2)   h2/h1   tau_1[h] (h=1)   max x'' hour (ratio)")
    for c in order:
        print(f"{channels[c]:<8} {tau2[1][c]:>12.2f} {tau2[2][c]:>7.2f} {tau2[2][c] / tau2[1][c]:>7.3f}"
              f" {tau1[1][c]:>14.1f}   {int(np.argmax(hour_ratio[:, c])):02d} UTC ({hour_ratio[:, c].max():.2f})")
    print(f"\nmin tau_2 over channels (h=1): {tau2[1].min():.2f} h ({channels[order[0]]})")
    print("rms(x'') by UTC hour, median over channels, relative to each channel's median hour:")
    print("  " + " ".join(f"{hr:02d}:{np.median(hour_ratio[hr]):.2f}" for hr in range(24)))


if __name__ == "__main__":
    main()
