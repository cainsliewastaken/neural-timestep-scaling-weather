#!/usr/bin/env python3
"""Channel-wise temporal-increment stats and decorrelation times for hourly ERA5 HDF5.

Follows upstream's stats_<tag>.h5 conventions: channel-wise statistics, test years
2018-2022 excluded (validation year 2017 included), same ``temp_diff_*`` naming.

Random anchor times t0 are drawn from the non-test years; for each anchor the frames
x(t0 + L) are read for every lag L in LAGS_HOURS. One set of reads serves both outputs:

  * Euler normalization (full resolution, L <= max dt_scale): per-channel mean / std
    of the raw increment d = x(t0 + dt) - x(t0) for each dt in DT_SCALES, pooled over
    all grid points and anchors (unweighted and cos-lat weighted). State moments at t0
    are also accumulated to check which convention upstream's global_mean/std used.
  * Decorrelation (coarse grid, every --coarse-stride points, all lags): anomalies from
    a per-point climatology fit (trend + annual and diurnal harmonics + their
    interaction), autocorrelation rho(L) per point and pooled per channel, and the
    timescales tau_e (1/e crossing), integral time (to first zero crossing) and Taylor
    microscale (from rho at 1 h).

Two phases so it fans out over Slurm tasks without MPI:

  accumulate  task SLURM_PROCID of SLURM_NTASKS processes every world-th anchor and
              writes <out>/partials/part_<rank>_of_<world>.npz
  finalize    combines the partials and writes
                tendency_stats_<tag>.h5   read by models/helpers.py (model.tendency_stats)
                decorrelation_<tag>.h5    autocorrelations and timescale maps
                decorrelation_summary_<tag>.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import time

import h5py
import numpy as np

DEFAULT_DATA_DIR = "/pscratch/sd/s/shas1693/data/weather/era5/latlon_025deg_hdf5_1h"
DEFAULT_OUT_DIR = os.path.join(os.environ.get("SCRATCH", "."), "era5_stats")
EXCLUDE_YEARS = [2018, 2019, 2020, 2021, 2022]
DT_SCALES = [1, 2, 3, 4, 6, 8, 12]
LAGS_HOURS = [0, 1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48, 72, 96, 120, 168, 240, 336, 480, 720]
NLAT = 720  # the loaders drop the south pole row
HOURS_PER_YEAR = 365.2425 * 24.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=("accumulate", "finalize"))
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--tag", default="v1.0", help="Matches data.tag / upstream stats_<tag>.h5.")
    p.add_argument("--n-anchors", type=int, default=512, help="Random anchor times (upstream stats used 512 samples).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--coarse-stride", type=int, default=8, help="Grid stride for the decorrelation analysis.")
    p.add_argument("--rank", type=int, default=int(os.environ.get("SLURM_PROCID", 0)))
    p.add_argument("--world", type=int, default=int(os.environ.get("SLURM_NTASKS", 1)))
    p.add_argument("--chunk", type=int, default=20000, help="Grid points per chunk in the climatology fit.")
    return p.parse_args()


def _year(path: str) -> int:
    return int(os.path.basename(path).replace(".h5", ""))


class Timeline:
    """Hourly frames of the non-test years as one contiguous global index."""

    def __init__(self, data_dir: str, exclude_years: list[int]):
        paths = sorted(glob.glob(os.path.join(data_dir, "*.h5")), key=_year)
        self.paths = [q for q in paths if _year(q) not in exclude_years]
        self.hours = []  # hours since 1900-01-01 per year
        for q in self.paths:
            with h5py.File(q, "r") as f:
                self.hours.append(f["time"][:].astype(np.int64))
        all_hours = np.concatenate(self.hours)
        assert np.all(np.diff(all_hours) == 1), "timeline is not hourly-contiguous across the kept years"
        self.first_hour = int(all_hours[0])
        self.offsets = np.cumsum([0] + [len(h) for h in self.hours])
        self.total = int(self.offsets[-1])
        self._files: dict[int, h5py.File] = {}
        with h5py.File(self.paths[0], "r") as f:
            self.channels = [c.decode("utf-8").strip("\x00") for c in f["channel"][:]]
            self.latitudes = f["latitude"][:NLAT].astype(np.float64)
            self.longitudes = f["longitude"][:].astype(np.float64)

    def _data(self, i: int):
        if i not in self._files:
            self._files[i] = h5py.File(self.paths[i], "r")
        return self._files[i]["data"]

    def _locate(self, g: int) -> tuple[int, int]:
        i = int(np.searchsorted(self.offsets, g, side="right") - 1)
        return i, g - int(self.offsets[i])

    def hour(self, g: int) -> int:
        """Hours since 1900-01-01 of global index g."""
        return self.first_hour + g

    def read_full(self, g: int) -> np.ndarray:
        i, t = self._locate(g)
        return self._data(i)[t][:, :NLAT, :]  # one contiguous read, [C, 720, 1440]

    def read_coarse(self, g: int, stride: int) -> np.ndarray:
        i, t = self._locate(g)
        rows = self._data(i)[t, :, 0:NLAT:stride, :]  # whole rows, then subsample lon
        return rows[..., ::stride]


def _moments(d: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel sum and sum of squares of d [C, H, W]: unweighted and lat-weighted (w [H])."""
    s1 = d.sum(axis=2, dtype=np.float64)  # [C, H]
    s2 = np.square(d, dtype=np.float64).sum(axis=2)
    return s1.sum(axis=1), s2.sum(axis=1), s1 @ w, s2 @ w


def anchor_times(tl: Timeline, n_anchors: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.integers(0, tl.total - max(LAGS_HOURS), size=n_anchors))


def accumulate(args: argparse.Namespace) -> None:
    tl = Timeline(args.data_dir, EXCLUDE_YEARS)
    anchors = anchor_times(tl, args.n_anchors, args.seed)[args.rank :: args.world]
    stride = args.coarse_stride
    n_ch = len(tl.channels)
    w_full = np.cos(np.deg2rad(tl.latitudes))
    lat_c = tl.latitudes[::stride]
    lon_c = tl.longitudes[::stride]
    max_dt = max(DT_SCALES)

    with h5py.File(os.path.join(args.data_dir, "stats", f"stats_{args.tag}.h5"), "r") as f:
        ref_mean = f["global_mean"][:].astype(np.float32)  # shift for state moments (precision)

    coarse = np.zeros((len(anchors), len(LAGS_HOURS), n_ch, len(lat_c), len(lon_c)), dtype=np.float32)
    hours = np.zeros((len(anchors), len(LAGS_HOURS)), dtype=np.int64)
    inc = np.zeros((4, len(DT_SCALES), n_ch))  # s1, s2, s1_w, s2_w
    state = np.zeros((4, n_ch))

    t_start = time.time()
    for a, g0 in enumerate(anchors):
        full = {}
        for j, lag in enumerate(LAGS_HOURS):
            g = int(g0) + lag
            hours[a, j] = tl.hour(g)
            if lag <= max_dt:
                full[lag] = tl.read_full(g)
                coarse[a, j] = full[lag][:, ::stride, ::stride]
            else:
                coarse[a, j] = tl.read_coarse(g, stride)
        x0 = full[0]
        state += np.stack(_moments(x0 - ref_mean[:, None, None], w_full))
        for k, dt in enumerate(DT_SCALES):
            inc[:, k] += np.stack(_moments(full[dt] - x0, w_full))
        print(
            f"[rank {args.rank}] anchor {a + 1}/{len(anchors)} (global index {g0}) "
            f"{time.time() - t_start:.0f}s",
            flush=True,
        )

    n_points = NLAT * len(tl.longitudes)
    out = os.path.join(args.out_dir, "partials")
    os.makedirs(out, exist_ok=True)
    np.savez(
        os.path.join(out, f"part_{args.rank:04d}_of_{args.world:04d}.npz"),
        anchors=anchors,
        hours=hours,
        coarse=coarse,
        inc=inc,
        state=state,
        count=np.array([len(anchors) * n_points, len(anchors) * len(tl.longitudes) * w_full.sum()]),
        ref_mean=ref_mean,
        lat_coarse=lat_c,
        lon_coarse=lon_c,
    )
    print(f"[rank {args.rank}] done: {len(anchors)} anchors in {time.time() - t_start:.0f}s", flush=True)


def climatology_basis(hours_since_1900: np.ndarray, first_hour: int) -> np.ndarray:
    """Trend + annual (3 harmonics) + diurnal (2 harmonics) + annual-diurnal interaction."""
    t = (hours_since_1900 - first_hour).astype(np.float64)
    ann = 2 * np.pi * t / HOURS_PER_YEAR
    diu = 2 * np.pi * (t % 24) / 24.0
    cols = [np.ones_like(t), t / t.max()]
    for k in (1, 2, 3):
        cols += [np.cos(k * ann), np.sin(k * ann)]
    for k in (1, 2):
        cols += [np.cos(k * diu), np.sin(k * diu)]
    for fd in (np.cos(diu), np.sin(diu)):
        cols += [fd * np.cos(ann), fd * np.sin(ann)]
    return np.stack(cols, axis=1)


def tau_efold(rho: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """First crossing of 1/e (interpolated in log rho); nan if rho stays above 1/e."""
    thresh = np.exp(-1.0)
    below = rho < thresh
    j = np.argmax(below, axis=-1)
    has = below.any(axis=-1) & (j > 0)
    j = np.clip(j, 1, len(lags) - 1)
    r0 = np.take_along_axis(rho, (j - 1)[..., None], axis=-1)[..., 0]
    r1 = np.take_along_axis(rho, j[..., None], axis=-1)[..., 0]
    l0, l1 = lags[j - 1], lags[j]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_log = (np.log(r0) + 1.0) / (np.log(r0) - np.log(r1))
        frac_lin = (r0 - thresh) / (r0 - r1)
    frac = np.where(r1 > 0, frac_log, frac_lin)
    return np.where(has, l0 + frac * (l1 - l0), np.nan)


def tau_integral(rho: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Trapezoid integral of rho up to its first non-positive lag (lower bound if none)."""
    positive = np.cumprod(rho > 0, axis=-1).astype(bool)
    r = np.where(positive, rho, 0.0)
    return np.sum(0.5 * (r[..., 1:] + r[..., :-1]) * np.diff(lags) * positive[..., 1:], axis=-1)


def tau_taylor(rho_1h: np.ndarray) -> np.ndarray:
    """Taylor microscale from rho(1 h) = 1 - (1 h)^2 / (2 tau_T^2)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(rho_1h < 1, 1.0 / np.sqrt(2.0 * (1.0 - rho_1h)), np.nan)


def finalize(args: argparse.Namespace) -> None:
    tl = Timeline(args.data_dir, EXCLUDE_YEARS)
    parts = sorted(glob.glob(os.path.join(args.out_dir, "partials", "part_*.npz")))
    assert parts, f"no partials in {args.out_dir}/partials"
    loaded = [np.load(q) for q in parts]
    anchors = np.concatenate([z["anchors"] for z in loaded])
    expected = anchor_times(tl, args.n_anchors, args.seed)
    assert np.array_equal(np.sort(anchors), expected), "partials do not cover the expected anchor set"
    n_ch = len(tl.channels)
    lags = np.asarray(LAGS_HOURS, dtype=np.float64)

    # ---- increment and state stats (full resolution) ----
    inc = sum(z["inc"] for z in loaded)
    state = sum(z["state"] for z in loaded)
    n_unw, n_w = sum(z["count"] for z in loaded)
    ref_mean = loaded[0]["ref_mean"].astype(np.float64)

    def mean_std(s1, s2, n):
        m = s1 / n
        return m, np.sqrt(np.maximum(s2 / n - m**2, 0.0))

    diff_mean, diff_std = mean_std(inc[0], inc[1], n_unw)
    diff_mean_w, diff_std_w = mean_std(inc[2], inc[3], n_w)
    st_mean, st_std = mean_std(state[0], state[1], n_unw)
    st_mean_w, st_std_w = mean_std(state[2], state[3], n_w)
    st_mean, st_mean_w = st_mean + ref_mean, st_mean_w + ref_mean

    with h5py.File(os.path.join(args.data_dir, "stats", f"stats_{args.tag}.h5"), "r") as f:
        up_mean = f["global_mean"][:].astype(np.float64)
        up_std = f["global_std"][:].astype(np.float64)
        up_diff = {k: f[k][:].astype(np.float64) for k in ("temp_diff_std", "temp_diff_std_length_02")}

    os.makedirs(args.out_dir, exist_ok=True)
    stats_path = os.path.join(args.out_dir, f"tendency_stats_{args.tag}.h5")
    with h5py.File(stats_path, "w") as f:
        f.attrs["tag"] = args.tag
        f.attrs["exclude_years"] = np.asarray(EXCLUDE_YEARS, dtype=np.int32)
        f.attrs["num_samples"] = np.int64(len(anchors))
        f.attrs["seed"] = np.int64(args.seed)
        f.attrs["description"] = (
            "Raw increment x(t + dt_scale * 1h) - x(t) per channel, pooled over all grid points "
            "(south pole row dropped) and anchor times. temp_diff_* unweighted, *_area_weighted cos-lat."
        )
        f.create_dataset("channel", data=np.asarray(tl.channels, dtype="S5"))
        f.create_dataset("dt_scales", data=np.asarray(DT_SCALES, dtype=np.int32))
        f.create_dataset("temp_diff_mean", data=diff_mean.astype(np.float32))
        f.create_dataset("temp_diff_std", data=diff_std.astype(np.float32))
        f.create_dataset("temp_diff_mean_area_weighted", data=diff_mean_w.astype(np.float32))
        f.create_dataset("temp_diff_std_area_weighted", data=diff_std_w.astype(np.float32))
        f.create_dataset("state_mean", data=st_mean.astype(np.float32))
        f.create_dataset("state_std", data=st_std.astype(np.float32))
        f.create_dataset("state_mean_area_weighted", data=st_mean_w.astype(np.float32))
        f.create_dataset("state_std_area_weighted", data=st_std_w.astype(np.float32))
    print(f"wrote {stats_path}", flush=True)

    # ---- decorrelation (coarse grid) ----
    coarse_shape = loaded[0]["coarse"].shape[2:]  # C, hc, wc
    n_frames_per_anchor = len(LAGS_HOURS)
    n_points = int(np.prod(coarse_shape))
    n_anchor = len(anchors)
    y = np.empty((n_anchor, n_frames_per_anchor, n_points), dtype=np.float32)
    hours = np.empty((n_anchor, n_frames_per_anchor), dtype=np.int64)
    row = 0
    for z in loaded:
        k = len(z["anchors"])
        y[row : row + k] = z["coarse"].reshape(k, n_frames_per_anchor, n_points)
        hours[row : row + k] = z["hours"]
        row += k
    lat_c, lon_c = loaded[0]["lat_coarse"], loaded[0]["lon_coarse"]
    del loaded

    x = climatology_basis(hours.reshape(-1), tl.first_hour)  # [A*L, nb]
    # least-squares via pseudo-inverse: stays defined when few anchors leave the basis rank-deficient
    x_pinv = np.linalg.pinv(x)
    cov = np.empty((n_frames_per_anchor, n_points))
    var = np.empty(n_points)
    y2 = y.reshape(-1, n_points)
    for c0 in range(0, n_points, args.chunk):
        c1 = min(c0 + args.chunk, n_points)
        yc = y2[:, c0:c1].astype(np.float64)
        beta = x_pinv @ yc
        anom = (yc - x @ beta).reshape(n_anchor, n_frames_per_anchor, c1 - c0)
        var[c0:c1] = np.mean(anom**2, axis=(0, 1))
        cov[:, c0:c1] = np.mean(anom[:, :1] * anom, axis=0)
    del y, y2

    cov = cov.reshape(n_frames_per_anchor, *coarse_shape)
    var = var.reshape(coarse_shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_point = np.moveaxis(cov / var, 0, -1)  # [C, hc, wc, L]
    rho_point[..., 0] = 1.0
    w_c = np.cos(np.deg2rad(lat_c))[None, :, None]
    rho_ch = (cov * w_c).sum(axis=(2, 3)) / (var * w_c).sum(axis=(1, 2))  # [L, C]
    rho_ch = rho_ch.T  # [C, L]
    rho_ch[:, 0] = 1.0

    i1 = LAGS_HOURS.index(1)
    te_ch, ti_ch, tt_ch = tau_efold(rho_ch, lags), tau_integral(rho_ch, lags), tau_taylor(rho_ch[:, i1])
    te_pt, ti_pt, tt_pt = tau_efold(rho_point, lags), tau_integral(rho_point, lags), tau_taylor(rho_point[..., i1])

    decor_path = os.path.join(args.out_dir, f"decorrelation_{args.tag}.h5")
    with h5py.File(decor_path, "w") as f:
        f.attrs["tag"] = args.tag
        f.attrs["exclude_years"] = np.asarray(EXCLUDE_YEARS, dtype=np.int32)
        f.attrs["num_samples"] = np.int64(n_anchor)
        f.attrs["coarse_stride"] = np.int64(args.coarse_stride)
        f.attrs["description"] = (
            "Anomaly autocorrelation (per-point fit of trend + annual/diurnal harmonics removed). "
            "acf_channel pools covariance and variance with cos-lat weights. Timescales in hours; "
            "tau_efold nan = rho stays above 1/e out to the max lag; tau_integral stops at the first "
            "non-positive lag (lower bound if none)."
        )
        f.create_dataset("channel", data=np.asarray(tl.channels, dtype="S5"))
        f.create_dataset("lags_hours", data=lags)
        f.create_dataset("latitude", data=lat_c)
        f.create_dataset("longitude", data=lon_c)
        f.create_dataset("acf_channel", data=rho_ch.astype(np.float32))
        f.create_dataset("acf_point", data=rho_point.astype(np.float32))
        f.create_dataset("anomaly_variance_point", data=var.astype(np.float32))
        for name, val in (("tau_efold", te_ch), ("tau_integral", ti_ch), ("tau_taylor", tt_ch)):
            f.create_dataset(f"{name}_channel", data=val.astype(np.float32))
        for name, val in (("tau_efold", te_pt), ("tau_integral", ti_pt), ("tau_taylor", tt_pt)):
            f.create_dataset(f"{name}_point", data=val.astype(np.float32))
    print(f"wrote {decor_path}", flush=True)

    # ---- summary ----
    k6, k12 = DT_SCALES.index(6), DT_SCALES.index(12)
    summary_path = os.path.join(args.out_dir, f"decorrelation_summary_{args.tag}.csv")
    fields = [
        "channel",
        "tau_efold_h", "tau_integral_h", "tau_taylor_h", "tau_efold_point_median_h",
        "state_std", "state_std_area_weighted", "upstream_global_std",
        "temp_diff_std_6h", "upstream_temp_diff_std",
        "temp_diff_std_12h", "upstream_temp_diff_std_length_02",
    ] + [f"temp_diff_std_{dt}h" for dt in DT_SCALES]
    with open(summary_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)
        for c, name in enumerate(tl.channels):
            writer.writerow(
                [name, te_ch[c], ti_ch[c], tt_ch[c], np.nanmedian(te_pt[c]),
                 st_std[c], st_std_w[c], up_std[c],
                 diff_std[k6, c], up_diff["temp_diff_std"][c],
                 diff_std[k12, c], up_diff["temp_diff_std_length_02"][c]]
                + [diff_std[k, c] for k in range(len(DT_SCALES))]
            )
    print(f"wrote {summary_path}", flush=True)

    def rel(a, b):
        return np.abs(a - b) / np.abs(b)

    print("\nupstream convention check (median relative difference over channels):")
    print(f"  state mean  unweighted {np.median(rel(st_mean, up_mean)):.3%}  area-weighted {np.median(rel(st_mean_w, up_mean)):.3%}")
    print(f"  state std   unweighted {np.median(rel(st_std, up_std)):.3%}  area-weighted {np.median(rel(st_std_w, up_std)):.3%}")
    print(f"  temp_diff_std           vs our 6h:  {np.median(rel(diff_std[k6], up_diff['temp_diff_std'])):.3%}"
          f"  vs our 1h: {np.median(rel(diff_std[0], up_diff['temp_diff_std'])):.3%}")
    print(f"  temp_diff_std_length_02 vs our 12h: {np.median(rel(diff_std[k12], up_diff['temp_diff_std_length_02'])):.3%}"
          f"  vs our 2h: {np.median(rel(diff_std[1], up_diff['temp_diff_std_length_02'])):.3%}")
    print("\nchannel    tau_e[h]  tau_int[h]  tau_T[h]")
    for c, name in enumerate(tl.channels):
        print(f"{name:<8} {te_ch[c]:>9.1f} {ti_ch[c]:>11.1f} {tt_ch[c]:>9.1f}")


def main() -> None:
    args = parse_args()
    if args.phase == "accumulate":
        accumulate(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
