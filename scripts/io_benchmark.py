#!/usr/bin/env python3
"""ERA5 HDF5 read benchmark that mimics the DALI loader's per-sample reads.

Each reader process repeatedly reads one training sample: ``frames`` time steps at a stride of
``dt_scale`` hours from a random start in a year file, all 71 channels, using one of three patterns:

* full      : whole 721x1440 grid (what read_local=false does before slicing)
* lat_half  : contiguous half of the latitude rows (read_local=true with an sp1=2 split)
* lon_half  : half of the longitude columns (read_local=true with an sp2=2 split; strided reads)

Readers either spread over random training years or all hit one year file. Prints one CSV row per
(pattern, readers, files) case with aggregate and per-reader GB/s and per-sample latency percentiles.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import time

import h5py
import numpy as np

DATA = "/pscratch/sd/s/shas1693/data/weather/era5/latlon_025deg_hdf5_1h"
TRAIN_YEARS = list(range(1979, 2017))


def selection(pattern: str):
    if pattern == "full":
        return np.s_[:, :], (721, 1440)
    if pattern == "lat_half":
        return np.s_[0:360, :], (360, 1440)
    if pattern == "lon_half":
        return np.s_[:, 0:720], (721, 720)
    raise ValueError(pattern)


def reader(pattern, years, frames, dt_scale, seconds, seed, out_q):
    rng = random.Random(seed)
    sel, shape = selection(pattern)
    handles = {}
    buf = np.empty((frames, 71) + shape, dtype=np.float32)
    nbytes, lat = 0, []
    t_end = time.time() + seconds
    while time.time() < t_end:
        year = rng.choice(years)
        f = handles.get(year)
        if f is None:
            f = handles[year] = h5py.File(os.path.join(DATA, f"{year}.h5"), "r")
        d = f["data"]
        t0 = rng.randrange(0, d.shape[0] - frames * dt_scale)
        t_sel = slice(t0, t0 + frames * dt_scale, dt_scale)
        s = time.time()
        d.read_direct(buf, np.s_[t_sel, :, sel[0], sel[1]])
        lat.append(time.time() - s)
        nbytes += buf.nbytes
    for f in handles.values():
        f.close()
    out_q.put((nbytes, lat))


def run_case(pattern, n, files, frames, dt_scale, seconds):
    years = TRAIN_YEARS if files == "spread" else [2000]
    q = mp.Queue()
    procs = [mp.Process(target=reader, args=(pattern, years, frames, dt_scale, seconds, 1000 * n + i, q)) for i in range(n)]
    t = time.time()
    for p in procs:
        p.start()
    res = [q.get() for _ in procs]
    for p in procs:
        p.join()
    wall = time.time() - t
    total = sum(r[0] for r in res)
    lats = np.array([x for r in res for x in r[1]])
    agg = total / wall / 2**30
    return agg, agg / n, len(lats), np.percentile(lats, 50), np.percentile(lats, 95), lats.max()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=int, default=13)
    p.add_argument("--dt-scale", type=int, default=2)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--readers", default="1,4,16,32")
    p.add_argument("--patterns", default="full,lat_half,lon_half")
    p.add_argument("--files", default="spread,single")
    a = p.parse_args()
    mp.set_start_method("spawn")
    print("pattern,readers,files,agg_GBps,per_reader_GBps,samples,p50_s,p95_s,max_s", flush=True)
    for files in a.files.split(","):
        for pattern in a.patterns.split(","):
            for n in map(int, a.readers.split(",")):
                r = run_case(pattern, n, files, a.frames, a.dt_scale, a.seconds)
                print(f"{pattern},{n},{files},{r[0]:.2f},{r[1]:.3f},{r[2]},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f}", flush=True)


if __name__ == "__main__":
    main()
