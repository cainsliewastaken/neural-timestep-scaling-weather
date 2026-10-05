#!/usr/bin/env python3
"""Byte-for-byte copy of a large file into a pre-striped Lustre target, then verify it.

The target must already exist with the wanted layout (``lfs setstripe -c 32 -S 4M <dst>``).
``--workers`` processes each copy one contiguous byte range with pread/pwrite, which keeps the
number of concurrent readers on the single-stripe source low (it collapses at ~16 readers).
Verification compares file sizes and ``--verify-steps`` random time steps of the HDF5 ``data``
dataset between source and copy.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import sys
import time

CHUNK = 64 * 2**20


def copy_range(src: str, dst: str, start: int, end: int, idx: int, q) -> None:
    fs, fd = os.open(src, os.O_RDONLY), os.open(dst, os.O_WRONLY)
    off, last = start, time.time()
    while off < end:
        n = min(CHUNK, end - off)
        buf = os.pread(fs, n, off)
        if len(buf) != n:
            raise IOError(f"worker {idx}: short read at {off}")
        os.pwrite(fd, buf, off)
        off += n
        if time.time() - last > 60:
            q.put((idx, off - start, end - start))
            last = time.time()
    os.fsync(fd)
    os.close(fs)
    os.close(fd)
    q.put((idx, end - start, end - start))


def verify(src: str, dst: str, steps: int) -> bool:
    import h5py
    import numpy as np

    if os.path.getsize(src) != os.path.getsize(dst):
        print("VERIFY FAIL: sizes differ", flush=True)
        return False
    with h5py.File(src, "r") as a, h5py.File(dst, "r") as b:
        da, db = a["data"], b["data"]
        if da.shape != db.shape or da.dtype != db.dtype:
            print(f"VERIFY FAIL: shape/dtype {da.shape}/{db.shape}", flush=True)
            return False
        ts = sorted({0, da.shape[0] - 1} | set(random.Random(0).sample(range(da.shape[0]), steps)))
        for t in ts:
            if not np.array_equal(da[t], db[t]):
                print(f"VERIFY FAIL: time step {t} differs", flush=True)
                return False
        for name in a:
            if name != "data" and not np.array_equal(a[name][()], b[name][()]):
                print(f"VERIFY FAIL: dataset {name} differs", flush=True)
                return False
    print(f"VERIFY OK: size {os.path.getsize(dst)} bytes, {len(ts)} time steps and all other datasets identical", flush=True)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--verify-steps", type=int, default=24)
    a = p.parse_args()
    size = os.path.getsize(a.src)
    with open(a.dst, "r+b") as f:
        f.truncate(size)
    bounds = [size * i // a.workers for i in range(a.workers + 1)]
    bounds = [b - b % CHUNK if 0 < b < size else b for b in bounds]
    q = mp.Queue()
    procs = [mp.Process(target=copy_range, args=(a.src, a.dst, bounds[i], bounds[i + 1], i, q)) for i in range(a.workers)]
    t0 = time.time()
    for pr in procs:
        pr.start()
    done = {}
    while any(pr.is_alive() for pr in procs) or not q.empty():
        try:
            idx, d, tot = q.get(timeout=5)
            done[idx] = d
            copied = sum(done.values())
            print(f"{time.strftime('%H:%M:%S')} copied {copied / 2**40:.2f} / {size / 2**40:.2f} TiB "
                  f"({100 * copied / size:.0f}%), {copied / (time.time() - t0) / 2**30:.2f} GiB/s", flush=True)
        except Exception:
            pass
    for pr in procs:
        pr.join()
    if any(pr.exitcode != 0 for pr in procs):
        print("COPY FAIL: a worker exited with an error", flush=True)
        sys.exit(1)
    print(f"COPY DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
    sys.exit(0 if verify(a.src, a.dst, a.verify_steps) else 2)


if __name__ == "__main__":
    main()
