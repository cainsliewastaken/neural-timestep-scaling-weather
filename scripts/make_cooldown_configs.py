#!/usr/bin/env python3
"""Split isoflop sweep configs into upstream's two-phase LR recipe.

For a config with ``optimizer.max_iterations = N`` this writes:

* ``<name>_main.yaml``: same run_tag, ``optimizer.max_iterations = N - floor(f * N)``,
  default ``fixed_warmup`` scheduler (warmup, then constant LR).
* ``<name>_cool.yaml``: run_tag ``<tag>-cool``, branches from the main run's
  ``ckpt_iter{N - floor(f * N)}.tar`` with ``optimizer.scheduler=cooldown``
  (1 - sqrt decay to 0 over the last ``f`` of the N iterations), as in upstream's README.

Total iterations (and so training FLOPs) stay N. Everything else is copied unchanged.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="Directory of generated sweep YAMLs.")
    p.add_argument("--dst", required=True, help="Output directory for *_main.yaml / *_cool.yaml.")
    p.add_argument("--cooldown-fraction", type=float, default=0.05, help="Upstream/paper default: 0.05.")
    p.add_argument("--run-name", default=None, help="Optional new run_name for both phases.")
    return p.parse_args()


def split_config(cfg: dict, fraction: float, run_name: str | None) -> tuple[dict, dict]:
    o = cfg["hydra_overrides"]
    total = int(o["optimizer.max_iterations"])
    cooldown_steps = int(total * fraction)
    start = total - cooldown_steps
    if run_name is not None:
        o["run_name"] = run_name
    name, tag = o["run_name"], o["run_tag"]

    main = copy.deepcopy(cfg)
    main["hydra_overrides"]["optimizer.max_iterations"] = start
    main.setdefault("metadata", {})["lr_phase"] = "main (fixed_warmup)"
    main["metadata"]["cooldown_from_iter"] = start
    main["metadata"]["total_iterations"] = total

    cool = copy.deepcopy(cfg)
    co = cool["hydra_overrides"]
    co["run_tag"] = f"{tag}-cool"
    co["train.branch_from"] = f"/expts/{name}/{tag}/checkpoints/ckpt_iter{start}.tar"
    co["optimizer.scheduler"] = "cooldown"
    co["optimizer.cooldown_from_iter"] = start
    co["optimizer.cooldown_to_iter"] = total
    co["optimizer.cooldown_fraction"] = fraction
    cool.setdefault("metadata", {})["lr_phase"] = "cooldown (1-sqrt)"
    cool["metadata"]["cooldown_from_iter"] = start
    cool["metadata"]["total_iterations"] = total
    return main, cool


def main() -> None:
    args = parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        if "hydra_overrides" not in cfg:
            continue
        main_cfg, cool_cfg = split_config(cfg, args.cooldown_fraction, args.run_name)
        for suffix, c in (("main", main_cfg), ("cool", cool_cfg)):
            out = dst / f"{path.stem}_{suffix}.yaml"
            out.write_text(yaml.safe_dump(c, sort_keys=False))
        m = main_cfg["metadata"]
        print(f"{path.stem}: main 0->{m['cooldown_from_iter']}, cooldown {m['cooldown_from_iter']}->{m['total_iterations']}")


if __name__ == "__main__":
    main()
