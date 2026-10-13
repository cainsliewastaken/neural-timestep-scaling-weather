#!/usr/bin/env python3
"""Read the RMSE-vs-lead-time curves out of the upstream paper's Fig. 7 (an SVG) into JSON.

Source: Subramanian et al. 2026, "On Neural Scaling Laws for Weather Emulation through Continual
Training", arXiv:2603.25687v1, Fig. 7 (rmse_vs_time_flops_rev.svg): RMSE vs lead time of the closest
compute-optimal Swin at each compute budget, for u10m, q700, t850 and z500.

The curves are matplotlib paths in plot coordinates; each panel is calibrated on its gridlines
(x: 0/100/200 h; y: the tick values read off the rendered figure). Accuracy is about a line width.

    python scripts/digitize_upstream_fig7.py [--svg rmse_vs_time_flops_rev.svg] [--out scripts/data/upstream_fig7_rmse.json]
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

URL = "https://arxiv.org/html/2603.25687v1/rmse_vs_time_flops_rev.svg"

# panel: (variable, x at 0 h, x at 100 h, (y_raw, value) for two gridlines); raw y increases upward
PANELS = [
    ("u10m", 33.08, 68.82, (112.69, 2.0), (148.32, 4.0)),
    ("q700", 168.88, 204.62, (93.48, 0.0005), (151.35, 0.0020)),
    ("t850", 304.68, 340.42, (96.28, 1.0), (153.95, 4.0)),
    ("z500", 440.48, 476.22, (82.15, 0.0), (157.23, 800.0)),
]
# line colours of the budgets (legend order of the figure)
BUDGETS = {"#f6bc99": "6E17", "#f58860": "1E18", "#ec4c3e": "3E18", "#cb1b4f": "6E18",
           "#971c5b": "1E19", "#611f53": "3E19", "#30173a": "6E19"}
PARAMS = {"6E17": "15.1M", "1E18": "15.1M", "3E18": "26.4M", "6E18": "39.0M",
          "1E19": "60.5M", "3E19": "115.1M", "6E19": "203.8M"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--svg", type=Path, default=None, help="local copy of the SVG (downloaded if omitted)")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data" / "upstream_fig7_rmse.json")
    a = p.parse_args()
    s = a.svg.read_text() if a.svg else urllib.request.urlopen(URL).read().decode()

    # gridlines must sit where the calibration expects them
    grid = [[float(x) for x in re.findall(r"-?\d+\.?\d*", m.group(1))]
            for m in re.finditer(r'<path[^>]*stroke="#b0b0b0"[^>]* d="([^"]+)"', s)]
    starts = {round(g[0], 2) for g in grid} | {round(g[1], 2) for g in grid}
    for var, x0, x100, (ya, _), (yb, _) in PANELS:
        for v in (x0, x100, ya, yb):
            assert any(abs(v - g) < 0.02 for g in starts), f"{var}: gridline {v} not found; figure changed?"

    rmse = {}
    for colour, budget in BUDGETS.items():
        curves = []
        for m in re.finditer(r'<path[^>]*stroke="%s"[^>]*>' % colour, s):
            d = re.search(r' d="([^"]+)"', m.group(0)).group(1)
            if "H" in d:  # legend swatch
                continue
            nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", d)]
            curves.append(list(zip(nums[0::2], nums[1::2])))
        assert len(curves) == len(PANELS), f"{budget}: expected {len(PANELS)} curves, got {len(curves)}"
        curves.sort(key=lambda c: c[0][0])
        for (var, x0, x100, (ya, va), (yb, vb)), curve in zip(PANELS, curves):
            rmse.setdefault(budget, {})[var] = {
                int(round((x - x0) / (x100 - x0) * 100)): round(va + (y - ya) * (vb - va) / (yb - ya), 8)
                for x, y in curve
            }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"source": __doc__.split("\n\n")[1].replace("\n", " "), "url": URL, "params": PARAMS,
               "units": {"u10m": "m/s", "q700": "kg/kg", "t850": "K", "z500": "m2/s2"},
               "lead_hours_note": "keys are lead times in hours (6 h steps)", "rmse": rmse},
              open(a.out, "w"), indent=1)
    print(f"wrote {a.out}; 3E18 u10m 6/24/72 h:",
          [rmse["3E18"]["u10m"][h] for h in (6, 24, 72)])


if __name__ == "__main__":
    main()
