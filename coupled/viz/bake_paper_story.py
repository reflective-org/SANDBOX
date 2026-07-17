# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Bake the data blob for paper_story.html (the tabbed Process-Uncertainties story page).

Reuses bake_plume_dynamics for the 30-case grid (adds nothing new there; the shared
assemble_case now carries an `oh` series for the diurnal-chemistry panel) and adds a
`sens` section extracted from the 810-run paper ensemble: the matched microphysics
sensitivity tuple around the baseline case (30N / 20 km, aged-air background, D2 medium
dilution, midnight start), i.e. the runs behind the paper's Figs. 17-18:

    nucleation rate x {0.01, 1, 100}, coagulation kernel x {0.5, 1, 2},
    condensation accommodation alpha in {0.5, 1.0}

For each: kept-grid time series of total wet surface area and total particle number,
plus dry dSA/dlogDp snapshots at end of days 2, 5 and 10.

No coast/USSA/wind sections (this page has no globe). Same size guard (2.0 MB) and the
same /*__D1_DATA_BEGIN__*/ markers so the injector is shared.

Run (from anywhere): python coupled/viz/bake_paper_story.py [--runs-root ...] [--html ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bake_plume_dynamics as bp  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))

# the paper's microphysics sensitivity tuple (ensemble case-id token grammar)
_SENS_BASE = "30N_20km__sabr220__D2med"
_SENS_AXES = [
    dict(key="nucleation", label="Nucleation rate",
         levels=[("nuc0p01", "×0.01"), ("nuc1", "×1"), ("nuc100", "×100")],
         tokens=lambda lv: f"a1p0__{lv}__cg1"),
    dict(key="coagulation", label="Coagulation kernel",
         levels=[("cg0p5", "×0.5"), ("cg1", "×1"), ("cg2", "×2")],
         tokens=lambda lv: f"a1p0__nuc1__{lv}"),
    dict(key="condensation", label="Condensation α",
         levels=[("a0p5", "α = 0.5"), ("a1p0", "α = 1.0")],
         tokens=lambda lv: f"{lv}__nuc1__cg1"),
]
_SNAP_DAYS = (2.0, 5.0, 10.0)


def _sens_case(runs_root, tokens):
    z = np.load(os.path.join(runs_root, "runs", f"{_SENS_BASE}__{tokens}", "state.npz"))
    t = np.asarray(z["t"], float)
    days = t / 86400.0
    keep = bp._keep_indices(days)
    dp = np.asarray(z["dp_mid_um"], float)
    dNdlogDp = np.asarray(z["dNdlogDp"], float)
    out = dict(
        days=bp._sig(days[keep], 5),
        sa=bp._sig(np.asarray(z["SA"], float)[keep], 3),      # wet, um^2/cm^3
        n=bp._sig(np.asarray(z["total_n"], float)[keep], 3),  # /cm^3
        dist={},
    )
    for d in _SNAP_DAYS:
        i = min(len(t) - 1, int(np.searchsorted(t, d * 86400.0 - 1e-6)))
        dsa = dNdlogDp[i] * np.pi * dp ** 2                   # dry dSA/dlogDp [um^2/cm^3]
        out["dist"][f"d{d:.0f}"] = bp._sig(dsa, 3)
    return out


def build_sens(runs_root):
    dp = None
    axes = []
    for ax in _SENS_AXES:
        levels = []
        for lv_key, lv_label in ax["levels"]:
            case = _sens_case(runs_root, ax["tokens"](lv_key))
            levels.append(dict(key=lv_key, label=lv_label, **case))
            if dp is None:
                z = np.load(os.path.join(runs_root, "runs",
                                         f"{_SENS_BASE}__{ax['tokens'](lv_key)}", "state.npz"))
                dp = bp._sig(np.asarray(z["dp_mid_um"], float), 4)
        axes.append(dict(key=ax["key"], label=ax["label"], levels=levels))
        print(f"  sens axis {ax['key']}: {len(levels)} levels")
    return dict(axes=axes, dp_um=dp, snap_days=list(_SNAP_DAYS),
                base=_SENS_BASE, note="matched tuple, other knobs at x1; midnight start")


def assemble(runs_root):
    cases, dp_um = {}, None
    for site in bp.SITES:
        for bgd in bp.BACKGROUNDS:
            for regime in bp.REGIMES:
                npz = bp._npz_path(runs_root, site["key"], bgd["key"], regime["key"])
                ctrl = bp._ctrl_npz_path(runs_root, site["key"], bgd["key"], regime["key"])
                case, r = bp.assemble_case(npz, site, bgd, regime, ctrl)
                cases[f"{site['key']}|{bgd['key']}|{regime['key']}"] = case
                if dp_um is None:
                    dp_um = bp._sig(np.asarray(r["dp_mid_um"], float), 4)
                print(f"  {site['key']:9s} {bgd['key']:8s} {regime['key']:8s} "
                      f"end={case['meta']['days_total']:6.2f} d", flush=True)
    return dict(
        sites=bp.SITES, backgrounds=bp.BACKGROUNDS, regimes=bp.REGIMES,
        default=bp._DEFAULT_CASE,
        dist_dp_um=dp_um, dist_qscale=14,
        cases=cases,
        sens=build_sens(runs_root),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", default=bp._DEFAULT_RUNS)
    ap.add_argument("--html", default=os.path.join(_HERE, "paper_story.html"))
    args = ap.parse_args()
    data = assemble(args.runs_root)
    js = json.dumps(data, separators=(",", ":"))

    def kb(obj):
        return len(json.dumps(obj, separators=(",", ":")).encode()) / 1024.0
    print(f"sections [KB]: cases {kb(data['cases']):.1f} | sens {kb(data['sens']):.1f} | "
          f"TOTAL {len(js.encode())/1024.0:.1f}")
    total = bp.inject(args.html, js)
    print(f"injected -> {args.html}   file = {total/1024.0:.1f} KB")
    if total > bp._SIZE_LIMIT:
        sys.exit(f"FILE TOO BIG: {total} > {bp._SIZE_LIMIT}")


if __name__ == "__main__":
    main()
