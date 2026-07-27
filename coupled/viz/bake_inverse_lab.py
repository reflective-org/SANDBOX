# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Bake the data blob for inverse_lab.html (the inverse-modeling sampling lab).

Unlike the story pages, this one is a MEASURING INSTRUMENT: every number it shows is read off
at a stated time, so the blob carries a UNIFORM 30-MINUTE SAMPLING GRID (`grid`) instead of the
visualization keep grid, and the client never interpolates a series. That matters because the
keep grid (bake_plume_dynamics._keep_indices) coarsens to 4 h late in a run, which aliases the
morning nucleation spikes in particle number by up to a factor of 8, and it resolves the first
hour of the burst only every 20 min (~30% error in mean radius if interpolated).

Grid values are sampled from the model's own ~600 s output by linear interpolation onto the
exact 30-min marks (log-space for the positive-definite, many-decade quantities), so each is a
faithful instantaneous sample. Series ride in the blob as base64 float32 (about half the bytes
of JSON text, and no significant-figure rounding).

The size-distribution raster is rebuilt on the same clock: column j is exactly day
j/(48*sub) with sub in 1..3, so distribution time is exact by construction. The shared
bake_plume_dynamics raster instead assumes a uniform 600 s model step; the real mean step is
~592 s (outer intervals snap to the terminator), which drifts its time axis by ~1.4% -- 0.5 d
at day 36. Harmless in a story, not in a lab.

Reuses bake_plume_dynamics for the case grid, the t* truncation, the day/night intervals and
the meta block. No captions (no narrative), no coast/USSA/wind (no globe). Same size guard and
the same /*__D1_DATA_BEGIN__*/ markers so the injector is shared.

Run (from anywhere): python coupled/viz/bake_inverse_lab.py [--runs-root ...] [--html ...]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bake_plume_dynamics as bp  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))

GRID_PER_DAY = 48                      # sampling clock: 48/day = every 30 minutes
_DIST_MAX_COLS = 2200                  # raster width cap (matches the story pages)
_DIST_E0, _DIST_E1 = -1, 8             # log10 dN/dlogDp range of the 8-bit raster (no clipping)

# quantities that span many decades and stay positive: interpolate in log space
_LOG_SERIES = {"V", "so2_ppt", "h2so4", "part_s", "total_n", "sa", "reff_um"}


def _b64f32(a):
    return base64.b64encode(np.asarray(a, "<f4").tobytes()).decode("ascii")


def _sample(marks_s, t, y, log=False):
    """Linear interpolation of a model series onto exact sampling marks (log space if asked)."""
    y = np.asarray(y, float)
    if log:
        out = 10.0 ** np.interp(marks_s, t, np.log10(np.maximum(y, 1e-30)))
        return np.where(out <= 1e-29, 0.0, out)
    return np.interp(marks_s, t, y)


def _dist_png_uniform(dN, t, marks_s, sub):
    """Size-distribution raster on the sampling clock: column j is day j/(48*sub), exactly.

    8-bit log encoding as in bake_plume_dynamics._dist_png (0 = below the floor; 1..255 map
    log10(dN/dlogDp) linearly over [_DIST_E0, _DIST_E1]), but over a range that CLIPS NOTHING:
    the story pages' [-1, 7] saturates the nucleation-burst peak (measured max 3.3e7), which
    cost up to 25% on the peak bins and ~40% of the integrated number. [-1, 8] leaves the
    quantization step at 9/254 decades, i.e. +-4.3% per bin, and the exponents ride in the JSON
    so the page decodes from them rather than a hardcoded range.

    Columns are interpolated in log space onto the sub-grid; sub is chosen so column spacing
    never falls below the ~600 s model step, so this resamples neighbouring model output rather
    than inventing structure.
    """
    from PIL import Image

    sub_s = np.arange(0, (len(marks_s) - 1) * sub + 1) * (marks_s[1] - marks_s[0]) / sub
    d = np.empty((len(sub_s), dN.shape[1]))
    for b in range(dN.shape[1]):
        d[:, b] = _sample(sub_s, t, dN[:, b], log=True)
    with np.errstate(divide="ignore"):
        lg = np.log10(np.maximum(d, 1e-30))
    span = _DIST_E1 - _DIST_E0
    v = np.clip((lg - _DIST_E0) / span * 254 + 1, 1, 255)
    v[lg < _DIST_E0] = 0
    clipped = int((lg > _DIST_E1).sum())
    buf = io.BytesIO()
    # rint, not a bare cast: truncation would bias every bin down by up to a full quantization
    # step (~8.5%) instead of half of one, and drag the integrated number ~4% light
    Image.fromarray(np.rint(v).astype(np.uint8).T[::-1], "L").save(buf, "PNG", optimize=True)
    return dict(png_b64=base64.b64encode(buf.getvalue()).decode(),
                n=int(len(sub_s)), sub=int(sub), e0=_DIST_E0, e1=_DIST_E1), clipped


def _grid(case, r, ctrl_path):
    """Uniform 30-min sampling grid of every quantity the lab reports, plus the raster."""
    meta = case["meta"]
    n_full = meta["n_full"]
    t = np.asarray(r["t"], float)[:n_full]
    sp = [str(s) for s in r["species"]]
    x = np.asarray(r["x"], float)[:n_full]
    M = float(r["M"])

    nt = int(np.floor(meta["days_total"] * GRID_PER_DAY))
    marks_s = np.arange(nt + 1) / GRID_PER_DAY * 86400.0
    marks_s[-1] = min(marks_s[-1], t[-1])                  # never extrapolate past the run

    V = np.asarray(r["V_ratio"], float)[:n_full]
    partS = np.asarray(r["particulate_S"], float)[:n_full]
    raw = dict(
        V=V,
        so2_ppt=x[:, sp.index("SO2")] / M * 1e12,
        oh=x[:, sp.index("OH")],
        h2so4=x[:, sp.index("H2SO4")],
        part_s=partS,
        total_n=np.asarray(r["total_n"], float)[:n_full],
        sa=np.asarray(r["SA"], float)[:n_full],
        reff_um=np.asarray(r["radius_cm"], float)[:n_full] * 1e4,
    )

    # dilution-corrected sulfur budget vs the PAIRED no-injection control (same k_dil(t), so the
    # dilution terms cancel): the injected tonne's own SO2 / particulate split, in tonnes.
    c = np.load(ctrl_path, allow_pickle=True)
    cs = [str(s) for s in c["species"]]
    cx = np.asarray(c["x"], float)[:n_full]
    to_t = bp._V0_CM3 * V / bp._AVOG * 64.0 / 1e6
    raw["so2_t"] = np.maximum((raw["so2_ppt"] * 1e-12 * M - cx[:, cs.index("SO2")]) * to_t, 0.0)
    raw["sulf_t"] = np.maximum(
        (partS - np.asarray(c["particulate_S"], float)[:n_full]) * to_t
        + (raw["h2so4"] - cx[:, cs.index("H2SO4")]) * to_t, 0.0)

    series, worst = {}, 0.0
    for name, y in raw.items():
        s = _sample(marks_s, t, y, log=name in _LOG_SERIES)
        series[name] = _b64f32(s)
        # honesty check: the sample must sit between its bracketing model steps
        i = np.clip(np.searchsorted(t, marks_s), 1, n_full - 1)
        lo = np.minimum(y[i - 1], y[i]) - 1e-12
        hi = np.maximum(y[i - 1], y[i]) + 1e-12
        span = np.maximum(hi - lo, 1e-30)
        worst = max(worst, float(np.max(np.maximum(lo - s, s - hi) / span)))

    dN = np.asarray(r["dNdlogDp"], float)[:n_full]
    sub = 1
    while sub < 3 and (nt * (sub + 1) + 1) <= _DIST_MAX_COLS and \
            (86400.0 / GRID_PER_DAY / (sub + 1)) >= 600.0:
        sub += 1
    dist, clipped = _dist_png_uniform(dN, t, marks_s, sub)
    return dict(per_day=GRID_PER_DAY, nt=nt, series=series), dist, worst, clipped


def assemble(runs_root):
    cases, dp_um, worst, clip = {}, None, 0.0, 0
    for site in bp.SITES:
        for bgd in bp.BACKGROUNDS:
            for regime in bp.REGIMES:
                npz = bp._npz_path(runs_root, site["key"], bgd["key"], regime["key"])
                ctrl = bp._ctrl_npz_path(runs_root, site["key"], bgd["key"], regime["key"])
                case, r = bp.assemble_case(npz, site, bgd, regime, ctrl)
                grid, dist, w, c = _grid(case, r, ctrl)
                worst = max(worst, w)
                clip += c
                # the keep-grid series and the narrative are for the story pages, not the lab
                for k in ("captions", "series"):
                    case.pop(k, None)
                case["grid"], case["dist"] = grid, dist
                cases[f"{site['key']}|{bgd['key']}|{regime['key']}"] = case
                if dp_um is None:
                    dp_um = bp._sig(np.asarray(r["dp_mid_um"], float), 4)
                print(f"  {site['key']:9s} {bgd['key']:8s} {regime['key']:8s} "
                      f"end={case['meta']['days_total']:6.2f} d  "
                      f"samples={grid['nt'] + 1:5d}  dist={dist['n']:5d} cols (sub={dist['sub']})",
                      flush=True)
    print(f"  worst sampling excursion outside the bracketing model steps: {worst:.2e} "
          f"(0 = every sample lies between its neighbours)")
    if clip:
        sys.exit(f"{clip} distribution cells clip at 1e{_DIST_E1} -- raise _DIST_E1")
    print(f"  distribution raster: 0 cells clipped at the 1e{_DIST_E1} ceiling")
    return dict(
        sites=bp.SITES, backgrounds=bp.BACKGROUNDS, regimes=bp.REGIMES,
        default=bp._DEFAULT_CASE,
        dist_dp_um=dp_um,
        cases=cases,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", default=bp._DEFAULT_RUNS)
    ap.add_argument("--html", default=os.path.join(_HERE, "inverse_lab.html"))
    args = ap.parse_args()
    data = assemble(args.runs_root)
    js = json.dumps(data, separators=(",", ":"))

    def kb(obj):
        return len(json.dumps(obj, separators=(",", ":")).encode()) / 1024.0
    grid_kb = sum(kb(c["grid"]) for c in data["cases"].values())
    dist_kb = sum(kb(c["dist"]) for c in data["cases"].values())
    print(f"sections [KB]: cases {kb(data['cases']):.1f} "
          f"(grid {grid_kb:.1f} | dist {dist_kb:.1f}) | TOTAL {len(js.encode())/1024.0:.1f}")
    total = bp.inject(args.html, js)
    print(f"injected -> {args.html}   file = {total/1024.0:.1f} KB")
    if total > bp._SIZE_LIMIT:
        sys.exit(f"FILE TOO BIG: {total} > {bp._SIZE_LIMIT}")


if __name__ == "__main__":
    main()
