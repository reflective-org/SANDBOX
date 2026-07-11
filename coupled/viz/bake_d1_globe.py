# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Bake the paper-ensemble dilution runs into the self-contained web visualization.

v2 of the bake: instead of the single D1 clean-stratosphere run, this bakes a GRID of cases --
every dilution parameterization (D1 low Kz, D2 med, D3 high, D5 very high, turbulence burst) at
every start location (30N/20 km, 60N/15 km) -- into one JSON blob with dropdown metadata. Each
case is truncated where the plume stops being distinguishable from the background: the first time
the wet aerosol surface area has stayed within 10% of the background SA for 24 h straight (the
same run-time stop criterion as coupled/paper_ensemble/run_60day.py / run_bgstop.py on the
paper/simulations branch). That endpoint differs per regime (~5 d for D5 up to ~35 d for D1).

Sources (sibling working copy, paper/simulations branch):
  * D1low / D2med / burst : paper_ensemble/runs_bgstop/<site>__sabr220__<reg>__h06_bgstop/
    (60-day-max runs with the run-time background stop; they end at the criterion)
  * D3high / D5vhigh      : paper_ensemble/runs_start_time/<site>__sabr220__<reg>__h06/
    (10-day runs; they reach the criterion in-window and are truncated here)

All cases share one scenario family (run_ensemble ICs, SABR-220 background, 1 t SO2 into
10 m x 10 m x 15 km, release 06:00 local, doy 172, alpha x1 / nuc x1 / coag x1).

The chemistry / aerosol arrays are REAL model output. The wind table (used only to advect and
shear the plume for the globe view) is a hand-specified illustrative June climatology -- NOT
model output; it is labelled as such in the JSON and surfaced in the page legend.

Run:   python coupled/viz/bake_d1_globe.py [--runs-root ...] [--html ...] [--no-net]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import urllib.request

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# run outputs live in the sibling working copy (paper/simulations branch)
_DEFAULT_RUNS = os.path.abspath(os.path.join(
    _HERE, "..", "..", "..", "..", "SANDBOX", "coupled", "paper_ensemble"))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_USSA_DIR = os.path.join(_REPO, "data", "profiles", "atmosphere")
_COAST_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
              "geojson/ne_110m_coastline.geojson")

_MARK_A = "/*__D1_DATA_BEGIN__*/"
_MARK_B = "/*__D1_DATA_END__*/"
_SIZE_LIMIT = 800_000

_AVOG = 6.02214076e23
_V0_CM3 = 1.5e12                     # 10 m x 10 m x 15 km (run_ensemble.py)
_SO2_INJ_T = 1.0                     # tonnes injected
_SO2_BG_PPT = 20.0                   # SABR-220 background SO2
_SIDE0_M = 10.0
_TRACK_KM = 15.0

# stop criterion (mirror of run_60day.py/run_bgstop.py make_stop)
_SA_FRAC, _HOLD_S, _SKIP_S = 1.10, 24 * 3600.0, 2 * 86400.0

SITES = [
    dict(key="30N_20km", label="30°N · 20 km", lat0=30.0, lon0=0.0, alt_km=20.0, p_hpa=55.0),
    dict(key="60N_15km", label="60°N · 15 km", lat0=60.0, lon0=0.0, alt_km=15.0, p_hpa=120.0),
]
REGIMES = [   # key, dropdown label (Schumann k or burst; TABLE_dilution_parameters.md)
    dict(key="D1low",   label="D1 — low Kz (slowest mixing)"),
    dict(key="D2med",   label="D2 — medium Kz (default)"),
    dict(key="D3high",  label="D3 — high Kz"),
    dict(key="D5vhigh", label="D5 — very high Kz (fastest)"),
    dict(key="burst",   label="Turbulence burst (~14 h of strong mixing)"),
]
_FROM_BGSTOP = {"D1low", "D2med", "burst"}
_DEFAULT_CASE = dict(site="30N_20km", regime="D2med")

# illustrative June ~50 hPa climatology (subtropical summer easterlies). NOT model output.
_WIND = dict(
    lat=[0, 10, 20, 30, 40, 50, 60],
    u_ms=[-12.0, -11.0, -10.0, -8.0, -5.0, -3.0, -2.0],
    v_ms=[0.0, 0.05, 0.10, 0.15, 0.20, 0.20, 0.20],
    shear_ms_per_km=2.0,
    source="hand-specified June ~50 hPa climatology, NOT model output",
)


def _npz_path(runs_root, site, regime):
    if regime in _FROM_BGSTOP:
        return os.path.join(runs_root, "runs_bgstop",
                            f"{site}__sabr220__{regime}__h06_bgstop", "state.npz")
    return os.path.join(runs_root, "runs_start_time",
                        f"{site}__sabr220__{regime}__h06", "state.npz")


def _t_star(t, sa):
    """Start [s] of the first 24 h window with SA <= 1.1 x background (SA[0]), after 2 d spin-up.

    Returns None if the run never sustains the criterion (the bgstop runs end exactly when it
    fires, so their last 24 h ARE the window).
    """
    sa_bg = sa[np.isfinite(sa)][0]
    run0 = None
    for ti, si in zip(t, sa):
        if np.isfinite(si) and si <= _SA_FRAC * sa_bg and ti > _SKIP_S:
            if run0 is None:
                run0 = ti
            if ti - run0 >= _HOLD_S:
                return run0
        else:
            run0 = None
    return None


def _keep_indices(days: np.ndarray) -> np.ndarray:
    """Subsample: dense through the burst (day 0-0.5), progressively coarser, plus the last step."""
    idx = []
    for d0, d1, step in ((0.0, 0.5, 2), (0.5, 1.0, 4), (1.0, 3.0, 8), (3.0, 10.0, 12),
                         (10.0, 1e9, 24)):
        i0, i1 = np.searchsorted(days, (d0, d1))
        idx.extend(range(i0, i1, step))
    idx.append(len(days) - 1)
    return np.array(sorted(set(idx)), dtype=int)


def _day_night(J, jt, days):
    """Model day/night (any J > 1e-3*max => daylight), plus night intervals [day]."""
    jmax = J.max(axis=1)
    lit = jmax > 1e-3 * jmax.max()
    is_day = lit[np.clip(np.searchsorted(jt, days) - 1, 0, len(jt) - 1)]
    tr = np.flatnonzero(np.diff(lit.astype(int)))
    jt_days = jt / 86400.0
    edges = [float(e) for e in jt_days[tr] if e < days[-1]]
    if not lit[0]:
        edges.insert(0, float(days[0]))
    edges.append(float(days[-1]))
    nights = [[round(edges[i], 4), round(edges[i + 1], 4)]
              for i in range(0, len(edges) - 1, 2)]
    return is_day.astype(int).tolist(), nights


def _sig(arr, n=4):
    """Round to n significant figures; plain list of floats for compact JSON."""
    a = np.asarray(arr, dtype=float)
    out = []
    for v in a:
        if not np.isfinite(v) or v == 0.0:
            out.append(0.0)
        else:
            out.append(float(f"{v:.{n}g}"))
    return out


def _quantize_dist(dN):
    """dN/dlogDp (nkeep x nbin) -> base64 int8 of round(14*log10(dN)), floor 1e-2, sentinel -128."""
    q = np.full(dN.shape, -128, dtype=np.int8)
    m = dN > 1e-2
    q[m] = np.clip(np.round(14.0 * np.log10(dN[m])), -127, 127).astype(np.int8)
    return base64.b64encode(q.tobytes()).decode("ascii")


def _captions(site, tend, t_burst):
    """Narrative beats, generated per case (days are data-driven where they can be)."""
    alt = int(site["alt_km"])
    return [
        dict(day=0.0, html=f"06:00, day 0 &mdash; 1 tonne of SO<sub>2</sub> released at "
                           f"{alt}&nbsp;km over {site['label'].replace(' · ', ', ')}: a plume "
                           f"just 10&nbsp;m wide"),
        dict(day=0.015, html="Sunlight makes OH radicals; OH oxidizes SO<sub>2</sub> gas into "
                             "sulfuric-acid vapour (H<sub>2</sub>SO<sub>4</sub>)"),
        dict(day=round(max(0.03, min(t_burst, 0.6)), 3),
             html="Nucleation burst: H<sub>2</sub>SO<sub>4</sub> + water form millions of "
                  "brand-new ~2&nbsp;nm particles per cm<sup>3</sup> &mdash; within hours"),
        dict(day=0.35, html="The new particles grow &mdash; H<sub>2</sub>SO<sub>4</sub> "
                            "condenses onto them, and they merge with each other (coagulation)"),
        dict(day=1.0, html="Chemistry runs by day, coagulation and mixing around the clock; the "
                           "plume keeps spreading and drifting on the stratospheric winds"),
        dict(day=round(min(4.0, 0.45 * tend), 2),
             html="Dilution thins the plume; particle number falls as small particles merge, "
                  "but their size keeps growing"),
        dict(day=round(0.9 * tend, 2),
             html=f"Day {tend:.0f}: the plume's surface area is back at the background level "
                  f"&mdash; under this mixing regime the plume took {tend:.0f} days to fade"),
    ]


def assemble_case(npz_path, site, regime):
    r = np.load(npz_path, allow_pickle=True)
    t = np.asarray(r["t"], float)
    sa = np.asarray(r["SA"], float)

    ts = _t_star(t, sa)
    if ts is None:
        # bgstop runs end exactly when the stop fires: last 24 h are the window
        ts = t[-1] - _HOLD_S
        i_end = len(t) - 1
    else:
        i_end = int(np.searchsorted(t, ts + _HOLD_S))
        i_end = min(i_end, len(t) - 1)
    t = t[:i_end + 1]
    days = t / 86400.0

    species = [str(s) for s in r["species"]]
    M = float(r["M"])
    keep = _keep_indices(days)
    dk = days[keep]

    V = np.asarray(r["V_ratio"], float)[:i_end + 1]
    Vt_cm3 = _V0_CM3 * V
    x_so2 = np.asarray(r["x"], float)[:i_end + 1, species.index("SO2")]
    # plume-integrated EXCESS (above-background) masses: dilution relaxes conc toward the
    # background, so excess x V is the conserved plume budget; absolute conc x V would blow up
    # once V is huge and the plume is nearly background (entrained background sulfur dominates).
    so2_bg = _SO2_BG_PPT * 1e-12 * float(r["M"])
    so2_t = np.maximum(x_so2 - so2_bg, 0.0) * Vt_cm3 / _AVOG * 64.0 / 1e6      # tonnes SO2
    # converted injected sulfur = injected - still-gas excess. Excess x V is exactly conserved
    # under the model's dilution, so this is the injected tonne's own budget; the converted share
    # lives in the particles (trace gas H2SO4 is kilograms). Any absolute conc x V (particulate_S,
    # gas H2SO4) would instead count processed ENTRAINED background sulfur, which exceeds 1 t once
    # V ~ 1e7 -- misleading, so it is not baked.
    sulf_t = np.maximum(so2_t[0] - so2_t, 0.0)                                 # SO2-equiv tonnes
    total_n = np.asarray(r["total_n"], float)[:i_end + 1]

    series = dict(
        so2_ppt=_sig(x_so2[keep] / M * 1e12),
        side_m=_sig(_SIDE0_M * np.sqrt(V[keep])),
        V=_sig(V[keep]),
        so2_t=_sig(so2_t[keep]),
        sulf_t=_sig(sulf_t[keep]),
        total_n=_sig(total_n[keep]),
        reff_um=_sig(np.asarray(r["radius_cm"], float)[:i_end + 1][keep] * 1e4),
        sa=_sig(sa[:i_end + 1][keep]),
    )
    is_day, nights = _day_night(np.asarray(r["J"]), np.asarray(r["J_tmid"], float), dk)
    series["is_day"] = is_day

    tend = float(round(days[-1], 4))
    t_burst = float(days[:len(total_n)][int(np.argmax(total_n))])
    meta = dict(
        scenario=f"{site['key']} / {regime['key']} — SABR-220, 1 t SO2, 06:00 release, 80-bin",
        npz=os.path.relpath(npz_path, os.path.dirname(os.path.dirname(npz_path))),
        n_keep=int(len(keep)), n_full=int(len(days)),
        lat0=site["lat0"], lon0=site["lon0"], alt_km=site["alt_km"], p_hpa=site["p_hpa"],
        doy=172, start_utc_hour=6.0, side0_m=_SIDE0_M, track_km=_TRACK_KM,
        so2_bg_ppt=_SO2_BG_PPT, so2_inj_t=_SO2_INJ_T,
        days_total=tend, t_star_day=float(round(ts / 86400.0, 4)),
        sa_bg=float(f"{sa[np.isfinite(sa)][0]:.4g}"),
        n_bg_cm3=float(f"{series['total_n'][0]:.3g}"),
        wind_note="Plume drift, shear and shape are an illustrative climatological sketch, NOT "
                  "model output. Chemistry and aerosol microphysics ARE model output.",
    )
    return dict(meta=meta, days=_sig(dk, 5), series=series,
                dist=dict(q_b64=_quantize_dist(np.asarray(r["dNdlogDp"])[:i_end + 1][keep])),
                nights=nights, captions=_captions(site, tend, t_burst)), r


def _load_ussa():
    def col(name, c):
        p = os.path.join(_USSA_DIR, name)
        rows = [ln.split() for ln in open(p) if ln.strip() and not ln.startswith("#")]
        return {int(float(r[0])): float(r[c]) for r in rows}
    T = col("ussa.temp", 1)
    D = col("ussa.dens", 1)
    z = list(range(0, 31))
    return dict(z_km=z, T_K=[round(T[k], 2) for k in z],
                log10_n=[round(float(np.log10(D[k])), 3) for k in z])


def _fetch_coastline(cache, allow_net):
    if not os.path.exists(cache):
        if not allow_net:
            sys.exit(f"coastline cache missing ({cache}) and --no-net set")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        print(f"downloading coastlines -> {cache}")
        urllib.request.urlretrieve(_COAST_URL, cache)
    return json.load(open(cache))


def _pack_coastline(geo, q=10, min_bbox_deg=1.5):
    """Natural Earth 110m -> delta-encoded int16. Drops polylines with bbox diagonal < min_bbox_deg.

    Layout of the int16 stream, per kept segment: [lon0*q, lat0*q, dlon1, dlat1, ...] (quantized
    degrees). ``seg_pts`` gives the point count per segment so JS can walk the stream.
    """
    seg_pts, vals = [], []
    for feat in geo["features"]:
        g = feat["geometry"]
        lines = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
        for line in lines:
            pts = np.array(line, dtype=float)  # (n, 2) lon,lat
            if len(pts) < 2:
                continue
            span = pts.max(0) - pts.min(0)
            if np.hypot(*span) < min_bbox_deg:
                continue
            qp = np.round(pts * q).astype(int)
            keep = np.concatenate([[True], np.any(np.diff(qp, axis=0) != 0, axis=1)])
            qp = qp[keep]
            if len(qp) < 2:
                continue
            seg_pts.append(len(qp))
            vals.append(qp[0, 0]); vals.append(qp[0, 1])
            d = np.diff(qp, axis=0)
            for dx, dy in d:
                vals.append(int(dx)); vals.append(int(dy))
    arr = np.array(vals, dtype="<i2")
    if (np.abs(arr) > 32767).any():
        sys.exit("coastline delta overflow int16 -- raise q resolution handling")
    return dict(q=q, seg_pts=seg_pts,
                xy_b64=base64.b64encode(arr.tobytes()).decode("ascii"))


def assemble(runs_root, coast_cache, allow_net):
    cases, dp_um = {}, None
    for site in SITES:
        for reg in REGIMES:
            p = _npz_path(runs_root, site["key"], reg["key"])
            if not os.path.exists(p):
                sys.exit(f"missing run: {p}")
            case, r = assemble_case(p, site, reg)
            cases[f"{site['key']}|{reg['key']}"] = case
            dp = _sig(np.asarray(r["dp_mid_um"], float), 4)
            if dp_um is None:
                dp_um = dp
            elif dp != dp_um:
                sys.exit("dp grid differs between cases -- per-case dp not implemented")
            print(f"  {site['key']:9s} {reg['key']:8s} end={case['meta']['days_total']:6.2f} d  "
                  f"t*={case['meta']['t_star_day']:6.2f} d  frames={case['meta']['n_keep']}")
    return dict(
        sites=SITES, regimes=REGIMES, default=_DEFAULT_CASE,
        dist_dp_um=dp_um, dist_qscale=14,
        cases=cases,
        ussa=_load_ussa(),
        coast=_pack_coastline(_fetch_coastline(coast_cache, allow_net)),
        wind=_WIND,
    )


def inject(html_path, json_str):
    html = open(html_path, encoding="utf-8").read()
    if html.count(_MARK_A) != 1 or html.count(_MARK_B) != 1:
        sys.exit(f"expected exactly one {_MARK_A}...{_MARK_B} span in {html_path}")
    safe = json_str.replace("</", "<\\/").replace("<!--", "<\\!--")
    pat = re.compile(re.escape(_MARK_A) + ".*?" + re.escape(_MARK_B), re.DOTALL)
    # lambda replacement: the JSON contains \u escapes that re.sub would parse as templates
    new = pat.sub(lambda m: _MARK_A + safe + _MARK_B, html, count=1)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(new)
    return len(new.encode("utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", default=_DEFAULT_RUNS,
                    help="paper_ensemble dir holding runs_bgstop/ + runs_start_time/")
    ap.add_argument("--html", default=os.path.join(_HERE, "d1_globe.html"))
    ap.add_argument("--coast", default=os.path.join(_HERE, "cache", "ne_110m_coastline.geojson"))
    ap.add_argument("--no-net", action="store_true")
    args = ap.parse_args()

    data = assemble(args.runs_root, args.coast, not args.no_net)
    js = json.dumps(data, separators=(",", ":"))

    def kb(obj):
        return len(json.dumps(obj, separators=(",", ":")).encode()) / 1024.0
    print("sections [KB]:")
    for k in ("cases", "coast", "ussa", "wind"):
        print(f"   {k:10s} {kb(data[k]):7.1f}")
    print(f"   {'TOTAL json':10s} {len(js.encode())/1024.0:7.1f}")

    total = inject(args.html, js)
    print(f"injected -> {args.html}   file = {total/1024.0:.1f} KB")
    if total > _SIZE_LIMIT:
        sys.exit(f"FILE TOO BIG: {total} > {_SIZE_LIMIT}")


if __name__ == "__main__":
    main()
