# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Bake a saved D1 dilution run (d1_clean.npz) into the self-contained web visualization.

Converts the model output + Natural Earth coastlines + USSA profile into a compact JSON blob and
injects it, minified, between the markers in ``d1_globe.html``. The HTML then opens directly via
``file://`` (or as a Claude Artifact) with no build step, no server, and no external requests.

The chemistry / aerosol arrays are REAL model output. The wind table (used only to advect and shear
the plume for the globe view) is a hand-specified illustrative June climatology -- NOT model output;
it is labelled as such in the JSON and surfaced in the page legend.

Run:   python coupled/viz/bake_d1_globe.py [npz_path] [--html ...] [--coast ...] [--no-net]
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
# run outputs live in the sibling working copy (see run_dilution_d1_clean.py); overridable positionally
_DEFAULT_NPZ = os.path.abspath(os.path.join(
    _HERE, "..", "..", "..", "..", "SANDBOX", "coupled", "analyses", "d1_clean_80bin", "d1_clean.npz"))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_USSA_DIR = os.path.join(_REPO, "data", "profiles", "atmosphere")
_COAST_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
              "geojson/ne_110m_coastline.geojson")

_MARK_A = "/*__D1_DATA_BEGIN__*/"
_MARK_B = "/*__D1_DATA_END__*/"
_SIZE_LIMIT = 800_000

# geography / schedule -- hardcoded from coupled/run_dilution_d1_clean.py (not stored in the npz)
_GEO = dict(lat0=30.0, lon0=0.0, alt_km=20.0, p_hpa=55.0, doy=172, start_utc_hour=6.0,
            side0_m=10.0, track_km=30.0, so2_bg_ppt=15.0)

# the seven v1 phase captions (coupled/animate_d1_plume.py), $..$ math -> HTML
_CAPTIONS = [
    (0.00, "06:00, day 0 &mdash; a fresh injection trail at 20&nbsp;km: 1.7 tonnes of SO<sub>2</sub> "
           "in a plume only 10&nbsp;m wide"),
    (0.015, "Sunlight makes OH radicals; OH oxidizes SO<sub>2</sub> gas into sulfuric-acid vapour "
            "(H<sub>2</sub>SO<sub>4</sub>)"),
    (0.05, "Nucleation burst: H<sub>2</sub>SO<sub>4</sub> + water form millions of brand-new "
           "~2&nbsp;nm particles per cm<sup>3</sup> &mdash; within hours"),
    (0.35, "The new particles grow &mdash; H<sub>2</sub>SO<sub>4</sub> condenses onto them, and they "
           "merge with each other (coagulation)"),
    (1.00, "Chemistry runs by day, coagulation and mixing around the clock; the plume keeps "
           "spreading and drifting west on the summer easterlies"),
    (4.00, "Dilution thins the plume; particle number falls as small particles merge, but their size "
           "keeps growing"),
    (8.00, "Day 10: a thousand-kilometre veil of 0.1&ndash;0.2&nbsp;&micro;m sulfate droplets "
           "&mdash; near the sweet-spot size for scattering sunlight"),
]

# illustrative June ~50 hPa climatology (subtropical summer easterlies). NOT model output.
_WIND = dict(
    lat=[0, 10, 20, 30, 40, 50, 60],
    u_ms=[-12.0, -11.0, -10.0, -8.0, -5.0, -3.0, -2.0],
    v_ms=[0.0, 0.05, 0.10, 0.15, 0.20, 0.20, 0.20],
    shear_ms_per_km=2.0,
    source="hand-specified June ~50 hPa climatology, NOT model output",
)


def _keep_indices(days: np.ndarray) -> np.ndarray:
    """Subsample: dense through the burst (day 0-0.5), progressively coarser, plus the last step."""
    idx = []
    for d0, d1, step in ((0.0, 0.5, 1), (0.5, 1.0, 2), (1.0, 3.0, 4), (3.0, 1e9, 8)):
        i0, i1 = np.searchsorted(days, (d0, d1))
        idx.extend(range(i0, i1, step))
    idx.append(len(days) - 1)
    return np.array(sorted(set(idx)), dtype=int)


def _day_night(J, jt, days):
    """Model day/night (v1 rule: any J > 1e-3*max => daylight), plus night intervals [day]."""
    jmax = J.max(axis=1)
    lit = jmax > 1e-3 * jmax.max()
    is_day = lit[np.clip(np.searchsorted(jt, days) - 1, 0, len(jt) - 1)]
    tr = np.flatnonzero(np.diff(lit.astype(int)))
    jt_days = jt / 86400.0
    edges = list(jt_days[tr])
    if not lit[0]:
        edges.insert(0, float(days[0]))
    edges.append(float(days[-1]))
    nights = [[round(edges[i], 4), round(edges[i + 1], 4)]
              for i in range(0, len(edges) - 1, 2)]
    return is_day.astype(int).tolist(), nights


def _sig(arr, n=4):
    """Round to n significant figures; return a plain list of floats for compact JSON."""
    a = np.asarray(arr, dtype=float)
    out = []
    for v in a:
        if not np.isfinite(v) or v == 0.0:
            out.append(0.0)
        else:
            out.append(float(f"{v:.{n}g}"))
    return out


def _quantize_dist(dN):
    """dN/dlogDp (nkeep x 80) -> base64 little-endian int16 of round(100*log10(dN)), floor 1e-2."""
    q = np.full(dN.shape, -30000, dtype="<i2")
    m = dN > 1e-2
    q[m] = np.clip(np.round(100.0 * np.log10(dN[m])), -29999, 32767).astype("<i2")
    return base64.b64encode(q.tobytes()).decode("ascii")


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
            # drop consecutive duplicate quantized points
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


def assemble(npz_path, coast_cache, allow_net):
    r = np.load(npz_path, allow_pickle=True)
    t = r["t"]; days = t / 86400.0
    species = [str(s) for s in r["species"]]
    M = float(r["M"])
    keep = _keep_indices(days)
    dk = days[keep]

    so2_ppt = r["x"][:, species.index("SO2")] / M * 1e12
    side_m = _GEO["side0_m"] * np.sqrt(r["V_ratio"])
    series = dict(
        so2_ppt=_sig(so2_ppt[keep]),
        side_m=_sig(side_m[keep]),
        V=_sig(r["V_ratio"][keep]),
        so2_t=_sig(r["so2_kg"][keep] / 1000.0),
        sulf_t=_sig(r["sulfate_kg"][keep] * 64.0 / 98.0 / 1000.0),   # SO2-equivalent tonnes (v1)
        h2so4_t=_sig(r["h2so4_kg"][keep] * 64.0 / 98.0 / 1000.0),
        total_n=_sig(r["total_n"][keep]),
        reff_um=_sig(r["radius_cm"][keep] * 1e4),
    )
    is_day, nights = _day_night(r["J"], r["J_tmid"], dk)
    series["is_day"] = is_day

    dist = dict(dp_um=_sig(r["dp_mid_um"], 4),
                q_b64=_quantize_dist(np.asarray(r["dNdlogDp"])[keep]))

    data = dict(
        meta=dict(scenario="D1 clean stratosphere, 80-bin",
                  npz=os.path.basename(os.path.dirname(npz_path)) + "/" + os.path.basename(npz_path),
                  n_keep=int(len(keep)), n_full=int(len(days)), **_GEO,
                  days_total=float(round(days[-1], 4)),
                  n_bg_cm3=float(f"{series['total_n'][0]:.3g}"),
                  wind_note="Plume drift, shear and shape are an illustrative climatological "
                            "sketch, NOT model output. Chemistry and aerosol microphysics ARE "
                            "model output."),
        days=_sig(dk, 5),
        series=series,
        dist=dist,
        nights=nights,
        ussa=_load_ussa(),
        coast=_pack_coastline(_fetch_coastline(coast_cache, allow_net)),
        wind=_WIND,
        captions=[dict(day=d, html=h) for d, h in _CAPTIONS],
    )
    return data


def inject(html_path, json_str):
    html = open(html_path, encoding="utf-8").read()
    if html.count(_MARK_A) != 1 or html.count(_MARK_B) != 1:
        sys.exit(f"expected exactly one {_MARK_A}...{_MARK_B} span in {html_path}")
    safe = json_str.replace("</", "<\\/").replace("<!--", "<\\!--")
    pat = re.compile(re.escape(_MARK_A) + ".*?" + re.escape(_MARK_B), re.DOTALL)
    new = pat.sub(_MARK_A + safe + _MARK_B, html, count=1)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(new)
    return len(new.encode("utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", nargs="?", default=_DEFAULT_NPZ)
    ap.add_argument("--html", default=os.path.join(_HERE, "d1_globe.html"))
    ap.add_argument("--coast", default=os.path.join(_HERE, "cache", "ne_110m_coastline.geojson"))
    ap.add_argument("--no-net", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.npz):
        sys.exit(f"npz not found: {args.npz}")
    data = assemble(args.npz, args.coast, not args.no_net)
    js = json.dumps(data, separators=(",", ":"))

    # per-section byte report
    def kb(obj):
        return len(json.dumps(obj, separators=(",", ":")).encode()) / 1024.0
    print(f"npz: {args.npz}")
    print(f"n_keep={data['meta']['n_keep']} of {data['meta']['n_full']}   sections [KB]:")
    for k in ("series", "dist", "coast", "ussa", "captions", "nights", "wind", "meta"):
        print(f"   {k:10s} {kb(data[k]):7.1f}")
    print(f"   {'TOTAL json':10s} {len(js.encode())/1024.0:7.1f}")

    total = inject(args.html, js)
    print(f"injected -> {args.html}   file = {total/1024.0:.1f} KB")
    if total > _SIZE_LIMIT:
        sys.exit(f"FILE TOO BIG: {total} > {_SIZE_LIMIT}")


if __name__ == "__main__":
    main()
