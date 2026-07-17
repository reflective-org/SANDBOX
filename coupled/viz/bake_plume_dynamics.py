# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Bake the paper-ensemble dilution runs into the self-contained web visualization.

v2 of the bake: instead of the single D1 clean-stratosphere run, this bakes a GRID of cases --
every dilution parameterization (D1 low Kz, D2 med, D3 high, D5 very high, turbulence burst) x
start location (30N/20 km, 60N/15 km) x background aerosol (SABR-220 N2O-aged, SABR-330
N2O-young, AER-2D geoengineered) -- into one JSON blob with dropdown metadata. Each case is
truncated where the plume stops being distinguishable from the background: the first time the
wet aerosol surface area has stayed within 10% of the background SA for 24 h straight (the same
run-time stop criterion as coupled/paper_ensemble/run_60day.py / run_bgstop.py on the
paper/simulations branch). That endpoint differs per regime (~5 d for D5 up to ~5 weeks for D1).

Sources (sibling working copy, paper/simulations branch):
  * paper_ensemble/runs_bgstop/<site>__<bg>__<reg>__h06_bgstop/ -- 60-day-max runs with the
    run-time background stop (all cases except the two below);
  * paper_ensemble/runs_start_time/<site>__sabr220__{D3high,D5vhigh}__h06/ -- 10-day runs that
    reach the criterion in-window and are truncated here;
  * paper_ensemble/runs_bgstop_ctrl/<site>__<bg>__<reg>__h06_ctrl/ -- PAIRED no-injection
    control runs (same k_dil), for the dilution-corrected sulfur budget.

All cases share one scenario family (run_ensemble ICs, 1 t SO2 into 10 m x 10 m x 15 km,
release 06:00 local, doy 172, alpha x1 / nuc x1 / coag x1).

The chemistry / aerosol arrays are REAL model output. The wind table (used only to advect and
shear the plume for the globe view) is a hand-specified illustrative June climatology -- NOT
model output; it is labelled as such in the JSON and surfaced in the page legend.

Run:   python coupled/viz/bake_plume_dynamics.py [--runs-root ...] [--html ...] [--no-net]
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
# run outputs live in this working copy (paper/simulations merged to main 2026-07-16)
_DEFAULT_RUNS = os.path.abspath(os.path.join(_HERE, "..", "paper_ensemble"))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
# the US-standard-atmosphere profile ships with the tuvx-jax submodule (legacy: repo-root data/)
_USSA_DIR = os.path.join(_REPO, "tuvx-jax", "data", "profiles", "atmosphere")
if not os.path.isdir(_USSA_DIR):
    _USSA_DIR = os.path.join(_REPO, "data", "profiles", "atmosphere")
_COAST_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
              "geojson/ne_110m_coastline.geojson")

_MARK_A = "/*__D1_DATA_BEGIN__*/"
_MARK_B = "/*__D1_DATA_END__*/"
# 30-case grid + embedded day/night Earth textures (2048x1024, per Ali 2026-07-16) + fonts +
# logo; still loads fast as a local file or static page
_SIZE_LIMIT = 2_000_000

_AVOG = 6.02214076e23
_V0_CM3 = 1.5e12                     # 10 m x 10 m x 15 km (run_ensemble.py)
_SO2_INJ_T = 1.0                     # tonnes injected
_SIDE0_M = 10.0
_TRACK_KM = 15.0

# stop criterion (mirror of run_60day.py/run_bgstop.py make_stop)
_SA_FRAC, _HOLD_S, _SKIP_S = 1.10, 24 * 3600.0, 2 * 86400.0

SITES = [
    dict(key="30N_20km", label="30°N · 20 km", lat0=30.0, lon0=0.0, alt_km=20.0, p_hpa=55.0),
    dict(key="60N_15km", label="60°N · 15 km", lat0=60.0, lon0=0.0, alt_km=15.0, p_hpa=120.0),
]
BACKGROUNDS = [   # key, dropdown label, background SO2 [pptv] (run_ensemble/run_geo_ensemble)
    # "aergeo" is the AER-2D geoengineered distribution; per Ali the AER-2D name stays out of
    # everything user-facing, so the label frames it as the deployed-SAI stratosphere.
    # Anchored in NOAA SABRE campaign measurements (csl.noaa.gov/projects/sabre; Science
    # doi:10.1126/science.adw8939) -- provenance surfaced via the page's info tooltip, not
    # the labels (SABRE-220/-330 read as jargon in the dropdown, per Ali)
    dict(key="sabr220", label="Aged air", so2_bg_ppt=20.0,
         phrase="clean aged-air background"),
    dict(key="sabr330", label="Young air", so2_bg_ppt=20.0,
         phrase="younger, particle-richer background"),
    dict(key="aergeo",  label="SAI deployed", so2_bg_ppt=100.0,
         phrase="background of a stratosphere where SAI is already deployed"),
]
REGIMES = [   # key, dropdown label (Schumann k or burst; TABLE_dilution_parameters.md)
    dict(key="D1low",   label="D1 · slowest mixing"),
    dict(key="D2med",   label="D2 · medium"),
    dict(key="D3high",  label="D3 · fast"),
    dict(key="D5vhigh", label="D5 · fastest"),
    dict(key="burst",   label="Turbulence burst · ~14 h"),
]
_FROM_BGSTOP = {"D1low", "D2med", "burst"}
_DEFAULT_CASE = dict(site="30N_20km", background="sabr220", regime="D2med")

# illustrative June ~50 hPa climatology (subtropical summer easterlies). NOT model output.
_WIND = dict(
    lat=[0, 10, 20, 30, 40, 50, 60],
    u_ms=[-12.0, -11.0, -10.0, -8.0, -5.0, -3.0, -2.0],
    v_ms=[0.0, 0.05, 0.10, 0.15, 0.20, 0.20, 0.20],
    shear_ms_per_km=2.0,
    source="hand-specified June ~50 hPa climatology, NOT model output",
)


def _npz_path(runs_root, site, bg, regime):
    if bg == "sabr220" and regime not in _FROM_BGSTOP:
        return os.path.join(runs_root, "runs_start_time",
                            f"{site}__sabr220__{regime}__h06", "state.npz")
    return os.path.join(runs_root, "runs_bgstop",
                        f"{site}__{bg}__{regime}__h06_bgstop", "state.npz")


def _ctrl_npz_path(runs_root, site, bg, regime):
    return os.path.join(runs_root, "runs_bgstop_ctrl",
                        f"{site}__{bg}__{regime}__h06_ctrl", "state.npz")


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


# beat 6, one voice per mixing regime (editorial-scientific, no em-dashes, per Ali 2026-07-16)
_BEAT6 = {
    "D1low":   "Mixing is slow here. The plume stays concentrated, so particles keep colliding "
               "and the count falls fast even as sizes climb.",
    "D2med":   "Steady mixing thins the plume. Particle numbers fall; the survivors keep growing.",
    "D3high":  "Vigorous mixing dilutes the plume quickly, and the background starts to show "
               "through.",
    "D5vhigh": "Mixing this strong tears the plume apart within days.",
    "burst":   "A burst of turbulence, about fourteen hours of violent mixing, rips through "
               "the plume.",
}


def _beat7(regime_key, bg_phrase, tend, conv_pct):
    """Closing beat: the case's own lifetime and conversion, phrased per regime."""
    n = f"{tend:.0f}"
    p = f"{conv_pct:.0f}"
    weeks = tend / 7.0
    if regime_key == "D1low":
        opener = (f"Five weeks on ({n} days)," if 4.5 <= weeks <= 5.5 else f"After {n} days,")
        return (f"{opener} the plume is finally indistinguishable from the {bg_phrase}. Slow "
                f"mixing kept it alive longer than any other regime here; roughly {p}% of the "
                f"tonne became particles.")
    if regime_key == "D2med":
        return (f"After {n} days the {bg_phrase} can explain everything that remains. About "
                f"{p}% of the tonne ended up in particles.")
    if regime_key == "burst":
        return (f"The burst finished the job by day {n}; after it, nothing remains that the "
                f"{bg_phrase} cannot explain. {p}% of the tonne converted.")
    return (f"By day {n} the plume is gone into the {bg_phrase}. Fast mixing left little time "
            f"for chemistry: only {p}% of the tonne converted.")


def _captions(site, regime_key, bg_phrase, tend, t_burst, conv_pct):
    """Narrative beats, written per case: site opener, shared chemistry spine, regime-specific
    dilution and closing beats with the case's own lifetime and conversion numbers."""
    alt = int(site["alt_km"])
    lat = site["label"].split(" · ")[0]
    so2 = "SO<sub>2</sub>"
    beats = [
        dict(day=0.0, html=f"At 06:00 local, one tonne of {so2} goes into the stratosphere at "
                           f"{alt}&nbsp;km over {lat}. The trail is fifteen kilometers long and "
                           f"ten meters wide."),
        dict(day=0.015, html=f"Sunlight builds OH radicals, and OH starts turning the {so2} "
                             "into sulfuric acid vapor."),
        dict(day=round(max(0.03, min(t_burst, 0.6)), 3),
             html="Within hours the vapor bursts into new particles. Millions per cubic "
                  "centimeter, each about 2&nbsp;nanometers across."),
        dict(day=0.35, html="The newborn particles grow. Acid vapor condenses onto them; "
                            "collisions merge them into fewer, larger ones."),
        dict(day=1.0, html="Chemistry works only in daylight. Coagulation and mixing never stop."),
        dict(day=round(min(4.0, 0.45 * tend), 2), html=_BEAT6[regime_key]),
        dict(day=round(0.9 * tend, 2), html=_beat7(regime_key, bg_phrase, tend, conv_pct)),
    ]
    for b in beats:   # template hygiene: no unexpanded placeholder or em-dash may survive
        assert "{" not in b["html"] and "}" not in b["html"], b
        assert "—" not in b["html"] and "&mdash;" not in b["html"], b
    return beats


def assemble_case(npz_path, site, bgd, regime, ctrl_path):
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
    x_so2 = np.asarray(r["x"], float)[:i_end + 1, species.index("SO2")]
    x_h2so4 = np.asarray(r["x"], float)[:i_end + 1, species.index("H2SO4")]
    partS = np.asarray(r["particulate_S"], float)[:i_end + 1]
    # sulfur partition of PLUME AIR, per cm^3, deliberately NOT dilution-corrected: the share of
    # sulfur (S atoms) still in SO2 gas vs already in particles (+ trace acid vapour). A
    # dilution-corrected excess x V budget claims 100% conversion the moment the plume SO2 conc
    # touches background -- chemically wrong on short runs, where the SO2 mostly blends away.
    # This share instead converges to the BACKGROUND sulfur partition (mostly particulate).
    so2_frac = x_so2 / (x_so2 + x_h2so4 + partS)
    so2_bg = bgd["so2_bg_ppt"] * 1e-12 * float(r["M"])
    bg_so2_frac = so2_bg / (so2_bg + partS[0])                   # background partition (h2so4 ~ 0)
    total_n = np.asarray(r["total_n"], float)[:i_end + 1]

    # dilution-corrected budget vs the PAIRED no-injection control (same site, same regime, same
    # k_dil(t), per Ali): the dilution terms cancel in the difference, so (plume - control) x V is
    # the injected tonne's own budget and decays only through genuine chemistry differences.
    c = np.load(ctrl_path, allow_pickle=True)
    tc = np.asarray(c["t"], float)
    assert len(tc) >= i_end + 1 and np.allclose(tc[:i_end + 1], t), \
        f"control t grid mismatch: {ctrl_path}"
    cs = [str(s) for s in c["species"]]
    to_t = _V0_CM3 * V / _AVOG * 64.0 / 1e6                      # conc -> SO2-equiv tonnes
    dso2 = (x_so2 - np.asarray(c["x"], float)[:i_end + 1, cs.index("SO2")]) * to_t
    dpart = (partS - np.asarray(c["particulate_S"], float)[:i_end + 1]) * to_t
    dh2so4 = (x_h2so4 - np.asarray(c["x"], float)[:i_end + 1, cs.index("H2SO4")]) * to_t
    so2_t = np.maximum(dso2, 0.0)
    sulf_t = np.maximum(dpart + dh2so4, 0.0)
    cons = dso2 + dpart + dh2so4                                 # should stay ~= injected 1 t
    print(f"      budget conservation vs control: total in [{cons.min():.3f}, {cons.max():.3f}] t")
    # end-of-life conversion of the injected tonne (feeds the closing narration beat)
    conv_pct = 100.0 * (1.0 - so2_t[-1] / max(so2_t[-1] + sulf_t[-1], 1e-12))

    series = dict(
        so2_ppt=_sig(x_so2[keep] / M * 1e12),
        side_m=_sig(_SIDE0_M * np.sqrt(V[keep])),
        V=_sig(V[keep]),
        so2_frac=_sig(so2_frac[keep]),
        so2_t=_sig(so2_t[keep]),
        sulf_t=_sig(sulf_t[keep]),
        total_n=_sig(total_n[keep]),
        sa=_sig(sa[:i_end + 1][keep]),
        reff_um=_sig(np.asarray(r["radius_cm"], float)[:i_end + 1][keep] * 1e4),
        oh=_sig(np.asarray(r["x"], float)[:i_end + 1, species.index("OH")][keep], 3),
    )
    is_day, nights = _day_night(np.asarray(r["J"]), np.asarray(r["J_tmid"], float), dk)
    series["is_day"] = is_day

    tend = float(round(days[-1], 4))
    t_burst = float(days[:len(total_n)][int(np.argmax(total_n))])
    meta = dict(
        scenario=f"{site['key']} / {bgd['key']} / {regime['key']}, 1 t SO2, 06:00 release, 80-bin",
        npz=os.path.relpath(npz_path, os.path.dirname(os.path.dirname(npz_path))),
        ctrl_npz=os.path.relpath(ctrl_path, os.path.dirname(os.path.dirname(ctrl_path))),
        n_keep=int(len(keep)), n_full=int(len(days)),
        lat0=site["lat0"], lon0=site["lon0"], alt_km=site["alt_km"], p_hpa=site["p_hpa"],
        doy=172, start_utc_hour=6.0, side0_m=_SIDE0_M, track_km=_TRACK_KM,
        so2_bg_ppt=bgd["so2_bg_ppt"], so2_inj_t=_SO2_INJ_T,
        bg_so2_frac=float(f"{bg_so2_frac:.4g}"),
        days_total=tend, t_star_day=float(round(ts / 86400.0, 4)),
        sa_bg=float(f"{sa[np.isfinite(sa)][0]:.4g}"),
        n_bg_cm3=float(f"{series['total_n'][0]:.3g}"),
        wind_note="Plume drift, shear and shape are an illustrative climatological sketch, NOT "
                  "model output. Chemistry and aerosol microphysics ARE model output.",
    )
    return dict(meta=meta, days=_sig(dk, 5), series=series,
                dist=dict(q_b64=_quantize_dist(np.asarray(r["dNdlogDp"])[:i_end + 1][keep])),
                nights=nights,
                captions=_captions(site, regime["key"], bgd["phrase"], tend, t_burst, conv_pct)), r


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
        for bgd in BACKGROUNDS:
            for reg in REGIMES:
                p = _npz_path(runs_root, site["key"], bgd["key"], reg["key"])
                pc = _ctrl_npz_path(runs_root, site["key"], bgd["key"], reg["key"])
                for path in (p, pc):
                    if not os.path.exists(path):
                        sys.exit(f"missing run: {path}")
                case, r = assemble_case(p, site, bgd, reg, pc)
                cases[f"{site['key']}|{bgd['key']}|{reg['key']}"] = case
                dp = _sig(np.asarray(r["dp_mid_um"], float), 4)
                if dp_um is None:
                    dp_um = dp
                elif dp != dp_um:
                    sys.exit("dp grid differs between cases -- per-case dp not implemented")
                print(f"  {site['key']:9s} {bgd['key']:8s} {reg['key']:8s} "
                      f"end={case['meta']['days_total']:6.2f} d  "
                      f"t*={case['meta']['t_star_day']:6.2f} d  frames={case['meta']['n_keep']}")
    return dict(
        sites=SITES, backgrounds=BACKGROUNDS, regimes=REGIMES, default=_DEFAULT_CASE,
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
    ap.add_argument("--html", default=os.path.join(_HERE, "plume_dynamics.html"))
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
