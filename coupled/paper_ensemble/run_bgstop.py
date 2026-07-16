# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Background-relaxation-stop runs for the viz case grid (+ t* analysis).

Same scenario family as run_start_time.py (run_ensemble ICs, alpha x1 / nuc x1 / coag x1,
release 06:00 local) but days = 60 max with the run-time stop from run_60day.py: end once the
wet aerosol SA has stayed within 10% of the background SA for 24 h straight.

Case axes: background (SABR-220 aged / SABR-330 young / AER-2D geoengineered) x site
(30N_20km, 60N_15km) x dilution regime. For SABR-220 only the slow regimes (D1low / D2med /
burst) are here — D3high / D5vhigh reach background inside the existing 10-day start-time runs
and the viz truncates those directly; the other backgrounds have no h06 runs, so all five
regimes run here. Already-completed cases (state.npz present) are skipped.

Output -> runs_bgstop/<site>__<bg>__<regime>__h06_bgstop/state.npz
(run_ensemble state.npz layout + V_ratio).

CLI (from SANDBOX/): python -m coupled.paper_ensemble.run_bgstop <case index>
                     python -m coupled.paper_ensemble.run_bgstop plan
"""
import dataclasses
import os
import sys

import numpy as np

from coupled.paper_ensemble import run_ensemble as _re
from coupled.driver import run_coupled
from coupled import dilution as dl
from config import IDX, air_number_density

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(_HERE, "runs_bgstop")
BGS = {"sabr220": ("sabr220", "sabr_220", 20.0),
       "sabr330": ("sabr330", "sabr_330", 20.0),
       "aergeo":  ("aergeo", "aer_geo", 100.0)}
_X1 = dict(sticking=("a1p0", 1.0), nucleation=("nuc1", 1.0), coag=("cg1", 1.0))

_SITES = ("30N_20km", "60N_15km")
_ALL = ("D1low", "D2med", "D3high", "D5vhigh", "burst")
CASES = ([(site, "sabr220", reg) for site in _SITES for reg in ("D1low", "D2med", "burst")]
         + [(site, bg, reg) for bg in ("sabr330", "aergeo") for site in _SITES
            for reg in _ALL])


def build_scenario(site, bg, reg):
    la = next(l for l in _re.LAT_ALT if l[0] == site)
    dil = next(d for d in _re.DILUTION if d[0] == reg)
    sc = _re.build_scenario(dict(lat_alt=la, background=BGS[bg], dilution=dil, **_X1))
    return dataclasses.replace(sc, start_utc_hour=6.0, days=60)


def make_stop(sc):
    """Stop once wet SA has stayed within 10% of the background SA for a full 24 h."""
    from coupled.tomas_bridge import initial_tomas_state
    from coupled.aerosol_props import het_inputs
    sa_bg = float(het_inputs(initial_tomas_state(sc))["SA"])
    state = {"run": 0}

    def stop(t1, sa):
        if not np.isfinite(sa):
            return False
        state["run"] = state["run"] + 1 if sa <= 1.10 * sa_bg else 0
        return state["run"] >= 144 and t1 > 2.0 * 86400.0   # 144 x 600 s = 24 h; skip spin-up
    print(f"[stop] background SA = {sa_bg:.4g} um^2/cm^3; stopping after 24 h within 10%",
          flush=True)
    return stop


def main(i):
    site, bg, reg = CASES[i]
    cid = f"{site}__{bg}__{reg}__h06_bgstop"
    out = os.path.join(_OUT, cid)
    if os.path.exists(os.path.join(out, "state.npz")):
        print(f"SKIP {cid} (state.npz exists)", flush=True)
        return
    os.makedirs(out, exist_ok=True)
    sc = build_scenario(site, bg, reg)
    t, x, aero, st_final, sd, jrec = run_coupled(
        sc, return_aerosol=True, return_state=True, return_size_dist=True,
        return_photolysis=True, stop_condition=make_stop(sc))
    M = air_number_density(sc.P, sc.T)
    V = dl.volume_ratio(t, sc.dilution_regime)
    from coupled.tomas_bridge import _bad
    dp_edges = _bad._xk_to_dp_um(np.asarray(st_final.xk))
    logdp = np.log10(dp_edges)
    dp_mid = 10 ** (0.5 * (logdp[:-1] + logdp[1:]))
    np.savez(os.path.join(out, "state.npz"),
             t=t, x=x, species=list(IDX), M=M,
             SA=aero["SA"], radius_cm=aero["radius_cm"], h2so4wp=aero["h2so4wp"],
             particulate_S=aero["particulate_S"], T=aero["T"],
             n_cm3=sd["n_cm3"], Dp_m=sd["Dp_m"], dp_mid_um=dp_mid,
             dNdlogDp=sd["n_cm3"] / np.diff(logdp)[None, :],
             V_ratio=V, total_n=sd["n_cm3"].sum(axis=1),
             J_tmid=jrec["t_mid"], J=jrec["J"], J_equations=jrec["equations"])
    print(f"DONE {cid}: reached t = {t[-1]/86400.0:.2f} d ({len(t)} steps), "
          f"SA(end) = {aero['SA'][-1]:.4g}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "plan":
        for k, (s, b, rg) in enumerate(CASES):
            print(k, s, b, rg)
    else:
        main(int(sys.argv[1]))
