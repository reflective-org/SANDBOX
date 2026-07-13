# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""No-injection CONTROL runs paired to the viz plume cases, for dilution correction.

Same scenario as the paired plume run (site, dilution regime, SABR-220 background, 06:00
release-hour clock, alpha x1 / nuc x1 / coag x1) but with NO SO2 spike -- the box starts at the
background composition (SO2 = the 20 pptv dilution background) and just breathes photochemically
while the same V(t) dilution relaxes it toward the static background state.

Purpose (per Ali): dilution-correct the plume sulfur budget by differencing against a control
that shares the SAME k_dil(t). Then d[(C_plume - C_ctrl) V]/dt = V (chem_plume - chem_ctrl):
the dilution terms cancel exactly, so the excess budget decays only through genuine chemistry
differences -- it cannot fake 100% conversion when the plume merely blends away. A control with
a DIFFERENT regime would leave a residual k V (C_ctrl - C_bg) term that grows with V, which is
why one control per (site, regime) is required.

Duration = the paired plume run's length (read from its state.npz), rounded up to whole days.
Output -> runs_bgstop_ctrl/<site>__sabr220__<regime>__h06_ctrl/state.npz (plume-run layout).

CLI (from SANDBOX/): python -m coupled.paper_ensemble.run_bgstop_control <0..9>
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
_OUT = os.path.join(_HERE, "runs_bgstop_ctrl")
_BG = ("sabr220", "sabr_220", 20.0)
_X1 = dict(sticking=("a1p0", 1.0), nucleation=("nuc1", 1.0), coag=("cg1", 1.0))

CASES = [(site, reg) for site in ("30N_20km", "60N_15km")
         for reg in ("D1low", "D2med", "D3high", "D5vhigh", "burst")]
_FROM_BGSTOP = {"D1low", "D2med", "burst"}


def paired_plume_npz(site, reg):
    if reg in _FROM_BGSTOP:
        return os.path.join(_HERE, "runs_bgstop", f"{site}__sabr220__{reg}__h06_bgstop",
                            "state.npz")
    return os.path.join(_HERE, "runs_start_time", f"{site}__sabr220__{reg}__h06", "state.npz")


def build_scenario(site, reg, days):
    la = next(l for l in _re.LAT_ALT if l[0] == site)
    dil = next(d for d in _re.DILUTION if d[0] == reg)
    sc = _re.build_scenario(dict(lat_alt=la, background=_BG, dilution=dil, **_X1))
    conc = dict(sc.concentrations)
    conc["SO2"] = 20.0                       # background SO2, NOT the spike
    return dataclasses.replace(sc, start_utc_hour=6.0, days=days, concentrations=conc)


def main(i):
    site, reg = CASES[i]
    plume = np.load(paired_plume_npz(site, reg), allow_pickle=True)
    days = int(np.ceil(plume["t"][-1] / 86400.0))
    cid = f"{site}__sabr220__{reg}__h06_ctrl"
    out = os.path.join(_OUT, cid)
    os.makedirs(out, exist_ok=True)
    sc = build_scenario(site, reg, days)
    print(f"[{i}] {cid}: control for {days} d (paired plume ends "
          f"{plume['t'][-1]/86400.0:.2f} d)", flush=True)
    t, x, aero, st_final, sd, jrec = run_coupled(
        sc, return_aerosol=True, return_state=True, return_size_dist=True,
        return_photolysis=True)
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
    so2_ppt = x[-1, IDX["SO2"]] / M * 1e12
    print(f"DONE {cid}: {t[-1]/86400.0:.2f} d ({len(t)} steps), SO2(end) = {so2_ppt:.2f} pptv, "
          f"SA(end) = {aero['SA'][-1]:.4g}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
