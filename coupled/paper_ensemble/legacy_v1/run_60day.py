# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""60-day (max) plume runs with spun-up initial conditions and background-relaxation stop.

Configuration (per Ali, 2026-07-08):
  * 30N / 20 km / 210 K / 55 hPa, SABR-220 aged-air background (bg SO2 = 20 ppt), summer
    (doy 172), release at 06:00 local; alpha x1, nuc x1, coag x1; 1 t SO2 into V0.
  * Initial gas composition = frank-model 60-day CONTROL run state (30N, doy 172, sza
    photolysis, P = 60 hPa preset -> transferred as mixing ratios), sampled on the last day
    at 06:00. Two IC sets (CLI tag):
      - ss20IC (default, 2026-09-24): the NO-INJECTION control, SO2 held at 20 ppt -- steady
        state (<0.4 %/day drift) -> runs_60day/control_so2_20ppt_ic.json.
      - frankIC (the 2026-07-08 runs): SO2 NOT held, so the P=60 preset's 954 ppb decays to
        572 ppb and H2SO4 builds to 382 ppb; its HOx/O3/Cl partitioning is sulfur-perturbed
        (O3 +15 %, OH -40 %, HO2 x4 vs ss20IC) -> runs_60day/frank_control_ic.json.
    Overrides: SO2 -> the injection spike;
    gas H2SO4 and SO3 -> 0 (standalone-control accumulation artifacts: frank has no aerosol
    sink); H2O from the scenario WTR (RH = 3%).
  * days = 60 max, EARLY STOP when the wet SA has stayed within 10% of the background SA
    for 24 h straight (the t*-analysis criterion, enforced at run time).
  * Dilution: D2 med and D1 low (cases 0 and 1); D3 high, D5 vhigh, burst (cases 2-4).
  * Case 5 (noinj): same ICs/dilution but SO2 = 20 ppt background -- the coupled-model
    no-injection baseline (36 d, no early stop).

Output -> runs_60day/30N_20km__sabr220__<regime>__60day_h06_<ic tag>/state.npz (same layout as the ensemble runs).

CLI (from SANDBOX/): python -m coupled.paper_ensemble.run_60day <0..5> [ss20IC|frankIC]
"""
import dataclasses
import json
import os
import sys

import numpy as np

from coupled import CoupledScenario
from coupled.coupled_scenario import Switches
from coupled.driver import run_coupled
from coupled import dilution as dl
from config import IDX, air_number_density

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(_HERE, "runs_60day")
_T, _P, _WTR = 210.0, 55.0, 6.9104
_SO2_SPIKE = 6.273063291666667e15            # 1 t SO2 in V0 [molec/cm^3]
_V0_CM3 = 1.5e12
IC_SETS = {"ss20IC": "control_so2_20ppt_ic.json", "frankIC": "frank_control_ic.json"}

CASES = [("D2med", "D2"), ("D1low", "D1"),
         ("D3high", "D3"), ("D5vhigh", "D5"), ("burst", "burst"),   # 2-4 added 2026-09-24
         ("noinj", "D1")]      # 5: NO-injection twin (SO2 = 20 ppt bg), 36 d, no stop


def build_scenario(dil_regime, inject=True, ic_tag="ss20IC"):
    M = air_number_density(_P, _T)
    ic = json.load(open(os.path.join(_OUT, IC_SETS[ic_tag])))["ppt"]
    conc = {k: float(v) for k, v in ic.items() if k in IDX}
    conc.pop("H2O", None)                       # H2O comes from WTR (RH = 3%)
    conc["H2SO4"] = 0.0                         # standalone-control artifacts (no aerosol sink)
    conc["SO3"] = 0.0
    conc["SO2"] = _SO2_SPIKE / M * 1e12 if inject else 20.0   # the injection spike / bg
    return CoupledScenario(
        T=_T, P=_P, WTR=_WTR, latitude=30.0, longitude=0.0, day_of_year=172,
        start_utc_hour=6.0, days=60, DT=600.0, dt_couple=600.0, photolysis="tuvx",
        tomas_nbins=80, background_dist="sabr_220", dilution_regime=dil_regime,
        dilution_zero_species=(), dilution_background={"SO2": 20.0},
        ion_pair_rate=30.0, so2_ho2_rate=1.0e-18,
        condensation_alpha=1.0, nucleation_rate_scale=1.0, coag_kernel_scale=1.0,
        switches=Switches(sulfur=True, nucleation=True, condensation=True, coagulation=True,
                          aerosol_to_j=False, heating_to_t=False, dilution=True),
        concentrations=conc)


def make_stop(sc):
    """Stop once wet SA has stayed within 10% of the background SA for a full 24 h."""
    from coupled.tomas_bridge import initial_tomas_state
    from coupled.aerosol_props import het_inputs
    sa_bg = float(het_inputs(initial_tomas_state(sc))["SA"])
    state = {"run": 0}

    def stop(diag):                      # run_coupled's end-of-interval diagnostics dict
        sa, t1 = diag["SA"], diag["t"]   # SA is the WET surface area; nan when TOMAS is inactive
        if not np.isfinite(sa):
            return False
        state["run"] = state["run"] + 1 if sa <= 1.10 * sa_bg else 0
        return state["run"] >= 144 and t1 > 2.0 * 86400.0   # 144 x 600 s = 24 h; skip spin-up

    print(f"[stop] background SA = {sa_bg:.4g} um^2/cm^3; stopping after 24 h within 10%",
          flush=True)
    return stop


def main(i, ic_tag="ss20IC"):
    name, regime = CASES[i]
    cid = f"30N_20km__sabr220__{name}__60day_h06_{ic_tag}"
    out = os.path.join(_OUT, cid)
    if os.path.exists(os.path.join(out, "state.npz")):
        print(f"SKIP {cid} (state.npz exists)", flush=True)
        return
    os.makedirs(out, exist_ok=True)
    inject = name != "noinj"
    sc = build_scenario(regime, inject=inject, ic_tag=ic_tag)
    if not inject:                   # coupled-model baseline after the frank-model hand-off
        sc = dataclasses.replace(sc, days=36)
    t, x, aero, st_final, sd, jrec = run_coupled(
        sc, return_aerosol=True, return_state=True, return_size_dist=True,
        return_photolysis=True, stop_condition=make_stop(sc) if inject else None)
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
    main(int(sys.argv[1]), *sys.argv[2:3])
