# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Paper SAI ensemble: full Cartesian sweep, 80-bin / 10-day / no-aerosol->J / heating-off.

Axes (810 = 3 x 3 x 5 x 2 x 3 x 3):
  lat_alt(3) x background+bgSO2(3) x dilution(5) x sticking(2) x nucleation(3) x coag_kernel(3)

Fixed: summer (doy 172), 1 tonne SO2 into 10m x 10m x 15km (see SO2 forcing below), SO2+HO2=1e-18,
RH=3% (WTR precomputed per baseline), ion-induced nucleation 30 cm^-3 s^-1, TUV-x photolysis (JPL 19-5
rates in jaxmodel/rates.py), PPM-JIT condensation, aerosol->J OFF, heating OFF, dilution ON, initial
plume gas = stratospheric background composition + the SO2 spike.

Saves EVERYTHING for analysis:
  * manifest.csv  -- one row per run with every parameter (resolved values).
  * <case_id>/state.npz -- full time series: all 36 gas species, aerosol (SA/radius/wt%/particulate S),
    size distribution (n_cm3, Dp, dN/dlogDp), dilution V(t)/kdil, per-reaction J, and metadata.
  * summary.csv   -- headline metrics per run (appended as runs finish).

See DECISIONS.md (same dir) for every modeling decision and its justification.

CLI (from SANDBOX/):
  python -m coupled.paper_ensemble.run_ensemble plan            # list cases + write manifest, no run
  python -m coupled.paper_ensemble.run_ensemble one <i>         # run a single case by index
  python -m coupled.paper_ensemble.run_ensemble run <lo> <hi>   # run cases [lo, hi)
"""
import csv
import itertools
import os
import sys

import numpy as np

from coupled import CoupledScenario
from coupled.coupled_scenario import Switches
from coupled.driver import run_coupled
from config import IDX, air_number_density

_OUT = os.path.join(os.path.dirname(__file__), "runs")
_AVOG = 6.02214076e23

# --- SO2 forcing: 1 tonne SO2 into V0 = 10 m x 10 m x 15 km (fixed NUMBER DENSITY; the pptv depends
# on the baseline's air density M). ---
_SO2_MASS_G = 1.0e6
_SO2_MW = 64.0
_V0_CM3 = (10.0 * 10.0 * 15000.0) * 1e6          # 1.5e12 cm^3
_SO2_NUMBER_DENSITY = _SO2_MASS_G / _SO2_MW * _AVOG / _V0_CM3   # 6.273e15 molec/cm^3

# --- stratospheric background gas composition [pptv] (initial plume = background + SO2 spike). ---
# NOTE (Ali, 2026-07-08) FOR ALL FUTURE PRODUCTION RUNS: initialize from SPUN-UP control-run
# chemistry instead of this static list, exactly as run_60day.py does -- run the frank-model
# (or gas-only) 60-d control at the SAME site/season, sample it at the release hour, transfer
# as mixing ratios, and zero gas H2SO4/SO3 (aerosol-free-control artifacts). One control per
# site is required (30N harvest: runs_60day/frank_control_ic.json, P=60 hPa preset; a 60N/15 km
# control must use the nearest preset, P=100 hPa). The static list below is retained only to
# reproduce the existing 810-run ensemble.
_BG_GAS_PPT = {"O2": 2.1e11, "O3": 1.18e6, "OH": 0.5, "HO2": 3.0,
               "NO": 450.0, "NO2": 450.0, "HCl": 777.0, "ClONO2": 127.0, "HNO3": 5000.0}

# --- AXES -------------------------------------------------------------------------------------------
# lat+alt baselines: (label, latitude, T[K], P[hPa], WTR[ppmv for RH=3% @ that T,P])
LAT_ALT = [
    ("30N_20km",      30.0, 210.0,  55.0,  6.9104),
    ("60N_15km",      60.0, 210.0, 120.0,  3.1673),
    ("30N_20km_213K", 30.0, 213.0,  55.0, 10.1834),
]
# background aerosol dist + co-varying background SO2 [pptv]
BACKGROUND = [
    ("sabr330", "sabr_330",  20.0),
    ("sabr220", "sabr_220",  20.0),
    ("cesm",    "cesm_g6",  100.0),
]
DILUTION = [("D1low", "D1"), ("D2med", "D2"), ("D3high", "D3"),
            ("burst", "burst"), ("D5vhigh", "D5")]
STICKING = [("a0p5", 0.5), ("a1p0", 1.0)]                     # condensation_alpha
NUCLEATION = [("nuc0p01", 0.01), ("nuc1", 1.0), ("nuc100", 100.0)]  # nucleation_rate_scale
COAG = [("cg0p5", 0.5), ("cg1", 1.0), ("cg2", 2.0)]          # coag_kernel_scale


def all_cases():
    """Ordered list of (case_id, axis-value-dict). Full Cartesian product = 810."""
    cases = []
    for la, bg, dl, st, nu, cg in itertools.product(LAT_ALT, BACKGROUND, DILUTION,
                                                    STICKING, NUCLEATION, COAG):
        cid = "__".join([la[0], bg[0], dl[0], st[0], nu[0], cg[0]])
        cases.append((cid, dict(lat_alt=la, background=bg, dilution=dl,
                                sticking=st, nucleation=nu, coag=cg)))
    return cases


def build_scenario(ax):
    la, bg, dl, st, nu, cg = ax["lat_alt"], ax["background"], ax["dilution"], \
        ax["sticking"], ax["nucleation"], ax["coag"]
    _, lat, T, P, WTR = la
    M = air_number_density(P, T)
    so2_ppt = _SO2_NUMBER_DENSITY / M * 1e12                  # fixed number density -> pptv @ this M
    conc = dict(_BG_GAS_PPT); conc["SO2"] = so2_ppt          # initial plume = background + SO2 spike
    return CoupledScenario(
        T=T, P=P, WTR=WTR, latitude=lat, longitude=0.0, day_of_year=172, start_utc_hour=0.0,
        days=10, DT=600.0, dt_couple=600.0, photolysis="tuvx", tomas_nbins=80,
        background_dist=bg[1], dilution_regime=dl[1], dilution_zero_species=(),
        dilution_background={"SO2": float(bg[2])},           # plume SO2 relaxes to background SO2
        ion_pair_rate=30.0, so2_ho2_rate=1.0e-18,
        condensation_alpha=float(st[1]), nucleation_rate_scale=float(nu[1]),
        coag_kernel_scale=float(cg[1]),
        switches=Switches(sulfur=True, nucleation=True, condensation=True, coagulation=True,
                          aerosol_to_j=False, heating_to_t=False, dilution=True),
        concentrations=conc)


def manifest_row(cid, ax, sc):
    M = air_number_density(sc.P, sc.T)
    return dict(
        case_id=cid, latitude=sc.latitude, T=sc.T, P=sc.P, WTR=sc.WTR,
        background_dist=sc.background_dist, bg_SO2_ppt=ax["background"][2],
        dilution_regime=sc.dilution_regime, condensation_alpha=sc.condensation_alpha,
        nucleation_rate_scale=sc.nucleation_rate_scale, coag_kernel_scale=sc.coag_kernel_scale,
        SO2_init_ppt=sc.concentrations["SO2"], SO2_init_molec_cm3=_SO2_NUMBER_DENSITY,
        SO2_injected_tonne=1.0, V0_cm3=_V0_CM3, tomas_nbins=sc.tomas_nbins, days=sc.days,
        DT=sc.DT, dt_couple=sc.dt_couple, photolysis=sc.photolysis, ion_pair_rate=sc.ion_pair_rate,
        so2_ho2_rate=sc.so2_ho2_rate, day_of_year=sc.day_of_year, start_utc_hour=sc.start_utc_hour,
        aerosol_to_j=sc.switches.aerosol_to_j, heating_to_t=sc.switches.heating_to_t, M_air=M)


def write_manifest():
    cases = all_cases()
    os.makedirs(_OUT, exist_ok=True)
    rows = [manifest_row(cid, ax, build_scenario(ax)) for cid, ax in cases]
    path = os.path.join(_OUT, "manifest.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return path, len(rows)


def run_one(cid, ax):
    import coupled.dilution as dl
    sc = build_scenario(ax)
    out = os.path.join(_OUT, cid)
    os.makedirs(out, exist_ok=True)
    t, x, aero, st_final, sd, jrec = run_coupled(
        sc, return_aerosol=True, return_state=True, return_size_dist=True, return_photolysis=True)
    M = air_number_density(sc.P, sc.T)
    days = t / 86400.0
    V = dl.volume_ratio(t, sc.dilution_regime)
    total_n = sd["n_cm3"].sum(axis=1)
    # dN/dlogDp on the (fixed) diameter bins
    from coupled.tomas_bridge import _bad
    dp_edges = _bad._xk_to_dp_um(np.asarray(st_final.xk)); logdp = np.log10(dp_edges)
    dp_mid = 10 ** (0.5 * (logdp[:-1] + logdp[1:])); dlogdp = logdp[1:] - logdp[:-1]
    dNdlogDp = sd["n_cm3"] / dlogdp[None, :]
    np.savez(os.path.join(out, "state.npz"),
             t=t, x=x, species=list(IDX), M=M,
             SA=aero["SA"], radius_cm=aero["radius_cm"], h2so4wp=aero["h2so4wp"],
             particulate_S=aero["particulate_S"], T=aero["T"],
             n_cm3=sd["n_cm3"], Dp_m=sd["Dp_m"], dp_mid_um=dp_mid, dNdlogDp=dNdlogDp,
             V_ratio=V, total_n=total_n,
             J_tmid=jrec["t_mid"], J=jrec["J"], J_equations=jrec["equations"])
    so2 = x[:, IDX["SO2"]] / M * 1e12
    return dict(case_id=cid, steps=len(t), SO2_end_ppt=float(so2[-1]),
                H2SO4_max_ppt=float(np.max(x[:, IDX["H2SO4"]] / M * 1e12)),
                N_max=float(total_n.max()), SA_max=float(np.nanmax(aero["SA"])),
                sulfate_end=float(aero["particulate_S"][-1]))


def _append_summary(row, path=None):
    path = path or os.path.join(_OUT, "summary.csv")
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)


def main(argv):
    os.makedirs(_OUT, exist_ok=True)
    cases = all_cases()
    mode = argv[0] if argv else "plan"
    if mode == "plan":
        path, n = write_manifest()
        print(f"{n} cases; manifest -> {path}")
        return
    if mode == "one":
        i = int(argv[1]); cid, ax = cases[i]
        print(f"[{i}] {cid} ...", flush=True)
        row = run_one(cid, ax); _append_summary(row)
        print(f"[{i}] done: {row}", flush=True)
        return
    if mode == "run":
        lo, hi = int(argv[1]), int(argv[2])
        # per-slice summary file so parallel workers never write the same CSV (merged post-hoc)
        spath = os.path.join(_OUT, f"summary_{lo}_{hi}.csv")
        for i in range(lo, min(hi, len(cases))):
            cid, ax = cases[i]
            if os.path.exists(os.path.join(_OUT, cid, "state.npz")):
                print(f"[{i}] {cid} SKIP (state.npz exists)", flush=True)
                continue
            try:
                print(f"[{i}] {cid} ...", flush=True)
                row = run_one(cid, ax); _append_summary(row, spath)
                print(f"[{i}] OK N_max={row['N_max']:.3e} SA_max={row['SA_max']:.1f}", flush=True)
            except Exception as e:
                _append_summary(dict(case_id=cid, steps=-1, SO2_end_ppt=float("nan"),
                                     H2SO4_max_ppt=float("nan"), N_max=float("nan"),
                                     SA_max=float("nan"), sulfate_end=float("nan")), spath)
                print(f"[{i}] FAILED: {type(e).__name__}: {str(e)[:150]}", flush=True)
        return
    print(main.__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
