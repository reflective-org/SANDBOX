# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Paper simulation matrix -- parameterized driver for the coupled model.

Runs a MATRIX of CoupledScenario cases (each defined by a dict of overrides) into its own
clearly-named output directory under coupled/analyses/paper/<case_name>/, saving the full npz +
the standard D1 plot set (reuses run_dilution_d1_clean's plotting/style). A summary CSV collects
the headline numbers across cases.

The matrix is defined in MATRIX below -- one dict per case. Everything not overridden falls back to
BASE (the corrected, heating-off, paper-quality config). Add axes (altitude/pressure, latitude,
season/day_of_year, injected SO2, dilution regime, tomas_nbins, so2_ho2_rate, ...) by adding cases.

Run one case:   python -m coupled.paper_matrix <case_name>
Run all:        python -m coupled.paper_matrix all
List cases:     python -m coupled.paper_matrix list
"""

import os
import sys

import numpy as np

from coupled import CoupledScenario
from coupled.coupled_scenario import Switches
from coupled.driver import run_coupled
from config import IDX, air_number_density

_OUT = os.path.join(os.path.dirname(__file__), "analyses", "paper")

# --- BASE config: the corrected, paper-quality defaults (edit here to change all cases) ---
#   heating OFF (SW-only, spurious drift); corrected SO2+HO2 -> SO3+OH at 1e-18 (JPL upper limit);
#   80-bin TOMAS; ion-induced nucleation on; 10 days; clean-stratosphere background.
_ZERO_BG = ("SO2", "SO3", "H2SO4", "OH", "HO2", "O1D", "O", "NO3", "Cl", "ClO", "Br", "BrO",
            "CH3", "OClO")


def _base_kwargs():
    return dict(
        T=215.0, P=55.0, WTR=4.5, latitude=30.0, longitude=0.0, day_of_year=172,
        start_utc_hour=6.0, days=10, DT=600.0, dt_couple=600.0, photolysis="tuvx",
        dilution_regime="D1", dilution_zero_species=_ZERO_BG,
        ion_pair_rate=30.0, tomas_nbins=80, so2_ho2_rate=1.0e-18,
        aerosol_thickness_km=1.0,
        switches=Switches(sulfur=True, nucleation=True, condensation=True, coagulation=True,
                          aerosol_to_j=True, heating_to_t=False, dilution=True),
        concentrations=None,   # filled per case from _concentrations() (needs M from T,P)
    )


def _concentrations(T, P):
    M = air_number_density(P, T)
    def ppt(molec_cm3):
        return molec_cm3 / M * 1e12
    return {"O2": 2.1e11, "O3": 1.18e6, "SO2": 2.9e9, "OH": 0.5, "HO2": 3.0,
            "NO": 450.0, "NO2": 450.0, "HCl": 777.0, "ClONO2": 127.0, "HNO3": 5000.0,
            "H2SO4": ppt(1.0e5)}, {"SO2": 15.0, "OH": ppt(5.0e5)}


# --- THE MATRIX: one entry per case. Keys override _base_kwargs(). ADD/EDIT cases here. ---
# Placeholder starter set (the established D1 reference). Awaiting the paper's matrix axes.
MATRIX = {
    "D1_ref": {},   # base as-is: D1, 20 km, 30N, summer, corrected chemistry, heating off, 80-bin
}


def build_scenario(overrides):
    kw = _base_kwargs()
    kw.update(overrides)
    conc, bg = _concentrations(kw["T"], kw["P"])
    kw["concentrations"] = conc
    return CoupledScenario(dilution_background=bg, **kw)


def run_case(name):
    from coupled import run_dilution_d1_clean as d1   # reuse its plotting + Helvetica style
    overrides = MATRIX[name]
    sc = build_scenario(overrides)
    out = os.path.join(_OUT, name)
    os.makedirs(out, exist_ok=True)
    d1._OUT = out                                     # redirect the shared plotters here
    d1._write_inputs_md(sc, __import__("coupled.aerosol_props", fromlist=["het_inputs"]).het_inputs(
        __import__("coupled.tomas_bridge", fromlist=["initial_tomas_state"]).initial_tomas_state(sc)))
    print(f"[{name}] running ({sc.tomas_nbins}-bin, {sc.days}d, regime {sc.dilution_regime}) -> {out}/",
          flush=True)
    t, x, aero, st, sd, jrec = run_coupled(sc, return_aerosol=True, return_state=True,
                                           return_size_dist=True, return_photolysis=True)
    M = x[0, IDX["O2"]] / 0.21
    so2 = x[:, IDX["SO2"]] / M * 1e12
    summary = dict(name=name, steps=len(t), SO2_end_ppt=so2[-1],
                   H2SO4_max_ppt=float(np.max(x[:, IDX["H2SO4"]] / M * 1e12)),
                   N_max=float(sd["n_cm3"].sum(axis=1).max()), SA_max=float(np.nanmax(aero["SA"])))
    np.savez(os.path.join(out, "state.npz"), t=t, x=x, species=list(IDX),
             SA=aero["SA"], radius_cm=aero["radius_cm"], h2so4wp=aero["h2so4wp"],
             particulate_S=aero["particulate_S"], T=aero["T"], n_cm3=sd["n_cm3"], Dp_m=sd["Dp_m"])
    print(f"[{name}] done: {summary}", flush=True)
    return summary


def main(argv):
    os.makedirs(_OUT, exist_ok=True)
    if not argv or argv[0] == "list":
        print("cases:", ", ".join(MATRIX)); return
    names = list(MATRIX) if argv[0] == "all" else [argv[0]]
    rows = [run_case(n) for n in names]
    import csv
    with open(os.path.join(_OUT, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} case(s) + summary.csv to {_OUT}/")


if __name__ == "__main__":
    main(sys.argv[1:])
