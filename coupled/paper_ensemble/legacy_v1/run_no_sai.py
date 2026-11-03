# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""No-SAI control run: stratospheric background chemistry, NO SO2 injection and NO dilution.

A closed box (dilution OFF) seeded with the background gas composition (SO2 at a clean 20 pptv, i.e.
no injected plume) at the reference 30N / 20 km / 210 K baseline, 80-bin TOMAS, 10 days. This shows
what the gas-phase species relax to (diurnal photochemical steady state) with no SAI forcing -- the
control against which the 810 injection runs are compared. Own directory: runs_no_sai/.

Run (from SANDBOX/): python -m coupled.paper_ensemble.run_no_sai
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coupled import CoupledScenario
from coupled.coupled_scenario import Switches
from coupled.driver import run_coupled
from config import IDX, air_number_density

_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_no_sai")
_BG_GAS_PPT = {"O2": 2.1e11, "O3": 1.18e6, "OH": 0.5, "HO2": 3.0, "NO": 450.0, "NO2": 450.0,
               "HCl": 777.0, "ClONO2": 127.0, "HNO3": 5000.0, "SO2": 20.0}   # SO2 = clean bg, NO injection

plt.rcParams.update({"font.family": "Helvetica", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3})


def scenario():
    return CoupledScenario(
        T=210.0, P=55.0, WTR=6.9104, latitude=30.0, longitude=0.0, day_of_year=172,
        start_utc_hour=0.0, days=10, DT=600.0, dt_couple=600.0, photolysis="tuvx", tomas_nbins=80,
        background_dist="sabr_330", ion_pair_rate=30.0, so2_ho2_rate=1.0e-18,
        switches=Switches(sulfur=True, nucleation=True, condensation=True, coagulation=True,
                          aerosol_to_j=False, heating_to_t=False, dilution=False),  # <-- no dilution
        concentrations=dict(_BG_GAS_PPT))


def main():
    os.makedirs(_OUT, exist_ok=True)
    sc = scenario()
    print("running no-SAI control (no injection, no dilution, 80-bin, 10 d) ...", flush=True)
    t, x, aero, st, sd = run_coupled(sc, return_aerosol=True, return_state=True, return_size_dist=True)
    M = air_number_density(sc.P, sc.T)
    days = t / 86400.0
    np.savez(os.path.join(_OUT, "state.npz"), t=t, x=x, species=list(IDX), M=M,
             SA=aero["SA"], radius_cm=aero["radius_cm"], h2so4wp=aero["h2so4wp"],
             particulate_S=aero["particulate_S"], n_cm3=sd["n_cm3"], Dp_m=sd["Dp_m"])
    # plot every species (ppt) vs time -> see which reach equilibrium
    sp = list(IDX)
    ppt = x / M * 1e12
    nonzero = [s for s in sp if np.nanmax(ppt[:, IDX[s]]) > 0]
    ncol = 5; nrow = int(np.ceil(len(nonzero) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.2 * nrow))
    for ax, s in zip(axs.ravel(), nonzero):
        ax.plot(days, ppt[:, IDX[s]], color="#2c7fb8", lw=1.0)
        ax.set_title(s, fontsize=9)
        ax.set_xlabel("days")
        end = ppt[-1, IDX[s]]
        ax.text(0.5, 0.92, f"end={end:.3g} ppt", transform=ax.transAxes, ha="center",
                va="top", fontsize=7, color="0.3")
    for ax in axs.ravel()[len(nonzero):]:
        ax.axis("off")
    fig.suptitle("No-SAI control: background gas species vs time (no injection, no dilution) — ppt",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "no_sai_species.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    # quick end-state print
    print("end-of-run (day 10) mixing ratios [ppt]:", flush=True)
    for s in nonzero:
        print(f"  {s:8s} {ppt[-1, IDX[s]]:.4g}", flush=True)
    print(f"wrote {_OUT}/state.npz + no_sai_species.png", flush=True)


if __name__ == "__main__":
    main()
