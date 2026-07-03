# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Phase 7 gate: ONE scenario, progressively more physics via switches. Runs the coupled box as
gas-only -> +microphysics -> +aerosol->J -> +heating -> +dilution and overlays the key responses,
demonstrating the single unified input reproduces every sub-case. Writes coupled/validation/phase7_*.png.

Nucleation is left OFF in every case (condensation+coagulation only) for a numerically robust demo
(nucleation can run away and stress the gas solver -- see CAVEATS). Run: python -m coupled.validate_phase7
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coupled import CoupledScenario
from coupled.coupled_scenario import Switches
from coupled.driver import run_coupled
from config import IDX

_OUT = os.path.join(os.path.dirname(__file__), "validation")

_BASE = dict(T=210.0, P=68.0, WTR=5.0, latitude=0.0, day_of_year=80, start_utc_hour=6.0,
             days=1, DT=3600.0, dt_couple=3600.0, photolysis="tuvx", aerosol_band_km=(10.0, 30.0),
             dilution_rate=3.0e-5,
             concentrations={"O2": 2.1e11, "O3": 1.18e6, "SO2": 1.0e4, "OH": 0.5, "HO2": 3.0,
                             "NO": 450.0, "NO2": 450.0, "HCl": 777.0, "ClONO2": 127.0})

# progressively enable physics (nucleation stays off for stability)
_CASES = [
    ("gas only", Switches(sulfur=True)),
    ("+microphysics", Switches(sulfur=True, condensation=True, coagulation=True)),
    ("+aerosol->J", Switches(sulfur=True, condensation=True, coagulation=True, aerosol_to_j=True)),
    ("+heating->T", Switches(sulfur=True, condensation=True, coagulation=True, aerosol_to_j=True,
                             heating_to_t=True)),
    ("+dilution", Switches(sulfur=True, condensation=True, coagulation=True, aerosol_to_j=True,
                           heating_to_t=True, dilution=True)),
]


def main():
    os.makedirs(_OUT, exist_ok=True)
    results = []
    for label, sw in _CASES:
        sc = CoupledScenario(switches=sw, **_BASE)
        t, x, aero = run_coupled(sc, return_aerosol=True)
        results.append((label, t / 86400.0, x, aero))
        M = x[0, IDX["O2"]] / 0.21
        print(f"{label:16s}: H2SO4_end={x[-1, IDX['H2SO4']]/M*1e12:8.1f} pptv  "
              f"SA_end={aero['SA'][-1]:.3f}  T_end={aero['T'][-1]:.3f} K")

    M = results[0][2][0, IDX["O2"]] / 0.21
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    for label, days, x, aero in results:
        axs[0, 0].plot(days, x[:, IDX["H2SO4"]] / M * 1e12, lw=1.3, label=label)
        axs[0, 1].plot(days, x[:, IDX["SO2"]] / M * 1e12, lw=1.3, label=label)
        axs[1, 0].plot(days, aero["SA"], lw=1.3, label=label)
        axs[1, 1].plot(days, aero["T"], lw=1.3, label=label)
    axs[0, 0].set_title("gas H2SO4 [pptv]"); axs[0, 1].set_title("SO2 [pptv]")
    axs[1, 0].set_title("aerosol surface area [um^2/cm^3]"); axs[1, 1].set_title("box T [K]")
    for ax in axs.flat:
        ax.set_xlabel("day"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle("Phase 7: one input, progressively more physics via switches")
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "phase7_subcases.png"), dpi=110); plt.close(fig)
    print(f"wrote plots to {_OUT}/")


if __name__ == "__main__":
    main()
