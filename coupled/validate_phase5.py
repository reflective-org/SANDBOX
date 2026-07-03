# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Phase 5 gate: radiative heating -> temperature. (1) diurnal dT/dt vs solar zenith angle, and
(2) a coupled run with heating_to_t showing the box T evolve. Writes coupled/validation/phase5_*.png.

Run:  python -m coupled.validate_phase5
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coupled import CoupledScenario
from coupled.coupled_scenario import Switches
from coupled.driver import run_coupled
from coupled import heating as ht
from coupled.model_bridge import to_model_config, initial_state

_OUT = os.path.join(os.path.dirname(__file__), "validation")


def main():
    # (1) dT/dt vs SZA: sweep the sun from overhead to the horizon (cheap; ~one radiation solve each).
    sc = CoupledScenario(T=210.0, P=68.0, WTR=5.0, latitude=0.0, day_of_year=80, start_utc_hour=12.0,
                         days=1, DT=3600.0, dt_couple=3600.0, photolysis="tuvx",
                         concentrations={"O2": 2.1e11, "O3": 1.18e6, "HCl": 777.0, "ClONO2": 127.0})
    cfg = to_model_config(sc)
    conc = initial_state(sc)
    # sweep start_utc_hour to move the sun (lat 0): build a cfg per hour via T-independent solar time
    hours = np.arange(0.0, 24.01, 1.0)
    dTdt = []
    for h in hours:
        cfg.start_utc_hour = float(h)
        dTdt.append(ht.box_dTdt(cfg, conc, 0.0) * 86400.0)   # K/day
    cfg.start_utc_hour = 12.0
    dTdt = np.array(dTdt)

    os.makedirs(_OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hours, dTdt, marker="o", lw=1.4)
    ax.set_xlabel("UTC hour (lat 0, day 80)"); ax.set_ylabel("O3 photochemical heating [K/day]")
    ax.set_title("Phase 5: diurnal box heating rate (night gated to 0)")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "phase5_dTdt_diurnal.png"), dpi=110); plt.close(fig)

    # (2) coupled run with heating -> T evolves. Condensation+coagulation (no nucleation -> stable).
    sc2 = CoupledScenario(T=210.0, P=68.0, WTR=5.0, latitude=0.0, day_of_year=80, start_utc_hour=6.0,
                          days=3, DT=3600.0, dt_couple=3600.0, photolysis="tuvx",
                          switches=Switches(sulfur=True, condensation=True, coagulation=True,
                                            heating_to_t=True),
                          concentrations={"O2": 2.1e11, "O3": 1.18e6, "SO2": 1.0e4, "OH": 0.5,
                                          "HO2": 3.0, "HCl": 777.0, "ClONO2": 127.0,
                                          "NO": 450.0, "NO2": 450.0})
    t, x, aero = run_coupled(sc2, return_aerosol=True)
    days = t / 86400.0
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(days, aero["T"], color="C3", lw=1.5)
    ax.set_xlabel("day"); ax.set_ylabel("box temperature [K]")
    ax.set_title(f"Phase 5: coupled box T with heating_to_t (dT={aero['T'][-1]-aero['T'][0]:+.2f} K)")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "phase5_box_temperature.png"), dpi=110); plt.close(fig)

    print("Phase 5 heating->T validation")
    print(f"diurnal O3 heating: max {np.max(dTdt):.3f} K/day at noon, night {dTdt[0]:.3f} K/day")
    print(f"coupled run: T {aero['T'][0]:.2f} -> {aero['T'][-1]:.2f} K over {sc2.days} d "
          f"(min {np.min(aero['T']):.2f}, max {np.max(aero['T']):.2f})")
    print(f"wrote plots to {_OUT}/")


if __name__ == "__main__":
    main()
