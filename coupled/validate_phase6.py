# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Phase 6 gate: dilution. (1) passive-tracer decay vs the analytic exp(-k t), and (2) a coupled run
with dilution on vs off showing SO2 relax toward background and H2SO4 suppressed. Writes
coupled/validation/phase6_*.png.  Run:  python -m coupled.validate_phase6
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coupled import CoupledScenario
from coupled.coupled_scenario import Switches
from coupled.driver import run_coupled
from coupled import dilution as dl
from config import IDX

_OUT = os.path.join(os.path.dirname(__file__), "validation")


def _sc(dilute, kdil=1.0e-5):
    return CoupledScenario(T=210.0, P=68.0, WTR=5.0, latitude=0.0, day_of_year=80, start_utc_hour=6.0,
                           days=3, DT=3600.0, dt_couple=3600.0, photolysis="sza",
                           dilution_rate=kdil,
                           switches=Switches(sulfur=True, dilution=dilute),
                           concentrations={"O2": 2.1e11, "O3": 1.18e6, "SO2": 9.5e5, "OH": 0.5,
                                           "HO2": 3.0, "NO": 450.0, "NO2": 450.0,
                                           "HCl": 777.0, "ClONO2": 127.0})


def main():
    os.makedirs(_OUT, exist_ok=True)

    # (1) passive tracer: simulate the analytic relaxation and overlay the closed form
    k = 1.0e-5
    t = np.arange(0.0, 3 * 86400.0 + 1, 3600.0)
    c = 100.0
    sim = [c]
    for _ in t[1:]:
        c = float(dl.dilute_gas(np.array([c]), np.array([0.0]), k, 3600.0)[0])
        sim.append(c)
    sim = np.array(sim)
    analytic = 100.0 * np.exp(-k * t)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(t / 86400.0, sim, "o", ms=3, label="stepped dilute_gas")
    ax.plot(t / 86400.0, analytic, "-", lw=1.2, label="analytic 100·exp(−k t)")
    ax.set_xlabel("day"); ax.set_ylabel("passive tracer"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(f"Phase 6: passive-tracer dilution (k={k:.0e}/s); max err "
                 f"{np.max(np.abs(sim-analytic)):.2e}")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "phase6_tracer.png"), dpi=110); plt.close(fig)

    # (2) coupled: dilution on vs off
    t0, x_off = run_coupled(_sc(False))
    t1, x_on = run_coupled(_sc(True))
    days = t0 / 86400.0
    M = x_off[0, IDX["O2"]] / 0.21

    def ppt(x, name):
        return x[:, IDX[name]] / M * 1e12

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.plot(days, ppt(x_off, "SO2"), lw=1.4, label="no dilution")
    a1.plot(days, ppt(x_on, "SO2"), lw=1.4, ls="--", label="dilution")
    a1.set_xlabel("day"); a1.set_ylabel("SO2 [pptv]"); a1.set_title("SO2"); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(days, ppt(x_off, "H2SO4"), lw=1.4, label="no dilution")
    a2.plot(days, ppt(x_on, "H2SO4"), lw=1.4, ls="--", label="dilution")
    a2.set_xlabel("day"); a2.set_ylabel("H2SO4 [pptv]"); a2.set_title("H2SO4 (product)")
    a2.legend(); a2.grid(alpha=0.3)
    fig.suptitle("Phase 6: dilution relaxes the plume toward background")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "phase6_dilution.png"), dpi=110); plt.close(fig)

    print("Phase 6 dilution validation")
    print(f"passive tracer max |sim-analytic| = {np.max(np.abs(sim-analytic)):.2e}")
    print(f"SO2   no-dil {ppt(x_off,'SO2')[-1]:.1f} -> dil {ppt(x_on,'SO2')[-1]:.1f} pptv")
    print(f"H2SO4 no-dil {ppt(x_off,'H2SO4')[-1]:.1f} -> dil {ppt(x_on,'H2SO4')[-1]:.1f} pptv")
    print(f"wrote plots to {_OUT}/")


if __name__ == "__main__":
    main()
