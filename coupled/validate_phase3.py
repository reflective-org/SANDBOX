# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Phase 3 gate: end-to-end coupled gas-chemistry <-> TOMAS microphysics run + plots.

Runs the coupled driver with TOMAS microphysics ON and reports/plots:
  * the sulfur budget across gas (SO2+SO3+H2SO4) and particulate (SO4) reservoirs + total-S drift,
  * aerosol surface area, effective radius, and H2SO4 weight-percent over time,
  * the aerosol size distribution before vs after (condensation growth + nucleation).
Writes to coupled/validation/. Run:  python -m coupled.validate_phase3
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coupled import CoupledScenario
from coupled.driver import run_coupled
from coupled import tomas_bridge as tb
from coupled.aerosol_props import _wet_diameters_m
from config import IDX

import jax.numpy as jnp

_OUT = os.path.join(os.path.dirname(__file__), "validation")


def main():
    # Moderate SO2 + dt_couple that keeps the per-step gas H2SO4 within TOMAS's stable range (AD-3.9).
    sc = CoupledScenario(T=210.0, P=68.0, WTR=5.0, latitude=0.0, longitude=0.0, day_of_year=80,
                         start_utc_hour=6.0, days=2, DT=3600.0, dt_couple=3600.0, photolysis="sza",
                         concentrations={"O2": 2.1e11, "O3": 1.18e6, "SO2": 1.0e4, "OH": 0.5,
                                         "HO2": 3.0, "HCl": 777.0, "ClONO2": 127.0,
                                         "NO": 450.0, "NO2": 450.0})
    sc.switches.nucleation = True
    sc.switches.condensation = True
    sc.switches.coagulation = True

    st0 = tb.initial_tomas_state(sc)
    t, x, aero, st_final = run_coupled(sc, return_aerosol=True, return_state=True)
    days = t / 86400.0

    gasS = x[:, IDX["SO2"]] + x[:, IDX["SO3"]] + x[:, IDX["H2SO4"]]
    partS = aero["particulate_S"]
    totS = gasS + partS
    drift = (totS[-1] - totS[0]) / totS[0]

    os.makedirs(_OUT, exist_ok=True)

    # 1) sulfur budget
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(days, gasS, label="gas S (SO2+SO3+H2SO4)", lw=1.5)
    ax.plot(days, partS, label="particulate S (SO4)", lw=1.5)
    ax.plot(days, totS, label="total S", lw=1.8, color="k")
    ax.set_xlabel("day"); ax.set_ylabel("S [molec/cm^3]")
    ax.set_title(f"Phase 3 sulfur budget (gas<->aerosol); total-S drift {drift:+.2e}")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "phase3_sulfur_budget.png"), dpi=110); plt.close(fig)

    # 2) SA + effective radius + weight percent
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.plot(days, aero["SA"], color="C2", lw=1.6)
    a1.set_xlabel("day"); a1.set_ylabel("surface area [um^2/cm^3]")
    a1.set_title("Aerosol surface area (feeds het chemistry)"); a1.grid(alpha=0.3)
    a2.plot(days, aero["radius_cm"] * 1e4, color="C3", lw=1.6, label="effective radius [um]")
    a2t = a2.twinx()
    a2t.plot(days, aero["h2so4wp"], color="C4", lw=1.2, ls="--", label="H2SO4 wt%")
    a2.set_xlabel("day"); a2.set_ylabel("effective radius [um]"); a2t.set_ylabel("H2SO4 wt%")
    a2.set_title("Effective radius + composition"); a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "phase3_aerosol_props.png"), dpi=110); plt.close(fig)

    # 3) size distribution before vs after
    Dp0, _ = _wet_diameters_m(st0)
    Dpf, _ = _wet_diameters_m(st_final)
    N0 = np.asarray(st0.Nk) / float(st0.boxvol)      # #/cm^3
    Nf = np.asarray(st_final.Nk) / float(st_final.boxvol)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.step(np.asarray(Dp0) * 1e6, N0, where="mid", label="initial", lw=1.5)
    ax.step(np.asarray(Dpf) * 1e6, Nf, where="mid", label=f"after {sc.days} d", lw=1.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("wet diameter [um]"); ax.set_ylabel("N per bin [cm^-3]")
    ax.set_title("Aerosol size distribution: condensation growth + nucleation")
    ax.legend(); ax.grid(alpha=0.3, which="both"); fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "phase3_size_distribution.png"), dpi=110); plt.close(fig)

    print(f"Phase 3 coupled run: {sc.days} d, dt_couple={sc.dt_couple}s, {len(t)} outer steps")
    print(f"SO2      {x[0, IDX['SO2']]:.4e} -> {x[-1, IDX['SO2']]:.4e} molec/cm^3")
    print(f"gas S    {gasS[0]:.4e} -> {gasS[-1]:.4e}")
    print(f"part. S  {partS[0]:.4e} -> {partS[-1]:.4e}")
    print(f"total S drift: {drift:+.3e}  (TOMAS-internal; coupling handoff is exact, see AD-3.10)")
    print(f"SA       {aero['SA'][0]:.3f} -> {aero['SA'][-1]:.3f} um^2/cm^3")
    print(f"r_eff    {aero['radius_cm'][0]*1e4:.4f} -> {aero['radius_cm'][-1]*1e4:.4f} um")
    print(f"H2SO4wp  {aero['h2so4wp'][0]:.2f} -> {aero['h2so4wp'][-1]:.2f} %")
    print(f"max gas H2SO4 {float(np.max(x[:, IDX['H2SO4']])):.3e} molec/cm^3 (< 1e9 stable)")
    print(f"wrote plots to {_OUT}/")


if __name__ == "__main__":
    main()
