# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Dilution 1 -- Low Latitude, High Altitude, Clean Stratosphere (20 km, 30N, summer, 10 days).

Full coupling (gas chemistry + TUV-x photolysis + TOMAS nucleation/condensation/coagulation +
aerosol->J + heating->T + dilution). A high-SO2 plume dilutes (regime D1 'Low Kz', Schumann volume
expansion) into CLEAN stratosphere: the background has NO SO2/SO3/H2SO4 and NO short-lived radicals.
Initial + background aerosol = the TOMAS Marianna 'redcircles' clean distribution.

Saves results to coupled/analyses/d1_clean.npz and writes plots to coupled/analyses/:
  banana, size distributions, species concentrations, dilution trends.  Run: python -m coupled.run_dilution_d1_clean
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
from coupled import tomas_bridge as tb
from config import IDX

_OUT = os.path.join(os.path.dirname(__file__), "analyses")

# clean-stratosphere background: no plume sulfur, no short-lived radicals (regenerate photochemically)
_ZERO_BG = ("SO2", "SO3", "H2SO4", "OH", "HO2", "O1D", "O", "NO3", "Cl", "ClO", "Br", "BrO",
            "CH3", "OClO")


def _scenario():
    return CoupledScenario(
        T=215.0, P=55.0, WTR=4.5,               # 20 km, 30N summer
        latitude=30.0, longitude=0.0, day_of_year=172, start_utc_hour=6.0,
        days=10, DT=600.0, dt_couple=600.0, photolysis="tuvx",
        dilution_regime="D1", dilution_zero_species=_ZERO_BG,
        switches=Switches(sulfur=True, nucleation=True, condensation=True, coagulation=True,
                          aerosol_to_j=True, heating_to_t=True, dilution=True),
        concentrations={"O2": 2.1e11, "O3": 1.18e6, "SO2": 9.5e5, "OH": 0.5, "HO2": 3.0,
                        "NO": 450.0, "NO2": 450.0, "HCl": 777.0, "ClONO2": 127.0, "HNO3": 5000.0})


def _bin_diam_edges_um():
    # diameter bin edges [um] from the TOMAS mass boundaries xk at the dry sulfate density (1770)
    xk = np.asarray(tb.tcfg.xk_boundaries(), dtype=float)         # kg, (nbins+1,)
    Dp_m = (6.0 * xk / (np.pi * 1770.0)) ** (1.0 / 3.0)
    return Dp_m * 1e6


def main():
    os.makedirs(_OUT, exist_ok=True)
    sc = _scenario()
    print("Running Dilution-1 clean-stratosphere (20 km, 30N, summer, 10 d, full coupling)...")
    try:
        # run_coupled appends extras in order: aerosol, state, size_dist
        t, x, aero, st_final, sd = run_coupled(sc, return_aerosol=True, return_state=True,
                                               return_size_dist=True)
    except RuntimeError as e:
        print(f"RUN STOPPED (reported, not swallowed): {e}")
        raise
    days = t / 86400.0
    M = x[0, IDX["O2"]] / 0.21

    def ppt(name):
        return x[:, IDX[name]] / M * 1e12

    # dN/dlogDp on fixed diameter bins: n_cm3(t,bin) / dlogDp(bin)
    edges = _bin_diam_edges_um()
    dp_mid = np.sqrt(edges[:-1] * edges[1:])
    dlogdp = np.log10(edges[1:] / edges[:-1])
    dNdlogDp = sd["n_cm3"] / dlogdp[None, :]                      # (n_t, nbins) cm^-3
    V = dl.volume_ratio(t, "D1")
    kdil = np.array([dl.kdil_from_regime("D1", t[i], t[i + 1]) for i in range(len(t) - 1)])

    np.savez(os.path.join(_OUT, "d1_clean.npz"), t=t, x=x, species=list(IDX),
             SA=aero["SA"], radius_cm=aero["radius_cm"], h2so4wp=aero["h2so4wp"],
             particulate_S=aero["particulate_S"], T=aero["T"],
             n_cm3=sd["n_cm3"], Dp_m=sd["Dp_m"], dp_mid_um=dp_mid, dNdlogDp=dNdlogDp,
             V_ratio=V, kdil=kdil, M=M)

    # 1) banana plot: dN/dlogDp(t, Dp)
    fig, ax = plt.subplots(figsize=(9, 5))
    pm = ax.pcolormesh(days, dp_mid, np.maximum(dNdlogDp, 1e-3).T, shading="auto",
                       norm=matplotlib.colors.LogNorm(vmin=1e-1, vmax=max(1e2, dNdlogDp.max())),
                       cmap="turbo")
    ax.set_yscale("log"); ax.set_ylabel("dry diameter [um]"); ax.set_xlabel("day")
    ax.set_title("Dilution 1 (clean strat): aerosol size distribution dN/dlogDp [cm$^{-3}$]")
    fig.colorbar(pm, ax=ax, label="dN/dlogDp [cm$^{-3}$]"); fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "d1_banana.png"), dpi=120); plt.close(fig)

    # 2) size distributions at selected days
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0):
        i = int(np.argmin(np.abs(days - d)))
        ax.step(dp_mid, dNdlogDp[i], where="mid", lw=1.4, label=f"{days[i]:.1f} d")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("dry diameter [um]"); ax.set_ylabel("dN/dlogDp [cm$^{-3}$]")
    ax.set_title("Size distributions"); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "d1_size_distributions.png"), dpi=120); plt.close(fig)

    # 3) species concentrations
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in ("SO2", "SO3", "H2SO4"):
        a1.plot(days, np.maximum(ppt(name), 1e-6), lw=1.4, label=name)
    a1.set_yscale("log"); a1.set_xlabel("day"); a1.set_ylabel("pptv"); a1.set_title("Sulfur gases")
    a1.legend(); a1.grid(alpha=0.3)
    for name in ("OH", "HO2", "O3", "NO2", "HCl"):
        a2.plot(days, np.maximum(ppt(name), 1e-6), lw=1.2, label=name)
    a2.set_yscale("log"); a2.set_xlabel("day"); a2.set_ylabel("pptv"); a2.set_title("Radicals / ambient")
    a2.legend(fontsize=8); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "d1_species.png"), dpi=120); plt.close(fig)

    # 4) dilution trends + aerosol response
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(days, V, lw=1.6); axs[0, 0].set_yscale("log")
    axs[0, 0].set_title("plume volume V(t)/V0 (D1 Low Kz)"); axs[0, 0].set_xlabel("day")
    axs[0, 1].plot(days[1:], kdil, lw=1.4, color="C1"); axs[0, 1].set_yscale("log")
    axs[0, 1].set_title("dilution rate k_dil(t) [1/s]"); axs[0, 1].set_xlabel("day")
    axs[1, 0].plot(days, ppt("SO2"), lw=1.5, color="C3")
    axs[1, 0].set_yscale("log"); axs[1, 0].set_title("SO2 [pptv] (plume -> clean bg)")
    axs[1, 0].set_xlabel("day")
    axs[1, 1].plot(days, aero["SA"], lw=1.5, label="surface area [um^2/cm^3]")
    axs[1, 1].plot(days, sd["n_cm3"].sum(axis=1), lw=1.2, ls="--", label="total N [cm^-3]")
    axs[1, 1].set_yscale("log"); axs[1, 1].set_title("aerosol response"); axs[1, 1].set_xlabel("day")
    axs[1, 1].legend(fontsize=8)
    for ax in axs.flat:
        ax.grid(alpha=0.3)
    fig.suptitle("Dilution 1 (Low Kz, clean stratosphere): dilution + aerosol trends")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "d1_dilution_trends.png"), dpi=120); plt.close(fig)

    print(f"steps={len(t)}  SO2 {ppt('SO2')[0]:.0f}->{ppt('SO2')[-1]:.1f} pptv  "
          f"H2SO4 max {np.max(ppt('H2SO4')):.1f} pptv  total N max {sd['n_cm3'].sum(axis=1).max():.2e}/cm3  "
          f"SA max {np.nanmax(aero['SA']):.3f}")
    print(f"wrote npz + 4 plots to {_OUT}/")


if __name__ == "__main__":
    main()
