# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Constant-OH vs diurnal-chemistry banana comparison (-> runs/plots/banana_OH_*.png).

Case: baseline 1 (30N / 20 km / 210 K), SABR-220 aged-air background, D2 med, alpha x1,
nuc x1, coag x1. The diurnal panel is the standard ensemble run; the constant-OH panel is a
NEW run with OH pinned to 5e5 molec/cm^3 in the gas ODE: the vector field is wrapped so every
rate evaluation sees OH = 5e5 and dOH/dt = 0 (all other species, photolysis of everything
else, dilution, and the microphysics are untouched).

Run (from SANDBOX/): python -m coupled.paper_ensemble.make_constant_oh_banana [replot]
The pinned run is cached in runs_special/30N_20km__sabr220__D2med__constOH5e5/.
"""
import dataclasses
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from coupled.paper_ensemble.make_paper_candidate_plots import _RUNS, case_dir

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(_RUNS, "plots")
_SPECIAL = os.path.join(_HERE, "runs_special")
_CID_DIURNAL = "30N_20km__sabr220__D2med__a1p0__nuc1__cg1"
_CID_CONST = "30N_20km__sabr220__D2med__constOH5e5"
OH_CONST = 5.0e5     # molec/cm^3
IDX_OH = 18

plt.rcParams.update({"font.family": "Helvetica", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130})


def run_const_oh():
    from coupled import driver                   # sets up sys.path for the flat gas model
    import jaxmodel.model as jm
    orig_factory = jm.make_frozen_vf

    def pinned_factory(opt):
        vf = orig_factory(opt)

        def vf_pinned(t, y, args):
            y_pin = y.at[IDX_OH].set(OH_CONST)   # every rate sees OH = 5e5
            dy = vf(t, y_pin, args)
            return dy.at[IDX_OH].set(0.0)        # ... and OH itself never moves

        return vf_pinned

    jm.make_frozen_vf = pinned_factory
    from coupled.paper_ensemble import run_ensemble as _re
    driver.make_frozen_vf = pinned_factory       # BDF stall fallback uses the same pinned RHS

    _re._OUT = _SPECIAL
    ax = dict(lat_alt=_re.LAT_ALT[0], background=("sabr220", "sabr_220", 20.0),
              dilution=("D2med", "D2"), sticking=("a1p0", 1.0), nucleation=("nuc1", 1.0),
              coag=("cg1", 1.0))
    from config import air_number_density
    build = _re.build_scenario
    sc_ppt_oh = OH_CONST / air_number_density(55.0, 210.0) * 1e12

    def build_pinned(a):
        sc = build(a)
        return dataclasses.replace(sc, concentrations={**sc.concentrations, "OH": sc_ppt_oh})

    _re.build_scenario = build_pinned
    row = _re.run_one(_CID_CONST, ax)
    print("const-OH run done:", {k: row[k] for k in ("case_id", "steps", "N_max", "SA_max")})
    z = np.load(os.path.join(_SPECIAL, _CID_CONST, "state.npz"))
    oh = z["x"][:, IDX_OH]
    print(f"OH pinned check: min={oh.min():.4g} max={oh.max():.4g} (target {OH_CONST:.4g})")


def fig(sa=False):
    panels = [(os.path.join(case_dir(_CID_DIURNAL), "state.npz"),
               "interactive chemistry (diurnal OH)"),
              (os.path.join(_SPECIAL, _CID_CONST, "state.npz"),
               f"constant OH = 5×10$^5$ molec cm$^{{-3}}$")]
    norm = LogNorm(1e-2, 2e3) if sa else LogNorm(1e2, 1e8)
    label = "dSA/dlogD$_p$ [µm$^2$ cm$^{-3}$]" if sa else "dN/dlogD$_p$ [cm$^{-3}$]"
    fig_, axs = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=True, sharey=True)
    pc = None
    for ax, (path, lab) in zip(axs, panels):
        z = np.load(path)
        t, dp = z["t"] / 86400.0, z["dp_mid_um"]
        field = z["dNdlogDp"] * np.pi * dp ** 2 if sa else z["dNdlogDp"]
        pc = ax.pcolormesh(t, dp, np.maximum(field, 1e-9).T, norm=norm,
                           cmap="viridis", rasterized=True, shading="nearest")
        ax.set_yscale("log")
        ax.set_ylim(dp[0], 3.0)
        ax.set_ylabel("dry diameter [µm]")
        ax.text(0.02, 0.95, lab, transform=ax.transAxes, fontsize=11, color="white",
                va="top", fontweight="bold")
    axs[-1].set_xlabel("day")
    fig_.suptitle("Diurnal vs constant-OH oxidation — 30°N / 20 km / 210 K, SABR-220, "
                  "D2 med, α ×1, nuc ×1, coag ×1", y=0.99, fontsize=12)
    fig_.tight_layout(rect=(0, 0, 0.88, 1))
    cax = fig_.add_axes((0.90, 0.12, 0.02, 0.76))
    fig_.colorbar(pc, cax=cax, label=label)
    fname = f"banana_OH_const_vs_diurnal{'_SA' if sa else ''}.png"
    fig_.savefig(os.path.join(_OUT, fname), bbox_inches="tight")
    plt.close(fig_)
    print(f"  {fname}")


def fig_snapshots(linear=False):
    """N (top) and SA (bottom) spectra at days 2/5/10 (t=2/5/10 d = midnight closing each
    day), diurnal vs constant OH. ``linear=True`` autoscales each panel linearly."""
    cases = [(os.path.join(case_dir(_CID_DIURNAL), "state.npz"),
              "diurnal OH", "#2a78d6", "-"),
             (os.path.join(_SPECIAL, _CID_CONST, "state.npz"),
              "constant OH = 5×10$^5$", "#e34948", "--")]
    days = [2, 5, 10]
    fig_, axs = plt.subplots(2, 3, figsize=(11.5, 7.0), sharex=True,
                             sharey=False if linear else "row")
    for k, day in enumerate(days):
        for path, lab, col, ls in cases:
            z = np.load(path)
            t, dp, nd = z["t"] / 86400.0, z["dp_mid_um"], z["dNdlogDp"]
            i = np.searchsorted(t, day - 1e-9)
            axs[0, k].plot(dp, np.maximum(nd[i], 1e-6), ls, lw=2.0, color=col, label=lab)
            axs[1, k].plot(dp, np.maximum(nd[i] * np.pi * dp ** 2, 1e-6), ls, lw=2.0,
                           color=col)
        for r in (0, 1):
            ax = axs[r, k]
            ax.set_xscale("log")
            ax.set_xlim(2e-3, 20)
            ax.grid(True, alpha=0.3, lw=0.5, color="#e1e0d9")
            if linear:
                ax.set_ylim(bottom=0)
            else:
                ax.set_yscale("log")
        axs[0, k].set_title(f"day {day} (midnight)", fontsize=11)
        if not linear:
            axs[0, k].set_ylim(1e-2, 3e6)
            axs[1, k].set_ylim(1e-4, 3e3)
        axs[1, k].set_xlabel("dry diameter [µm]")
    axs[0, 0].set_ylabel("dN/dlogD$_p$ [cm$^{-3}$]")
    axs[1, 0].set_ylabel("dSA/dlogD$_p$ [µm$^2$ cm$^{-3}$]")
    (axs[0, 0] if linear else axs[0, 2]).legend(fontsize=9, loc="upper right")
    fig_.suptitle("Diurnal vs constant-OH size distributions at midnight — 30°N / 20 km / "
                  "210 K, SABR-220, D2 med, α ×1, nuc ×1, coag ×1 "
                  f"({'linear y, per-panel scale' if linear else 'log y'})", y=1.0, fontsize=12)
    fig_.tight_layout(rect=(0, 0, 1, 0.97))
    fname = f"sizedist_OH_const_vs_diurnal{'_linear' if linear else ''}.png"
    fig_.savefig(os.path.join(_OUT, fname), bbox_inches="tight")
    plt.close(fig_)
    print(f"  {fname}")


if __name__ == "__main__":
    if "replot" not in sys.argv and not os.path.exists(
            os.path.join(_SPECIAL, _CID_CONST, "state.npz")):
        run_const_oh()
    fig()
    fig(sa=True)
    fig_snapshots()
    fig_snapshots(linear=True)
