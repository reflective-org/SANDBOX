# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Background-comparison banana plot (-> runs/plots/banana_backgrounds_D2med.png):
dN/dlogDp evolution for the SAME plume (baseline 1: 30N / 20 km / 210 K, D2 med,
alpha x1, nuc x1, coag x1) released into three backgrounds:

  SABR-330 (young air, high N2O)  |  SABR-220 (aged air, low N2O)  |  AER-2D geoengineered

SABR cases come from runs/, the aer_geo case from runs_geo/ (via case_dir).

Run (from SANDBOX/): python -m coupled.paper_ensemble.make_background_banana
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from coupled.paper_ensemble.make_paper_candidate_plots import _RUNS, case_dir

_OUT = os.path.join(_RUNS, "plots")
CASE = "30N_20km__{bg}__D2med__a1p0__nuc1__cg1"
BGS = [("sabr330", "SABR-330 (young air, high N$_2$O)"),
       ("sabr220", "SABR-220 (aged air, low N$_2$O)"),
       ("aergeo", "AER-2D geoengineered stratosphere")]

plt.rcParams.update({"font.family": "Helvetica", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130})


BG_COLOR = {"sabr330": "#2a78d6", "sabr220": "#1baf7a", "aergeo": "#eda100"}


def fig_background_dists(linear=False, stp=False):
    """The three seeded background distributions (t=0 state of each run): number (left) and
    surface area (right) on the 80-bin grid. ``stp=True`` rescales ambient -> STP number
    concentrations (divide by the STP->ambient factor at 210 K / 55 hPa; the SABR curves then
    match their source plots' STP values, and aer_geo shows its STP-equivalent)."""
    from coupled.tomas_bridge import _bad
    f = _bad.stp_to_ambient_factor(210.0, 5500.0) if stp else 1.0
    fig, (ax_n, ax_sa) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for bg, lab in BGS:
        z = np.load(os.path.join(case_dir(CASE.format(bg=bg)), "state.npz"))
        dp, nd0 = z["dp_mid_um"], z["dNdlogDp"][0] / f
        dlog = np.log10(dp[1] / dp[0])
        n_tot = nd0.sum() * dlog
        sa0 = nd0 * np.pi * dp ** 2
        sa_tot = sa0.sum() * dlog
        ax_n.plot(dp, np.maximum(nd0, 1e-6), lw=2.0, color=BG_COLOR[bg],
                  label=f"{lab}  (N = {n_tot:.0f} cm$^{{-3}}$)")
        ax_sa.plot(dp, np.maximum(sa0, 1e-6), lw=2.0, color=BG_COLOR[bg],
                   label=f"{lab}  (SA = {sa_tot:.1f} µm$^2$ cm$^{{-3}}$)")
    for ax, ylab in ((ax_n, "dN/dlogD$_p$ [cm$^{-3}$]"),
                     (ax_sa, "dSA/dlogD$_p$ [µm$^2$ cm$^{-3}$]")):
        ax.set_xscale("log")
        ax.set_xlim(2e-3, 20)
        ax.set_xlabel("dry diameter [µm]")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3, lw=0.5, color="#e1e0d9")
        if linear:
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=8, loc="upper right")
        else:
            ax.set_yscale("log")
            ax.legend(fontsize=8, loc="lower center")
    if not linear:
        ax_n.set_ylim(1e-4 / f, 3e2 / f)
        ax_sa.set_ylim(1e-6 / f, 3e2 / f)
    ax_n.set_title("background number distribution", fontsize=11)
    ax_sa.set_title("background surface-area distribution", fontsize=11)
    basis = "STP number concentrations" if stp else "ambient"
    scale = "linear y" if linear else "log y"
    fig.suptitle(f"Background aerosol distributions as seeded ({basis}, 30°N / 20 km / 210 K / "
                 f"55 hPa; dry diameter; {scale})", y=1.02, fontsize=12)
    fig.tight_layout()
    fname = f"background_sizedists{'_STP' if stp else ''}{'_linear' if linear else ''}.png"
    fig.savefig(os.path.join(_OUT, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


# 06:00-release variants of the three cases (sabr330 run ad hoc; the others come from the
# start-time batches)
_HERE = os.path.dirname(os.path.abspath(__file__))
H06_DIRS = {
    "sabr330": os.path.join(_HERE, "runs_start_time_extra", "30N_20km__sabr330__D2med__h06"),
    "sabr220": os.path.join(_HERE, "runs_start_time", "30N_20km__sabr220__D2med__h06"),
    "aergeo": os.path.join(_HERE, "runs_start_time_geo", "30N_20km__aergeo__D2med__h06"),
}


def fig_banana(h06=False, sa=False):
    """3-background banana; ``sa=True`` colors by dSA/dlogDp (= pi*Dp^2 * dN/dlogDp)."""
    norm = LogNorm(1e-2, 2e3) if sa else LogNorm(1e2, 1e8)
    label = "dSA/dlogD$_p$ [µm$^2$ cm$^{-3}$]" if sa else "dN/dlogD$_p$ [cm$^{-3}$]"
    fig, axs = plt.subplots(3, 1, figsize=(9.0, 10.5), sharex=True, sharey=True)
    pc = None
    for ax, (bg, lab) in zip(axs, BGS):
        d = H06_DIRS[bg] if h06 else case_dir(CASE.format(bg=bg))
        z = np.load(os.path.join(d, "state.npz"))
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
    when = "06:00 release" if h06 else "00:00 release"
    fig.suptitle(f"Plume evolution vs background aerosol — 30°N / 20 km / 210 K, D2 med, "
                 f"α ×1, nuc ×1, coag ×1 ({when})", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    cax = fig.add_axes((0.90, 0.12, 0.02, 0.76))
    fig.colorbar(pc, cax=cax, label=label)
    fname = f"banana_backgrounds_D2med{'_h06' if h06 else ''}{'_SA' if sa else ''}.png"
    fig.savefig(os.path.join(_OUT, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


def fig_dilution_banana(sa=False):
    """2-panel banana: D2 med vs burst dilution, SABR-220 (aged air) background, baseline 1.
    ``sa=True`` colors by dSA/dlogDp."""
    dils = [("D2med", "D2 med dilution"), ("burst", "burst dilution")]
    norm = LogNorm(1e-2, 2e3) if sa else LogNorm(1e2, 1e8)
    label = "dSA/dlogD$_p$ [µm$^2$ cm$^{-3}$]" if sa else "dN/dlogD$_p$ [cm$^{-3}$]"
    fig, axs = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=True, sharey=True)
    pc = None
    for ax, (dil, lab) in zip(axs, dils):
        cid = f"30N_20km__sabr220__{dil}__a1p0__nuc1__cg1"
        z = np.load(os.path.join(case_dir(cid), "state.npz"))
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
    fig.suptitle("Plume evolution: D2 med vs burst dilution — 30°N / 20 km / 210 K, "
                 "SABR-220 (aged air), α ×1, nuc ×1, coag ×1", y=0.99, fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    cax = fig.add_axes((0.90, 0.12, 0.02, 0.76))
    fig.colorbar(pc, cax=cax, label=label)
    fname = f"banana_dilution_D2med_burst{'_SA' if sa else ''}.png"
    fig.savefig(os.path.join(_OUT, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


# site + background sensitivity panels (all D2 med, alpha x1, nuc x1, coag x1)
SITE_BG_PANELS = [
    ("30N_20km__sabr220__D2med__a1p0__nuc1__cg1",
     "(a)  30°N, 20 km, 210 K, 55 hPa — SABR-220 (aged air)"),
    ("60N_15km__sabr330__D2med__a1p0__nuc1__cg1",
     "(b)  60°N, 15 km, 210 K, 120 hPa — SABR-330 (young air)"),
    ("30N_20km_213K__aergeo__D2med__a1p0__nuc1__cg1",
     "(c)  30°N, 20 km, 213 K, 55 hPa — AER-2D geoengineered"),
]


def fig_site_bg_banana(sa=False):
    """3-panel site+background sensitivity banana; ``sa=True`` colors by dSA/dlogDp."""
    norm = LogNorm(1e-2, 2e3) if sa else LogNorm(1e2, 1e8)
    label = "dSA/dlogD$_p$ [µm$^2$ cm$^{-3}$]" if sa else "dN/dlogD$_p$ [cm$^{-3}$]"
    fig, axs = plt.subplots(3, 1, figsize=(9.0, 10.5), sharex=True, sharey=True)
    pc = None
    for ax, (cid, lab) in zip(axs, SITE_BG_PANELS):
        z = np.load(os.path.join(case_dir(cid), "state.npz"))
        t, dp = z["t"] / 86400.0, z["dp_mid_um"]
        field = z["dNdlogDp"] * np.pi * dp ** 2 if sa else z["dNdlogDp"]
        pc = ax.pcolormesh(t, dp, np.maximum(field, 1e-9).T, norm=norm,
                           cmap="viridis", rasterized=True, shading="nearest")
        ax.set_yscale("log")
        ax.set_ylim(dp[0], 3.0)
        ax.set_ylabel("dry diameter [µm]")
        ax.text(0.02, 0.95, lab, transform=ax.transAxes, fontsize=10.5, color="white",
                va="top", fontweight="bold")
    axs[-1].set_xlabel("day")
    fig.suptitle("Plume evolution vs stratospheric background state — D2 med, α ×1, "
                 "nuc ×1, coag ×1", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    cax = fig.add_axes((0.90, 0.12, 0.02, 0.76))
    fig.colorbar(pc, cax=cax, label=label)
    fname = f"banana_site_background{'_SA' if sa else ''}.png"
    fig.savefig(os.path.join(_OUT, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


def main():
    os.makedirs(_OUT, exist_ok=True)
    fig_site_bg_banana()
    fig_site_bg_banana(sa=True)
    fig_dilution_banana()
    fig_dilution_banana(sa=True)
    fig_background_dists()
    fig_background_dists(linear=True)
    fig_background_dists(stp=True, linear=True)
    fig_banana()
    fig_banana(sa=True)
    if all(os.path.exists(os.path.join(d, "state.npz")) for d in H06_DIRS.values()):
        fig_banana(h06=True)
    else:
        print("  SKIP banana_backgrounds_D2med_h06.png (h06 runs incomplete)")


if __name__ == "__main__":
    main()
