# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Banana / size-distribution figures for the D2-med case study
(-> runs/plots/nucleation/banana_D2med*.png, sizedist_snapshots_D2med.png).

Cases: baseline 1 (30N / 20 km / 210 K), dilution D2 med, condensation alpha x1,
coagulation x1; backgrounds SABR-220 (clean aged air, from measurements) vs CESM-G6 (SAI);
nucleation x0.01 / x1 / x100.

  banana_D2med.png          -- absolute dN/dlogDp, 3 rows (nucleation) x 2 cols (background).
  banana_D2med_ylgnbu.png   -- same, light-background colormap variant.
  banana_D2med_ratio.png    -- clean x1 panel absolute; the other five as the RATIO to it
                               (diverging scale centered at 1; cells where both runs are
                               near-empty are masked gray).
  sizedist_snapshots_D2med.png -- dN/dlogDp (row 1) and dSA/dlogDp (row 2) at day 2 / 5 / 10;
                               lines = nucleation levels, solid = SABR-220, dashed = CESM-G6.

Run (from SANDBOX/): python -m coupled.paper_ensemble.make_banana_nucleation
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from coupled.paper_ensemble.make_paper_candidate_plots import (
    _RUNS, case_dir, SAI_BG, SAI_LABEL, SAI_SHORT, SAI_SUFFIX)

_OUT = os.path.join(_RUNS, "plots", f"nucleation{SAI_SUFFIX}")

plt.rcParams.update({"font.family": "Helvetica", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130})

BGS = [("sabr220", "SABR-220 (clean aged air) background"), (SAI_BG, SAI_LABEL)]
NUCS = [("nuc0p01", "×0.01"), ("nuc1", "×1"), ("nuc100", "×100")]
CASE = "30N_20km__{bg}__D2med__a1p0__{nuc}__cg1"
NUC_COLOR = {"nuc0p01": "#2a78d6", "nuc1": "#1baf7a", "nuc100": "#eda100"}


def load(bg, nuc):
    z = np.load(os.path.join(case_dir(CASE.format(bg=bg, nuc=nuc)), "state.npz"))
    return z["t"] / 86400.0, z["dp_mid_um"], z["dNdlogDp"]


def render(cmap, fname, sa=False):
    """Banana grid; ``sa=True`` colors by dSA/dlogDp (= pi*Dp^2 * dN/dlogDp) instead of number."""
    fig, axs = plt.subplots(3, 2, figsize=(10.0, 9.6), sharex=True, sharey=True)
    pc = None
    # number: vmax covers the strongest burst core (7.3e7 in the x100 runs) without clipping;
    # surface area: peak dSA/dlogDp is ~1.15e3 um^2/cm^3 in all six runs
    norm = LogNorm(vmin=1e-2, vmax=2e3) if sa else LogNorm(vmin=1e2, vmax=1e8)
    label = ("dSA/dlogD$_p$ [µm$^2$ cm$^{-3}$]" if sa else "dN/dlogD$_p$ [cm$^{-3}$]")
    for i, (nuc, nlab) in enumerate(NUCS):
        for j, (bg, blab) in enumerate(BGS):
            t_days, dp, dnd = load(bg, nuc)
            field = dnd * np.pi * dp ** 2 if sa else dnd
            nd = np.maximum(field, 1e-9).T                  # (bins, time)
            ax = axs[i, j]
            pc = ax.pcolormesh(t_days, dp, nd, norm=norm,
                               cmap=cmap, rasterized=True, shading="nearest")
            ax.set_yscale("log")
            ax.set_ylim(dp[0], 2.0)
            if i == 0:
                ax.set_title(blab, fontsize=11)
            if i == 2:
                ax.set_xlabel("day")
            if j == 0:
                ax.set_ylabel(f"nucleation {nlab}\ndry diameter [µm]", fontsize=10)
    fig.suptitle("Plume size-distribution evolution — D2 med dilution, 30°N / 20 km / 210 K, "
                 "α ×1, coag ×1", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    cax = fig.add_axes((0.92, 0.12, 0.02, 0.76))
    fig.colorbar(pc, cax=cax, label=label)
    fig.savefig(os.path.join(_OUT, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


def fig_ratio(sa=False):
    """Clean x1 as the absolute reference; the other five panels as run/reference ratios.

    ``sa=True`` shows the reference as dSA/dlogDp and masks cells that carry no meaningful
    surface area. The ratio VALUES are identical to the number version (pi*Dp^2 cancels
    bin-by-bin); only the reference panel and the mask change.
    """
    t_days, dp, nd_ref = load("sabr220", "nuc1")
    to_field = (lambda nd: nd * np.pi * dp ** 2) if sa else (lambda nd: nd)
    ref = to_field(nd_ref)
    norm_abs = LogNorm(vmin=1e-2, vmax=2e3) if sa else LogNorm(vmin=1e2, vmax=1e8)
    label_abs = ("reference dSA/dlogD$_p$ [µm$^2$ cm$^{-3}$]" if sa
                 else "reference dN/dlogD$_p$ [cm$^{-3}$]")
    floor = 1e-2 if sa else 1e1                  # "near-empty" mask threshold per quantity
    fig, axs = plt.subplots(3, 2, figsize=(10.0, 9.6), sharex=True, sharey=True)
    pc_abs = pc_rat = None
    cmap_rat = plt.get_cmap("RdBu_r").copy()
    cmap_rat.set_bad("0.88")                     # both runs near-empty: no meaningful ratio
    for i, (nuc, nlab) in enumerate(NUCS):
        for j, (bg, blab) in enumerate(BGS):
            ax = axs[i, j]
            if (bg, nuc) == ("sabr220", "nuc1"):
                pc_abs = ax.pcolormesh(t_days, dp, np.maximum(ref, 1e-9).T, norm=norm_abs,
                                       cmap="viridis", rasterized=True, shading="nearest")
                ax.text(0.02, 0.96, "REFERENCE (absolute)", transform=ax.transAxes,
                        fontsize=8.5, va="top", color="white", fontweight="bold")
            else:
                _, _, nd = load(bg, nuc)
                fld = to_field(nd)
                ratio = np.clip(fld / np.maximum(ref, 1e-30), 1e-2, 1e2)
                ratio[(fld < floor) & (ref < floor)] = np.nan
                pc_rat = ax.pcolormesh(t_days, dp, ratio.T, norm=LogNorm(1e-2, 1e2),
                                       cmap=cmap_rat, rasterized=True, shading="nearest")
            ax.set_yscale("log")
            ax.set_ylim(dp[0], 2.0)
            if i == 0:
                ax.set_title(blab, fontsize=11)
            if i == 2:
                ax.set_xlabel("day")
            if j == 0:
                ax.set_ylabel(f"nucleation {nlab}\ndry diameter [µm]", fontsize=10)
    fig.suptitle("Nucleation / background effect relative to the clean ×1 case — D2 med, "
                 "30°N / 20 km / 210 K, α ×1, coag ×1", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    cax1 = fig.add_axes((0.90, 0.55, 0.018, 0.33))
    fig.colorbar(pc_abs, cax=cax1, label=label_abs)
    cax2 = fig.add_axes((0.90, 0.12, 0.018, 0.33))
    fig.colorbar(pc_rat, cax=cax2, label="ratio to reference (gray = both near-empty)")
    fname = "banana_D2med_ratio_SA.png" if sa else "banana_D2med_ratio.png"
    fig.savefig(os.path.join(_OUT, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


def fig_snapshots(linear=False):
    """dN/dlogDp (top) and dSA/dlogDp (bottom) spectra at days 2 / 5 / 10.

    ``linear=True`` uses a linear y axis, autoscaled per panel (the spectra decay ~100x from
    day 2 to day 10, so shared linear rows would flatten the later panels): it compares
    nucleation levels within a day; the log version compares across days.
    """
    days = [2, 5, 10]
    data = {(bg, nuc): load(bg, nuc) for bg, _ in BGS for nuc, _ in NUCS}
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 7.0), sharex=True,
                            sharey=False if linear else "row")
    for k, day in enumerate(days):
        for nuc, nlab in NUCS:
            for bg, blab in BGS:
                t_days, dp, nd = data[(bg, nuc)]
                it = np.searchsorted(t_days, day - 1e-9)
                spec_n = np.maximum(nd[it], 1e-6)
                # floor AFTER the pi*Dp^2 conversion, else the floor grows a fake coarse tail
                spec_sa = np.maximum(nd[it] * np.pi * dp ** 2, 1e-6)  # µm^2 cm^-3 per dlogDp
                ls = "-" if bg == "sabr220" else "--"
                lab = f"{nlab}, {'clean' if bg == 'sabr220' else 'SAI'}"
                axs[0, k].plot(dp, spec_n, ls, color=NUC_COLOR[nuc], lw=2.0, label=lab)
                axs[1, k].plot(dp, spec_sa, ls, color=NUC_COLOR[nuc], lw=2.0)
        for r in (0, 1):
            ax = axs[r, k]
            ax.set_xscale("log")
            ax.set_xlim(2e-3, 20)
            ax.grid(True, alpha=0.3, lw=0.5, color="#e1e0d9")
            if linear:
                ax.set_ylim(bottom=0)
            else:
                ax.set_yscale("log")
        axs[0, k].set_title(f"day {day}", fontsize=11)
        if not linear:
            axs[0, k].set_ylim(1e-2, 3e6)
            axs[1, k].set_ylim(1e-4, 3e3)
        axs[1, k].set_xlabel("dry diameter [µm]")
    axs[0, 0].set_ylabel("dN/dlogD$_p$ [cm$^{-3}$]")
    axs[1, 0].set_ylabel("dSA/dlogD$_p$ [µm$^2$ cm$^{-3}$]")
    leg_ax = axs[0, 0] if linear else axs[0, 2]
    leg_ax.legend(fontsize=7.5, ncols=2, loc="upper right", title="nucleation, background",
                  title_fontsize=7.5)
    scale_note = "linear y, per-panel scale" if linear else "log y"
    fig.suptitle("Number (top) and surface-area (bottom) size distributions — D2 med, "
                 "30°N / 20 km / 210 K, α ×1, coag ×1\n"
                 f"solid: SABR-220 clean; dashed: {SAI_SHORT} ({scale_note})",
                 y=1.0, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fname = f"sizedist_snapshots_D2med{'_linear' if linear else ''}.png"
    fig.savefig(os.path.join(_OUT, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


def fig_totals():
    """Total (integrated) surface area (left) and number (right) vs time for the six runs."""
    fig, (ax_sa, ax_n) = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)
    bg_ref = {}                                  # t=0 state = the background distribution
    for nuc, nlab in NUCS:
        for bg, blab in BGS:
            z = np.load(os.path.join(case_dir(CASE.format(bg=bg, nuc=nuc)), "state.npz"))
            t_days = z["t"] / 86400.0
            ls = "-" if bg == "sabr220" else "--"
            lab = f"{nlab}, {'clean' if bg == 'sabr220' else 'SAI'}"
            ax_sa.plot(t_days, z["SA"], ls, color=NUC_COLOR[nuc], lw=1.7, label=lab)
            ax_n.plot(t_days, z["total_n"], ls, color=NUC_COLOR[nuc], lw=1.7)
            bg_ref[bg] = (float(z["SA"][0]), float(z["total_n"][0]))
    for bg, blab in BGS:
        ls = "-" if bg == "sabr220" else "--"
        lab = f"background ({'clean' if bg == 'sabr220' else 'SAI'})"
        ax_sa.axhline(bg_ref[bg][0], ls=ls, color="#898781", lw=1.3, label=lab)
        ax_n.axhline(bg_ref[bg][1], ls=ls, color="#898781", lw=1.3)
    for ax, ylab, title in ((ax_sa, "surface area (wet) [µm$^2$ cm$^{-3}$]", "total surface area"),
                            (ax_n, "number [cm$^{-3}$]", "total number")):
        ax.set_yscale("log")
        ax.set_xlim(0, 10)
        ax.set_xlabel("day")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3, lw=0.5, color="#e1e0d9")
    ax_n.set_ylim(bottom=0.5 * min(v[1] for v in bg_ref.values()))
    ax_sa.set_ylim(bottom=0.5 * min(v[0] for v in bg_ref.values()))
    ax_sa.legend(fontsize=7.5, ncols=2, loc="lower left", title="nucleation, background",
                 title_fontsize=7.5)
    fig.suptitle("Integrated aerosol surface area and number — D2 med, 30°N / 20 km / 210 K, "
                 f"α ×1, coag ×1 (solid: SABR-220 clean; dashed: {SAI_SHORT})", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "totals_D2med.png"), bbox_inches="tight")
    plt.close(fig)
    print("  totals_D2med.png")


def main():
    os.makedirs(_OUT, exist_ok=True)
    render("viridis", "banana_D2med.png")
    render("YlGnBu", "banana_D2med_ylgnbu.png")
    render("viridis", "banana_D2med_SA.png", sa=True)
    fig_ratio()
    fig_ratio(sa=True)
    fig_snapshots()
    fig_snapshots(linear=True)
    fig_totals()


if __name__ == "__main__":
    main()
