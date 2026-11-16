# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Start-time-of-day figures for the D2-med case study (-> runs_start_time/plots/).

Subset: baseline 1 (30N / 20 km / 210 K / 55 hPa), SABR-220 clean background, D2 med
dilution, condensation alpha x1, nucleation x1, coag x1; start hours 00..21 UTC every 3 h
(longitude 0 -> local solar time; the runs otherwise share everything).

  sizedist_by_start.png -- dN/dlogDp (top) and dSA/dlogDp (bottom) at plume age 2 / 5 / 10 d,
                           one line per start hour. NOTE: equal AGE = different local clock
                           time per run; that phase offset is the point of the comparison.
  banana_grid.png       -- dN/dlogDp evolution for all 8 start hours (shared color scale),
                           x = elapsed days since release.

Run (from SANDBOX/): python -m coupled.paper_ensemble.make_start_time_plots
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# importing installs the PNG+PDF savefig hook
import coupled.paper_ensemble.make_paper_candidate_plots  # noqa: F401

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNS = os.path.join(_HERE, "runs_start_time")       # rebound per bg in main()
_OUT = os.path.join(_HERE, "runs_start_time", "plots")
HOURS = [0, 3, 6, 9, 12, 15, 18, 21]
SITE, SITE_LABEL, SUFFIX = "30N_20km", "30°N / 20 km / 210 K", ""   # set by main()
BG, BG_LABEL, RUNS_DIR = "sabr220", "SABR-220", "runs_start_time"   # set by main()
DIL, DIL_LABEL = "D2med", "D2 med"                                  # set by main()
CASE = "{site}__{bg}__{dil}__h{h:02d}"
# 8-slot categorical palette in its validated fixed order
HOUR_COLOR = dict(zip(HOURS, ["#2a78d6", "#1baf7a", "#eda100", "#008300",
                              "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]))

plt.rcParams.update({"font.family": "Helvetica", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
                     "grid.linewidth": 0.5, "grid.color": "#e1e0d9", "axes.axisbelow": True,
                     "figure.dpi": 130, "legend.frameon": False})


def load(h):
    z = np.load(os.path.join(_RUNS, CASE.format(site=SITE, bg=BG, dil=DIL, h=h), "state.npz"))
    return z["t"] / 86400.0, z["dp_mid_um"], z["dNdlogDp"]


def fig_sizedist(linear=False):
    """N (top) and SA (bottom) spectra at ages 2/5/10 d; ``linear=True`` autoscales each
    panel linearly (the spectra decay ~100x across ages, so shared linear rows would
    flatten the later panels)."""
    days = [2, 5, 10]
    data = {h: load(h) for h in HOURS}
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 7.0), sharex=True,
                            sharey=False if linear else "row")
    for k, day in enumerate(days):
        for h in HOURS:
            t, dp, nd = data[h]
            i = np.searchsorted(t, day - 1e-9)
            axs[0, k].plot(dp, np.maximum(nd[i], 1e-6), lw=1.7, color=HOUR_COLOR[h],
                           label=f"{h:02d}:00")
            axs[1, k].plot(dp, np.maximum(nd[i] * np.pi * dp ** 2, 1e-6), lw=1.7,
                           color=HOUR_COLOR[h])
        for r in (0, 1):
            ax = axs[r, k]
            ax.set_xscale("log")
            ax.set_xlim(2e-3, 20)
            if linear:
                ax.set_ylim(bottom=0)
            else:
                ax.set_yscale("log")
        axs[0, k].set_title(f"plume age {day} d", fontsize=11)
        if not linear:
            axs[0, k].set_ylim(1e-2, 3e6)
            axs[1, k].set_ylim(1e-4, 3e3)
        axs[1, k].set_xlabel("dry diameter [µm]")
    axs[0, 0].set_ylabel("dN/dlogD$_p$ [cm$^{-3}$]")
    axs[1, 0].set_ylabel("dSA/dlogD$_p$ [µm$^2$ cm$^{-3}$]")
    leg_ax = axs[0, 0] if linear else axs[0, 2]
    leg_ax.legend(fontsize=8, ncols=2, title="release time (local)", title_fontsize=8,
                  loc="upper right")
    scale_note = "linear y, per-panel scale" if linear else "log y"
    fig.suptitle(f"Size distribution vs release time of day — {SITE_LABEL}, {BG_LABEL}, "
                 f"{DIL_LABEL}, α ×1, nuc ×1, coag ×1\n(equal plume AGE per panel; local clock time "
                 f"at sampling differs by release time; {scale_note})", y=1.0, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fname = f"sizedist_by_start{'_linear' if linear else ''}{SUFFIX}.png"
    fig.savefig(os.path.join(_OUT, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


def fig_totals():
    """Integrated (wet) surface area and total number vs plume age for all release hours."""
    fig, (ax_sa, ax_n) = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)
    bg = None
    for h in HOURS:
        z = np.load(os.path.join(_RUNS, CASE.format(site=SITE, bg=BG, dil=DIL, h=h), "state.npz"))
        t = z["t"] / 86400.0
        ax_sa.plot(t, z["SA"], lw=1.6, color=HOUR_COLOR[h], label=f"{h:02d}:00")
        ax_n.plot(t, z["total_n"], lw=1.6, color=HOUR_COLOR[h])
        bg = (float(z["SA"][0]), float(z["total_n"][0]))    # same background for all runs
    ax_sa.axhline(bg[0], color="#898781", lw=1.3, label="background")
    ax_n.axhline(bg[1], color="#898781", lw=1.3)
    for ax, ylab, title in ((ax_sa, "surface area (wet) [µm$^2$ cm$^{-3}$]",
                             "total surface area"),
                            (ax_n, "number [cm$^{-3}$]", "total number")):
        ax.set_yscale("log")
        ax.set_xlim(0, 10)
        ax.set_xlabel("days since release")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11)
    ax_sa.set_ylim(bottom=0.5 * bg[0])
    ax_n.set_ylim(bottom=0.5 * bg[1])
    ax_sa.legend(fontsize=7.5, ncols=3, loc="lower left", title="release time (local)",
                 title_fontsize=7.5)
    fig.suptitle("Integrated aerosol surface area and number vs release time of day — "
                 f"{SITE_LABEL}, {BG_LABEL}, {DIL_LABEL}, α ×1, nuc ×1, coag ×1",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, f"totals_by_start{SUFFIX}.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  totals_by_start{SUFFIX}.png")


def fig_banana_grid():
    fig, axs = plt.subplots(4, 2, figsize=(10.5, 12.0), sharex=True, sharey=True)
    pc = None
    for ax, h in zip(axs.ravel(), HOURS):
        t, dp, nd = load(h)
        pc = ax.pcolormesh(t, dp, np.maximum(nd, 1e-3).T, norm=LogNorm(1e2, 1e8),
                           cmap="viridis", rasterized=True, shading="nearest")
        ax.set_yscale("log")
        ax.set_ylim(dp[0], 2.0)
        ax.text(0.02, 0.95, f"release {h:02d}:00", transform=ax.transAxes, fontsize=10,
                color="white", va="top", fontweight="bold")
    for ax in axs[-1]:
        ax.set_xlabel("days since release")
    for ax in axs[:, 0]:
        ax.set_ylabel("dry diameter [µm]")
    fig.suptitle(f"Plume evolution vs release time of day — {SITE_LABEL}, {BG_LABEL}, "
                 f"{DIL_LABEL}, α ×1, nuc ×1, coag ×1", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    cax = fig.add_axes((0.92, 0.12, 0.02, 0.76))
    fig.colorbar(pc, cax=cax, label="dN/dlogD$_p$ [cm$^{-3}$]")
    fig.savefig(os.path.join(_OUT, f"banana_grid{SUFFIX}.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  banana_grid{SUFFIX}.png")


_SITES = {"30N_20km": ("30°N / 20 km / 210 K", ""),
          "60N_15km": ("60°N / 15 km / 210 K", "_60N")}
# (bg, bg_label, runs_dir, dil, dil_label, sites)
COMBOS = [
    ("sabr220", "SABR-220 clean", "runs_start_time", "D2med", "D2 med",
     ["30N_20km", "60N_15km"]),
    ("sabr220", "SABR-220 clean", "runs_start_time", "burst", "burst dilution",
     ["30N_20km"]),
    ("aergeo", "AER-2D geoengineered", "runs_start_time_geo", "D2med", "D2 med",
     ["30N_20km", "60N_15km"]),
]


def main():
    global SITE, SITE_LABEL, SUFFIX, BG, BG_LABEL, DIL, DIL_LABEL, _RUNS
    os.makedirs(_OUT, exist_ok=True)
    for BG, BG_LABEL, runs_dir, DIL, DIL_LABEL, sites in COMBOS:
        _RUNS = os.path.join(_HERE, runs_dir)
        for SITE in sites:
            SITE_LABEL, site_sfx = _SITES[SITE]
            probe = os.path.join(_RUNS, CASE.format(site=SITE, bg=BG, dil=DIL, h=0),
                                 "state.npz")
            if not os.path.exists(probe):
                print(f"  SKIP {BG}/{DIL}/{SITE} (runs not present yet)")
                continue
            SUFFIX = (("" if BG == "sabr220" else "_aergeo")
                      + ("" if DIL == "D2med" else f"_{DIL}") + site_sfx)
            fig_sizedist()
            fig_sizedist(linear=True)
            fig_totals()
            fig_banana_grid()


if __name__ == "__main__":
    main()
