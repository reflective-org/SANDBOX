# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Banana plots for the 60-day-max runs (-> runs_60day/plots/).

Two panels on a SHARED time axis so the plume lifetimes compare visually: D2 med (stopped
at 16.8 d) and D1 low (stopped at 35.9 d); each run ends where the wet SA stayed within 10%
of the background for 24 h. 30N / 20 km / 210 K, SABR-220, alpha x1, nuc x1, coag x1,
06:00 release, frank-model spun-up ICs.

Run (from SANDBOX/): python -m coupled.paper_ensemble.make_60day_plots
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import coupled.paper_ensemble.make_paper_candidate_plots  # noqa: F401  (PNG+PDF hook)

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNS = os.path.join(_HERE, "runs_60day")
_OUT = os.path.join(_RUNS, "plots")
CASES = [("30N_20km__sabr220__D2med__60day_h06_frankIC", "D2 med dilution"),
         ("30N_20km__sabr220__D1low__60day_h06_frankIC", "D1 low dilution")]

plt.rcParams.update({"font.family": "Helvetica", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130})


def fig_banana(sa=False):
    norm = LogNorm(1e-2, 2e3) if sa else LogNorm(1e2, 1e8)
    label = "dSA/dlogD$_p$ [µm$^2$ cm$^{-3}$]" if sa else "dN/dlogD$_p$ [cm$^{-3}$]"
    fig, axs = plt.subplots(2, 1, figsize=(11.0, 7.4), sharex=True, sharey=True)
    pc = None
    for ax, (cid, lab) in zip(axs, CASES):
        z = np.load(os.path.join(_RUNS, cid, "state.npz"))
        t, dp = z["t"] / 86400.0, z["dp_mid_um"]
        field = z["dNdlogDp"] * np.pi * dp ** 2 if sa else z["dNdlogDp"]
        pc = ax.pcolormesh(t, dp, np.maximum(field, 1e-9).T, norm=norm,
                           cmap="viridis", rasterized=True, shading="nearest")
        ax.axvline(t[-1], color="white", lw=1.0, ls="--")
        ax.set_yscale("log")
        ax.set_ylim(dp[0], 3.0)
        ax.set_xlim(0, 37)
        ax.set_ylabel("dry diameter [µm]")
        ax.text(0.015, 0.94, f"{lab} — plume life {t[-1]:.1f} d", transform=ax.transAxes,
                fontsize=11, color="white", va="top", fontweight="bold")
    axs[-1].set_xlabel("days since release")
    fig.suptitle("Full plume life cycle — 30°N / 20 km / 210 K, SABR-220, α ×1, nuc ×1, "
                 "coag ×1, 06:00 release, spun-up ICs\n(each run stops once wet SA stays "
                 "within 10% of background for 24 h)", y=0.995, fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    cax = fig.add_axes((0.90, 0.12, 0.02, 0.76))
    fig.colorbar(pc, cax=cax, label=label)
    fname = f"banana_60day{'_SA' if sa else ''}.png"
    fig.savefig(os.path.join(_OUT, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


def fig_sulfur():
    """Plume-integrated sulfur budget, normalized by injected S: excess-over-background pool
    x V(t)/V0 / N_injected. Total is conserved at 1 (mass-balance check). Shaded where the
    static-background attribution degrades (background-SO2-in-plume-volume > 10% of injected:
    V amplification makes ambient-level residuals O(injected))."""
    N_INJ = 6.273063291666667e15
    fig, axs = plt.subplots(2, 1, figsize=(11.0, 7.4), sharex=True, sharey=True)
    for ax, (cid, lab) in zip(axs, CASES):
        z = np.load(os.path.join(_RUNS, cid, "state.npz"))
        t = z["t"] / 86400.0
        n_bg = 20e-12 * float(z["M"])
        V = z["V_ratio"]
        exc_so2 = (z["x"][:, 32] - n_bg) * V / N_INJ
        exc_gas = (z["x"][:, 34] + z["x"][:, 35]) * V / N_INJ
        exc_sulf = (z["particulate_S"] - z["particulate_S"][0]) * V / N_INJ
        amp = n_bg * V / N_INJ                     # attribution-degradation measure
        i_bad = int(np.searchsorted(amp, 0.10))
        ax.axvspan(t[min(i_bad, len(t) - 1)], t[-1], color="#e1e0d9", alpha=0.55, lw=0,
                   zorder=0)
        ax.plot(t, exc_so2, lw=2.0, color="#eda100", label="SO$_2$ (excess)")
        ax.plot(t, exc_sulf, lw=2.0, color="#d03b3b", label="particulate sulfate (excess)")
        ax.plot(t, exc_gas, lw=1.6, ls="--", color="#eb6834", label="gas SO$_3$+H$_2$SO$_4$")
        ax.plot(t, exc_so2 + exc_gas + exc_sulf, lw=1.4, color="#0b0b0b",
                label="total (=1: mass balance)")
        ax.axhline(0.0, color="#898781", lw=0.8)
        ax.set_ylabel("plume-integrated S / injected S")
        ax.text(0.015, 0.95, lab, transform=ax.transAxes, fontsize=11, va="top",
                fontweight="bold")
        ax.text(t[min(i_bad, len(t) - 1)] + 0.3, -0.45,
                "static-background attribution degrades", fontsize=8, color="#52514e")
        ax.grid(True, alpha=0.3, lw=0.5, color="#e1e0d9")
        i_ok = max(i_bad - 1, 0)
        print(f"  {lab}: attribution valid to day {t[i_ok]:.1f}; SO2 reacted by then = "
              f"{(1.0 - exc_so2[i_ok]) * 100:.1f}%")
    axs[0].legend(fontsize=8.5, loc="center left")
    axs[-1].set_xlabel("days since release")
    axs[0].set_ylim(-0.75, 1.35)
    fig.suptitle("Plume-integrated sulfur budget per unit injected sulfur — 60-day runs\n"
                 "(excess over static background × V(t)/V$_0$; shaded: dilution amplifies "
                 "ambient residuals, pool attribution unreliable)", y=0.99, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "sulfur_budget_60day.png"), bbox_inches="tight")
    plt.close(fig)
    print("  sulfur_budget_60day.png")


S_POOLS = [("SO$_2$", "#eda100", "-"), ("SO$_3$", "#008300", "-"),
           ("H$_2$SO$_4$ (gas)", "#eb6834", "--"), ("particulate sulfate", "#d03b3b", "-")]


def fig_sulfur_inbox():
    """In-box pure-sulfur budget per run, same layout/styling as
    runs/plots/oxidation/sulfur_budget_D2med.png: absolute ug S/cm^3 (left, log) and
    stacked fraction of in-box sulfur (right)."""
    s_ug = 32.065 / 6.02214076e23 * 1e6
    for cid, lab in CASES:
        z = np.load(os.path.join(_RUNS, cid, "state.npz"))
        t = z["t"] / 86400.0
        x = z["x"]
        pools = [x[:, 32] * s_ug, x[:, 34] * s_ug, x[:, 35] * s_ug,
                 z["particulate_S"] * s_ug]
        total = np.sum(pools, axis=0)

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
        a1.plot(t, total, color="#0b0b0b", lw=2.6, label="total in-box S")
        for (name, col, ls), pl in zip(S_POOLS, pools):
            a1.plot(t, np.maximum(pl, 1e-30), ls, color=col, lw=2.2, label=name)
        a1.set_yscale("log")
        a1.set_ylim(1e-11, 2)
        a1.set_xlim(0, t[-1])
        a1.grid(True, which="major", alpha=0.6, lw=0.6, color="#c9c8c1")
        a1.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=range(2, 10),
                                                                numticks=200))
        a1.grid(True, which="minor", axis="y", alpha=0.25, lw=0.35, color="#c9c8c1")
        a1.set_yticks([10.0 ** e for e in range(-11, 1)])
        a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, pos: (f"$10^{{{int(round(np.log10(v)))}}}$"
                            if int(round(np.log10(v))) % 2 else "")))
        a1.set_xlabel("day")
        a1.set_ylabel("sulfur mass [µg S cm$^{-3}$]")
        a1.set_title("in-box sulfur by reservoir (dilution included)", fontsize=11)
        a1.legend(fontsize=8, loc="upper right")

        fracs = [pl / total for pl in pools]
        a2.stackplot(t, *fracs, colors=[c for _, c, _ in S_POOLS], alpha=0.85)
        a2.set_xlim(0, t[-1])
        a2.set_ylim(0, 1)
        yticks = np.arange(0.0, 1.01, 0.1)
        a2.set_yticks(yticks, [f"{v:.1f}" if round(v * 10) % 2 == 0 else "" for v in yticks])
        a2.grid(True, which="major", color="white", alpha=0.35, lw=0.7)
        a2.set_axisbelow(False)
        a2.text(0.30 * t[-1], 0.45, "SO$_2$", fontsize=13, color="#0b0b0b",
                ha="center", va="center")
        a2.text(0.80 * t[-1], 0.90, "particulate sulfate", fontsize=10, color="white",
                ha="center", va="center")
        a2.set_xlabel("day")
        a2.set_ylabel("fraction of in-box sulfur")
        a2.set_title("sulfur partitioning (fractions sum to 1)", fontsize=11)

        fig.suptitle(f"Pure-sulfur budget — {lab}, 30°N / 20 km / 210 K, SABR-220, "
                     "α ×1, nuc ×1, coag ×1, 06:00 release, spun-up ICs", y=1.02, fontsize=12)
        fig.tight_layout()
        fname = f"sulfur_budget_inbox_{cid.split('__')[2]}.png"
        fig.savefig(os.path.join(_OUT, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  {fname}")


def main():
    os.makedirs(_OUT, exist_ok=True)
    fig_banana()
    fig_banana(sa=True)
    fig_sulfur()
    fig_sulfur_inbox()


if __name__ == "__main__":
    main()
