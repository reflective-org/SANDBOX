# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Oxidation-side figures (-> runs/plots/oxidation/).

Subset: baseline 30N / 20 km / 210 K / 55 hPa, nucleation x1, condensation alpha x1, coag x1,
background SABR-220 (the 220-230 ppbv N2O aged-air observation; bg SO2 = 20 ppt).

  OH_by_dilution.png      -- OH vs time for all five dilution regimes.
  sulfur_budget_D2med.png -- pure-sulfur budget of the D2 med run: absolute in-box S per
                             species (left; every species carries one S atom, so molec/cm^3
                             of the species IS its sulfur content) and fraction of total
                             in-box S (right, stacked).

Run (from SANDBOX/): python -m coupled.paper_ensemble.make_oxidation_plots
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# importing installs the PNG+PDF savefig hook
from coupled.paper_ensemble.make_paper_candidate_plots import (
    _RUNS, REGIMES, REGIME_LABEL, REGIME_COLOR)

_OUT = os.path.join(_RUNS, "plots", "oxidation")
_OH, _SO2, _SO3, _H2SO4 = 18, 32, 34, 35
CASE = "30N_20km__sabr220__{reg}__a1p0__nuc1__cg1"

plt.rcParams.update({"font.family": "Helvetica", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.35,
                     "grid.linewidth": 0.5, "grid.color": "#e1e0d9", "axes.axisbelow": True,
                     "figure.dpi": 130, "legend.frameon": False})

# AMS-inspired reservoir colors (particulate sulfate in the canonical AMS red):
# SO2 yellow(amber for print) / SO3 green / gas H2SO4 orange / particulate sulfate red
# gas H2SO4 is DASHED in the line panel so its orange never reads as the sulfate red
S_POOLS = [("SO$_2$", "#eda100", "-"), ("SO$_3$", "#008300", "-"),
           ("H$_2$SO$_4$ (gas)", "#eb6834", "--"), ("particulate sulfate", "#d03b3b", "-")]


def load(reg):
    return np.load(os.path.join(_RUNS, CASE.format(reg=reg), "state.npz"))


# OH figure uses the 5-slot categorical palette (validated fixed order) instead of the
# ordinal blue dilution ramp used elsewhere
OH_COLOR = {"D1low": "#2a78d6", "D2med": "#1baf7a", "D3high": "#eda100",
            "D5vhigh": "#008300", "burst": "#4a3aa7"}


def fig_oh():
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for reg in REGIMES:
        z = load(reg)
        ax.plot(z["t"] / 86400.0, z["x"][:, _OH], color=OH_COLOR[reg], lw=1.8,
                label=REGIME_LABEL[reg])
    ax.set_ylim(bottom=0)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(6, 6), useMathText=True)
    ax.set_xlim(0, 10)
    ax.set_xlabel("day")
    ax.set_ylabel("OH [molec cm$^{-3}$]")
    ax.legend(fontsize=8.5, title="dilution", title_fontsize=8.5, loc="upper left", ncols=2)
    ax.set_title("OH — 30°N / 20 km / 210 K / 55 hPa, SABR-220 background, "
                 "nucleation ×1, α ×1, coag ×1", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "OH_by_dilution.png"), bbox_inches="tight")
    plt.close(fig)
    print("  OH_by_dilution.png")


def fig_sulfur_budget():
    z = load("D2med")
    t = z["t"] / 86400.0
    x = z["x"]
    # each species carries one S atom: atoms/cm^3 -> µg S / cm^3 via 32.065 g/mol
    s_ug = 32.065 / 6.02214076e23 * 1e6
    pools = [x[:, _SO2] * s_ug, x[:, _SO3] * s_ug, x[:, _H2SO4] * s_ug,
             z["particulate_S"] * s_ug]
    total = np.sum(pools, axis=0)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    a1.plot(t, total, color="#0b0b0b", lw=2.6, label="total in-box S")
    for (lab, col, ls), p in zip(S_POOLS, pools):
        a1.plot(t, np.maximum(p, 1e-30), ls, color=col, lw=2.2, label=lab)
    a1.set_yscale("log")
    a1.set_ylim(1e-11, 2)
    a1.set_xlim(0, 10)
    a1.grid(True, which="major", alpha=0.6, lw=0.6, color="#c9c8c1")
    # >8 decades: matplotlib prunes minor log ticks entirely, so force the locator
    a1.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=range(2, 10),
                                                            numticks=200))
    a1.grid(True, which="minor", axis="y", alpha=0.25, lw=0.35, color="#c9c8c1")
    # ticks at every decade, labels only on the odd exponents (10^-1, 10^-3, ... 10^-11)
    a1.set_yticks([10.0 ** e for e in range(-11, 1)])
    a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, pos: (f"$10^{{{int(round(np.log10(v)))}}}$"
                        if int(round(np.log10(v))) % 2 else "")))
    a1.set_xlabel("day")
    a1.set_ylabel("sulfur mass [µg S cm$^{-3}$]")
    a1.set_title("in-box sulfur by reservoir (dilution included)", fontsize=11)
    a1.legend(fontsize=8, loc="upper right")

    fracs = [p / total for p in pools]
    a2.stackplot(t, *fracs, colors=[c for _, c, _ in S_POOLS], alpha=0.85,
                 labels=[l for l, _, _ in S_POOLS])
    a2.set_xlim(0, 10)
    a2.set_ylim(0, 1)
    a2.set_xticks(range(0, 11))
    yticks = np.arange(0.0, 1.01, 0.1)           # tick every 0.1, label every 0.2
    a2.set_yticks(yticks, [f"{v:.1f}" if round(v * 10) % 2 == 0 else "" for v in yticks])
    a2.grid(True, which="major", color="white", alpha=0.35, lw=0.7)
    a2.set_axisbelow(False)                      # white grid must sit ON TOP of the fills
    a2.text(3.0, 0.45, "SO$_2$", fontsize=13, color="#0b0b0b", ha="center", va="center")
    a2.text(8.5, 0.93, "particulate sulfate", fontsize=10, color="white",
            ha="center", va="center")
    a2.set_xlabel("day")
    a2.set_ylabel("fraction of in-box sulfur")
    a2.set_title("sulfur partitioning (fractions sum to 1)", fontsize=11)

    fig.suptitle("Pure-sulfur budget — D2 med, 30°N / 20 km / 210 K, SABR-220 background, "
                 "nucleation ×1, α ×1, coag ×1", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "sulfur_budget_D2med.png"), bbox_inches="tight")
    plt.close(fig)
    print("  sulfur_budget_D2med.png")


def main():
    os.makedirs(_OUT, exist_ok=True)
    fig_oh()
    fig_sulfur_budget()


if __name__ == "__main__":
    main()
