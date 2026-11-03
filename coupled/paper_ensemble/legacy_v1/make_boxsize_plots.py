# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Banana plots for the initial-volume (emission density) sweep (-> runs/plots/boxsize/).

Panels: the standard V0 (x1, from the main ensemble) + 2/5/10/20/50/100x larger initial
volume (same 1 t SO2 -> initial concentration / factor). Baseline 1, SABR-220, D2 med,
alpha x1, nuc x1, coag x1, midnight release.

Run (from SANDBOX/): python -m coupled.paper_ensemble.make_boxsize_plots
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from coupled.paper_ensemble.make_paper_candidate_plots import _RUNS, case_dir

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOX = os.path.join(_HERE, "runs_boxsize")
_OUT = os.path.join(_RUNS, "plots", "boxsize")
FACTORS = [1, 2, 5, 10, 20, 50, 100]

plt.rcParams.update({"font.family": "Helvetica", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130})


DIL, DIL_LABEL = "D2med", "D2 med"                 # set by main()


def _path(f):
    if f == 1:
        return os.path.join(case_dir(f"30N_20km__sabr220__{DIL}__a1p0__nuc1__cg1"), "state.npz")
    return os.path.join(_BOX, f"30N_20km__sabr220__{DIL}__v{f}", "state.npz")


def fig_banana_grid(sa=False):
    norm = LogNorm(1e-2, 2e3) if sa else LogNorm(1e2, 1e8)
    label = "dSA/dlogD$_p$ [µm$^2$ cm$^{-3}$]" if sa else "dN/dlogD$_p$ [cm$^{-3}$]"
    fig, axs = plt.subplots(4, 2, figsize=(10.5, 12.0), sharex=True, sharey=True)
    pc = None
    for ax, f in zip(axs.ravel(), FACTORS):
        z = np.load(_path(f))
        t, dp = z["t"] / 86400.0, z["dp_mid_um"]
        field = z["dNdlogDp"] * np.pi * dp ** 2 if sa else z["dNdlogDp"]
        pc = ax.pcolormesh(t, dp, np.maximum(field, 1e-9).T, norm=norm,
                           cmap="viridis", rasterized=True, shading="nearest")
        ax.set_yscale("log")
        ax.set_ylim(dp[0], 3.0)
        note = "standard V$_0$" if f == 1 else f"V$_0$ ×{f}  (SO$_2$ ÷{f})"
        ax.text(0.02, 0.95, note, transform=ax.transAxes, fontsize=10, color="white",
                va="top", fontweight="bold")
    axs.ravel()[-1].axis("off")
    for ax in axs[-1]:
        ax.set_xlabel("day")
    axs[2, 1].set_xlabel("day")           # bottom-right panel is blank; label the one above
    axs[2, 1].tick_params(labelbottom=True)
    for ax in axs[:, 0]:
        ax.set_ylabel("dry diameter [µm]")
    fig.suptitle("Plume evolution vs initial volume (1 t SO$_2$; larger V$_0$ = lower initial "
                 f"concentration)\n30°N / 20 km / 210 K, SABR-220, {DIL_LABEL}, α ×1, nuc ×1, "
                 "coag ×1", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    cax = fig.add_axes((0.90, 0.12, 0.02, 0.76))
    fig.colorbar(pc, cax=cax, label=label)
    dsfx = '' if DIL == 'D2med' else f'_{DIL}'
    fname = f"banana_boxsize{dsfx}{'_SA' if sa else ''}.png"
    fig.savefig(os.path.join(_OUT, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


def main():
    global DIL, DIL_LABEL
    os.makedirs(_OUT, exist_ok=True)
    for DIL, DIL_LABEL in [("D2med", "D2 med"), ("D1low", "D1 low")]:
        if not os.path.exists(_path(2)):
            print(f"  SKIP {DIL} (runs not present)")
            continue
        fig_banana_grid()
        fig_banana_grid(sa=True)


if __name__ == "__main__":
    main()
