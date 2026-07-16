# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Combined SO2+HO2 x aerosol->photolysis comparison (all cases on one figure).

Overlays the two saved sweeps -- coupled/analyses/so2ho2_sweep (aerosol->J ON) and
so2ho2_sweep_noaer (aerosol->J OFF) -- for OH/HO2/SO2/H2SO4. Encoding: COLOR = SO2+HO2 rate,
LINESTYLE = solid (aerosol->J ON) vs dashed (aerosol->J OFF). Run both sweeps first:
    python -m coupled.run_so2ho2_sweep          # ON
    python -m coupled.run_so2ho2_sweep noaer    # OFF
    python -m coupled.plot_so2ho2_aer_combined
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import coupled.run_dilution_d1_clean  # noqa: F401  (Helvetica + no-spine style)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import IDX

_A = os.path.join(os.path.dirname(__file__), "analyses")
_COLORS = ["#4C4C4C", "#1E88E5", "#F5A623", "#D64545"]


def main():
    on = np.load(os.path.join(_A, "so2ho2_sweep", "so2ho2_sweep.npz"), allow_pickle=True)
    off = np.load(os.path.join(_A, "so2ho2_sweep_noaer", "so2ho2_sweep.npz"), allow_pickle=True)
    days = on["days"]
    labels = [str(s) for s in on["labels"]]
    species = [("OH", "OH [molec cm$^{-3}$]"), ("HO2", "HO2 [molec cm$^{-3}$]"),
               ("SO2", "SO2 [molec cm$^{-3}$]"), ("H2SO4", "H2SO4 [molec cm$^{-3}$]")]

    fig, axs = plt.subplots(2, 2, figsize=(13.5, 9), sharex=True)
    for ax, (name, ylab) in zip(axs.flat, species):
        for i, color in enumerate(_COLORS):
            ax.plot(days, np.maximum(on[f"x_{i}"][:, IDX[name]], 1e-2), lw=1.5, color=color)
            ax.plot(days, np.maximum(off[f"x_{i}"][:, IDX[name]], 1e-2), lw=1.3, color=color,
                    ls="--", alpha=0.85)
        ax.set_yscale("log")
        ax.set_title(name)
        ax.set_ylabel(ylab)
    for ax in axs[-1, :]:
        ax.set_xlabel("day")

    # two legends: color = SO2+HO2 rate; linestyle = aerosol->J on/off
    rate_handles = [Line2D([0], [0], color=c, lw=2) for c in _COLORS]
    style_handles = [Line2D([0], [0], color="0.35", lw=2, ls="-"),
                     Line2D([0], [0], color="0.35", lw=2, ls="--")]
    leg1 = axs[0, 0].legend(rate_handles, labels, fontsize=8.5,
                            title="SO2 + HO2 $\\rightarrow$ SO3 + OH", loc="lower center")
    axs[0, 0].add_artist(leg1)
    axs[0, 1].legend(style_handles, ["aerosol$\\rightarrow$J ON", "aerosol$\\rightarrow$J OFF"],
                     fontsize=8.5, loc="lower center")
    fig.suptitle("D1 dilution: SO2+HO2 rate x aerosol$\\rightarrow$photolysis (heating off)", y=0.996)
    fig.tight_layout()
    out = os.path.join(_A, "so2ho2_aerosol_combined.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
