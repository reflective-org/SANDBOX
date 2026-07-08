# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Surface-area distribution per unit injected sulfur vs time
(-> runs/plots/dAdlogDp_per_injectedS.png).

y = [dA/dlogDp(t) - dA/dlogDp(0)] x V(t)/V0 / N_S,injected   [m^2 per injected S atom]

i.e. the plume-integrated EXCESS surface area per sulfur atom injected -- dilution-fair
(concentration x volume) and background-subtracted. Snapshots at plume age 2 / 5 / 10 d.
A regime's curve is DROPPED at snapshots after its plume has relaxed to within 10% of the
background surface area (24 h-smoothed), i.e. once the excess is no longer meaningful.

Cases: 30N / 20 km / 210 K, SABR-220, alpha x1, nuc x1, coag x1, all five dilution regimes.
Injected S = 1 t SO2 -> 6.273e15 SO2 molecules/cm^3 in V0 (one S atom each).

Run (from SANDBOX/): python -m coupled.paper_ensemble.make_area_per_injected_s
"""
import os

import numpy as np
from scipy.ndimage import median_filter
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coupled.paper_ensemble.make_paper_candidate_plots import (
    _RUNS, REGIMES, REGIME_LABEL, REGIME_COLOR, case_dir)

_OUT = os.path.join(_RUNS, "plots")
HOURS = [48, 120, 240]            # 2 / 5 / 10 d
N_S_INJ = 6.273063291666667e15          # SO2 molecules/cm^3 in V0 (1 S atom each)
CASE = "30N_20km__sabr220__{reg}__a1p0__nuc1__cg1"

plt.rcParams.update({"font.family": "Helvetica", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
                     "grid.linewidth": 0.5, "grid.color": "#e1e0d9", "axes.axisbelow": True,
                     "figure.dpi": 130, "legend.frameon": False})


def main():
    os.makedirs(_OUT, exist_ok=True)
    data = {}
    for reg in REGIMES:
        z = np.load(os.path.join(case_dir(CASE.format(reg=reg)), "state.npz"))
        t_d = z["t"] / 86400.0
        # 10%-above-background crossing (24 h-smoothed SA excess), as in the t* analysis
        e = median_filter(z["SA"] / z["SA"][0] - 1.0, size=145, mode="nearest")
        idx = np.where(e < 0.10)[0]
        idx = idx[idx > 50]
        t_cross_h = t_d[idx[0]] * 24.0 if len(idx) else np.inf
        data[reg] = (z, t_cross_h)

    # age-2d panel gets its own tighter y-scale (peaks ~1e-21) so the five curves
    # separate; panels 2-3 share the full scale
    fig, axs = plt.subplots(1, 3, figsize=(12.0, 4.4), sharex=True)
    for k, hour in enumerate(HOURS):
        ax = axs.ravel()[k]
        for reg in REGIMES:
            z, t_cross_h = data[reg]
            if hour >= t_cross_h:
                continue                     # plume within 10% of background: excluded
            t_d = z["t"] / 86400.0
            it = int(np.searchsorted(t_d, hour / 24.0 - 1e-9))
            dp_um = z["dp_mid_um"]
            da = z["dNdlogDp"] * np.pi * dp_um ** 2 * 1e-12        # m^2/cm^3 per dlogDp
            y = (da[it] - da[0]) * float(z["V_ratio"][it]) / N_S_INJ
            ax.plot(dp_um * 1e3, y, lw=1.8, color=REGIME_COLOR[reg],
                    label=REGIME_LABEL[reg])
        ax.set_xscale("log")
        ax.set_xlim(1.0, 2e4)
        ax.set_title(f"plume age {hour // 24} d", fontsize=12)
        ax.axhline(0.0, color="#898781", lw=0.8)
        ax.set_xlabel("D$_p$ [nm]")
    axs[0].set_ylim(0, 1.0e-21)
    for a in axs[1:]:
        a.set_ylim(0, 5.6e-21)
    axs[2].tick_params(labelleft=False)
    axs[0].set_ylabel("dA/dlogD$_p$ ÷ injected S  [m$^2$]")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncols=5, fontsize=9, bbox_to_anchor=(0.5, 0.90))
    fig.suptitle("Plume-integrated excess surface area per injected sulfur atom — "
                 "30°N / 20 km / 210 K, SABR-220, α ×1, nuc ×1, coag ×1\n"
                 "(curves dropped once the plume is within 10% of the background SA; "
                 "dry diameter)", y=1.0, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    fig.savefig(os.path.join(_OUT, "dAdlogDp_per_injectedS.png"), bbox_inches="tight")
    plt.close(fig)
    for reg in REGIMES:
        print(f"  {reg}: 10% crossing at {data[reg][1]:.1f} h"
              if np.isfinite(data[reg][1]) else f"  {reg}: never crosses")
    print("  dAdlogDp_per_injectedS.png")


if __name__ == "__main__":
    main()
