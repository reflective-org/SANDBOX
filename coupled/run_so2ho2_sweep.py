# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""SO2 + HO2 -> SO3 + OH sensitivity sweep on the D1 dilution scenario.

JPL 19-5 gives only an UPPER LIMIT (~1e-18 cm^3/molec/s) for SO2+HO2 and recommends NO products, so
the SO3+OH channel is a genuine uncertainty. This runs the D1 clean-stratosphere injection plume
(heating OFF) four times, varying only ``so2_ho2_rate``:

    off   : channel eliminated (rate = 0)
    1e-18 : JPL upper-limit / model default
    1e-17 : 10x upper limit
    1e-16 : 100x upper limit

and overlays OH, HO2, SO2, H2SO4 (molec/cm^3) vs time in one 2x2 figure. All else is identical to
``run_dilution_d1_clean`` (20 km, 30N, summer, 10 d, full physics minus heating; TOMAS 40-bin).

Writes coupled/analyses/so2ho2_sweep/ : so2ho2_sweep.png + so2ho2_sweep.npz.
Run: python -m coupled.run_so2ho2_sweep
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reuse the D1 scenario + the shared Helvetica/publication style
from coupled.run_dilution_d1_clean import _scenario
from coupled.driver import run_coupled
from config import IDX

_OUT = os.path.join(os.path.dirname(__file__), "analyses", "so2ho2_sweep")

# label -> so2_ho2_rate [cm^3/molec/s]; None-like 0.0 = channel off. Ordered off -> strongest.
_CASES = [("SO2+HO2 off", 0.0),
          ("k = 1e-18 (JPL upper limit)", 1.0e-18),
          ("k = 1e-17", 1.0e-17),
          ("k = 1e-16", 1.0e-16)]
_COLORS = ["#4C4C4C", "#1E88E5", "#F5A623", "#D64545"]   # neutral -> blue -> amber -> red


def main():
    os.makedirs(_OUT, exist_ok=True)
    runs = []
    for label, rate in _CASES:
        sc = _scenario(40)
        sc.so2_ho2_rate = rate
        print(f"running: {label} (so2_ho2_rate={rate:g}) ...", flush=True)
        t, x = run_coupled(sc)[:2]
        runs.append((label, rate, t, x))
    days = runs[0][2] / 86400.0
    M = runs[0][3][0, IDX["O2"]] / 0.21

    # save raw for later replotting/analysis
    np.savez(os.path.join(_OUT, "so2ho2_sweep.npz"),
             days=days, M=M, labels=[c[0] for c in _CASES], rates=[c[1] for c in _CASES],
             **{f"x_{i}": r[3] for i, r in enumerate(runs)})

    species = [("OH", "OH [molec cm$^{-3}$]"), ("HO2", "HO2 [molec cm$^{-3}$]"),
               ("SO2", "SO2 [molec cm$^{-3}$]"), ("H2SO4", "H2SO4 [molec cm$^{-3}$]")]
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    for ax, (name, ylab) in zip(axs.flat, species):
        for (label, _rate, _t, x), color in zip(runs, _COLORS):
            ax.plot(days, np.maximum(x[:, IDX[name]], 1e-2), lw=1.5, color=color, label=label)
        ax.set_yscale("log")
        ax.set_title(name)
        ax.set_ylabel(ylab)
    for ax in axs[-1, :]:
        ax.set_xlabel("day")
    axs[0, 0].legend(fontsize=9, title="SO2 + HO2 $\\rightarrow$ SO3 + OH")
    fig.suptitle("D1 dilution: SO2+HO2 rate sensitivity (heating off)", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "so2ho2_sweep.png"))
    plt.close(fig)

    # concise end-of-run summary
    print("\ncase                              end SO2       end H2SO4     day-10 OH(noon)")
    for label, rate, t, x in runs:
        oh = x[:, IDX["OH"]]
        print(f"{label:33s} {x[-1, IDX['SO2']]:.3e}   {x[-1, IDX['H2SO4']]:.3e}   {oh.max():.3e}")
    print(f"\nwrote so2ho2_sweep.png + .npz to {_OUT}/")


if __name__ == "__main__":
    main()
