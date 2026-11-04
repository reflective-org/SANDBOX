# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Figure: plume dilution V(t)/V0 of the eight v2 sheared-Gaussian cases (L19), over the 10-day runs.

(a) log-log from 1 s, to show the early (diffusive, 2Kz t / sigma_z0^2) and late (shear, t^2) regimes;
(b) linear days 0-10 with the evaluation days 1, 2, 7, 10 marked and the C3/C3a burst (48 h).
Colour = Kz family, solid = across shear (sigma_L = 25 m), dashed = along shear (sigma_L = 15 km/sqrt12).

    python -m coupled.paper_ensemble.make_dilution_v2_plot     # -> runs_v2/plots/dilution_v2.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from coupled.dilution import GAUSSIAN_CASES, volume_ratio  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "runs_v2", "plots", "dilution_v2.png")
DAY = 86400.0
EVAL_DAYS = (1, 2, 7, 10)
COLOR = {"C1": "#1f77b4", "C2": "#1f77b4", "C3": "#d62728", "C3a": "#d62728",
         "C4": "#2ca02c", "C4a": "#2ca02c", "C5": "#9467bd", "C5a": "#9467bd"}
FAMILY = {"C1": "Kz 0.01", "C3": "Kz 0.001 + burst Kz 10 at 48 h", "C4": "Kz 0.001", "C5": "Kz 0.1"}


def _style(name):
    return dict(color=COLOR[name], ls="--" if name.endswith("a") or name == "C2" else "-", lw=1.8)


def main(out=OUT):
    t_log = np.logspace(np.log10(60.0), np.log10(10 * DAY), 2000)
    t_lin = np.linspace(0, 10 * DAY, 4001)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for name in GAUSSIAN_CASES:
        ax1.loglog(t_log / DAY, volume_ratio(t_log, name), label=name, **_style(name))
        ax2.semilogy(t_lin / DAY, volume_ratio(t_lin, name), **_style(name))
    for ax in (ax1, ax2):
        for d in EVAL_DAYS:
            ax.axvline(d, color="0.6", lw=0.8, ls=":")
        ax.set_xlabel("time since release (days)")
        ax.grid(True, which="major", alpha=0.3)
    ax1.set_ylabel("V(t) / V0")
    ax1.set_xlim(60.0 / DAY, 10)
    ax1.text(0.02, 0.62, "C3 = C4 and C3a = C4a until the burst at 48 h", transform=ax1.transAxes,
             fontsize=8, color="0.3")
    sec = ax1.secondary_xaxis("top", functions=(lambda d: d * 24, lambda h: h / 24))
    sec.set_xlabel("hours")
    ax2.set_xlim(0, 10)
    c3 = GAUSSIAN_CASES["C3"]
    ax2.axvspan(c3.t_burst_start / DAY, (c3.t_burst_start + c3.tau_burst) / DAY,
                color="#d62728", alpha=0.15, lw=0)
    ax2.set_xticks(range(11))
    ax1.set_title("(a) dilution, log time")
    ax2.set_title("(b) dilution over the 10-day runs (dotted: evaluation days 1, 2, 7, 10)")
    # legend: one entry per case, plus a key for colour (Kz) and line style (shear direction)
    ax1.legend(ncol=2, fontsize=8, loc="upper left", title="case (solid across, dashed along)",
               title_fontsize=8)
    handles = [plt.Line2D([], [], color=COLOR[k], lw=2, label=v) for k, v in FAMILY.items()]
    ax2.legend(handles=handles, fontsize=8, loc="lower right")
    nudge = {"C1": 5, "C3": -5, "C2": 5, "C3a": -5}      # pairs that end close together
    for name in GAUSSIAN_CASES:                          # day-10 value labels
        v = float(volume_ratio(np.array([10 * DAY]), name)[0])
        ax2.annotate(f"{name} {v:.2g}", (10, v), xytext=(3, nudge.get(name, 0)),
                     textcoords="offset points",
                     fontsize=7, va="center", color=COLOR[name], annotation_clip=False)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160)
    print(out)
    for name in GAUSSIAN_CASES:
        vals = volume_ratio(np.array(EVAL_DAYS, float) * DAY, name)
        print(name, " ".join(f"d{d}={v:.3g}" for d, v in zip(EVAL_DAYS, vals)))


if __name__ == "__main__":
    main()
