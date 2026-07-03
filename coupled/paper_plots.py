# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Cross-run plots for the paper ensemble (coupled/paper_matrix.py output).

Reads coupled/analyses/paper/{manifest.csv, summary.csv, <case>/state.npz} and produces, per swept
axis, (a) time-series overlays of the key quantities across that axis's levels, and (b) a tornado of
the headline metrics' %-change vs BASE across all cases. Reuses the paper Helvetica / no-top-right-
spine style. Robust to missing cases (only plots what ran).

Run: python -m coupled.paper_plots
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coupled.paper_matrix import AXES, _OUT
from config import IDX

_H2SO4, _SO2, _O2 = IDX["H2SO4"], IDX["SO2"], IDX["O2"]

# paper style: Helvetica, only x/y axes (no top/right spine), light grid
plt.rcParams.update({
    "font.family": "Helvetica", "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "figure.dpi": 130,
})


def _load_case(name):
    p = os.path.join(_OUT, name, "state.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    t = d["t"]; x = d["x"]
    M = x[0, _O2] / 0.21
    return dict(
        t_days=t / 86400.0,
        SO2=x[:, _SO2] / M * 1e12,        # pptv
        H2SO4=x[:, _H2SO4] / M * 1e12,    # pptv
        N=d["n_cm3"].sum(axis=1),         # #/cm^3
        SA=d["SA"],                       # um^2/cm^3
    )


_SERIES = [("N", "N  [# cm$^{-3}$]", True), ("SA", "surface area  [$\\mu$m$^2$ cm$^{-3}$]", False),
           ("H2SO4", "gas H$_2$SO$_4$  [pptv]", False), ("SO2", "SO$_2$  [pptv]", True)]


def per_axis_timeseries():
    """One 2x2 (N/SA/H2SO4/SO2 vs time) figure per axis, overlaying BASE + that axis's levels."""
    base = _load_case("BASE")
    for axis, levels in AXES.items():
        names = ["BASE"] + [f"{axis}={lab}" for lab in levels]
        loaded = [(n, _load_case(n)) for n in names]
        loaded = [(n, d) for n, d in loaded if d is not None]
        if len(loaded) < 2:
            continue
        colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(loaded)))
        fig, axs = plt.subplots(2, 2, figsize=(10, 7))
        for (key, ylab, logy), ax in zip(_SERIES, axs.ravel()):
            for (n, d), c in zip(loaded, colors):
                ax.plot(d["t_days"], d[key], color=c, lw=1.6,
                        label=n.replace(f"{axis}=", ""))
            ax.set_ylabel(ylab); ax.set_xlabel("time  [days]")
            if logy:
                ax.set_yscale("log")
        axs[0, 0].legend(title=axis, fontsize=8, title_fontsize=9, frameon=False)
        fig.suptitle(f"Sensitivity to {axis}", fontweight="bold")
        fig.tight_layout()
        out = os.path.join(_OUT, f"axis_{axis}.png")
        fig.savefig(out, bbox_inches="tight"); plt.close(fig)
        print(f"wrote {out}", flush=True)


_METRICS = [("N_max", "peak N"), ("SA_max", "peak SA"),
            ("H2SO4_max_ppt", "peak H$_2$SO$_4$"), ("SO2_end_ppt", "end SO$_2$")]


def tornado_vs_base():
    """%-change vs BASE for each headline metric, one horizontal-bar panel per metric."""
    sp = os.path.join(_OUT, "summary.csv")
    if not os.path.exists(sp):
        print("no summary.csv yet; skipping tornado"); return
    rows = {r["name"]: r for r in csv.DictReader(open(sp))}
    if "BASE" not in rows:
        print("no BASE row; skipping tornado"); return
    cases = [n for n in rows if n != "BASE"]
    fig, axs = plt.subplots(1, len(_METRICS), figsize=(4 * len(_METRICS), 0.4 * len(cases) + 2),
                            sharey=True)
    y = np.arange(len(cases))
    for (mkey, mlab), ax in zip(_METRICS, np.atleast_1d(axs)):
        b = float(rows["BASE"][mkey])
        pct = [100.0 * (float(rows[n][mkey]) / b - 1.0) if b else 0.0 for n in cases]
        ax.barh(y, pct, color=["#c0392b" if p < 0 else "#2c7fb8" for p in pct], height=0.7)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel(f"% Δ {mlab} vs BASE"); ax.set_yticks(y); ax.set_yticklabels(cases, fontsize=7)
    fig.suptitle("OAT sensitivity: % change vs BASE", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(_OUT, "tornado_vs_base.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}", flush=True)


def main():
    per_axis_timeseries()
    tornado_vs_base()


if __name__ == "__main__":
    main()
