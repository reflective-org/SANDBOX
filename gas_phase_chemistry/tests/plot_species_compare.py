"""Per-species comparison of the Python port against the Octave reference.

For every species, draws a two-panel column:
  * top:    absolute concentration -- Octave (thick pale) and Python (thin) overlaid
  * bottom: ratio Python / Octave, with a reference line at 1.0

This makes per-species differences easy to see. Species that are identically zero in the
scenario (e.g. bromine at P=68) have an undefined ratio; those panels are annotated.

Usage:
    python3 tests/plot_species_compare.py [output.png]
"""

import csv
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src-python"))
from config import IDX, SPECIES, ModelConfig  # noqa: E402
from driver import run  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures")          # input CSV fixtures
FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")  # default output dir


def main(out_path):
    rows = list(csv.DictReader(open(os.path.join(FIX, "trajectory.csv"))))
    meta = {r[0]: r[1] for r in csv.reader(open(os.path.join(FIX, "trajectory_meta.csv")))
            if r and r[0] != "key"}
    M = float(meta["M"])
    DT = float(meta["DT"])
    t_hr = np.array([float(r["time"]) for r in rows]) / 3600.0

    cfg = ModelConfig(T=float(meta["T"]), P=float(meta["P"]), M=M, SA=float(meta["SA"]),
                      WTR=float(meta["WTR"]), Yn2o5=float(meta["Yn2o5"]),
                      opt=int(float(meta["opt"])))
    t, x = run(cfg, td=float(meta["td"]), tn=float(meta["tn"]),
               days=int(meta["days"]), DT=DT)

    # Layout: species arranged in a grid of `ncol` columns; each species occupies a
    # 2-row block (absolute on top, ratio below).
    ncol = 4
    nsp = len(SPECIES)
    nrow_species = int(np.ceil(nsp / ncol))
    fig = plt.figure(figsize=(4.2 * ncol, 3.0 * nrow_species))
    outer = gridspec.GridSpec(nrow_species, ncol, figure=fig, hspace=0.55, wspace=0.35)

    for k, name in enumerate(SPECIES):
        r, c = divmod(k, ncol)
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[r, c], height_ratios=[2, 1], hspace=0.1)
        ax_abs = fig.add_subplot(inner[0])
        ax_ratio = fig.add_subplot(inner[1], sharex=ax_abs)

        xo = np.array([float(rr[name]) for rr in rows])   # octave, molec/cm^3
        xp = x[:, IDX[name]]                               # python, molec/cm^3

        # Absolute panel.
        ax_abs.plot(t_hr, xo, color="0.6", lw=4, alpha=0.5)
        ax_abs.plot(t_hr, xp, color="C0", lw=1.2)
        ax_abs.set_title(name, fontsize=10, fontweight="bold")
        ax_abs.tick_params(labelbottom=False, labelsize=7)
        ax_abs.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        ax_abs.yaxis.get_offset_text().set_fontsize(6)
        ax_abs.grid(alpha=0.25)

        # Difference panel: relative difference (Python/Octave - 1), centered on 0.
        # Plotting (ratio - 1) rather than the ratio itself avoids matplotlib's
        # "1e-6 +1" offset label, which otherwise clutters the gap between panels
        # because the ratio sits so close to 1.0. Undefined where Octave is ~0.
        scale = max(np.max(np.abs(xo)), 1e-30)
        if scale <= 1e-29:  # species is identically zero in this scenario
            ax_ratio.text(0.5, 0.5, "= 0 (no diff)", ha="center", va="center",
                          transform=ax_ratio.transAxes, fontsize=7, color="0.5")
            ax_ratio.set_yticks([])
        else:
            reldiff = np.where(np.abs(xo) > 1e-6 * scale,
                               xp / np.where(xo == 0, np.nan, xo) - 1.0, np.nan)
            ax_ratio.plot(t_hr, reldiff, color="C3", lw=1.0)
            ax_ratio.axhline(0.0, color="0.4", lw=0.8, ls="--")
            ax_ratio.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax_ratio.yaxis.get_offset_text().set_fontsize(6)
            finite = reldiff[np.isfinite(reldiff)]
            if finite.size:
                dev = max(np.max(np.abs(finite)), 1e-12)
                ax_ratio.set_ylim(-1.3 * dev, 1.3 * dev)
                ax_ratio.text(0.02, 0.08, f"max |dev| {dev:.1e}", transform=ax_ratio.transAxes,
                              fontsize=6, color="C3")
        ax_ratio.tick_params(labelsize=7)
        ax_ratio.set_ylabel("P/O − 1", fontsize=7)
        ax_ratio.grid(alpha=0.25)
        if r == nrow_species - 1 or k + ncol >= nsp:
            ax_ratio.set_xlabel("Time (hr)", fontsize=7)

    fig.suptitle(
        f"Python vs Octave per species  (abs: Octave thick/pale, Python thin C0; diff: P/O - 1)\n"
        f"T={cfg.T:g} K, P={cfg.P:g} mbar, H2O={cfg.WTR:g} ppm, output step DT={DT:g} s",
        fontsize=12,
    )
    fig.subplots_adjust(top=0.95, bottom=0.03, left=0.06, right=0.98)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=110)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(FIGDIR, "species_compare.png")
    main(out)
