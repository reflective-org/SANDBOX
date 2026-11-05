# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Paper-style figures of the v2 base set (8 dilution cases x 5 V0, default microphysics).

The v1 paper figure set (legacy_v1/make_paper_candidate_plots.py, make_banana_nucleation.py) redrawn
in the Reflective brand style (``make_dilution_v2_plot.brand_style``: Open Sans, Gambarino titles,
Charcoal text) with the same colour per Kz family as the dilution figure: C1/C2 Blue, C3/C3a Clay
Red, C4/C4a Green, C5/C5a Gold. Bananas use a brand sequential map (Charcoal -> Blue -> Green ->
Yellow, lightness rising monotonically) with the v1 colour limits. Left column = across shear
(sigma_L = 25 m), right = along shear.
Bold line = mass-weighted-mean V0 (the L30 baseline); shading = range over the five V0 variants.

Box (not plume-integrated) quantities: plume-integrated, control-subtracted metrics (L16) need the
ctrl runs, which are not made yet.

    python -m coupled.paper_ensemble.make_v2_base_plots [runs_dir]    # default runs_v2/base
    -> <runs_dir>/../plots/base/{timeseries_v2_base, sizedist_day10_v2_grid, sizedist_evaldays_v2,
                                 banana_v2_base, banana_v2_base_SA}.{png,pdf}
"""
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

from coupled.paper_ensemble import run_v2 as rv  # noqa: E402
from coupled.paper_ensemble.make_dilution_v2_plot import (  # noqa: E402
    BLUE, CHARCOAL, CLAY, COLOR, GOLD, GREEN, SAGE, SLATE, brand_style)

_HERE = os.path.dirname(os.path.abspath(__file__))
GRID = "#E6E8E7"
YELLOW = "#FFBB00"                            # fill-only brand colour (never text)
CMAP = LinearSegmentedColormap.from_list("reflective_seq", [CHARCOAL, BLUE, GREEN, YELLOW])
HEADLINE = {"family": None}                   # set by _style()

DAY = 86400.0
EVAL_DAYS = (1, 2, 7, 10)
ACROSS = ("C4", "C1", "C5", "C3")             # low -> high mixing, then burst
ALONG = ("C4a", "C2", "C5a", "C3a")
LABEL = {"C4": "C4  Kz 0.001", "C1": "C1  Kz 0.01", "C5": "C5  Kz 0.1", "C3": "C3  Kz 0.001 + burst",
         "C4a": "C4a  Kz 0.001", "C2": "C2  Kz 0.01", "C5a": "C5a  Kz 0.1", "C3a": "C3a  Kz 0.001 + burst"}
COL_TITLE = (r"Across shear ($\sigma_L$ = 25 m)", r"Along shear ($\sigma_L$ = 15 km/$\sqrt{12}$)")
DAY_SHADE = {1: SAGE, 2: SLATE, 7: BLUE, 10: CHARCOAL}
V0_LS = {"peak": "-", "c1sig": (0, (5, 2)), "tophat": (0, (5, 1.5, 1, 1.5)), "c2sig": (0, (1, 1.5))}
XLIM = (1.5e-3, 3.0)                          # dry diameter [um]: smallest bin edge to past the tail


def _style():
    HEADLINE["family"] = brand_style()
    plt.rcParams.update({"axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.axisbelow": True, "figure.dpi": 130, "font.size": 9.5})


def _title(ax, text, **kw):
    ax.set_title(text, fontfamily=HEADLINE["family"], fontsize=kw.pop("fontsize", 11.5),
                 color=kw.pop("color", CHARCOAL), **kw)


def _suptitle(fig, text, y=0.995):
    """Headline (first line) in the headline font; the rest as a wrapped Slate subtitle below it,
    offset in points so it never overlaps whatever the figure height."""
    import textwrap
    from matplotlib.transforms import ScaledTranslation
    head, _, rest = text.partition("\n")
    fig.text(0.5, y, head, ha="center", va="top", fontfamily=HEADLINE["family"], fontsize=15,
             color=CHARCOAL)
    if rest:
        width = int(fig.get_figwidth() * 13)          # ~characters per line at 9 pt
        fig.text(0.5, y, "\n".join(textwrap.wrap(rest, width)), ha="center", va="top",
                 fontsize=9, color=SLATE, linespacing=1.4,
                 transform=fig.transFigure + ScaledTranslation(0, -24 / 72, fig.dpi_scale_trans))


def _savefig(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png/.pdf")


def load(runs):
    """{(dil, v0): series dict} for every finished state.npz under ``runs``."""
    data = {}
    for f in sorted(glob.glob(os.path.join(runs, "*", "state.npz"))):
        parts = os.path.basename(os.path.dirname(f)).split("__")
        dil, v0 = parts[2], parts[6][2:]
        z = np.load(f)
        n, dp = z["n_cm3"], z["dp_mid_um"]
        species = list(z["species"])
        data[dil, v0] = dict(
            t=z["t"] / DAY, dp=dp, dndlogdp=z["dNdlogDp"],
            N3=n[:, dp >= 0.003].sum(1), N10=n[:, dp >= 0.01].sum(1), SA=z["SA"],
            reff=z["radius_cm"] * 1e4, SO2=z["x"][:, species.index("SO2")],
            conv=z["so2_chem_loss_cum"] / rv.SO2_SPIKE_MOLEC_CM3[v0])
    return data


def _band(ax, data, dil, key, transform=None):
    runs = [data[k] for k in data if k[0] == dil]
    if not runs:
        return
    t = runs[0]["t"]
    f = transform or (lambda s, y: y)
    ys = np.array([np.interp(t, r["t"], f(r, r[key])) for r in runs])
    ax.fill_between(t, ys.min(0), ys.max(0), color=COLOR[dil], alpha=0.16, lw=0)
    if (dil, rv.DEFAULT_V0) in data:
        r = data[dil, rv.DEFAULT_V0]
        ax.plot(r["t"], f(r, r[key]), color=COLOR[dil], lw=1.7, label=LABEL[dil])


def fig_timeseries(data, out, tag):
    rows = [("N3", "N > 3 nm [cm$^{-3}$]", "log"), ("N10", "N > 10 nm [cm$^{-3}$]", "log"),
            ("SA", "surface area [µm$^2$ cm$^{-3}$]", "log"),
            ("reff", "effective radius (wet) [µm]", "linear"),
            ("conv", "SO$_2$ converted\n(fraction of injected)", "linear")]
    fig, axs = plt.subplots(len(rows), 2, figsize=(11, 13), sharex=True, sharey="row")
    for j, cases in enumerate((ACROSS, ALONG)):
        for i, (key, ylab, scale) in enumerate(rows):
            ax = axs[i, j]
            for dil in cases:
                _band(ax, data, dil, key)
            ax.set_yscale(scale)
            for d in EVAL_DAYS:
                ax.axvline(d, color=SAGE, lw=0.8, ls=":", zorder=0)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=10)
            if i == 0:
                _title(ax, COL_TITLE[j])
                ax.legend(fontsize=8.5, loc="lower left")
            if i == len(rows) - 1:
                ax.set_xlabel("day since release (06:00 local)")
    axs[0, 0].set_xlim(0, 10)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    _suptitle(fig, "Plume box evolution by dilution case\nv2 base set. Bold: mass-weighted-mean V0; "
              f"shading: range over the 5 V0 variants; dotted: evaluation days 1, 2, 7, 10.{tag}",
              y=0.995)
    _savefig(fig, out, "timeseries_v2_base")


def _background(ax, data, dil):
    """The SABR-220 background (the t = 0 distribution, also the entrained air) as a reference line."""
    r = next((v for k, v in data.items() if k[0] == dil), None)
    if r is not None:
        ax.loglog(r["dp"], r["dndlogdp"][0], color=CHARCOAL, lw=1.1, ls=(0, (4, 2)),
                  label="background (SABR-220)", zorder=2)


def _fig_legend(fig, ax):
    """One legend row under the subtitle, so no panel's curves are covered."""
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncols=len(h), fontsize=8.5, bbox_to_anchor=(0.5, 0.925),
               handlelength=2.6, columnspacing=1.6)


def _nearest(r, day):
    return int(np.argmin(np.abs(r["t"] - day)))


def fig_sizedist_day10(data, out, tag):
    fig, axs = plt.subplots(4, 2, figsize=(10, 12), sharex=True, sharey=True)
    for j, cases in enumerate((ACROSS, ALONG)):
        for i, dil in enumerate(cases):
            ax = axs[i, j]
            _background(ax, data, dil)
            for (d, v0), r in sorted(data.items()):
                if d != dil:
                    continue
                y = r["dndlogdp"][_nearest(r, 10)]
                if v0 == rv.DEFAULT_V0:
                    ax.loglog(r["dp"], y, color=COLOR[dil], lw=1.9, zorder=3)
                else:
                    ax.loglog(r["dp"], y, color=SLATE, lw=0.9, alpha=0.8, ls=V0_LS[v0], label=v0)
            ax.set_ylim(1e-2, 3e5)
            ax.set_xlim(*XLIM)
            _title(ax, LABEL[dil], fontsize=10.5, loc="left", color=COLOR[dil])
            if j == 0:
                ax.set_ylabel("dN/dlogD$_p$ [cm$^{-3}$]", fontsize=9.5)
            if i == 3:
                ax.set_xlabel("dry diameter [µm]")
    _fig_legend(fig, axs[0, 0])
    for j in range(2):
        axs[0, j].annotate(COL_TITLE[j], (0.5, 1.22), xycoords="axes fraction", ha="center",
                           fontsize=12, fontfamily=HEADLINE["family"], color=CHARCOAL)
    fig.tight_layout(rect=(0, 0, 1, 0.895))
    _suptitle(fig, "Day-10 size distribution\nBold: mass-weighted-mean V0; grey: the other V0 "
              f"variants; dashed: the SABR-220 background (t = 0, also the entrained air).{tag}",
              y=0.995)
    _savefig(fig, out, "sizedist_day10_v2_grid")


def fig_sizedist_evaldays(data, out, tag):
    fig, axs = plt.subplots(4, 2, figsize=(10, 12), sharex=True, sharey=True)
    for j, cases in enumerate((ACROSS, ALONG)):
        for i, dil in enumerate(cases):
            ax = axs[i, j]
            r = data.get((dil, rv.DEFAULT_V0))
            if r is None:
                continue
            _background(ax, data, dil)
            for d in EVAL_DAYS:
                ax.loglog(r["dp"], r["dndlogdp"][_nearest(r, d)], color=DAY_SHADE[d], lw=1.7,
                          label=f"day {d}")
            ax.set_ylim(1e-2, 3e5)
            ax.set_xlim(*XLIM)
            _title(ax, LABEL[dil], fontsize=10.5, loc="left", color=COLOR[dil])
            if j == 0:
                ax.set_ylabel("dN/dlogD$_p$ [cm$^{-3}$]", fontsize=9.5)
            if i == 3:
                ax.set_xlabel("dry diameter [µm]")
    _fig_legend(fig, axs[0, 0])
    for j in range(2):
        axs[0, j].annotate(COL_TITLE[j], (0.5, 1.22), xycoords="axes fraction", ha="center",
                           fontsize=12, fontfamily=HEADLINE["family"], color=CHARCOAL)
    fig.tight_layout(rect=(0, 0, 1, 0.895))
    _suptitle(fig, "Size distribution on the evaluation days\nAt 06:00 local, mass-weighted-mean V0; "
              f"dashed: the SABR-220 background (t = 0, also the entrained air).{tag}", y=0.995)
    _savefig(fig, out, "sizedist_evaldays_v2")


def fig_banana(data, out, tag, sa=False):
    norm = LogNorm(vmin=1e-2, vmax=2e3) if sa else LogNorm(vmin=1e2, vmax=1e8)
    label = "dSA/dlogD$_p$ [µm$^2$ cm$^{-3}$]" if sa else "dN/dlogD$_p$ [cm$^{-3}$]"
    fig, axs = plt.subplots(4, 2, figsize=(11, 12), sharex=True, sharey=True)
    pc = None
    for j, cases in enumerate((ACROSS, ALONG)):
        for i, dil in enumerate(cases):
            ax = axs[i, j]
            ax.grid(False)
            r = data.get((dil, rv.DEFAULT_V0))
            if r is None:
                continue
            field = r["dndlogdp"] * np.pi * r["dp"] ** 2 if sa else r["dndlogdp"]
            pc = ax.pcolormesh(r["t"], r["dp"], np.maximum(field, 1e-9).T, norm=norm,
                               cmap=CMAP, rasterized=True, shading="nearest")
            ax.set_yscale("log")
            ax.set_ylim(r["dp"][0], 2.0)
            ax.text(0.02, 0.93, LABEL[dil], transform=ax.transAxes, color="white", fontsize=10,
                    va="top")
            if i == 0:
                _title(ax, COL_TITLE[j])
            if i == 3:
                ax.set_xlabel("day since release (06:00 local)")
            if j == 0:
                ax.set_ylabel("dry diameter [µm]")
    fig.tight_layout(rect=(0, 0, 0.9, 0.93))
    _suptitle(fig, "Plume size-distribution evolution\nv2 base set, mass-weighted-mean V0, default "
              f"microphysics.{tag}", y=0.995)
    cax = fig.add_axes((0.92, 0.12, 0.02, 0.76))
    fig.colorbar(pc, cax=cax, label=label)
    _savefig(fig, out, "banana_v2_base_SA" if sa else "banana_v2_base")


def main(runs=None):
    runs = runs or os.path.join(rv.set_dir("base"))
    out = os.path.join(os.path.dirname(os.path.abspath(runs)), "plots", "base")
    _style()
    data = load(runs)
    n_mw = sum(1 for k in data if k[1] == rv.DEFAULT_V0)
    print(f"{len(data)} runs from {runs} ({n_mw} of 8 mass-weighted-mean) -> {out}")
    if not data:
        return
    tag = "" if len(data) == 40 else f" {len(data)} of 40 runs finished."
    if "pre_L31" in runs:
        tag += " Runs with 10 TOMAS coagulation sub-steps (pre-L31; within 0.2% of L31)."
    fig_timeseries(data, out, tag)
    fig_sizedist_day10(data, out, tag)
    fig_sizedist_evaldays(data, out, tag)
    fig_banana(data, out, tag)
    fig_banana(data, out, tag, sa=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
