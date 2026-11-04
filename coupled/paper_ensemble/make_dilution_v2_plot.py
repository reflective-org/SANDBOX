# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Figure: plume dilution V(t)/V0 of the eight v2 sheared-Gaussian cases (L19), over the 10-day runs.

(a) log-log from 1 min, to show the early (diffusive, 2Kz t / sigma_z0^2) and late (shear, t^2) regimes;
(b) linear days 0-10 with the evaluation days 1, 2, 7, 10 marked and the C3/C3a burst (48 h).
Colour = Kz family, solid = across shear (sigma_L = 25 m), dashed = along shear (sigma_L = 15 km/sqrt12).

Reflective brand style (Brand Guidelines 2026), as in the reflective-org/presentations theme:
Open Sans for all text (font files from the presentations repo's ``theme/fonts``), panel titles in
Gambarino (installed locally; Fontshare licence, not redistributable) or its official replacement
Libre Caslon Text (``theme/fonts``), Charcoal text, brand Blue / Clay Red / Green / Gold for the four
Kz families. Point ``REFLECTIVE_THEME_DIR`` at the presentations repo's ``theme/`` if it is not at
``~/Documents/GitHub/presentations/theme``; without it matplotlib's default fonts are used (and said so).

    python -m coupled.paper_ensemble.make_dilution_v2_plot   # -> runs_v2/plots/dilution_v2.{pdf,png}
"""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from coupled.dilution import GAUSSIAN_CASES, volume_ratio  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "runs_v2", "plots", "dilution_v2")      # + .pdf / .png
THEME = Path(os.environ.get("REFLECTIVE_THEME_DIR",
                            Path.home() / "Documents" / "GitHub" / "presentations" / "theme"))
DAY = 86400.0
EVAL_DAYS = (1, 2, 7, 10)

# Reflective palette (brand guidelines 2026)
CHARCOAL, SLATE, SAGE, SAND = "#282C2E", "#8B8D8E", "#BBC2BD", "#F6EFEA"
BLUE, CLAY, GREEN, GOLD = "#466F8D", "#A0503C", "#858F6A", "#A47F52"
COLOR = {"C1": BLUE, "C2": BLUE, "C3": CLAY, "C3a": CLAY,
         "C4": GREEN, "C4a": GREEN, "C5": GOLD, "C5a": GOLD}
FAMILY = {"C1": "Kz 0.01", "C3": "Kz 0.001 + burst Kz 10 at 48 h", "C4": "Kz 0.001", "C5": "Kz 0.1"}


def brand_style():
    """Register the brand fonts and set the rcParams; returns the headline font family."""
    body = "DejaVu Sans"
    for f in sorted((THEME / "fonts" / "OpenSans").glob("OpenSans-*.ttf")):
        font_manager.fontManager.addfont(str(f))
        body = "Open Sans"
    headline = body
    for f in sorted((THEME / "fonts" / "LibreCaslonText").glob("*.ttf")):
        font_manager.fontManager.addfont(str(f))
        headline = "Libre Caslon Text"
    for f in (Path.home() / "Library" / "Fonts" / "Gambarino-Regular.otf",
              Path("/Library/Fonts/Gambarino-Regular.otf")):
        if f.exists():
            font_manager.fontManager.addfont(str(f))
            headline = "Gambarino"
            break
    if body != "Open Sans":
        print(f"[brand] Open Sans not found under {THEME}/fonts; using {body}")
    plt.rcParams.update({
        "font.family": body, "font.size": 9, "text.color": CHARCOAL,
        "axes.labelcolor": CHARCOAL, "axes.edgecolor": SAGE, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": SAGE, "ytick.color": SAGE, "xtick.labelcolor": CHARCOAL,
        "ytick.labelcolor": CHARCOAL, "legend.frameon": False, "axes.unicode_minus": True,
        "pdf.fonttype": 42, "ps.fonttype": 42,          # embed TrueType outlines, text stays text
        "figure.facecolor": "white", "axes.facecolor": "white"})
    return headline


def _style(name):
    return dict(color=COLOR[name], ls=(0, (5, 2.5)) if name.endswith("a") or name == "C2" else "-",
                lw=1.7)


def main(out=OUT):
    headline = brand_style()
    title_kw = dict(fontfamily=headline, fontsize=12, loc="left", color=CHARCOAL, pad=10)
    t_log = np.logspace(np.log10(60.0), np.log10(10 * DAY), 2000)
    t_lin = np.linspace(0, 10 * DAY, 4001)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for name in GAUSSIAN_CASES:
        ax1.loglog(t_log / DAY, volume_ratio(t_log, name), label=name, **_style(name))
        ax2.semilogy(t_lin / DAY, volume_ratio(t_lin, name), **_style(name))
    for ax in (ax1, ax2):
        for d in EVAL_DAYS:
            ax.axvline(d, color=SAGE, lw=0.8, ls=":", zorder=0)
        ax.set_xlabel("Time since release (days)")
        ax.grid(True, which="major", color="#E6E8E7", lw=0.6, zorder=0)
    ax1.set_ylabel("V(t) / V0")
    ax1.set_xlim(60.0 / DAY, 10)
    ax1.text(0.02, 0.62, "C3 = C4 and C3a = C4a until the burst at 48 h", transform=ax1.transAxes,
             fontsize=8, color=SLATE)
    sec = ax1.secondary_xaxis("top", functions=(lambda d: d * 24, lambda h: h / 24))
    sec.set_xlabel("Hours", color=SLATE)
    sec.spines["top"].set_color(SAGE)
    ax2.set_xlim(0, 10)
    c3 = GAUSSIAN_CASES["C3"]
    ax2.axvspan(c3.t_burst_start / DAY, (c3.t_burst_start + c3.tau_burst) / DAY,
                color=SAND, lw=0, zorder=0)
    ax2.set_xticks(range(11))
    ax1.set_title("(a) Plume dilution, log time", **title_kw)
    ax2.set_title("(b) Over the 10-day runs", **title_kw)
    ax2.text(1.0, 1.02, "dotted: evaluation days 1, 2, 7, 10", transform=ax2.transAxes,
             ha="right", va="bottom", fontsize=8, color=SLATE)
    ax1.legend(ncol=2, fontsize=8, loc="upper left", title="Case (solid across, dashed along)",
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
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200)
        print(f"{out}.{ext}")
    print(f"[brand] body font {plt.rcParams['font.family'][0]}, headline font {headline}")
    for name in GAUSSIAN_CASES:
        vals = volume_ratio(np.array(EVAL_DAYS, float) * DAY, name)
        print(name, " ".join(f"d{d}={v:.3g}" for d, v in zip(EVAL_DAYS, vals)))


if __name__ == "__main__":
    main()
