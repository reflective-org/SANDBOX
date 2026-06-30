"""Plotting for the box model -- Python versions of the MATLAB figures.

Python port of the active figures in ``runconcs_het.m``:
  * chlorine / NO2 partitioning (MATLAB Figure 1)
  * ozone as a fraction of its starting value (Figure 2)
  * HOx: OH and HO2 (Figure 3)
  * bromine family (Figure 5)
  * SO2 and H2O2 (the FK-added Figure 8 panels)

Each function takes the model output and draws into a Matplotlib Axes, so they can be used
individually or assembled into the overview produced by ``plot_overview``.

Run as a script to integrate the default scenario and save an overview figure:
    python3 src-python/plotting.py            # writes overview.png
"""

from __future__ import annotations

import numpy as np

import matplotlib.pyplot as plt

from config import IDX, SPECIES


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------
def states_to_pptv(states: np.ndarray, M: float) -> dict:
    """Convert a (n_times, 34) state array (molec/cm^3) to pptv time series by name."""
    return {name: states[:, IDX[name]] / M * 1e12 for name in SPECIES}


def total_inorganic_chlorine(pptv: dict) -> np.ndarray:
    """Cly = total inorganic chlorine (pptv), as defined in runconcs_het.m."""
    return (pptv["Cl"] + pptv["ClO"] + 2 * pptv["ClOOCl"] + pptv["ClONO2"]
            + pptv["HCl"] + 2 * pptv["Cl2"] + pptv["HOCl"] + pptv["OClO"]
            + pptv["BrCl"])


def night_intervals(td: float, tn: float, days: int) -> list[tuple[float, float]]:
    """Return (start_hr, end_hr) of each nighttime period, for shading.

    The run begins with one daytime period (0..td), then alternates night/day.
    """
    out = []
    t = td
    for _ in range(days):
        out.append((t, t + tn))   # night
        t += tn + td              # skip the following day
    return out


def _shade_nights(ax, nights):
    for i, (a, b) in enumerate(nights):
        ax.axvspan(a, b, color="0.85", zorder=0,
                   label="night" if i == 0 else None)


def _scenario_text(ax, cfg):
    txt = (f"T = {cfg.T:g} K\nP = {cfg.P:g} mbar\n"
           f"H2O = {cfg.WTR:g} ppm\nSA = {cfg.SA:g} $\\mu$m$^2$/cm$^3$")
    ax.text(0.015, 0.97, txt, transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.7))


# ---------------------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------------------
def plot_chlorine(ax, t_hr, pptv, cfg=None, nights=None):
    """Chlorine / NO2 partitioning (MATLAB Figure 1), including total Cly."""
    if nights:
        _shade_nights(ax, nights)
    for name, c in [("HCl", "r"), ("ClONO2", "g"), ("NO2", "m"), ("ClO", "b"),
                    ("ClOOCl", "c"), ("Cl2", "y"), ("NO", "k")]:
        ax.plot(t_hr, pptv[name], c, lw=1.8, label=name)
    ax.plot(t_hr, total_inorganic_chlorine(pptv), "0.4", lw=1.5, ls="--", label="Cly")
    ax.set_title("Chlorine / NO$_2$ partitioning")
    ax.set_xlabel("Time (hr)"); ax.set_ylabel("Mixing ratio (pptv)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    if cfg is not None:
        _scenario_text(ax, cfg)


def plot_ozone_fraction(ax, t_hr, pptv, cfg=None, nights=None):
    """Ozone as a fraction of its starting value (MATLAB Figure 2)."""
    if nights:
        _shade_nights(ax, nights)
    o3 = pptv["O3"]
    ax.plot(t_hr, o3 / o3[0], "b", lw=1.8)
    ax.set_title("Ozone (fraction of start)")
    ax.set_xlabel("Time (hr)"); ax.set_ylabel("O$_3$ / O$_3$(0)")
    ax.grid(alpha=0.3)
    if cfg is not None:
        _scenario_text(ax, cfg)


def plot_hox(ax, t_hr, pptv, cfg=None, nights=None):
    """HOx: OH and HO2 (MATLAB Figure 3)."""
    if nights:
        _shade_nights(ax, nights)
    ax.plot(t_hr, pptv["OH"], "b", lw=1.8, label="OH")
    ax.plot(t_hr, pptv["HO2"], "r", lw=1.8, label="HO2")
    ax.set_title("HOx")
    ax.set_xlabel("Time (hr)"); ax.set_ylabel("Mixing ratio (pptv)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


def plot_bromine(ax, t_hr, pptv, cfg=None, nights=None):
    """Bromine family (MATLAB Figure 5)."""
    if nights:
        _shade_nights(ax, nights)
    for name, c in [("BrO", "b"), ("BrCl", "r"), ("BrONO2", "g"),
                    ("Br", "m"), ("HBr", "k"), ("HOBr", "c")]:
        ax.plot(t_hr, pptv[name], c, lw=1.8, label=name)
    ax.set_title("Bromine")
    ax.set_xlabel("Time (hr)"); ax.set_ylabel("Mixing ratio (pptv)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)


def plot_so2_h2o2(ax, t_hr, pptv, cfg=None, nights=None):
    """SO2 and H2O2 (the FK-added Figure 8 panels), on twin y-axes."""
    if nights:
        _shade_nights(ax, nights)
    ax.plot(t_hr, pptv["SO2"], "g", lw=1.8, label="SO2")
    ax.set_xlabel("Time (hr)"); ax.set_ylabel("SO$_2$ (pptv)", color="g")
    ax.tick_params(axis="y", labelcolor="g")
    ax2 = ax.twinx()
    ax2.plot(t_hr, pptv["H2O2"], "purple", lw=1.8, label="H2O2")
    ax2.set_ylabel("H$_2$O$_2$ (pptv)", color="purple")
    ax2.tick_params(axis="y", labelcolor="purple")
    ax.set_title("SO$_2$ and H$_2$O$_2$"); ax.grid(alpha=0.3)


# ---------------------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------------------
def plot_overview(t_seconds, states, cfg, td=14.0, tn=10.0, days=5):
    """Assemble all active figures into one overview Figure and return it."""
    t_hr = np.asarray(t_seconds) / 3600.0
    pptv = states_to_pptv(states, cfg.M)
    nights = night_intervals(td, tn, days)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(
        f"Box model overview  (T={cfg.T:g} K, P={cfg.P:g} mbar, "
        f"H2O={cfg.WTR:g} ppm, SA={cfg.SA:g} um^2/cm^3, opt={cfg.opt})",
        fontsize=13,
    )
    plot_chlorine(axes[0, 0], t_hr, pptv, cfg, nights)
    plot_hox(axes[0, 1], t_hr, pptv, cfg, nights)
    plot_ozone_fraction(axes[0, 2], t_hr, pptv, cfg, nights)
    plot_bromine(axes[1, 0], t_hr, pptv, cfg, nights)
    plot_so2_h2o2(axes[1, 1], t_hr, pptv, cfg, nights)
    axes[1, 2].axis("off")  # spare panel
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def main():
    import os

    import matplotlib
    matplotlib.use("Agg")
    from config import ModelConfig
    from driver import run

    cfg = ModelConfig(T=210.0, P=68.0, SA=2.0, WTR=5.0, Yn2o5=0.1, opt=1)
    td, tn, days = 14.0, 10.0, 5
    t, states = run(cfg, td=td, tn=tn, days=days, DT=600.0)
    fig = plot_overview(t, states, cfg, td=td, tn=tn, days=days)

    figdir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, "overview.png")
    fig.savefig(out, dpi=110)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
