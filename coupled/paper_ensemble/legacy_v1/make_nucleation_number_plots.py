"""Particle-number figures for the D2-med nucleation sensitivity
(-> runs/plots/nucleation/plume_number_D2med.png, mode_number_D2med.png).

Cases: baseline 1 (30N / 20 km / 210 K), dilution D2 med, condensation alpha x1,
coagulation x1, SABR-220 (clean aged air) background; nucleation x0.01 / x1 / x100.

  plume_number_D2med.png -- total particle number: plume-integrated count N(t)*V(t)
                            (left; V0 from manifest.csv) and number concentration (right).
  mode_number_D2med.png  -- number concentration in the nucleation mode (Dp < 10 nm),
                            the Aitken mode (10-100 nm) and their sum (< 100 nm).
  plume_number_D2med_daily.png
                         -- daily-mean particles in plume as a ratio to the nucleation x1 run.
  plume_number_D2med_daily_rel0p01.png
                         -- the same, relative to the nucleation x0.01 run.
  mode_number_D2med_daily.png
                         -- the mode concentrations as daily means (day 1 = t 0-1 d, ...). Means
                            are time-weighted (trapezoidal): the output grid is 10 min plus
                            extra samples at the sunrise/sunset photolysis breakpoints.
  mode_plume_number_D2med.png, mode_plume_number_D2med_daily.png
                         -- the mode figures as plume-integrated counts N_mode(t)*V(t) [#].
  plume_sa_D2med_daily.png
                         -- daily-mean plume-integrated total WET surface area SA(t)*V(t) [km^2]
                            (SA as stored by the driver, the area the het. chemistry sees).
  plume_sa_by_dilution_daily.png
                         -- the same, one panel per dilution regime (D1/D2/D3/D5/burst). The dotted
                            line is the background air's SA in the plume volume; days after the
                            regime's evaluation day t* (make_nucleation_plots.EVAL_DAY) are shaded,
                            since the plume has relaxed to background by then.

In the time-series figures every panel's y-axis floor is 5% below the lowest value any case reaches during day 2
(t = 1-2 d), so the day-1 spin-up transient and the background level (both lower) are cut off.

Modes are split on the DRY bin-mid diameter (dp_mid_um, the fixed TOMAS grid), so each bin
falls wholly in one mode; the nearest mids are 9.08/10.19 nm and 91.5/102.7 nm.

Run (from SANDBOX/): python -m coupled.paper_ensemble.make_nucleation_number_plots
"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coupled.paper_ensemble.make_paper_candidate_plots import (
    _RUNS, REGIMES, REGIME_LABEL, case_dir)
from coupled.paper_ensemble.make_nucleation_plots import EVAL_DAY, NUC_COLOR, NUC_LAB

_OUT = os.path.join(_RUNS, "plots", "nucleation")

CASE = "30N_20km__sabr220__{reg}__a1p0__{nuc}__cg1"
NUCS = [("nuc0p01", 0.01), ("nuc1", 1.0), ("nuc100", 100.0)]
NUC_CUT_NM, AIT_CUT_NM = 10.0, 100.0
TITLE = "D2 med dilution, 30°N / 20 km / 210 K, α ×1, coag ×1, SABR-220 clean background"


def v0_cm3(cid):
    with open(os.path.join(_RUNS, "manifest.csv")) as f:
        for r in csv.DictReader(f):
            if r["case_id"] == cid:
                return float(r["V0_cm3"])
    raise KeyError(cid)


def load(nuc):
    """Time [d], per-bin number [cm^-3], dry mids [nm], V(t) [cm^3] for one case."""
    cid = CASE.format(reg="D2med", nuc=nuc)
    z = np.load(os.path.join(case_dir(cid), "state.npz"))
    return z["t"] / 86400.0, z["n_cm3"], z["dp_mid_um"] * 1e3, v0_cm3(cid) * z["V_ratio"]


PANELS = [("nucleation mode (D$_p$ < 10 nm)", lambda d: d < NUC_CUT_NM),
          ("Aitken mode (10–100 nm)", lambda d: (d >= NUC_CUT_NM) & (d < AIT_CUT_NM)),
          ("nucleation + Aitken (< 100 nm)", lambda d: d < AIT_CUT_NM)]


def plume_series(nuc):
    """Time [d], plume-integrated count N(t)*V(t) [#], total concentration [cm^-3]."""
    t, n, _, vol = load(nuc)
    ntot = n.sum(axis=1)
    return t, ntot * vol, ntot


def mode_series(nuc, integrated=False):
    """Time [d] and the number in each of PANELS' diameter ranges: concentration [cm^-3],
    or the plume-integrated count N_mode(t)*V(t) [#] if ``integrated``."""
    t, n, dd, vol = load(nuc)
    w = vol if integrated else 1.0
    return t, [n[:, sel(dd)].sum(axis=1) * w for _, sel in PANELS]


def _mode_text(integrated, daily=False):
    """(y-label, suptitle lead, file stem) for the concentration / plume-integrated variants."""
    mean = "daily-mean " if daily else ""
    if integrated:
        return (f"{mean}particles in plume, N·V [#]",
                "Sub-100 nm particles in plume (dry diameter)", "mode_plume_number_D2med")
    return (f"{mean}number concentration [cm$^{{-3}}$]",
            "Sub-100 nm number concentration (dry diameter)", "mode_number_D2med")


def plume_sa_series(nuc, reg="D2med"):
    """Time [d], plume-integrated total wet surface area SA(t)*V(t) [km^2], and the background
    air's share of it SA_bg*V(t) [km^2] (the t=0 state is the background distribution)."""
    cid = CASE.format(reg=reg, nuc=nuc)
    z = np.load(os.path.join(case_dir(cid), "state.npz"))
    vol_km2 = v0_cm3(cid) * z["V_ratio"] / 1e18          # cm^3 * um^2/cm^3 -> km^2
    return z["t"] / 86400.0, z["SA"] * vol_km2, z["SA"][0] * vol_km2


def daily_mean(t, y):
    """Time-weighted mean of y over each whole day; returns (day number 1..N, means)."""
    trapz = getattr(np, "trapezoid", None) or np.trapz
    days = np.arange(int(round(t[-1])))
    means = []
    for d in days:
        sel = (t >= d) & (t <= d + 1)
        means.append(trapz(y[sel], t[sel]) / (t[sel][-1] - t[sel][0]))
    return days + 1, np.array(means)


def _label_ends(ax, series, min_gap=0.055):
    """Direct-label each line at its right end (the legend carries the same identity).

    Call after the y-limits are final: labels are spread apart to at least ``min_gap`` of the
    (log) axis height so near-coincident end values stay readable."""
    lo, hi = np.log10(ax.get_ylim())
    series = sorted(series, key=lambda s: s[0])
    # clamp into the axis: an end value below the floor is labeled at the floor
    pos = [max((np.log10(y) - lo) / (hi - lo), 0.03) for y, _, _ in series]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    # ...and back down from the top if the stack overflowed it
    pos[-1] = min(pos[-1], 0.97)
    for i in range(len(pos) - 2, -1, -1):
        pos[i] = min(pos[i], pos[i + 1] - min_gap)
    for p, (_, lab, _) in zip(pos, series):
        ax.annotate(lab, (1.0, p), xycoords="axes fraction", xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color="#3b3a36")


def _ymin_day2(t, ys, pad=0.95):
    """Axis floor: ``pad`` x the minimum over all series during day 2 (1 <= t <= 2 d)."""
    sel = (t >= 1.0) & (t <= 2.0)
    return pad * min(y[sel].min() for y in ys)


def _style_time_axis(ax):
    ax.set_yscale("log")
    ax.set_xlim(0, 10)
    ax.set_xticks(range(0, 11, 2))
    ax.set_xlabel("day")


def fig_plume_number():
    fig, (ax_int, ax_c) = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    ends_int, ends_c, ys_int, ys_c = [], [], [], []
    for nuc, scale in NUCS:
        t, nint, ntot = plume_series(nuc)
        col = NUC_COLOR[scale]
        ax_int.plot(t, nint, color=col, lw=1.7, label=f"nucleation {NUC_LAB[scale]}")
        ax_c.plot(t, ntot, color=col, lw=1.7)
        ends_int.append((nint[-1], NUC_LAB[scale], col))
        ends_c.append((ntot[-1], NUC_LAB[scale], col))
        ys_int.append(nint)
        ys_c.append(ntot)
    for ax, ys in ((ax_int, ys_int), (ax_c, ys_c)):
        _style_time_axis(ax)
        ax.set_ylim(bottom=_ymin_day2(t, ys))
    _label_ends(ax_int, ends_int)
    _label_ends(ax_c, ends_c)
    ax_int.set_ylabel("particles in plume, N(t)·V(t) [#]")
    ax_int.set_title("plume-integrated particle number", fontsize=11)
    ax_c.set_ylabel("number concentration [cm$^{-3}$]")
    ax_c.set_title("total number concentration", fontsize=11)
    ax_int.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Total particle number: {TITLE}", fontsize=11.5)
    fig.savefig(os.path.join(_OUT, "plume_number_D2med.png"), bbox_inches="tight")
    plt.close(fig)
    print("  plume_number_D2med.png")


def fig_mode_number(integrated=False):
    ylab, lead, stem = _mode_text(integrated)
    data = {nuc: mode_series(nuc, integrated) for nuc, _ in NUCS}
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    ends = []
    for k, (ax, (title, _)) in enumerate(zip(axs, PANELS)):
        ends.append([])
        ys = []
        for nuc, scale in NUCS:
            t, nm = data[nuc][0], data[nuc][1][k]
            ax.plot(t, nm, color=NUC_COLOR[scale], lw=1.7, label=f"nucleation {NUC_LAB[scale]}")
            ends[-1].append((nm[-1], NUC_LAB[scale], NUC_COLOR[scale]))
            ys.append(nm)
        _style_time_axis(ax)
        ax.set_ylim(bottom=_ymin_day2(t, ys))
        ax.set_title(title, fontsize=11)
    for ax, e in zip(axs, ends):
        _label_ends(ax, e)
    axs[0].set_ylabel(ylab)
    axs[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"{lead}: {TITLE}", fontsize=11.5)
    fig.savefig(os.path.join(_OUT, f"{stem}.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}.png")


def _style_day_axis(ax):
    ax.set_yscale("log")
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("day of run")


def _plot_daily(ax, t, y, scale):
    days, m = daily_mean(t, y)
    ax.plot(days, m, "-o", color=NUC_COLOR[scale], lw=1.7, ms=5.5, mec="white", mew=1.0,
            label=f"nucleation {NUC_LAB[scale]}")
    return m[-1]


# reference run -> (file suffix, y-limits, y-ticks) for the ratio figure
RATIO_REFS = {"nuc1": ("", (0.1, 5), [0.1, 0.2, 0.5, 1, 2, 5]),
              "nuc0p01": ("_rel0p01", (0.5, 20), [0.5, 1, 2, 5, 10, 20])}


def fig_plume_number_daily(ref_nuc="nuc1"):
    """Daily-mean particles in the plume, as a ratio to the ``ref_nuc`` run.

    V(t) is identical across nucleation levels (same dilution regime), so this is also the
    concentration ratio to within the intra-day V weighting (<3% here); one panel suffices."""
    from matplotlib.ticker import FixedLocator, NullLocator
    suffix, ylim, ticks = RATIO_REFS[ref_nuc]
    ref_lab = NUC_LAB[dict(NUCS)[ref_nuc]]
    ref_t, ref_int, _ = plume_series(ref_nuc)
    ref = daily_mean(ref_t, ref_int)[1]
    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    ends = []
    for nuc, scale in NUCS:
        t, nint, _ = plume_series(nuc)
        days, m = daily_mean(t, nint)
        r = m / ref
        ax.plot(days, r, "-o", color=NUC_COLOR[scale], lw=1.7, ms=5.5, mec="white", mew=1.0,
                label=f"nucleation {NUC_LAB[scale]}")
        ends.append((r[-1], NUC_LAB[scale], None))
    _style_day_axis(ax)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([f"{v:g}" for v in ticks])
    _label_ends(ax, ends)
    ax.set_ylabel(f"particles in plume, relative to nucleation {ref_lab}")
    ax.legend(fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.72 if ref_nuc == "nuc1"
                                                                 else 0.3))
    ax.set_title(f"Plume particle number relative to nucleation {ref_lab}, daily means\n"
                 "D2 med, 30°N / 20 km / 210 K, α ×1, coag ×1, SABR-220 clean", fontsize=10.5)
    fname = f"plume_number_D2med_daily{suffix}.png"
    fig.savefig(os.path.join(_OUT, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


def fig_mode_number_daily(integrated=False):
    ylab, lead, stem = _mode_text(integrated, daily=True)
    data = {nuc: mode_series(nuc, integrated) for nuc, _ in NUCS}
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for k, (ax, (title, _)) in enumerate(zip(axs, PANELS)):
        ends = [(_plot_daily(ax, data[nuc][0], data[nuc][1][k], scale), NUC_LAB[scale], None)
                for nuc, scale in NUCS]
        _style_day_axis(ax)
        _label_ends(ax, ends)
        ax.set_title(title, fontsize=11)
    axs[0].set_ylabel(ylab)
    axs[0].legend(fontsize=8, loc="upper left" if integrated else "lower left")
    fig.suptitle(f"{lead}, daily means: {TITLE}", fontsize=11.5)
    fig.savefig(os.path.join(_OUT, f"{stem}_daily.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}_daily.png")


def fig_plume_sa_daily():
    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    ends = []
    for nuc, scale in NUCS:
        t, sa, _ = plume_sa_series(nuc)
        ends.append((_plot_daily(ax, t, sa, scale), NUC_LAB[scale], None))
    _style_day_axis(ax)
    _label_ends(ax, ends)
    ax.set_ylabel("daily-mean plume surface area, SA·V [km$^2$]")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Plume-integrated total surface area (wet), daily means\n"
                 "D2 med, 30°N / 20 km / 210 K, α ×1, coag ×1, SABR-220 clean", fontsize=10.5)
    fig.savefig(os.path.join(_OUT, "plume_sa_D2med_daily.png"), bbox_inches="tight")
    plt.close(fig)
    print("  plume_sa_D2med_daily.png")


def fig_plume_sa_by_dilution_daily():
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.6), constrained_layout=True)
    for ax, reg in zip(axs.flat, REGIMES):
        ends = []
        for nuc, scale in NUCS:
            t, sa, sa_bg = plume_sa_series(nuc, reg)
            ends.append((_plot_daily(ax, t, sa, scale), NUC_LAB[scale], None))
        days, bg = daily_mean(t, sa_bg)                  # identical across nucleation levels
        ax.plot(days, bg, ":", color="#898781", lw=1.4, label="background air in plume volume")
        _style_day_axis(ax)
        if EVAL_DAY[reg] < 10:
            ax.axvspan(EVAL_DAY[reg] + 0.5, 10.5, color="#e1e0d9", alpha=0.45, lw=0, zorder=0,
                       label="after t* (plume ≈ background)")
        _label_ends(ax, ends)
        ax.set_title(f"{REGIME_LABEL[reg]}  (t* = day {EVAL_DAY[reg]})", fontsize=11)
    for ax in axs[:, 0]:
        ax.set_ylabel("daily-mean plume SA·V [km$^2$]")
    leg = axs.flat[-1]
    leg.axis("off")
    handles, labels = axs.flat[3].get_legend_handles_labels()     # D5: has the t* shading
    leg.legend(handles, labels, loc="center", fontsize=9.5, title="nucleation rate scale")
    fig.suptitle("Plume-integrated total surface area (wet), daily means, by dilution regime\n"
                 "30°N / 20 km / 210 K, α ×1, coag ×1, SABR-220 clean background", fontsize=12)
    fig.savefig(os.path.join(_OUT, "plume_sa_by_dilution_daily.png"), bbox_inches="tight")
    plt.close(fig)
    print("  plume_sa_by_dilution_daily.png")


def main():
    os.makedirs(_OUT, exist_ok=True)
    fig_plume_number()
    fig_mode_number()
    fig_plume_number_daily()
    fig_plume_number_daily("nuc0p01")
    fig_mode_number_daily()
    fig_mode_number(integrated=True)
    fig_mode_number_daily(integrated=True)
    fig_plume_sa_daily()
    fig_plume_sa_by_dilution_daily()


if __name__ == "__main__":
    main()
