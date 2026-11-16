# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Nucleation, coagulation and condensation sensitivity figures of the v2 ensemble, in the
Reflective style (the v1 set of legacy_v1/make_nucleation_plots.py + make_axis_effect_plots.py).

Ensemble figures compare runs that are identical except for the varied knob (matched sets), so the
other knobs cancel. Matched sets use the mass-weighted-mean V0 members only (L30): per dilution
case 6 nucleation triplets (coag x alpha), 6 coagulation triplets (nuc x alpha) and 9 alpha pairs
(nuc x coag). Metrics at the evaluation days 1, 2, 7, 10 (L26) plus 10-day peaks. Case studies:
C1 and C2 (mwmean), the other two knobs at x1. Box quantities.

Per knob, in <tree>/plots/<knob>/ (PNG + PDF):
  slopegraph, amplification_ratios, elasticity, sizedist_by_level_day{1,2,7,10},
  time_resolved_ratio, banana_C1_C2(_SA), banana_ratio_C1_C2(_SA), sizedist_snapshots_C1_C2
  (+_linear), totals_C1_C2

    python -m coupled.paper_ensemble.make_v2_axis_plots [nucleation|coagulation|condensation] [--tree DIR]
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

from coupled.paper_ensemble import v2_paper_lib as L  # noqa: E402
from coupled.paper_ensemble.make_v2_base_plots import HEADLINE  # noqa: E402
from coupled.paper_ensemble.v2_paper_lib import (  # noqa: E402
    ACROSS, ALONG, BLUE, CASES, CHARCOAL, CLAY, CMAP, COL_TITLE, COLOR, DAY_SHADE, EVAL_DAYS,
    GRID, LABEL, SAGE, SLATE, XLIM, _finish, _savefig, _title)

_LIGHT_BLUE, _DARK_BLUE = "#A9C0D1", "#1F3547"
AXES = {
    "nucleation": dict(key="nuc", name="nucleation rate", short="nucleation",
                       levels=(0.01, 1.0, 100.0), lab={0.01: "×0.01", 1.0: "×1", 100.0: "×100"},
                       color={0.01: _LIGHT_BLUE, 1.0: BLUE, 100.0: _DARK_BLUE},
                       fixed="coagulation ×1, α = 1", headline="Nucleation"),
    "coagulation": dict(key="cg", name="coagulation kernel", short="coagulation",
                        levels=(0.5, 1.0, 2.0), lab={0.5: "×0.5", 1.0: "×1", 2.0: "×2"},
                        color={0.5: _LIGHT_BLUE, 1.0: BLUE, 2.0: _DARK_BLUE},
                        fixed="nucleation ×1, α = 1", headline="Coagulation"),
    "condensation": dict(key="alpha", name="accommodation coefficient α", short=r"condensation $\alpha$",
                         levels=(0.5, 1.0), lab={0.5: "α = 0.5", 1.0: "α = 1"},
                         color={0.5: _LIGHT_BLUE, 1.0: BLUE},
                         fixed="nucleation ×1, coagulation ×1", headline="Condensation"),
}
_KI = {"alpha": 1, "nuc": 2, "cg": 3}                 # position of each knob in (dil, a, nuc, cg)
DAY_METRICS = [("N3", "N > 3 nm [cm$^{-3}$]"), ("N10", "N > 10 nm [cm$^{-3}$]"),
               ("SA", "surface area [µm$^2$ cm$^{-3}$]"), ("reff", "r$_{eff}$ (wet) [µm]")]
PEAK_METRICS = [("N3", "peak N > 3 nm"), ("SA", "peak surface area"),
                ("h2so4", "peak gas H$_2$SO$_4$")]
CASE_STUDY = ("C1", "C2")
RATIO_CMAP = LinearSegmentedColormap.from_list("reflective_div", [BLUE, "#F6EFEA", CLAY])
RATIO_CMAP.set_bad(GRID)


def matched_sets(ens, knob):
    """{(dil, other knobs...): {level: series}} for one knob; every set is complete."""
    k = _KI[knob]
    sets = {}
    for key, r in ens.items():
        rest = key[:k] + key[k + 1:]
        sets.setdefault(rest, {})[key[k]] = r
    n = len(L.LEVELS[knob])
    assert all(len(v) == n for v in sets.values()), "incomplete matched set"
    return sets


def day_value(r, metric, day):
    return r[metric][L.iday(r, day)]


def peak_value(r, metric):
    return np.max(L.h2so4_ppt(r)) if metric == "h2so4" else np.max(r[metric])


def _tag(ax_def, sets):
    n = len(sets) // 8
    return f"{n} matched {'pairs' if len(ax_def['levels']) == 2 else 'triplets'} per dilution case"


class _CaseLegend:
    """Stands in for _finish's legend axis: the 8 dilution cases in one row."""
    def get_legend_handles_labels(self):
        h = L.case_legend_handles()
        return h, [x.get_label() for x in h]


def _header(fig, text):
    _finish(fig, text, legend_ax=_CaseLegend())


# ------------------------------------------------------------------------- 1. slopegraph
def fig_slopegraph(a, sets, out, tag):
    lv = a["levels"]
    fig, axs = plt.subplots(len(DAY_METRICS), len(EVAL_DAYS), figsize=(13, 12), sharex=True,
                            sharey="row")
    for i, (m, ylab) in enumerate(DAY_METRICS):
        for j, d in enumerate(EVAL_DAYS):
            ax = axs[i, j]
            for dil in CASES:
                st = L.case_style(dil)
                ys = np.array([[day_value(s[x], m, d) for x in lv]
                               for key, s in sets.items() if key[0] == dil])
                for y in ys:
                    ax.plot(lv, y, color=st["color"], ls=st["ls"], lw=0.5, alpha=0.25)
                ax.plot(lv, np.median(ys, 0), color=st["color"], ls=st["ls"], lw=1.9,
                        marker="o", ms=4, mfc=st["mfc"], mec=st["color"])
            ax.set_xscale("log")
            ax.set_yscale("linear" if m == "reff" else "log")
            ax.set_xticks(lv, [a["lab"][x] for x in lv])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            if i == 0:
                _title(ax, f"Day {d}")
            if j == 0:
                ax.set_ylabel(ylab)
            if i == len(DAY_METRICS) - 1:
                ax.set_xlabel(a["name"])
    for i, (m, _) in enumerate(DAY_METRICS):         # shared rows: label once all data are in
        if m != "reff":
            for ax in axs[i]:
                ax.relim()
                ax.autoscale_view()
            L.plain_log_labels(axs[i, 0])
    _header(fig, f"{a['headline']} sensitivity within matched runs\nThin: each matched set "
                 f"({_tag(a, sets)}); bold: per-case median. Solid / filled: across shear; dashed / "
                 f"open: along shear. Mass-weighted-mean V0.{tag}")
    _savefig(fig, out, "slopegraph")


# --------------------------------------------------------------- 2. amplification ratios
def _strip(ax, vals_by_case, logy=True):
    rng = np.random.default_rng(1)
    for dil in CASES:
        v = np.asarray(vals_by_case[dil])
        st = L.case_style(dil)
        x0 = L.case_x(dil)
        ax.scatter(x0 + rng.uniform(-0.17, 0.17, v.size), v, s=13, color=st["color"],
                   facecolors=st["mfc"], linewidths=0.8, zorder=3)
        ax.plot([x0 - 0.3, x0 + 0.3], [np.median(v)] * 2, color=CHARCOAL, lw=1.6, zorder=4)
    ax.set_xticks([L.case_x(d) for d in CASES], CASES, fontsize=8)
    ax.set_xlim(-0.6, L.case_x(CASES[-1]) + 0.6)
    if logy:
        ax.set_yscale("log")
        ax.autoscale_view()
        L.plain_log_labels(ax)


def fig_ratios(a, sets, out, tag):
    lo, hi = a["levels"][0], a["levels"][-1]
    rows = [(m, lab, d) for m, lab in DAY_METRICS for d in EVAL_DAYS]
    fig, axs = plt.subplots(5, 4, figsize=(13, 14.5), sharex=True)
    for p, (m, lab, d) in enumerate(rows):
        ax = axs.ravel()[p]
        vals = {dil: [day_value(s[hi], m, d) / day_value(s[lo], m, d)
                      for key, s in sets.items() if key[0] == dil] for dil in CASES}
        _strip(ax, vals)
        ax.axhline(1.0, color=SLATE, lw=0.9, ls=(0, (4, 2)), zorder=1)
        _title(ax, f"{lab.split(' [')[0]}, day {d}", fontsize=10)
        if p % 4 == 0:
            ax.set_ylabel(f"ratio {a['lab'][hi]} / {a['lab'][lo]}")
    for p, (m, lab) in enumerate(PEAK_METRICS):
        ax = axs[4, p]
        vals = {dil: [peak_value(s[hi], m) / peak_value(s[lo], m)
                      for key, s in sets.items() if key[0] == dil] for dil in CASES}
        _strip(ax, vals)
        ax.axhline(1.0, color=SLATE, lw=0.9, ls=(0, (4, 2)), zorder=1)
        _title(ax, f"{lab} (10 days)", fontsize=10)
        if p == 0:
            ax.set_ylabel(f"ratio {a['lab'][hi]} / {a['lab'][lo]}")
    axs[4, 3].axis("off")
    axs[4, 3].text(0.02, 0.9, f"Each point: one matched set,\nmetric at {a['lab'][hi]} over the "
                   f"same\nmetric at {a['lab'][lo]}. Bar: median.\n{_tag(a, sets).capitalize()}.",
                   va="top", fontsize=9, color=SLATE, transform=axs[4, 3].transAxes)
    _header(fig, f"{a['headline']} amplification by dilution case\nResponse of each metric to "
                 f"{a["lab"][lo]} to {a["lab"][hi]} in {a['name']}, within matched runs. Filled: "
                 f"across shear; open: along shear. Mass-weighted-mean V0.{tag}")
    _savefig(fig, out, "amplification_ratios")


# ----------------------------------------------------------------------- 3. elasticity
def fig_elasticity(a, sets, out, tag):
    lo, hi = a["levels"][0], a["levels"][-1]
    dlog = np.log10(hi / lo)
    cols = [(f"Day {d}", [(m, lab, d) for m, lab in DAY_METRICS]) for d in EVAL_DAYS]
    cols.append(("10-day peak", [(m, lab, None) for m, lab in PEAK_METRICS]))
    fig, axs = plt.subplots(1, len(cols), figsize=(15, 5.4), sharex=True)
    for ax, (title, rows) in zip(axs, cols):
        for yi, (m, lab, d) in enumerate(rows):
            for k, dil in enumerate(CASES):
                st = L.case_style(dil)
                f = (lambda r: day_value(r, m, d)) if d else (lambda r: peak_value(r, m))
                e = np.array([np.log10(f(s[hi]) / f(s[lo])) / dlog
                              for key, s in sets.items() if key[0] == dil])
                y = yi + (k - 3.5) * 0.09
                ax.plot([e.min(), e.max()], [y, y], color=st["color"], ls=st["ls"], lw=1.3)
                ax.plot(np.median(e), y, "o", ms=5, color=st["color"], mfc=st["mfc"])
        ax.axvline(0.0, color=SLATE, lw=0.9, ls=(0, (4, 2)))
        ax.axvline(1.0, color=SLATE, lw=0.9, ls=":")
        ax.set_yticks(range(len(rows)), [lab.split(" [")[0] for _, lab, _ in rows], fontsize=9)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        _title(ax, title, fontsize=11)
        ax.set_xlabel("d log(metric) / d log(knob)")
    _header(fig, f"{a['headline']} elasticity\nd log(metric) / d log({a['name']}) between "
                 f"{a['lab'][lo]} and {a['lab'][hi]}: 0 = no response, 1 = proportional (dotted). "
                 f"Dot: median over the matched sets; line: their range. Mass-weighted-mean V0."
                 f"{tag}")
    _savefig(fig, out, "elasticity")


# -------------------------------------------------------------- 4. size dist by level
def _pm_ylim(Ys):
    top = max(np.nanmax(y) for y in Ys)
    return top / 3e8, top * 3


def fig_sizedist_by_level(a, ens, out, tag, day, per_mass=False):
    lv, ki = a["levels"], _KI[a["key"]]
    allY = []
    fig, axs = plt.subplots(4, 2, figsize=(10, 12), sharex=True, sharey=True)
    for j, cases in enumerate((ACROSS, ALONG)):
        for i, dil in enumerate(cases):
            ax = axs[i, j]
            any_r = next(r for k, r in ens.items() if k[0] == dil)
            ax.loglog(any_r["dp"], np.maximum(L.dist(any_r, 0, per_mass=per_mass), 1e-30),
                      color=CHARCOAL, lw=1.0, ls=(0, (4, 2)), label="background (SABR-220)",
                      zorder=2)
            for x in lv:
                Y = np.array([L.dist(r, L.iday(r, day), per_mass=per_mass)
                              for k, r in ens.items() if k[0] == dil and k[ki] == x])
                allY.append(Y)
                ax.fill_between(any_r["dp"], np.maximum(Y.min(0), 1e-30), Y.max(0),
                                color=a["color"][x], alpha=0.18, lw=0)
                ax.loglog(any_r["dp"], np.maximum(np.median(Y, 0), 1e-30), color=a["color"][x],
                          lw=1.8, label=f"{a['short']} {a['lab'][x]}")
            ax.set_xlim(*XLIM)
            _title(ax, LABEL[dil], fontsize=10.5, loc="left", color=COLOR[dil])
            if j == 0:
                ax.set_ylabel(L.dist_label(False, per_mass))
            if i == 3:
                ax.set_xlabel("dry diameter [µm]")
    for j in range(2):
        axs[0, j].annotate(COL_TITLE[j], (0.5, 1.22), xycoords="axes fraction", ha="center",
                           fontsize=12, fontfamily=HEADLINE["family"], color=CHARCOAL)
    axs[0, 0].set_ylim(*(_pm_ylim(allY) if per_mass else (1e-2, 3e6)))
    n = len(ens) // 8 // len(lv)
    pm = " per unit H2SO4 in particles" if per_mass else ""
    note = (" Per mass: each distribution divided by the particulate H2SO4 in the same box; the "
            "background by its own sulfate." if per_mass else "")
    _finish(fig, f"Day-{day} size distribution by {a['short']} level{pm}\nMedian (line) and range "
                 f"(shading) over the {n} runs per case and level (the other knobs vary). Dashed: "
                 f"the SABR-220 background. 06:00 local, mass-weighted-mean V0.{note}{tag}", axs[0, 0])
    _savefig(fig, out, f"sizedist_by_level_day{day}" + ("_per_mass" if per_mass else ""))


# -------------------------------------------------------------- 5. time-resolved ratio
def fig_time_ratio(a, sets, out, tag):
    lo, hi = a["levels"][0], a["levels"][-1]
    fig, axs = plt.subplots(len(DAY_METRICS), 2, figsize=(11, 12), sharex=True, sharey="row")
    yl = {}
    for m, _ in DAY_METRICS:                  # y range from t >= 6 h (first-hour spikes clipped)
        v = np.concatenate([s[hi][m][L.tdays(s[hi]) >= 0.25] / s[lo][m][L.tdays(s[hi]) >= 0.25]
                            for s in sets.values()])
        yl[m] = (min(v.min(), 1.0) / 1.15, max(v.max(), 1.0) * 1.15)
    for j, cases in enumerate((ACROSS, ALONG)):
        for i, (m, lab) in enumerate(DAY_METRICS):
            ax = axs[i, j]
            for dil in cases:
                ss = [s for key, s in sets.items() if key[0] == dil]
                t = L.tdays(ss[0][lo])
                R = np.array([s[hi][m] / s[lo][m] for s in ss])
                ax.fill_between(t, R.min(0), R.max(0), color=COLOR[dil], alpha=0.14, lw=0)
                ax.plot(t, np.median(R, 0), color=COLOR[dil], lw=1.7, label=LABEL[dil])
            ax.axhline(1.0, color=SLATE, lw=0.9, ls=(0, (4, 2)))
            for d in EVAL_DAYS:
                ax.axvline(d, color=SAGE, lw=0.8, ls=":", zorder=0)
            ax.set_yscale("log")
            ax.set_ylim(*yl[m])
            L.plain_log_labels(ax)
            if j == 0:
                ax.set_ylabel(f"{lab.split(' [')[0]}\n{a['lab'][hi]} / {a['lab'][lo]}")
            if i == 0:
                _title(ax, COL_TITLE[j])
            if i == len(DAY_METRICS) - 1:
                ax.set_xlabel("day since release (06:00 local)")
                ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncols=2)
    axs[0, 0].set_xlim(0, 10)
    _finish(fig, f"When {a['short']} matters\nRatio {a['lab'][hi]} / {a['lab'][lo]} within matched "
                 f"runs over time (y range set from 6 h on): median (line) and range (shading) over the {_tag(a, sets)}. "
                 f"Dotted: evaluation days. Mass-weighted-mean V0.{tag}")
    _savefig(fig, out, "time_resolved_ratio")


# ------------------------------------------------------------------- 6. case studies
def _case_run(ens, a, dil, x):
    key = [dil, 1.0, 1.0, 1.0]
    key[_KI[a["key"]]] = x
    return ens[tuple(key)]


def _field(r, sa):
    return r["nd"] * (np.pi * r["dp"] ** 2 if sa else 1.0)


_NORM = {False: (1e2, 1e8), True: (1e-2, 2e3)}
_FLAB = {False: "dN/dlogD$_p$ [cm$^{-3}$]", True: "dSA/dlogD$_p$ [µm$^2$ cm$^{-3}$]"}


def _case_axes_labels(axs, a, rows, i_last):
    for j, dil in enumerate(CASE_STUDY):
        side = "across shear" if dil in ACROSS else "along shear"
        _title(axs[0, j], f"{LABEL[dil]}  ({side})", fontsize=11)
    for i, x in enumerate(rows):
        axs[i, 0].set_ylabel(f"{a['short']} {a['lab'][x]}\ndry diameter [µm]")
    for j in range(2):
        axs[i_last, j].set_xlabel("day since release (06:00 local)")


def fig_banana(a, ens, out, tag, sa=False, per_mass=False):
    lv = a["levels"]
    fig, axs = plt.subplots(len(lv), 2, figsize=(11, 3.3 * len(lv) + 1.6), sharex=True,
                            sharey=True, squeeze=False)
    pc = None
    for i, x in enumerate(lv):
        for j, dil in enumerate(CASE_STUDY):
            r = _case_run(ens, a, dil, x)
            ax = axs[i, j]
            ax.grid(False)
            pc = ax.pcolormesh(L.tdays(r), r["dp"],
                               np.maximum(L.dist(r, sa=sa, per_mass=per_mass), 1e-9).T,
                               norm=LogNorm(*L.BANANA_NORM[sa, per_mass]), cmap=CMAP,
                               rasterized=True, shading="nearest")
            ax.set_yscale("log")
            ax.set_ylim(r["dp"][0], 2.0)
    _case_axes_labels(axs, a, lv, len(lv) - 1)
    pm = " per unit H2SO4 in particles" if per_mass else ""
    _finish(fig, f"{a['headline']}: plume {'surface-area' if sa else 'size'}-distribution "
                 f"evolution{pm}\nC1 and C2 (Kz 0.01), mass-weighted-mean V0, {a['fixed']}."
                 f"{L.PER_MASS_NOTE if per_mass else ''}{tag}", right=0.9)
    cax = fig.add_axes((0.92, 0.1, 0.02, 0.7))
    fig.colorbar(pc, cax=cax, label=L.dist_label(sa, per_mass),
                 extend="max" if per_mass else "neither")
    _savefig(fig, out, f"banana_C1_C2{'_SA' if sa else ''}{'_per_mass' if per_mass else ''}")


def fig_banana_ratio(a, ens, out, tag, sa=False):
    lv = a["levels"]
    floor = 1e-2 if sa else 1e1
    fig, axs = plt.subplots(len(lv), 2, figsize=(11, 3.3 * len(lv) + 1.6), sharex=True,
                            sharey=True, squeeze=False)
    pc_abs = pc_rat = None
    for i, x in enumerate(lv):
        for j, dil in enumerate(CASE_STUDY):
            ref = _field(_case_run(ens, a, dil, 1.0), sa)
            r = _case_run(ens, a, dil, x)
            ax = axs[i, j]
            ax.grid(False)
            if x == 1.0:
                pc_abs = ax.pcolormesh(L.tdays(r), r["dp"], np.maximum(ref, 1e-9).T,
                                       norm=LogNorm(*_NORM[sa]), cmap=CMAP, rasterized=True,
                                       shading="nearest")
                ax.text(0.02, 0.95, "reference (absolute)", transform=ax.transAxes, va="top",
                        fontsize=8.5, color="white")
            else:
                fld = _field(r, sa)
                ratio = np.clip(fld / np.maximum(ref, 1e-30), 1e-2, 1e2)
                ratio[(fld < floor) & (ref < floor)] = np.nan
                pc_rat = ax.pcolormesh(L.tdays(r), r["dp"], ratio.T, norm=LogNorm(1e-2, 1e2),
                                       cmap=RATIO_CMAP, rasterized=True, shading="nearest")
            ax.set_yscale("log")
            ax.set_ylim(r["dp"][0], 2.0)
    _case_axes_labels(axs, a, lv, len(lv) - 1)
    _finish(fig, f"{a['headline']}: change relative to the default run\nEach run's "
                 f"{'surface-area' if sa else 'number'} distribution over the default ({a['lab'][1.0]})"
                 f" run of the same case; grey: both below {floor:g}. C1 and C2, mass-weighted-mean "
                 f"V0, {a['fixed']}.{tag}", right=0.88)
    fig.colorbar(pc_abs, cax=fig.add_axes((0.9, 0.47, 0.018, 0.3)),
                 label="reference " + _FLAB[sa])
    fig.colorbar(pc_rat, cax=fig.add_axes((0.9, 0.08, 0.018, 0.3)), label="ratio to reference")
    _savefig(fig, out, f"banana_ratio_C1_C2{'_SA' if sa else ''}")


def fig_snapshots(a, ens, out, tag, linear=False, per_mass=False):
    lv = a["levels"]
    fig, axs = plt.subplots(2, len(EVAL_DAYS), figsize=(14, 7.6), sharex=True,
                            sharey=False if linear else "row")
    allY = ([], [])
    for k, d in enumerate(EVAL_DAYS):
        for x in lv:
            for dil in CASE_STUDY:
                r = _case_run(ens, a, dil, x)
                st = L.case_style(dil)
                i = L.iday(r, d)
                lab = f"{a['lab'][x]}, {dil}"
                for row, sa in ((0, False), (1, True)):
                    y = L.dist(r, i, sa=sa, per_mass=per_mass)
                    allY[row].append(y)
                    axs[row, k].plot(r["dp"], y if linear else np.maximum(y, 1e-30), ls=st["ls"],
                                     color=a["color"][x], lw=1.7, label=lab if row == 0 else None)
        r0 = _case_run(ens, a, "C1", 1.0)
        for row, sa in ((0, False), (1, True)):
            ax = axs[row, k]
            y = L.dist(r0, 0, sa=sa, per_mass=per_mass)
            ax.plot(r0["dp"], y, color=SLATE, lw=1.0, ls=":",
                    label="background (SABR-220)" if (row, k) == (0, 0) else None)
            ax.set_xscale("log")
            ax.set_xlim(*XLIM)
            if linear:
                ax.set_ylim(bottom=0)
            else:
                ax.set_yscale("log")
                ax.set_ylim(*(_pm_ylim(allY[row]) if per_mass
                              else (1e-2, 3e6) if row == 0 else (1e-4, 3e3)))
        _title(axs[0, k], f"Day {d}")
        axs[1, k].set_xlabel("dry diameter [µm]")
    axs[0, 0].set_ylabel(L.dist_label(False, per_mass))
    axs[1, 0].set_ylabel(L.dist_label(True, per_mass).replace("per mass ", "per mass\n"))
    pm = " per unit H2SO4 in particles" if per_mass else ""
    note = (" Per mass: divided by the particulate H2SO4 in the same box; the background by its "
            "own sulfate." if per_mass else "")
    _finish(fig, f"{a['headline']}: number and surface-area distributions{pm}\nSolid: C1 (across "
                 f"shear); dashed: C2 (along shear); dotted: the SABR-220 background. 06:00 local, "
                 f"mass-weighted-mean V0, {a['fixed']}"
                 f"{'; linear y, per-panel scale' if linear else ''}.{note}{tag}", axs[0, 0])
    _savefig(fig, out, f"sizedist_snapshots_C1_C2{'_linear' if linear else ''}"
             f"{'_per_mass' if per_mass else ''}")


def fig_totals(a, ens, out, tag):
    lv = a["levels"]
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8), sharex=True)
    for (m, lab), ax in zip(DAY_METRICS, axs.ravel()):
        for x in lv:
            for dil in CASE_STUDY:
                r = _case_run(ens, a, dil, x)
                st = L.case_style(dil)
                ax.plot(L.tdays(r), r[m], ls=st["ls"], color=a["color"][x], lw=1.6,
                        label=f"{a['lab'][x]}, {dil}")
        r0 = _case_run(ens, a, "C1", 1.0)
        ax.axhline(r0[m][0], color=SLATE, lw=1.0, ls=":", label="background (SABR-220)")
        ax.set_yscale("linear" if m == "reff" else "log")
        ax.set_ylabel(lab)
        for d in EVAL_DAYS:
            ax.axvline(d, color=SAGE, lw=0.8, ls=":", zorder=0)
    for ax in axs[1]:
        ax.set_xlabel("day since release (06:00 local)")
    axs[0, 0].set_xlim(0, 10)
    _finish(fig, f"{a['headline']}: plume totals\nSolid: C1 (across shear); dashed: C2 (along "
                 f"shear); dotted: the SABR-220 background. Mass-weighted-mean V0, {a['fixed']}."
                 f"{tag}", axs[0, 0])
    _savefig(fig, out, "totals_C1_C2")


def run_axis(name, data, tree, tag):
    a = AXES[name]
    ens = L.ensemble(data)
    sets = matched_sets(ens, a["key"])
    out = os.path.join(tree, "plots", name)
    print(f"[{name}] {len(sets)} matched sets -> {out}")
    fig_slopegraph(a, sets, out, tag)
    fig_ratios(a, sets, out, tag)
    fig_elasticity(a, sets, out, tag)
    for d in EVAL_DAYS:
        fig_sizedist_by_level(a, ens, out, tag, d)
        fig_sizedist_by_level(a, ens, out, tag, d, per_mass=True)
    fig_time_ratio(a, sets, out, tag)
    for sa in (False, True):
        fig_banana(a, ens, out, tag, sa)
        fig_banana(a, ens, out, tag, sa, per_mass=True)
        fig_banana_ratio(a, ens, out, tag, sa)
    fig_snapshots(a, ens, out, tag)
    fig_snapshots(a, ens, out, tag, linear=True)
    fig_snapshots(a, ens, out, tag, per_mass=True)
    fig_totals(a, ens, out, tag)


def main(argv):
    tree = L.TREE
    if "--tree" in argv:
        i = argv.index("--tree")
        tree = os.path.abspath(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    L._style()
    data = L.load(tree)
    tag = L.numerics_tag(tree)
    for name in argv or list(AXES):
        run_axis(name, data, tree, tag)


if __name__ == "__main__":
    main(sys.argv[1:])
