# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Dilution-sensitivity, cross-ensemble and oxidation figures of the v2 ensemble, in the Reflective
style (the v1 set of legacy_v1/make_paper_candidate_plots.py, make_boxsize_plots.py and
make_oxidation_plots.py).

Ensemble statistics use the 144 mass-weighted-mean (L30) members: per dilution case 18 microphysics
combinations (nucleation x coagulation x alpha). V0 figures use the 40 base runs (default knobs).
Box quantities; evaluation days 1, 2, 7, 10 (L26).

<tree>/plots/dilution/ (PNG + PDF):
  tornado_sensitivity, timeseries_by_case, sizedist_grid_day{1,2,7,10}(_per_mass), so2_conversion,
  npf_outcome_map, banana_V0_C1_C2(_SA)(_per_mass)
<tree>/plots/oxidation/:
  OH_by_case, HO2_by_case, H2O2_by_case, O3_by_case, sulfur_budget_C1_C2

    python -m coupled.paper_ensemble.make_v2_dilution_plots [--tree DIR]
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from coupled.paper_ensemble import v2_paper_lib as L  # noqa: E402
from coupled.paper_ensemble.make_v2_base_plots import HEADLINE  # noqa: E402
from coupled.paper_ensemble.v2_paper_lib import (  # noqa: E402
    ACROSS, ALONG, BLUE, CASES, CHARCOAL, CLAY, CMAP, COL_TITLE, COLOR, DAY_SHADE, EVAL_DAYS,
    GOLD, GREEN, LABEL, SAGE, SLATE, XLIM, _finish, _savefig, _title)

KZ_FAMILY = {"C4": "Kz 0.001", "C4a": "Kz 0.001", "C1": "Kz 0.01", "C2": "Kz 0.01",
             "C5": "Kz 0.1", "C5a": "Kz 0.1", "C3": "burst", "C3a": "burst"}
V0_LAB = {"peak": "peak", "c1sig": "1σ mean", "mwmean": "mass-weighted mean",
          "tophat": "top hat", "c2sig": "2σ mean"}


def _default(ens, dil):
    return ens[(dil, 1.0, 1.0, 1.0)]


def _members(ens, dil):
    return [r for k, r in ens.items() if k[0] == dil]


def _col_titles(axs):
    for j in range(2):
        axs[0, j].annotate(COL_TITLE[j], (0.5, 1.22), xycoords="axes fraction", ha="center",
                           fontsize=12, fontfamily=HEADLINE["family"], color=CHARCOAL)


# ---------------------------------------------------------------------------- tornado
TORNADO_AXES = [("dilution case (8)", lambda k: k[0]),
                ("shear orientation (across / along)", lambda k: k[0] in ACROSS),
                ("vertical mixing (Kz family)", lambda k: KZ_FAMILY[k[0]]),
                ("nucleation ×0.01 / ×1 / ×100", lambda k: k[2]),
                ("coagulation ×0.5 / ×1 / ×2", lambda k: k[3]),
                ("condensation α 0.5 / 1", lambda k: k[1])]
TORNADO_METRICS = [("N3", "N > 3 nm"), ("N10", "N > 10 nm"), ("SA", "surface area"),
                   ("reff", "r$_{eff}$ (wet)"), ("conv", "SO$_2$ converted")]


def fig_tornado(ens, out, tag):
    keys = list(ens)
    fig, axs = plt.subplots(2, 3, figsize=(14, 8.6))
    for (m, mlab), ax in zip(TORNADO_METRICS, axs.ravel()):
        for d_i, d in enumerate(EVAL_DAYS):
            v = np.log10(np.maximum([ens[k][m][L.iday(ens[k], d)] for k in keys], 1e-30))
            for a_i, (alab, f) in enumerate(TORNADO_AXES):
                g = np.array([f(k) for k in keys], dtype=object)
                med = [np.median(v[g == lev]) for lev in set(g.tolist())]
                span = max(med) - min(med)
                y = a_i + (d_i - 1.5) * 0.2
                ax.barh(y, span, height=0.18, color=DAY_SHADE[d],
                        label=f"day {d}" if a_i == 0 else None)
                if d == 10:
                    ax.text(span, y, f"  ×{10 ** span:.2g}", va="center", fontsize=7.5,
                            color=SLATE)
        ax.set_yticks(range(len(TORNADO_AXES)), [a for a, _ in TORNADO_AXES], fontsize=8.5)
        ax.set_ylim(len(TORNADO_AXES) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(right=ax.get_xlim()[1] * 1.18)
        _title(ax, mlab.replace("$_{eff}$", " eff").replace("$_2$", "2"), fontsize=11)
        ax.set_xlabel("main-effect range [decades]")
    ax = axs[1, 2]
    ax.axis("off")
    ax.text(0.0, 0.95, "Main effect of each knob: the spread of\nthe per-level medians of "
            "log10(metric),\nall other knobs pooled (144 runs).\n×N: fold change between the "
            "extreme\nlevels at day 10.\n\nThe dilution rows overlap: case = shear\n× mixing "
            "family.", va="top", fontsize=9.5, color=SLATE, transform=ax.transAxes)
    _finish(fig, "Which knob moves each metric\nMass-weighted-mean V0 ensemble: 8 dilution cases × "
                 f"18 microphysics combinations. Bars: evaluation days 1, 2, 7, 10.{tag}", axs[0, 0])
    _savefig(fig, out, "tornado_sensitivity")


# ---------------------------------------------------------------- time series by case
def fig_timeseries(ens, out, tag):
    rows = [("N3", "N > 3 nm [cm$^{-3}$]", "log"), ("N10", "N > 10 nm [cm$^{-3}$]", "log"),
            ("SA", "surface area [µm$^2$ cm$^{-3}$]", "log"),
            ("reff", "r$_{eff}$ (wet) [µm]", "linear"),
            ("conv", "SO$_2$ converted\n(fraction of injected)", "linear")]
    fig, axs = plt.subplots(len(rows), 2, figsize=(11, 13), sharex=True, sharey="row")
    for j, cases in enumerate((ACROSS, ALONG)):
        for i, (m, ylab, scale) in enumerate(rows):
            ax = axs[i, j]
            for dil in cases:
                Y = np.array([r[m] for r in _members(ens, dil)])
                r0 = _default(ens, dil)
                ax.fill_between(L.tdays(r0), Y.min(0), Y.max(0), color=COLOR[dil], alpha=0.15,
                                lw=0)
                ax.plot(L.tdays(r0), r0[m], color=COLOR[dil], lw=1.7, label=LABEL[dil])
            ax.set_yscale(scale)
            for d in EVAL_DAYS:
                ax.axvline(d, color=SAGE, lw=0.8, ls=":", zorder=0)
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 0:
                _title(ax, COL_TITLE[j])
            if i == len(rows) - 1:
                ax.set_xlabel("day since release (06:00 local)")
                ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncols=2)
    axs[0, 0].set_xlim(0, 10)
    _finish(fig, "Plume evolution by dilution case, with the microphysics range\nBold: default "
                 "microphysics; shading: range over the 18 nucleation × coagulation × α runs of "
                 f"the case. Mass-weighted-mean V0; dotted: evaluation days.{tag}")
    _savefig(fig, out, "timeseries_by_case")


# --------------------------------------------------------------- size dist grid by day
def fig_sizedist_grid(ens, out, tag, day, per_mass=False):
    fig, axs = plt.subplots(4, 2, figsize=(10, 12), sharex=True, sharey=True)
    tops = []
    for j, cases in enumerate((ACROSS, ALONG)):
        for i, dil in enumerate(cases):
            ax = axs[i, j]
            r0 = _default(ens, dil)
            ax.loglog(r0["dp"], np.maximum(L.dist(r0, 0, per_mass=per_mass), 1e-30),
                      color=CHARCOAL, lw=1.0, ls=(0, (4, 2)), label="background (SABR-220)",
                      zorder=2)
            for r in _members(ens, dil):
                y = L.dist(r, L.iday(r, day), per_mass=per_mass)
                tops.append(np.nanmax(y))
                ax.loglog(r["dp"], np.maximum(y, 1e-30), color=SLATE, lw=0.6, alpha=0.35,
                          rasterized=True)
            ax.loglog(r0["dp"], np.maximum(L.dist(r0, L.iday(r0, day), per_mass=per_mass),
                                           1e-30), color=COLOR[dil], lw=2.0,
                      label="default microphysics", zorder=3)
            ax.plot([], [], color=SLATE, lw=0.8, label="the 18 microphysics runs")
            ax.set_xlim(*XLIM)
            _title(ax, LABEL[dil], fontsize=10.5, loc="left", color=COLOR[dil])
            if j == 0:
                ax.set_ylabel(L.dist_label(False, per_mass))
            if i == 3:
                ax.set_xlabel("dry diameter [µm]")
    axs[0, 0].set_ylim(*((max(tops) / 3e8, max(tops) * 3) if per_mass else (1e-2, 3e6)))
    _col_titles(axs)
    pm = " per unit H2SO4 in particles" if per_mass else ""
    note = (" Per mass: divided by the particulate H2SO4 in the same box; the background by its "
            "own sulfate." if per_mass else "")
    _finish(fig, f"Day-{day} size distribution by dilution case{pm}\nColoured: default "
                 "microphysics; grey: the 18 nucleation × coagulation × α runs; dashed: the "
                 f"SABR-220 background. 06:00 local, mass-weighted-mean V0.{note}{tag}", axs[0, 0])
    _savefig(fig, out, f"sizedist_grid_day{day}" + ("_per_mass" if per_mass else ""))


# --------------------------------------------------------------------- SO2 conversion
def fig_so2_conversion(ens, out, tag):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.9), width_ratios=[1, 1, 1.15])
    for ax, cases, title in ((axs[0], ACROSS, COL_TITLE[0]), (axs[1], ALONG, COL_TITLE[1])):
        for dil in cases:
            Y = np.array([r["conv"] for r in _members(ens, dil)])
            r0 = _default(ens, dil)
            ax.fill_between(L.tdays(r0), Y.min(0), Y.max(0), color=COLOR[dil], alpha=0.15, lw=0)
            ax.plot(L.tdays(r0), r0["conv"], color=COLOR[dil], lw=1.8, label=LABEL[dil])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, None)
        ax.set_xlabel("day since release (06:00 local)")
        _title(ax, title)
        ax.legend(fontsize=8, loc="upper left")
    axs[0].set_ylabel("SO$_2$ converted (fraction of injected)")
    ax = axs[2]
    rng = np.random.default_rng(2)
    for d_i, d in enumerate(EVAL_DAYS):
        for dil in CASES:
            v = np.array([r["conv"][L.iday(r, d)] for r in _members(ens, dil)])
            x = L.case_x(dil) + (d_i - 1.5) * 0.18
            ax.scatter(x + rng.uniform(-0.05, 0.05, v.size), v, s=9, color=DAY_SHADE[d],
                       label=f"day {d}" if dil == CASES[0] else None, zorder=3)
    ax.set_xticks([L.case_x(d) for d in CASES], CASES, fontsize=8.5)
    ax.set_ylabel("SO$_2$ converted (fraction of injected)")
    ax.set_ylim(0, None)
    _title(ax, "At the evaluation days, all 18 runs per case")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=4)
    _finish(fig, "SO2 conversion by dilution case\nCumulative chemical SO2 loss (L27) over the "
                 "injected SO2. Lines: default microphysics; shading: range over the 18 "
                 f"microphysics runs. Mass-weighted-mean V0.{tag}")
    _savefig(fig, out, "so2_conversion")


# --------------------------------------------------------------------- NPF outcome map
def fig_npf_map(ens, out, tag):
    rows = [("peak N > 3 nm [cm$^{-3}$]", lambda r: np.max(r["N3"])),
            ("day-10 N > 3 nm [cm$^{-3}$]", lambda r: r["N3"][L.iday(r, 10)]),
            ("day-10 N > 10 nm [cm$^{-3}$]", lambda r: r["N10"][L.iday(r, 10)])]
    fig, axs = plt.subplots(len(rows), 2, figsize=(11, 11), sharex=True, sharey="row")
    rng = np.random.default_rng(0)
    for j, cases in enumerate((ACROSS, ALONG)):
        for i, (ylab, f) in enumerate(rows):
            ax = axs[i, j]
            for dil in cases:
                st = L.case_style(dil)
                ks = [k for k in ens if k[0] == dil]
                x = np.array([k[2] for k in ks]) * 10 ** rng.uniform(-0.15, 0.15, len(ks))
                ax.scatter(x, [f(ens[k]) for k in ks], s=16, color=st["color"],
                           facecolors=st["mfc"], linewidths=0.8, label=LABEL[dil])
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks([0.01, 1, 100], ["×0.01", "×1", "×100"])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            if i == 0:
                _title(ax, COL_TITLE[j])
            if j == 0:
                ax.set_ylabel(ylab)
            if i == len(rows) - 1:
                ax.set_xlabel("nucleation rate scale")
                ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2)
    for ax in axs[:, 0]:
        L.plain_log_labels(ax)
    _finish(fig, "New-particle-formation outcome vs nucleation rate\nEach point: one run (6 per "
                 "case and nucleation level: coagulation × α). Mass-weighted-mean V0; filled: "
                 f"across shear; open: along shear.{tag}")
    _savefig(fig, out, "npf_outcome_map")


# ---------------------------------------------------------------------- V0 bananas
def fig_banana_v0(data, out, tag, sa=False, per_mass=False):
    fig, axs = plt.subplots(len(L.V0S), 2, figsize=(11, 15), sharex=True, sharey=True)
    pc = None
    for i, v0 in enumerate(L.V0S):
        for j, dil in enumerate(("C1", "C2")):
            r = data[(dil, 1.0, 1.0, 1.0, v0)]
            ax = axs[i, j]
            ax.grid(False)
            pc = ax.pcolormesh(L.tdays(r), r["dp"],
                               np.maximum(L.dist(r, sa=sa, per_mass=per_mass), 1e-9).T,
                               norm=LogNorm(*L.BANANA_NORM[sa, per_mass]), cmap=CMAP,
                               rasterized=True, shading="nearest")
            ax.set_yscale("log")
            ax.set_ylim(r["dp"][0], 2.0)
            if j == 0:
                ax.set_ylabel(f"V0: {V0_LAB[v0]}\ndry diameter [µm]")
        for j, dil in enumerate(("C1", "C2")):
            side = "across shear" if dil in ACROSS else "along shear"
            _title(axs[0, j], f"{LABEL[dil]}  ({side})", fontsize=11)
            axs[-1, j].set_xlabel("day since release (06:00 local)")
    what = "surface-area" if sa else "size"
    pm = " per unit H2SO4 in particles" if per_mass else ""
    _finish(fig, f"Initial plume volume (V0): {what}-distribution evolution{pm}\nThe same 1 t SO2 "
                 "in the five V0 variants (L21; smallest first), default microphysics. C1 and C2 "
                 f"(Kz 0.01).{L.PER_MASS_NOTE if per_mass else ''}{tag}", right=0.9)
    fig.colorbar(pc, cax=fig.add_axes((0.92, 0.1, 0.02, 0.7)), label=L.dist_label(sa, per_mass),
                 extend="max" if per_mass else "neither")
    _savefig(fig, out, f"banana_V0_C1_C2{'_SA' if sa else ''}{'_per_mass' if per_mass else ''}")


# ------------------------------------------------------------------------- oxidation
OXIDANTS = [("OH", "OH"), ("HO2", "HO$_2$"), ("H2O2", "H$_2$O$_2$"), ("O3", "O$_3$")]


def fig_oxidant(ens, out, tag, sp, lab):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, cases, title in zip(axs, (ACROSS, ALONG), COL_TITLE):
        for dil in cases:
            Y = np.array([r[sp] for r in _members(ens, dil)])
            r0 = _default(ens, dil)
            ax.fill_between(L.tdays(r0), Y.min(0), Y.max(0), color=COLOR[dil], alpha=0.15, lw=0)
            ax.plot(L.tdays(r0), r0[sp], color=COLOR[dil], lw=1.5, label=LABEL[dil])
        ax.set_xlim(0, 10)
        ax.set_ylim(bottom=0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax.set_xlabel("day since release (06:00 local)")
        _title(ax, title)
        ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=4)
    axs[0].set_ylabel(f"{lab} [molec cm$^{{-3}}$]")
    name = lab.replace("$_2$", "2").replace("$_3$", "3")
    _finish(fig, f"{name} in the plume by dilution case\nLines: default microphysics; shading: "
                 f"range over the 18 microphysics runs. 30N, 20 km, 210 K, 55 hPa, SABR-220 "
                 f"background, mass-weighted-mean V0.{tag}")
    _savefig(fig, out, f"{sp}_by_case")


S_POOLS = [("SO2", "SO$_2$", GOLD, "-"), ("SO3", "SO$_3$", GREEN, "-"),
           ("H2SO4", "H$_2$SO$_4$ (gas)", BLUE, (0, (4, 2))),
           ("pS", "particulate sulfate", CLAY, "-")]


def fig_sulfur_budget(ens, out, tag):
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.6), sharex=True)
    for i, dil in enumerate(("C1", "C2")):
        r = _default(ens, dil)
        t = L.tdays(r)
        pools = [r[k] * L.S_UG for k, _, _, _ in S_POOLS]
        total = np.sum(pools, axis=0)
        a1, a2 = axs[i]
        a1.plot(t, total, color=CHARCOAL, lw=2.2, label="total in-box S")
        for (_, lab, col, ls), p in zip(S_POOLS, pools):
            a1.plot(t, np.maximum(p, 1e-30), color=col, ls=ls, lw=1.8, label=lab)
        a1.set_yscale("log")
        a1.set_ylim(1e-11, 2 * total.max())
        a1.set_ylabel(f"{dil}: sulfur [µg S cm$^{{-3}}$]")
        a2.stackplot(t, *[p / total for p in pools], colors=[c for _, _, c, _ in S_POOLS],
                     alpha=0.85, labels=[lab for _, lab, _, _ in S_POOLS])
        a2.set_ylim(0, 1)
        a2.set_ylabel(f"{dil}: fraction of in-box S")
        a2.grid(True, color="white", alpha=0.4)
        a2.set_axisbelow(False)
        side = "across shear" if dil in ACROSS else "along shear"
        _title(a1, f"{LABEL[dil]} ({side}): in-box sulfur by reservoir", fontsize=10.5)
        _title(a2, f"{dil}: sulfur partitioning", fontsize=10.5)
    for a in axs[1]:
        a.set_xlabel("day since release (06:00 local)")
    axs[0, 0].set_xlim(0, 10)
    _finish(fig, "Sulfur budget in the plume box\nDefault microphysics, mass-weighted-mean V0. "
                 "Box values: dilution lowers every pool and entrains background SO2 and "
                 f"aerosol.{tag}", axs[0, 0])
    _savefig(fig, out, "sulfur_budget_C1_C2")


def main(argv):
    tree = L.TREE
    if "--tree" in argv:
        tree = os.path.abspath(argv[argv.index("--tree") + 1])
    L._style()
    data = L.load(tree)
    ens = L.ensemble(data)
    tag = L.numerics_tag(tree)
    out = os.path.join(tree, "plots", "dilution")
    print(f"[dilution] {len(ens)} mwmean runs -> {out}")
    fig_tornado(ens, out, tag)
    fig_timeseries(ens, out, tag)
    for d in EVAL_DAYS:
        fig_sizedist_grid(ens, out, tag, d)
        fig_sizedist_grid(ens, out, tag, d, per_mass=True)
    fig_so2_conversion(ens, out, tag)
    fig_npf_map(ens, out, tag)
    for sa in (False, True):
        for pm in (False, True):
            fig_banana_v0(data, out, tag, sa, pm)
    out = os.path.join(tree, "plots", "oxidation")
    print(f"[oxidation] -> {out}")
    for sp, lab in OXIDANTS:
        fig_oxidant(ens, out, tag, sp, lab)
    fig_sulfur_budget(ens, out, tag)


if __name__ == "__main__":
    main(sys.argv[1:])
