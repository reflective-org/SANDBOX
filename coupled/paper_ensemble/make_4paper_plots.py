# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Paper figures in the layout Marianna set (-> ``4paper/``), rebuilt from the v2 run tree.

Conventions, taken from her edited ``dilution_timeseries_v2.pdf`` (2026-11-06):
  * colour = Kz family, in this order everywhere (legend and drawing):
    Low Kz (C4/C4a), Med Kz (C1/C2), High Kz (C5/C5a), Burst (C3/C3a, drawn from 48 h: before that it is Low Kz);
    the colours are her Illustrator CMYK swatches as they render on screen (sRGB).
  * solid = across shear (sigma_L = 25 m), dashed = along shear (15 km/sqrt12); labelled "across /
    along shear", not "fast / slow" (Ali, 2026-11-13).
  * single-column size (3.46 x 2.37 in), no title; Open Sans (Reflective body font).

    python -m coupled.paper_ensemble.make_4paper_plots [run_tree]
"""
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

from coupled.dilution import GAUSSIAN_CASES  # noqa: E402
from coupled.paper_ensemble.make_dilution_v2_plot import CHARCOAL, brand_style  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.join(_HERE, "runs_v2.1")
OUT = os.path.join(_HERE, "4paper")
DAY = 86400.0

#: (legend label, fast case, slow case, colour) in Marianna's order
FAMILIES = (("Low K$_\\mathregular{z}$", "C4", "C4a", "#5fa9d1"),
            ("Med K$_\\mathregular{z}$", "C1", "C2", "#2a357e"),
            ("High K$_\\mathregular{z}$", "C5", "C5a", "#17837f"),
            ("Burst", "C3", "C3a", "#9a498c"))
BURST = ("C3", "C3a")
LABEL_OF = {c: lab for lab, f, s, _ in FAMILIES for c in (f, s)}
FAST, SLOW = "-", (0, (5.0, 2.5))
REF_GREY = "#8B8D8E"                       # the fast/slow key lines
SPINE, GRIDC = "#B7BDBA", "#E3E6E4"
SIZE = (3.46, 2.37)                        # in, her page (248.9 x 170.7 pt)
OH_SIZE = (4.27, 1.64)                     # in, the v1 f08 page (307.6 x 118.2 pt)


def _paper_style():
    brand_style()
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": SPINE, "axes.linewidth": 0.8,
                         "xtick.color": SPINE, "ytick.color": SPINE,
                         "xtick.labelcolor": CHARCOAL, "ytick.labelcolor": CHARCOAL,
                         "xtick.major.size": 3, "ytick.major.size": 3, "ytick.minor.size": 1.6,
                         "axes.labelpad": 2, "savefig.dpi": 600})


def _save(fig, name, out=OUT):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  {os.path.join(out, name)}.pdf/.png")


def _base_run(tree, case):
    """t [s] and V/V0 of a default-microphysics mwmean base run, at every saved time."""
    f, = glob.glob(os.path.join(tree, "base", f"*__{case}__a1p0__nuc1__cg1__V0mwmean*",
                                "state.npz"))
    with np.load(f, allow_pickle=False) as z:
        return z["t"], z["V_ratio_applied"]


def _from_burst(case, t, *ys):
    """C3/C3a equal C4/C4a until the burst starts; drop that part so Low Kz shows (as she drew)."""
    if case not in BURST:
        return (t, *ys)
    keep = t >= GAUSSIAN_CASES[case].t_burst_start - 1e-6
    return (t[keep], *(y[keep] for y in ys))


def _base_species(tree, case, species):
    """t [s] and the box concentration [molec cm^-3] of ``species`` in the same base run."""
    f, = glob.glob(os.path.join(tree, "base", f"*__{case}__a1p0__nuc1__cg1__V0mwmean*",
                                "state.npz"))
    with np.load(f, allow_pickle=False) as z:
        sp = [str(s) for s in z["species"]]
        return z["t"], z["x"][:, sp.index(species)]


def fig_dilution_timeseries(tree=TREE, out=OUT):
    """V(t)/V0 of the eight cases over the 10-day runs (the applied dilution, from the runs)."""
    fig, ax = plt.subplots(figsize=SIZE)
    for d in range(11):
        ax.axvline(d, color=GRIDC, lw=0.6, ls=(0, (3, 2)), zorder=0)
    for _, fast, slow, col in FAMILIES:
        for case, ls in ((fast, FAST), (slow, SLOW)):
            t, v = _from_burst(case, *_base_run(tree, case))
            ax.semilogy(t / DAY, v, color=col, ls=ls, lw=1.5, solid_capstyle="butt")
    ax.set_xlim(0, 10)
    ax.set_xticks(range(11))
    ax.set_ylim(0.6, 3e5)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=12))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, axis="y", which="major", color=GRIDC, lw=0.6, zorder=0)
    ax.grid(False, axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Time since release (days)")
    ax.set_ylabel("V(t) / V$_0$")
    _family_legend(ax, bbox_to_anchor=(0.0, 1.06))
    _save(fig, "dilution_timeseries_v2", out)


def _family_legend(ax, row=False, **kw):
    """Her legend: the four families (colour) in one column, fast / slow (grey) in the next;
    ``row``: all six in one row above the axes (for strips with no free corner)."""
    fam = [Line2D([], [], color=c, lw=1.8, label=lab) for lab, _, _, c in FAMILIES]
    key = [Line2D([], [], color=REF_GREY, lw=1.8, ls=FAST, label="across shear"),
           Line2D([], [], color=REF_GREY, lw=1.8, ls=SLOW, label="along shear")]
    blank = Line2D([], [], lw=0, label=" ")
    opts = dict(ncols=2, loc="upper left", fontsize=8.5, handlelength=2.2, handletextpad=0.5,
                labelspacing=0.15, columnspacing=0.8, borderaxespad=0.0)
    if row:
        opts.update(ncols=6, loc="lower left", bbox_to_anchor=(-0.02, 1.02), fontsize=7,
                    handlelength=2.6, handletextpad=0.4, columnspacing=0.9)
        opts.update(kw)
        return ax.legend(handles=fam + key, **opts)
    opts.update(kw)
    return ax.legend(handles=fam + key + [blank] * (len(fam) - len(key)), **opts)


def _days_axis(ax, step=2):
    ax.set_xlim(0, 10)
    ax.set_xticks(range(0, 11, step))
    ax.spines[["top", "right"]].set_visible(False)


def fig_oh(tree=TREE, out=OUT):
    """Box OH vs time, eight cases, default microphysics (v1 f08_OH_dilution)."""
    fig, ax = plt.subplots(figsize=OH_SIZE)
    for _, fast, slow, col in FAMILIES:
        for case, ls in ((fast, FAST), (slow, SLOW)):
            t, oh = _from_burst(case, *_base_species(tree, case, "OH"))
            ax.plot(t / DAY, oh / 1e6, color=col, ls=ls, lw=1.1)
    _days_axis(ax)
    ax.set_ylim(0, None)
    ax.grid(True, color=GRIDC, lw=0.5, zorder=0)
    ax.set_xlabel("day", labelpad=0)
    ax.set_ylabel("OH (10$^6$ molec cm$^{-3}$)", fontsize=8)
    _family_legend(ax, row=True)
    _save(fig, "f08_OH_dilution", out)


#: v1 f09 reservoir colours (sampled from the figure) and styles, in drawing order
S_POOLS = (("SO$_2$", "SO2", "#f48223", "-"), ("SO$_3$", "SO3", "#4a8cdc", "-"),
           ("H$_2$SO$_4$ (gas)", "H2SO4", "#da3e93", (0, (6, 1.5, 1.5, 1.5))),
           ("particulate sulfate", "pS", "#c52228", "-"))
S_UG = 32.065 / 6.02214076e23 * 1e6                    # ug S per S atom
SB_SIZE = (8.04, 4.75)                                  # in: the v1 f09 width, two rows


def _sulfur_pools(tree, case):
    """t [s] and the in-box sulfur [ug S cm^-3] per reservoir (every species carries one S)."""
    f, = glob.glob(os.path.join(tree, "base", f"*__{case}__a1p0__nuc1__cg1__V0mwmean*",
                                "state.npz"))
    with np.load(f, allow_pickle=False) as z:
        sp = [str(s) for s in z["species"]]
        pools = {k: (z["particulate_S"] if k == "pS" else z["x"][:, sp.index(k)]) * S_UG
                 for _, k, _, _ in S_POOLS}
        return z["t"], pools


def fig_sulfur_budget(tree=TREE, out=OUT, rows=(("Across shear", "C1"), ("Along shear", "C2"))):
    """v1 f09, one row per dilution speed (Med Kz): in-box S by reservoir (log, left) and the
    fraction of in-box S per reservoir (stacked, right)."""
    fig, axs = plt.subplots(len(rows), 2, figsize=SB_SIZE, sharex=True, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.06, wspace=0.08)
    for r, ((speed, case), (a1, a2)) in enumerate(zip(rows, axs)):
        t, pools = _sulfur_pools(tree, case)
        d = t / DAY
        total = np.sum(list(pools.values()), axis=0)
        a1.semilogy(d, total, color="black", lw=1.8, label="total S", zorder=3)
        for lab, k, col, ls in S_POOLS:
            a1.semilogy(d, np.maximum(pools[k], 1e-30), color=col, ls=ls, lw=1.5, label=lab)
        a1.set_ylim(1e-11, 2)
        a1.set_yticks([10.0 ** e for e in range(-11, 1)])
        a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"10$^{{{round(np.log10(v))}}}$" if round(np.log10(v)) % 2 else ""))
        a1.yaxis.set_minor_formatter(NullFormatter())
        a1.grid(True, axis="y", which="major", color=GRIDC, lw=0.5, zorder=0)
        a1.set_ylabel("sulfur mass (µg S cm$^{-3}$)", fontsize=8.5)
        a1.text(-0.27, 0.5, f"{speed} ({LABEL_OF[case]})", transform=a1.transAxes,
                rotation=90, ha="center", va="center", fontsize=10, fontweight="semibold")
        a2.stackplot(d, *[pools[k] / total for _, k, _, _ in S_POOLS],
                     colors=[c for _, _, c, _ in S_POOLS], lw=0)
        a2.set_ylim(0, 1)
        a2.set_yticks(np.arange(0, 1.01, 0.2))
        a2.grid(True, color="white", alpha=0.45, lw=0.6)
        a2.set_axisbelow(False)                     # the white grid sits on the fills
        a2.set_ylabel("fraction of sulfur", fontsize=8.5)
        for ax in (a1, a2):
            _days_axis(ax)
            if r == len(rows) - 1:
                ax.set_xlabel("day", labelpad=0)
    axs[0, 0].set_title("In-box sulfur by reservoir (dilution included)", fontsize=10)
    axs[0, 1].set_title("Sulfur partitioning", fontsize=10)
    axs[0, 0].legend(loc="upper right", fontsize=7.5, handlelength=2.4, labelspacing=0.2)
    _save(fig, "f09_sulfur_budget", out)


UG_H2SO4 = 98.079 / 6.02214076e23 * 1e6               # ug H2SO4 per particulate-S molecule
#: v1 f13 rows, in its order: (row label, site token, background token)
BACKGROUNDS = (("Low Alt, High Lat, young background air", "60N_15km", "sabr220"),
               ("High Alt, Low Lat, old background air", "30N_20km", "sabr220"),
               ("AER-2D SAI ~1.5 °C cooling", "30N_20km", "aergeo"))
BB_SIZE = (7.0, 6.2)                                    # in: two columns of the v1 f13 panels
DB_SIZE = (7.0, 4.4)                                    # in: two columns of the v1 f14 panels


EXTRA_TREE = os.path.join(_HERE, "runs_v2")          # local run_v2 output (geo, site60N sets)


def _run_and_ctrl(tree, site, bg, case, variant=""):
    """(run, control) state.npz paths of the default-microphysics mwmean run, or None if either is
    missing. Looked up in every set (base/ctrl, geo/geo_ctrl, site60N/site60N_ctrl) of ``tree``
    first, then of ``runs_v2/`` (the geo and 60N runs were made there, 2026-11-06, at L31/L32
    numerics; runs_v2.1 has the 30N SABR-220 runs only). Run and control come from the same tree.
    ``variant``: case-id suffix, "" (interactive chemistry) or "__constOH" (OH pinned; its control
    is the pinned twin ``__ctrl__constOH``, L34). Ids are matched exactly at the end."""
    cid = f"{site}__{bg}__{case}__a1p0__nuc1__cg1__V0mwmean__h06_ss*IC"
    for tr in (tree, EXTRA_TREE):
        run = glob.glob(os.path.join(tr, "*", cid + variant, "state.npz"))
        ctrl = glob.glob(os.path.join(tr, "*", cid + "__ctrl" + variant, "state.npz"))
        if run and ctrl:
            return run[0], ctrl[0]
    return None


def plume_n_per_mass(run_file, ctrl_file):
    """t [d], dry diameter [um] and the plume's dN/dlogDp per plume H2SO4 [ug^-1]: (run - control)
    number distribution / (run - control) particulate H2SO4 (L33). t = 0 is NaN."""
    with np.load(run_file, allow_pickle=False) as r, np.load(ctrl_file, allow_pickle=False) as k:
        assert np.array_equal(r["t"], k["t"]) and np.allclose(r["dp_mid_um"], k["dp_mid_um"])
        d_n = r["dNdlogDp"] - k["dNdlogDp"]                               # cm^-3
        m = (r["particulate_S"] - k["particulate_S"]) * UG_H2SO4          # ug cm^-3
        t, dp = r["t"] / DAY, r["dp_mid_um"]
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(m[:, None] > 0, d_n / m[:, None], np.nan)
    return t, dp, y


def box_per_total_mass(run_file, _ctrl_file=None, sa=False):
    """t [d], dry diameter [um] and the BOX dN/dlogDp [ug^-1] or dSA/dlogDp [m^2 g^-1] per ALL
    particulate H2SO4 in the box at each time (background + plume; no control subtracted)."""
    with np.load(run_file, allow_pickle=False) as r:
        dp = r["dp_mid_um"]
        y = r["dNdlogDp"] * (np.pi * dp ** 2 if sa else 1.0)
        m = r["particulate_S"] * UG_H2SO4                                  # ug cm^-3
        t = r["t"] / DAY
    return t, dp, (y * 1e-12 / (m[:, None] * 1e-6) if sa else y / m[:, None])


def box_sa_per_total_mass(run_file, ctrl_file=None):
    return box_per_total_mass(run_file, ctrl_file, sa=True)


#: banana quantity -> (per-mass function, colour limits, colour-bar label)
BANANA_QTY = {
    "sa": ("plume_sa_per_mass", (1e-3, 5e2),
           "plume dSA/dlogD$_p$ per plume H$_2$SO$_4$ (m$^2$ g$^{-1}$)"),
    "n": ("plume_n_per_mass", (1e7, 1e13),
          "plume dN/dlogD$_p$ per plume H$_2$SO$_4$ (µg$^{-1}$)"),
    "sa_total": ("box_sa_per_total_mass", (1e-3, 5e2),
                 "dSA/dlogD$_p$ per total particulate H$_2$SO$_4$ (m$^2$ g$^{-1}$)"),
    "n_total": ("box_per_total_mass", (1e7, 1e13),
                "dN/dlogD$_p$ per total particulate H$_2$SO$_4$ (µg$^{-1}$)"),
}


def plume_sa_per_mass(run_file, ctrl_file):
    """t [d], dry diameter [um] and the plume's dSA/dlogDp per plume H2SO4 [m^2 g^-1]:
    (run - control) surface-area distribution / (run - control) particulate H2SO4 (L33).
    Times with no plume sulfate yet (t = 0) are NaN."""
    with np.load(run_file, allow_pickle=False) as r, np.load(ctrl_file, allow_pickle=False) as k:
        assert np.array_equal(r["t"], k["t"]) and np.allclose(r["dp_mid_um"], k["dp_mid_um"])
        dp = r["dp_mid_um"]
        d_sa = (r["dNdlogDp"] - k["dNdlogDp"]) * np.pi * dp ** 2          # um^2 cm^-3
        m = (r["particulate_S"] - k["particulate_S"]) * UG_H2SO4          # ug cm^-3
        t = r["t"] / DAY
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(m[:, None] > 0, d_sa * 1e-12 / (m[:, None] * 1e-6), np.nan)
    return t, dp, y


def _plume_banana_grid(tree, grid, col_titles, size, name, mark_burst=False, qty="sa"):
    """Rows x columns of plume-normalised SA bananas (L33) with one shared colour bar below.
    ``grid``: [(row label, [(site, bg, case) per column])]. A cell whose run or control is missing
    is drawn empty and marked. ``mark_burst``: thin line at each burst case's burst start."""
    from matplotlib.colors import LogNorm
    from coupled.paper_ensemble.make_v2_base_plots import CMAP
    cmap = CMAP.copy()
    cmap.set_bad(cmap(0.0))
    cmap.set_under(cmap(0.0))
    fn_name, lims, cb_label = BANANA_QTY[qty]
    fn = globals()[fn_name]
    norm = LogNorm(*lims)
    fig, axs = plt.subplots(len(grid), len(col_titles), figsize=size, sharex=True, sharey=True,
                            layout="constrained", squeeze=False)
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.03, hspace=0.03)
    for i, (row_label, cells) in enumerate(grid):
        for j, (site, bg, case, *variant) in enumerate(cells):
            ax = axs[i, j]
            pair = _run_and_ctrl(tree, site, bg, case, *variant)
            if pair is None:
                ax.set_facecolor("#F3F4F3")
                ax.text(0.5, 0.45, f"v2 runs not made yet\n({site}, {bg}, {case}"
                        f"{''.join(variant)} + control)",
                        transform=ax.transAxes, ha="center", va="center", fontsize=8,
                        color=REF_GREY)
                print(f"  [{name}] missing: {site} {bg} {case}{''.join(variant)} (run or control)")
            else:
                t, dp, y = fn(*pair)
                y = np.where(y > 0, y, np.nan)        # negatives: see L33 (lowest colour)
                ax.pcolormesh(t, dp, y.T, cmap=cmap, norm=norm, shading="nearest",
                              rasterized=True)
                if mark_burst and case in BURST:
                    ax.axvline(GAUSSIAN_CASES[case].t_burst_start / DAY, color="white",
                               lw=0.7, ls=(0, (2, 2)), alpha=0.8)
            ax.set_yscale("log")
            ax.set_ylim(1.5e-3, 3.0)
            ax.set_xlim(0, 10)
            ax.set_xticks(range(0, 11, 2))
            ax.grid(True, color="white", alpha=0.35, lw=0.5)
            ax.set_axisbelow(False)
            ax.text(0.03, 0.95, row_label, transform=ax.transAxes, ha="left", va="top",
                    fontsize=8, color="white" if pair else CHARCOAL)
            if i == 0:
                ax.set_title(col_titles[j], fontsize=10)
            if j == 0:
                ax.set_ylabel("dry diameter (µm)", fontsize=8.5)
            if i == len(grid) - 1:
                ax.set_xlabel("day", labelpad=0)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs,
                      orientation="horizontal", fraction=0.05 * 3 / len(grid), pad=0.02,
                      aspect=40, extend="both")
    cb.set_label(cb_label, fontsize=8.5)
    _save(fig, name)


def fig_background_bananas(tree=TREE, cols=(("Across shear", "C1"), ("Along shear", "C2")), qty="sa"):
    """v1 f13 (``qty`` "sa") / fS02 (``qty`` "n") in two columns (fast / slow, Med Kz), one row per
    background (v1 order), normalised to the plume (L33)."""
    grid = [(lab, [(site, bg, case) for _, case in cols]) for lab, site, bg in BACKGROUNDS]
    titles = [f"{speed} ({LABEL_OF[case]})" for speed, case in cols]
    name = {"sa": "f13_background_bananas", "n": "fS02_background_bananas_N",
            "sa_total": "f13_background_bananas_per_total_H2SO4",
            "n_total": "fS02_background_bananas_N_per_total_H2SO4"}[qty]
    _plume_banana_grid(tree, grid, titles, BB_SIZE, name, qty=qty)


def fig_dilution_bananas(tree=TREE, qty="sa"):
    """v1 f14 (``qty`` "sa") / fS03 ("n") in two columns (fast / slow), rows Med Kz (C1 / C2) and
    Burst (C3 / C3a), 30N SABR-220, per plume H2SO4 (L33) or, with the ``_total`` quantities, per
    all particulate H2SO4 in the box; dotted line = burst start (48 h)."""
    rows = [(LABEL_OF[f], [("30N_20km", "sabr220", f), ("30N_20km", "sabr220", s)])
            for lab, f, s, _ in FAMILIES if f in ("C1", "C3")]
    name = {"sa": "f14_dilution_bananas", "n": "fS03_dilution_bananas_N",
            "sa_total": "f14_dilution_bananas_per_total_H2SO4",
            "n_total": "fS03_dilution_bananas_N_per_total_H2SO4"}[qty]
    _plume_banana_grid(tree, rows, ["Across shear", "Along shear"], DB_SIZE, name,
                       mark_burst=True, qty=qty)


SNAP_DAYS = (2, 5, 10)                                  # v1 f15
SNAP_SIZE = (7.6, 4.3)                                  # in: the v1 f15 width, fast / slow rows
SNAP_NORM = {   # file suffix, y label (numerator = the plume: run - control)
    "per_particle_H2SO4": "dSA/dlogD$_p$ ÷ plume H$_2$SO$_4$\n(m$^2$ g$^{-1}$)",
    "per_injected_S": "dSA/dlogD$_p$ ÷ injected S\n(m$^2$ per g as H$_2$SO$_4$)",
}


#: snapshot timing -> (file suffix, panel-title suffix). Release is 06:00 local, so day N's
#: 06:00 is t = N d and its noon t = N - 0.75 d; the 24-h mean covers N - 1 .. N d.
SNAP_TIMING = {"sunrise": ("", "06:00"), "noon": ("_noon", "12:00"),
               "dailymean": ("_dailymean", "24-h mean"), "sunrise_noon": ("_sunrise_noon", "")}
FADED = dict(alpha=0.35)                                 # the 06:00 curves in "sunrise_noon"


def _dist_parts(run_file, ctrl_file, qty, norm):
    """t [d], dp [um], numerator (time x bins), denominator (time, or None) and factor so that
    y = num * factor / den (or num * factor): the snapshot quantity before any time averaging.
    qty "sa" | "n"; norm "box" (box values), "per_particle_H2SO4" (plume = run - control per plume
    particulate H2SO4, L33) or "per_injected_S" (plume SA per injected S as H2SO4 mass)."""
    from coupled.paper_ensemble import run_v2 as rv
    with np.load(run_file, allow_pickle=False) as r, np.load(ctrl_file, allow_pickle=False) as k:
        assert np.array_equal(r["t"], k["t"]) and np.allclose(r["dp_mid_um"], k["dp_mid_um"])
        t, dp = r["t"] / DAY, r["dp_mid_um"]
        geo = np.pi * dp ** 2 if qty == "sa" else 1.0
        if norm == "box":
            return t, dp, r["dNdlogDp"] * geo, None, 1.0
        num = (r["dNdlogDp"] - k["dNdlogDp"]) * geo                     # plume, cm^-3 or um^2 cm^-3
        if norm == "per_particle_H2SO4":
            den = (r["particulate_S"] - k["particulate_S"]) * UG_H2SO4  # ug cm^-3
        else:                                                           # per_injected_S
            v0 = os.path.basename(os.path.dirname(run_file)).split("__")[6][2:]
            den = rv.SO2_SPIKE_MOLEC_CM3[v0] / r["V_ratio_applied"] * UG_H2SO4
    return t, dp, num, den, (1e-12 / 1e-6 if qty == "sa" else 1.0)


def _snap(t, num, den, factor, day, timing):
    """The snapshot for ``day`` (one curve over dp): the instant at 06:00 or 12:00, or the 24-h
    mean over day N as the ratio of the time means (the mean of num / the mean of den)."""
    def at(tt):
        i = int(np.argmin(np.abs(t - tt)))
        assert abs(t[i] - tt) < 1e-3, (tt, t[i])
        return num[i], (None if den is None else den[i])
    if timing == "dailymean":
        m = (t >= day - 1 - 1e-9) & (t <= day + 1e-9)
        span = t[m][-1] - t[m][0]
        n_, d_ = (np.trapezoid(num[m], t[m], axis=0) / span,
                  None if den is None else np.trapezoid(den[m], t[m]) / span)
    else:
        n_, d_ = at(day if timing == "sunrise" else day - 0.75)
    if d_ is None:
        return n_ * factor
    return n_ * factor / d_ if d_ > 0 else np.full_like(n_, np.nan)


def _snap_curves(t, num, den, factor, day, timing):
    """[(curve, style overrides, window end [d])] for one panel: one curve, or noon + faded 06:00."""
    if timing == "sunrise_noon":
        return [(_snap(t, num, den, factor, day, "noon"), {}, day - 0.75),
                (_snap(t, num, den, factor, day, "sunrise"), FADED, day)]
    end = day - 0.75 if timing == "noon" else day
    return [(_snap(t, num, den, factor, day, timing), {}, end)]


def _timing_keys(timing):
    if timing != "sunrise_noon":
        return []
    return [Line2D([], [], color=REF_GREY, lw=1.6, label="12:00 local"),
            Line2D([], [], color=REF_GREY, lw=1.6, label="06:00 local", **FADED)]


def _day_title(day, timing):
    tag = SNAP_TIMING[timing][1]
    return f"day {day}" + (f", {tag}" if tag else "")


def fig_dilution_snaps(tree=TREE, norm="per_particle_H2SO4", one_row=False, timing="sunrise"):
    """v1 f15: the plume's dSA/dlogDp on days 2, 5, 10 for the four Kz families (30N SABR-220,
    default microphysics, mwmean V0), per plume H2SO4 in particles (L33) or per injected S (as
    H2SO4 mass). Fast / slow rows (y shared within each day column), or ``one_row``: fast (solid)
    and slow (dashed) in the same panels, y shared across the row, legend above. Linear y.
    ``timing`` (SNAP_TIMING): instant at 06:00 (as before), at 12:00, the 24-h mean, or 12:00 with
    06:00 faded. A burst case is left out while its window ends before the burst starts (it is
    still identical to Low Kz)."""
    n_rows = 1 if one_row else 2
    size = (SNAP_SIZE[0], 2.45) if one_row else SNAP_SIZE
    fig, axs = plt.subplots(n_rows, len(SNAP_DAYS), figsize=size, sharex=True,
                            sharey=True if one_row else "col", layout="constrained",
                            squeeze=False)
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.04, hspace=0.04)
    for lab, fast, slow, col in FAMILIES:
        for r, (case, ls) in enumerate(((fast, FAST), (slow, SLOW))):
            t, dp, num, den, fac = _dist_parts(*_run_and_ctrl(tree, "30N_20km", "sabr220", case),
                                               "sa", norm)
            for c, day in enumerate(SNAP_DAYS):
                for y, sty, end in _snap_curves(t, num, den, fac, day, timing):
                    if case in BURST and end * DAY <= GAUSSIAN_CASES[case].t_burst_start + 1e-6:
                        continue              # still identical to Low Kz: do not hide it
                    axs[0 if one_row else r, c].plot(dp, y, color=col, ls=ls, lw=1.3, **sty)
    speeds = ("",) if one_row else ("Across shear", "Along shear")
    for r, speed in enumerate(speeds):
        for c, day in enumerate(SNAP_DAYS):
            ax = axs[r, c]
            ax.set_xscale("log")
            ax.set_xlim(3e-3, 3.0)
            ax.set_ylim(bottom=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(True, color=GRIDC, lw=0.5)
            ax.text(0.04, 0.95, _day_title(day, timing) if r == 0 else "",
                    transform=ax.transAxes, ha="left", va="top", fontsize=10,
                    fontweight="semibold")
            if r == n_rows - 1:
                ax.set_xlabel("dry diameter (µm)", labelpad=0)
        axs[r, 0].set_ylabel(SNAP_NORM[norm], fontsize=7.5)
        if speed:
            axs[r, 0].text(-0.42, 0.5, f"{speed}", transform=axs[r, 0].transAxes,
                           rotation=90, ha="center", va="center", fontsize=10,
                           fontweight="semibold")
    tkeys = _timing_keys(timing)
    if one_row:                      # figure-level, so the layout does not widen one panel
        leg = _family_legend(axs[0, 0], row=True)
        handles = list(leg.legend_handles) + tkeys
        labels = [tx.get_text() for tx in leg.get_texts()] + [h.get_label() for h in tkeys]
        leg.remove()
        fig.legend(handles, labels, loc="outside upper center", ncols=len(handles), fontsize=7.5,
                   handlelength=2.6, handletextpad=0.4, columnspacing=1.2, frameon=False)
    else:
        _family_legend(axs[1, -1], loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=7.5)
        if tkeys:
            fig.legend(handles=tkeys, loc="outside upper center", ncols=2, fontsize=7.5,
                       frameon=False, handlelength=2.6, columnspacing=1.6)
    _save(fig, f"f15_dilution_snaps_{norm}{SNAP_TIMING[timing][0]}")


def fig_oh_bananas(tree=TREE, qty="sa"):
    """v1 f16 in two columns (fast / slow, Med Kz C1 / C2), 30N SABR-220: interactive chemistry
    (diurnal OH) vs OH pinned at 5e5 cm^-3, each normalised to its own plume (L33: run - control of
    the same chemistry, the constOH runs against their pinned twins, L34)."""
    from coupled.paper_ensemble.run_v2 import OH_PIN_MOLEC_CM3
    e = int(np.floor(np.log10(OH_PIN_MOLEC_CM3)))
    m = OH_PIN_MOLEC_CM3 / 10 ** e
    rows = [("interactive chemistry (diurnal OH)",
             [("30N_20km", "sabr220", "C1"), ("30N_20km", "sabr220", "C2")]),
            (f"constant OH = {m:g}×10$^{e}$ molec cm$^{{-3}}$",
             [("30N_20km", "sabr220", "C1", "__constOH"),
              ("30N_20km", "sabr220", "C2", "__constOH")])]
    _plume_banana_grid(tree, rows, [f"Across shear ({LABEL_OF['C1']})",
                                    f"Along shear ({LABEL_OF['C2']})"], DB_SIZE,
                       {"sa": "f16_OH_bananas", "n": "fS04_OH_bananas_N",
                        "sa_total": "f16_OH_bananas_per_total_H2SO4",
                        "n_total": "fS04_OH_bananas_N_per_total_H2SO4"}[qty], qty=qty)


#: v1 f17 rows: (knob key in the v2 case tuple, row title, levels low -> high)
TOTAL_AXES = (("alpha", "condensation", (0.5, 1.0)), ("nuc", "nucleation", (0.01, 1.0, 100.0)),
              ("cg", "coagulation", (0.5, 1.0, 2.0)))
LEVEL_COLORS = {2: ("#2a78d6", "#eda100"), 3: ("#2a78d6", "#1baf7a", "#eda100")}   # v1 f17
TOT_SIZE = (8.04, 4.8)                                  # in, the v1 f17 page (579 x 346 pt)


TOTAL_NORM = {   # file suffix -> (SA label, number label)
    "box": ("surface area, wet\n(µm$^2$ cm$^{-3}$)", "number (cm$^{-3}$)"),
    "per_particle_H2SO4": ("surface area, wet\n(m$^2$ g$^{-1}$)", "number (µg$^{-1}$)"),
    "per_total_H2SO4": ("surface area, wet\n(m$^2$ g$^{-1}$)", "number (µg$^{-1}$)"),
}
TOTAL_NOTE = {  # what the y axes are divided by (key row)
    "per_particle_H2SO4": "plume (run − control) ÷ plume particulate H$_2$SO$_4$",
    "per_total_H2SO4": "box ÷ all particulate H$_2$SO$_4$",
}


def _totals_series(run_file, ctrl_file, norm):
    """t [d], wet SA and total number of a member: box values ("box", µm^2 cm^-3 and cm^-3), the
    plume (run - control) per plume particulate H2SO4 ("per_particle_H2SO4", L33) or the box per
    all particulate H2SO4 in it ("per_total_H2SO4"); m^2 g^-1 and µg^-1."""
    with np.load(run_file, allow_pickle=False) as r, np.load(ctrl_file, allow_pickle=False) as k:
        assert np.array_equal(r["t"], k["t"])
        t = r["t"] / DAY
        sa, n, ms = r["SA"], r["n_cm3"].sum(1), r["particulate_S"]
        if norm == "per_particle_H2SO4":
            sa, n, ms = sa - k["SA"], n - k["n_cm3"].sum(1), ms - k["particulate_S"]
    if norm == "box":
        return t, sa, n
    m = ms * UG_H2SO4                                                   # ug cm^-3
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = m > 0
        return (t, np.where(ok, sa * 1e-12 / (m * 1e-6), np.nan),
                np.where(ok, n / m, np.nan))


def fig_totals(tree=TREE, norm="box"):
    """v1 f17 with fast (C1, solid) and slow (C2, dashed) in the same panels: wet surface area
    (left) and total particle number (right, as v1; Ali 2026-11-06, although L29: the total is
    mostly 1.8-3 nm clusters and moves 5-14% between the finest time steps) for each microphysics
    knob at its levels, the other knobs x1 (mwmean V0, 30N SABR-220). ``norm``: "box" (as v1;
    grey line = background SA at t = 0), "per_particle_H2SO4" (plume per plume H2SO4, L33) or
    "per_total_H2SO4" (box per all particulate H2SO4; grey line = the no-injection control's)."""
    idx = {"alpha": 0, "nuc": 1, "cg": 2}
    fig, axs = plt.subplots(len(TOTAL_AXES), 2, figsize=TOT_SIZE, sharex=True,
                            layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.03, wspace=0.06, hspace=0.03)
    for r, (knob, title, levels) in enumerate(TOTAL_AXES):
        cols = LEVEL_COLORS[len(levels)]
        late = []
        for case, ls in (("C1", FAST), ("C2", SLOW)):
            for lev, col in zip(levels, cols):
                knobs = [1.0, 1.0, 1.0]
                knobs[idx[knob]] = lev
                t, sa, n = _totals_series(*_member_files(tree, case, *knobs), norm)
                axs[r, 0].semilogy(t, sa, color=col, ls=ls, lw=1.2)
                axs[r, 1].semilogy(t, np.maximum(n, 1e-3), color=col, ls=ls, lw=1.2)
                late.append((sa[t > 0.25], n[t > 0.25]))
        run1, ctrl1 = _member_files(tree, "C1", 1.0, 1.0, 1.0)
        if norm == "box":
            with np.load(run1, allow_pickle=False) as z:
                axs[r, 0].axhline(float(z["SA"][0]), color=REF_GREY, lw=1.2, zorder=1)
        elif norm == "per_total_H2SO4":
            tc, sac, _ = _totals_series(ctrl1, ctrl1, norm)
            axs[r, 0].semilogy(tc, sac, color=REF_GREY, lw=1.2, zorder=1)
        hs = [Line2D([], [], color=c, lw=1.6, label=f"× {lev:g}") for lev, c in zip(levels, cols)]
        legend_at = (dict(loc="lower left", bbox_to_anchor=(0.12, 0.12)) if norm == "box"
                     else dict(loc="upper right"))
        axs[r, 0].legend(handles=hs, title=title, fontsize=7.5, title_fontsize=8,
                         alignment="left", labelspacing=0.15, handlelength=2.0, **legend_at)
        axs[r, 0].set_ylabel(TOTAL_NORM[norm][0], fontsize=8)
        axs[r, 1].set_ylabel(TOTAL_NORM[norm][1], fontsize=8)
        for ax in axs[r]:
            _days_axis(ax)
            ax.grid(True, color=GRIDC, lw=0.5)
        if norm != "box":                     # range from t > 6 h (the first hours divide by ~0)
            for c in (0, 1):
                v = np.concatenate([x[c][np.isfinite(x[c]) & (x[c] > 0)] for x in late])
                axs[r, c].set_ylim(v.min() / 2, v.max() * 2)
    for ax in axs[-1]:
        ax.set_xlabel("day", labelpad=0)
    if norm == "box":
        for ax in axs[:, 0]:
            ax.set_ylim(bottom=1e-1)
        for ax in axs[:, 1]:                  # as v1: from 10^3
            ax.set_ylim(bottom=1e3)
    key = [Line2D([], [], color=REF_GREY, lw=1.6, ls=FAST, label="across shear (Med K$_\\mathregular{z}$)"),
           Line2D([], [], color=REF_GREY, lw=1.6, ls=SLOW, label="along shear (Med K$_\\mathregular{z}$)")]
    if norm != "per_particle_H2SO4":
        key.append(Line2D([], [], color=REF_GREY, lw=1.2,
                          label="background SA" if norm == "box" else "no-injection control"))
    if norm in TOTAL_NOTE:
        key.append(Line2D([], [], lw=0, label="per mass: " + TOTAL_NOTE[norm]))
    fig.legend(handles=key, loc="outside upper center", ncols=len(key), fontsize=7.5,
               frameon=False, handlelength=2.6, columnspacing=1.4)
    _save(fig, "f17_totals_all" + ("" if norm == "box" else f"_{norm}"))


SIZE_SNAP_SIZE = (8.0, 6.6)                             # in, the v1 f18 page
SIZE_SNAP_NORM = {   # (quantity, file suffix) -> y label
    ("sa", "box"): "dSA/dlogD$_p$ (µm$^2$ cm$^{-3}$)",
    ("sa", "per_particle_H2SO4"): "dSA/dlogD$_p$ ÷ plume H$_2$SO$_4$\n(m$^2$ g$^{-1}$)",
    ("n", "box"): "dN/dlogD$_p$ (cm$^{-3}$)",
    ("n", "per_particle_H2SO4"): "dN/dlogD$_p$ ÷ plume H$_2$SO$_4$\n(µg$^{-1}$)",
}
SIZE_SNAP_NAME = {"sa": "f18_size_snaps_all", "n": "fS05_size_snaps_all_N"}


def _member_files(tree, case, alpha, nuc, cg):
    """(run, control) state.npz of a mwmean microphysics member at 30N SABR-220 (base/ens and
    ctrl/ens_ctrl of ``tree``)."""
    from coupled.paper_ensemble import run_v2 as rv
    spec = rv.CaseSpec(case, condensation_alpha=alpha, nucleation_rate_scale=nuc,
                       coag_kernel_scale=cg)
    run, = glob.glob(os.path.join(tree, "*", spec.case_id, "state.npz"))
    ctrl, = glob.glob(os.path.join(tree, "*", rv.paired_control(spec).case_id, "state.npz"))
    return run, ctrl


def fig_size_snaps(tree=TREE, norm="box", qty="sa", timing="sunrise"):
    """v1 f18 with fast (C1, solid) and slow (C2, dashed) in the same panels: dSA/dlogDp on days
    2, 5, 10 (columns) for each microphysics knob at its levels (rows, the other knobs x1; mwmean
    V0, 30N SABR-220). ``norm``: "box" (as v1) or "per_particle_H2SO4" (the plume, run - control,
    per gram of plume H2SO4 in particles, L33). ``qty``: "sa" (f18, linear y) or "n" (fS05, log y
    over 5 decades: the 1.8-3 nm bins reach ~1e7 cm^-3 and would flatten a linear axis). y shared
    along each row. ``timing``: see SNAP_TIMING."""
    idx = {"alpha": 0, "nuc": 1, "cg": 2}
    fig, axs = plt.subplots(len(TOTAL_AXES), len(SNAP_DAYS), figsize=SIZE_SNAP_SIZE, sharex=True,
                            sharey="row", layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.03, wspace=0.04, hspace=0.03)
    for r, (knob, title, levels) in enumerate(TOTAL_AXES):
        cols = LEVEL_COLORS[len(levels)]
        for case, ls in (("C1", FAST), ("C2", SLOW)):
            for lev, col in zip(levels, cols):
                knobs = [1.0, 1.0, 1.0]
                knobs[idx[knob]] = lev
                t, dp, num, den, fac = _dist_parts(*_member_files(tree, case, *knobs), qty, norm)
                for c, day in enumerate(SNAP_DAYS):
                    for y, sty, _ in _snap_curves(t, num, den, fac, day, timing):
                        axs[r, c].plot(dp, y, color=col, ls=ls, lw=1.1, **sty)
        hs = [Line2D([], [], color=c, lw=1.6, label=f"× {lev:g}") for lev, c in zip(levels, cols)]
        axs[r, 0].legend(handles=hs, title=title, loc="upper right" if qty == "n" else "upper left",
                         fontsize=7.5,
                         title_fontsize=8, alignment="left", labelspacing=0.15, handlelength=2.0)
        axs[r, 0].set_ylabel(SIZE_SNAP_NORM[(qty, norm)], fontsize=8)
        for ax in axs[r]:
            ax.set_xscale("log")
            ax.set_xlim(1.5e-3, 3.0)
            if qty == "n":
                ax.set_yscale("log")
            else:
                ax.set_ylim(bottom=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(True, color=GRIDC, lw=0.5)
    if qty == "n":                            # 5 decades below each row's maximum
        for r in range(len(TOTAL_AXES)):
            top = max(max(np.nanmax(l.get_ydata()) for l in ax.get_lines()) for ax in axs[r])
            axs[r, 0].set_ylim(top * 1e-5, top * 2)
    for c, day in enumerate(SNAP_DAYS):
        axs[0, c].set_title(_day_title(day, timing), fontsize=10, fontweight="semibold")
        axs[-1, c].set_xlabel("dry diameter (µm)", labelpad=0)
    key = [Line2D([], [], color=REF_GREY, lw=1.6, ls=FAST, label="across shear (Med K$_\\mathregular{z}$)"),
           Line2D([], [], color=REF_GREY, lw=1.6, ls=SLOW, label="along shear (Med K$_\\mathregular{z}$)")]
    key += _timing_keys(timing)
    fig.legend(handles=key, loc="outside upper center", ncols=len(key), fontsize=7.5,
               frameon=False, handlelength=2.6, columnspacing=1.6)
    _save(fig, f"{SIZE_SNAP_NAME[qty]}_{norm}{SNAP_TIMING[timing][0]}")


def main(tree=TREE):
    _paper_style()
    fig_dilution_timeseries(tree)
    fig_oh(tree)
    fig_sulfur_budget(tree)
    fig_background_bananas(tree)
    fig_background_bananas(tree, qty="n")
    fig_background_bananas(tree, qty="sa_total")         # box / all particulate H2SO4 (Ali)
    fig_background_bananas(tree, qty="n_total")
    for qty in BANANA_QTY:
        fig_dilution_bananas(tree, qty)
    for timing in SNAP_TIMING:                     # 06:00 / 12:00 / 24-h mean / both (Ali)
        fig_dilution_snaps(tree, "per_particle_H2SO4", one_row=True, timing=timing)
        fig_dilution_snaps(tree, "per_injected_S", timing=timing)
    for qty in BANANA_QTY:
        fig_oh_bananas(tree, qty)
    for norm in TOTAL_NORM:
        fig_totals(tree, norm)
    for qty, norm in SIZE_SNAP_NORM:
        for timing in SNAP_TIMING:
            fig_size_snaps(tree, norm, qty, timing)


if __name__ == "__main__":
    main(*sys.argv[1:])
