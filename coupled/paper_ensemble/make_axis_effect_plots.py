# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Condensation- and coagulation-impact figure sets, mirroring the nucleation set
(-> runs/plots/condensation/ and runs/plots/coagulation/).

Ensemble figures compare within matched tuples (runs identical except the varied axis), so
all other axes cancel exactly; t* metrics use the same per-regime evaluation days as the
nucleation set (day 10 D1/D2/burst, day 5 D3, day 3 D5). Case-study figures (bananas,
snapshots, totals) are the D2-med / 30N / 20 km / 210 K subset with BOTH other microphysics
knobs at x1 (condensation set: nucleation x1 + coag x1; coagulation set: nucleation x1 +
condensation alpha x1).

Per axis: slopegraph, amplification_ratios, elasticity, sizedist_by_level,
time_resolved_ratio, banana_D2med(+_SA, _ratio, _ratio_SA), sizedist_snapshots(+_linear),
totals_D2med.

Run (from SANDBOX/): python -m coupled.paper_ensemble.make_axis_effect_plots [condensation|coagulation]
(no argument = both). Reuses the caches from the nucleation scripts.
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from coupled.paper_ensemble.make_paper_candidate_plots import (
    _RUNS, REGIMES, REGIME_LABEL, REGIME_COLOR, _tokens, reduce_runs, load_summary, case_dir,
    SAI_BG, SAI_LABEL, SAI_SHORT, SAI_SUFFIX)
from coupled.paper_ensemble.make_nucleation_plots import (
    metrics_table, nd_eval, EVAL_DAY, EVAL_NOTE)

plt.rcParams.update({"font.family": "Helvetica", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.35,
                     "grid.linewidth": 0.5, "grid.color": "#e1e0d9", "axes.axisbelow": True,
                     "figure.dpi": 130, "legend.frameon": False})

BGS = [("sabr220", "SABR-220 (clean aged air) background"), (SAI_BG, SAI_LABEL)]

AXES = {
    "condensation": dict(
        tok=3, levels=[0.5, 1.0], name="condensation α",
        lab={0.5: "×0.5", 1.0: "×1.0"},
        ramp={0.5: "#86b6ef", 1.0: "#104281"},            # ordinal ends of the blue ramp
        cat={0.5: "#2a78d6", 1.0: "#eda100"},             # snapshot/totals line colors
        case_lv={0.5: "a0p5", 1.0: "a1p0"},
        case="30N_20km__{bg}__D2med__{lv}__nuc1__cg1",
        fixed_note="nucleation ×1, coag ×1",
        span_note="×2 in condensation α (×1.0 / ×0.5)"),
    "coagulation": dict(
        tok=5, levels=[0.5, 1.0, 2.0], name="coagulation kernel",
        lab={0.5: "×0.5", 1.0: "×1", 2.0: "×2"},
        ramp={0.5: "#86b6ef", 1.0: "#2a78d6", 2.0: "#104281"},
        cat={0.5: "#2a78d6", 1.0: "#1baf7a", 2.0: "#eda100"},
        case_lv={0.5: "cg0p5", 1.0: "cg1", 2.0: "cg2"},
        case="30N_20km__{bg}__D2med__a1p0__nuc1__{lv}",
        fixed_note="nucleation ×1, α ×1",
        span_note="×4 in the coagulation kernel (×2 / ×0.5)"),
}


def build_tuples(R, tok):
    """Key = all case tokens except the varied one -> {level: run_index}."""
    tup = {}
    for i, c in enumerate(R["case"]):
        parts = _tokens(c)
        key = tuple(p for k, p in enumerate(parts) if k != tok)
        tup.setdefault(key, {})[parts[tok]] = i
    return tup


def load_case(ax, bg, lv):
    case = ax["case"].format(bg=bg, lv=ax["case_lv"][lv])
    return np.load(os.path.join(case_dir(case), "state.npz"))


# ----------------------------------------------------------- ensemble figures
def fig_slopegraph(ax, out, R, tup, M):
    keys = list(M)[:3]                            # the three t* metrics
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 4.6))
    for a, key in zip(axs, keys):
        v = M[key]
        for tk, idx in tup.items():
            y = [v[idx[s]] for s in ax["levels"]]
            a.plot(ax["levels"], y, color=REGIME_COLOR[tk[2]], lw=0.55, alpha=0.16)
        for reg in REGIMES:
            med = [np.median([v[idx[s]] for tk, idx in tup.items() if tk[2] == reg])
                   for s in ax["levels"]]
            a.plot(ax["levels"], med, color=REGIME_COLOR[reg], lw=2.4, label=REGIME_LABEL[reg])
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xticks(ax["levels"], [ax["lab"][s] for s in ax["levels"]])
        a.set_xlabel(f"{ax['name']} scale")
        a.set_title(key, fontsize=10.5)
    axs[0].legend(fontsize=8, loc="lower right", title="dilution", title_fontsize=8)
    fig.suptitle(f"{ax['name'].capitalize()} impact within matched tuples (thin: "
                 f"{len(tup)} tuples; bold: per-regime median)\nt* = end of plume life: "
                 f"{EVAL_NOTE}", y=1.06, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "slopegraph.png"), bbox_inches="tight")
    plt.close(fig)


def fig_ratios(ax, out, tup, M):
    lo, hi = ax["levels"][0], ax["levels"][-1]
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 7.0), sharex=True)
    for a, (key, v) in zip(axs.ravel(), M.items()):
        for k, reg in enumerate(REGIMES):
            r = [v[idx[hi]] / v[idx[lo]] for tk, idx in tup.items() if tk[2] == reg]
            bp = a.boxplot(r, positions=[k], widths=0.6, patch_artist=True, showfliers=False,
                           medianprops={"color": "#0b0b0b"})
            bp["boxes"][0].set(facecolor=REGIME_COLOR[reg], alpha=0.8, lw=0.6)
        a.axhline(1.0, color="#898781", lw=0.9, ls="--")
        a.set_yscale("log")
        a.set_title(key, fontsize=10)
        a.set_xticks(range(len(REGIMES)),
                     [REGIME_LABEL[r].replace(" ", "\n") for r in REGIMES], fontsize=8.5)
        if a in axs[:, 0]:
            a.set_ylabel(f"metric({ax['lab'][hi]}) / metric({ax['lab'][lo]})")
    n_per = sum(1 for tk in tup if tk[2] == REGIMES[0])
    fig.suptitle(f"{ax['name'].capitalize()} amplification: within-tuple response to "
                 f"{ax['span_note']} ({n_per} tuples per box)\nt* metrics evaluated per regime "
                 f"({EVAL_NOTE.split(': ')[-1] if ': ' in EVAL_NOTE else EVAL_NOTE})",
                 y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "amplification_ratios.png"), bbox_inches="tight")
    plt.close(fig)


def fig_elasticity(ax, out, tup, M):
    lo, hi = ax["levels"][0], ax["levels"][-1]
    span = np.log10(hi / lo)
    fig, a = plt.subplots(figsize=(8.2, 4.8))
    names = list(M)
    for m, key in enumerate(names):
        v = M[key]
        for k, reg in enumerate(REGIMES):
            e = np.array([np.log10(v[idx[hi]] / v[idx[lo]]) / span
                          for tk, idx in tup.items() if tk[2] == reg])
            q1, med, q3 = np.percentile(e, [25, 50, 75])
            y = m + (k - 2) * 0.13
            a.plot([q1, q3], [y, y], color=REGIME_COLOR[reg], lw=1.6, alpha=0.85)
            a.plot(med, y, "o", color=REGIME_COLOR[reg], ms=6,
                   label=REGIME_LABEL[reg] if m == 0 else None)
    a.axvline(0.0, color="#898781", lw=0.9, ls="--")
    a.axvline(1.0, color="#898781", lw=0.9, ls=":")
    a.text(0.985, 0.03, "1:1 response ", fontsize=8, color="#898781", ha="right", va="bottom",
           rotation=90, transform=a.get_xaxis_transform())
    a.set_xlim(right=1.06)
    a.set_yticks(range(len(names)), names, fontsize=9.5)
    a.invert_yaxis()
    a.set_xlabel(f"elasticity  d log(metric) / d log({ax['name']} scale)")
    a.legend(fontsize=8.5, title="dilution", title_fontsize=8.5, loc="upper right",
             bbox_to_anchor=(0.985, 0.97))
    a.set_title(f"How buffered is each response to {ax['name']}? (median + IQR over matched "
                f"tuples)\nt* metrics {EVAL_NOTE}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "elasticity.png"), bbox_inches="tight")
    plt.close(fig)


def fig_sizedist_by_level(ax, out, R):
    toks = [_tokens(c) for c in R["case"]]
    nd = nd_eval(R)
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 6.6), sharex=True, sharey=True)
    for p, reg in enumerate(REGIMES):
        a = axs.ravel()[p]
        for s in ax["levels"]:
            sel = np.array([tk[2] == reg and tk[ax["tok"]] == s for tk in toks])
            med = np.median(nd[sel], axis=0)
            a.plot(R["dp_um"], np.maximum(med, 1e-6), color=ax["ramp"][s], lw=1.9,
                   label=f"{ax['name']} {ax['lab'][s]}")
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_ylim(1e-4, 3e5)
        a.set_xlim(2e-3, 20)
        a.set_title(f"{REGIME_LABEL[reg]} (day {EVAL_DAY[reg]})", fontsize=10.5,
                    color=REGIME_COLOR[reg])
        if p % 3 == 0:
            a.set_ylabel("dN/dlogD$_p$ [cm$^{-3}$]")
        if p >= 2:
            a.set_xlabel("dry diameter [µm]")
    axs[0, 0].legend(fontsize=8.5, loc="lower left")
    axs[1, 2].axis("off")
    n_per = sum(1 for tk in [_tokens(c) for c in R["case"]]
                if tk[2] == REGIMES[0] and tk[ax["tok"]] == ax["levels"][0])
    axs[1, 2].text(0.03, 0.8, f"Median end-of-plume-life distribution\nover the {n_per} "
                   "matched runs per regime ×\nlevel (other axes pooled). Each regime\n"
                   "is shown at its evaluation day.", fontsize=9.5, va="top", color="#52514e")
    fig.suptitle(f"End-of-life size distribution vs {ax['name']} scale", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sizedist_by_level.png"), bbox_inches="tight")
    plt.close(fig)


def fig_time_ratio(ax, out, R, tup):
    lo, hi = ax["levels"][0], ax["levels"][-1]
    t = R["t_days"]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)
    for a, (key, lab) in zip(axs, [("Ntot", "total number"), ("SA", "surface area")]):
        for reg in REGIMES:
            ratios = np.array([R[key][idx[hi]] / R[key][idx[lo]]
                               for tk, idx in tup.items() if tk[2] == reg])
            q1, med, q3 = np.percentile(ratios, [25, 50, 75], axis=0)
            a.fill_between(t, q1, q3, color=REGIME_COLOR[reg], alpha=0.14, lw=0)
            a.plot(t, med, color=REGIME_COLOR[reg], lw=1.8, label=REGIME_LABEL[reg])
        a.axhline(1.0, color="#898781", lw=0.9, ls="--")
        for reg in ("D5vhigh", "D3high"):
            a.axvline(EVAL_DAY[reg], color=REGIME_COLOR[reg], lw=1.1, ls=":")
            a.text(EVAL_DAY[reg], 0.985, f" t* {REGIME_LABEL[reg].split()[0]}", fontsize=7.5,
                   color=REGIME_COLOR[reg], ha="left", va="top",
                   transform=a.get_xaxis_transform())
        a.set_yscale("log")
        a.set_xlim(0, 10)
        a.set_xlabel("day")
        a.set_ylabel(f"{lab} ratio  ({ax['lab'][hi]} / {ax['lab'][lo]})")
        a.set_title(lab, fontsize=11)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncols=5, fontsize=9, bbox_to_anchor=(0.5, 0.93))
    n_per = sum(1 for tk in tup if tk[2] == REGIMES[0])
    fig.suptitle(f"When {ax['name']} matters: within-pair ratio vs time (median + IQR, "
                 f"{n_per} pairs per regime)", y=1.0, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(os.path.join(out, "time_resolved_ratio.png"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------- D2-med case-study figures
def fig_banana(ax, out, sa=False, cmap="viridis"):
    rows = ax["levels"]
    fig, axs = plt.subplots(len(rows), 2, figsize=(10.0, 3.3 * len(rows) + 0.4),
                            sharex=True, sharey=True, squeeze=False)
    norm = LogNorm(vmin=1e-2, vmax=2e3) if sa else LogNorm(vmin=1e2, vmax=1e8)
    label = ("dSA/dlogD$_p$ [µm$^2$ cm$^{-3}$]" if sa else "dN/dlogD$_p$ [cm$^{-3}$]")
    pc = None
    for i, lv in enumerate(rows):
        for j, (bg, blab) in enumerate(BGS):
            z = load_case(ax, bg, lv)
            t_days, dp, dnd = z["t"] / 86400.0, z["dp_mid_um"], z["dNdlogDp"]
            field = dnd * np.pi * dp ** 2 if sa else dnd
            a = axs[i, j]
            pc = a.pcolormesh(t_days, dp, np.maximum(field, 1e-9).T, norm=norm, cmap=cmap,
                              rasterized=True, shading="nearest")
            a.set_yscale("log")
            a.set_ylim(dp[0], 2.0)
            if i == 0:
                a.set_title(blab, fontsize=11)
            if i == len(rows) - 1:
                a.set_xlabel("day")
            if j == 0:
                a.set_ylabel(f"{ax['name']} {ax['lab'][lv]}\ndry diameter [µm]", fontsize=10)
    fig.suptitle(f"Plume size-distribution evolution — D2 med, 30°N / 20 km / 210 K, "
                 f"{ax['fixed_note']}", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    cax = fig.add_axes((0.92, 0.12, 0.02, 0.76))
    fig.colorbar(pc, cax=cax, label=label)
    fname = f"banana_D2med{'_SA' if sa else ''}.png"
    fig.savefig(os.path.join(out, fname), bbox_inches="tight")
    plt.close(fig)


def fig_banana_ratio(ax, out, sa=False):
    ref_z = load_case(ax, "sabr220", 1.0)
    t_days, dp = ref_z["t"] / 86400.0, ref_z["dp_mid_um"]
    to_field = (lambda nd: nd * np.pi * dp ** 2) if sa else (lambda nd: nd)
    ref = to_field(ref_z["dNdlogDp"])
    norm_abs = LogNorm(vmin=1e-2, vmax=2e3) if sa else LogNorm(vmin=1e2, vmax=1e8)
    label_abs = ("reference dSA/dlogD$_p$ [µm$^2$ cm$^{-3}$]" if sa
                 else "reference dN/dlogD$_p$ [cm$^{-3}$]")
    floor = 1e-2 if sa else 1e1
    rows = ax["levels"]
    fig, axs = plt.subplots(len(rows), 2, figsize=(10.0, 3.3 * len(rows) + 0.4),
                            sharex=True, sharey=True, squeeze=False)
    pc_abs = pc_rat = None
    cmap_rat = plt.get_cmap("RdBu_r").copy()
    cmap_rat.set_bad("0.88")
    for i, lv in enumerate(rows):
        for j, (bg, blab) in enumerate(BGS):
            a = axs[i, j]
            if (bg, lv) == ("sabr220", 1.0):
                pc_abs = a.pcolormesh(t_days, dp, np.maximum(ref, 1e-9).T, norm=norm_abs,
                                      cmap="viridis", rasterized=True, shading="nearest")
                a.text(0.02, 0.96, "REFERENCE (absolute)", transform=a.transAxes,
                       fontsize=8.5, va="top", color="white", fontweight="bold")
            else:
                fld = to_field(load_case(ax, bg, lv)["dNdlogDp"])
                ratio = np.clip(fld / np.maximum(ref, 1e-30), 1e-2, 1e2)
                ratio[(fld < floor) & (ref < floor)] = np.nan
                pc_rat = a.pcolormesh(t_days, dp, ratio.T, norm=LogNorm(1e-2, 1e2),
                                      cmap=cmap_rat, rasterized=True, shading="nearest")
            a.set_yscale("log")
            a.set_ylim(dp[0], 2.0)
            if i == 0:
                a.set_title(blab, fontsize=11)
            if i == len(rows) - 1:
                a.set_xlabel("day")
            if j == 0:
                a.set_ylabel(f"{ax['name']} {ax['lab'][lv]}\ndry diameter [µm]", fontsize=10)
    fig.suptitle(f"{ax['name'].capitalize()} / background effect relative to the clean ×1 case "
                 f"— D2 med, 30°N / 20 km / 210 K, {ax['fixed_note']}", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    cax1 = fig.add_axes((0.90, 0.55, 0.018, 0.33))
    fig.colorbar(pc_abs, cax=cax1, label=label_abs)
    cax2 = fig.add_axes((0.90, 0.12, 0.018, 0.33))
    fig.colorbar(pc_rat, cax=cax2, label="ratio to reference (gray = both near-empty)")
    fname = f"banana_D2med_ratio{'_SA' if sa else ''}.png"
    fig.savefig(os.path.join(out, fname), bbox_inches="tight")
    plt.close(fig)


def fig_snapshots(ax, out, linear=False):
    days = [2, 5, 10]
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 7.0), sharex=True,
                            sharey=False if linear else "row")
    for k, day in enumerate(days):
        for lv in ax["levels"]:
            for bg, blab in BGS:
                z = load_case(ax, bg, lv)
                t_days, dp, nd = z["t"] / 86400.0, z["dp_mid_um"], z["dNdlogDp"]
                it = np.searchsorted(t_days, day - 1e-9)
                spec_n = np.maximum(nd[it], 1e-6)
                spec_sa = np.maximum(nd[it] * np.pi * dp ** 2, 1e-6)
                ls = "-" if bg == "sabr220" else "--"
                lab = f"{ax['lab'][lv]}, {'clean' if bg == 'sabr220' else 'SAI'}"
                axs[0, k].plot(dp, spec_n, ls, color=ax["cat"][lv], lw=2.0, label=lab)
                axs[1, k].plot(dp, spec_sa, ls, color=ax["cat"][lv], lw=2.0)
        for r in (0, 1):
            a = axs[r, k]
            a.set_xscale("log")
            a.set_xlim(2e-3, 20)
            if linear:
                a.set_ylim(bottom=0)
            else:
                a.set_yscale("log")
        axs[0, k].set_title(f"day {day}", fontsize=11)
        if not linear:
            axs[0, k].set_ylim(1e-2, 3e6)
            axs[1, k].set_ylim(1e-4, 3e3)
        axs[1, k].set_xlabel("dry diameter [µm]")
    axs[0, 0].set_ylabel("dN/dlogD$_p$ [cm$^{-3}$]")
    axs[1, 0].set_ylabel("dSA/dlogD$_p$ [µm$^2$ cm$^{-3}$]")
    leg_ax = axs[0, 0] if linear else axs[0, 2]
    leg_ax.legend(fontsize=7.5, ncols=2, loc="upper right",
                  title=f"{ax['name']}, background", title_fontsize=7.5)
    scale_note = "linear y, per-panel scale" if linear else "log y"
    fig.suptitle(f"Number (top) and surface-area (bottom) size distributions — D2 med, "
                 f"30°N / 20 km / 210 K, {ax['fixed_note']}\n"
                 f"solid: SABR-220 clean; dashed: {SAI_SHORT} ({scale_note})",
                 y=1.0, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fname = f"sizedist_snapshots_D2med{'_linear' if linear else ''}.png"
    fig.savefig(os.path.join(out, fname), bbox_inches="tight")
    plt.close(fig)


def fig_totals(ax, out):
    fig, (ax_sa, ax_n) = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)
    bg_ref = {}                                  # t=0 state = the background distribution
    for lv in ax["levels"]:
        for bg, blab in BGS:
            z = load_case(ax, bg, lv)
            t_days = z["t"] / 86400.0
            ls = "-" if bg == "sabr220" else "--"
            lab = f"{ax['lab'][lv]}, {'clean' if bg == 'sabr220' else 'SAI'}"
            ax_sa.plot(t_days, z["SA"], ls, color=ax["cat"][lv], lw=1.7, label=lab)
            ax_n.plot(t_days, z["total_n"], ls, color=ax["cat"][lv], lw=1.7)
            bg_ref[bg] = (float(z["SA"][0]), float(z["total_n"][0]))
    for bg, blab in BGS:
        ls = "-" if bg == "sabr220" else "--"
        lab = f"background ({'clean' if bg == 'sabr220' else 'SAI'})"
        ax_sa.axhline(bg_ref[bg][0], ls=ls, color="#898781", lw=1.3, label=lab)
        ax_n.axhline(bg_ref[bg][1], ls=ls, color="#898781", lw=1.3)
    for a, ylab, title in ((ax_sa, "surface area (wet) [µm$^2$ cm$^{-3}$]",
                            "total surface area"),
                           (ax_n, "number [cm$^{-3}$]", "total number")):
        a.set_yscale("log")
        a.set_xlim(0, 10)
        a.set_xlabel("day")
        a.set_ylabel(ylab)
        a.set_title(title, fontsize=11)
    ax_n.set_ylim(bottom=0.5 * min(v[1] for v in bg_ref.values()))
    ax_sa.set_ylim(bottom=0.5 * min(v[0] for v in bg_ref.values()))
    ax_sa.legend(fontsize=7.5, ncols=2, loc="lower left",
                 title=f"{ax['name']}, background", title_fontsize=7.5)
    fig.suptitle(f"Integrated aerosol surface area and number — D2 med, 30°N / 20 km / 210 K, "
                 f"{ax['fixed_note']} (solid: SABR-220 clean; dashed: {SAI_SHORT})",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "totals_D2med.png"), bbox_inches="tight")
    plt.close(fig)


def run_axis(axis_name):
    ax = AXES[axis_name]
    out = os.path.join(_RUNS, "plots", f"{axis_name}{SAI_SUFFIX}")
    os.makedirs(out, exist_ok=True)
    R = reduce_runs()
    summary = load_summary()
    assert list(R["case"]) == [r["case_id"] for r in summary]
    tup = build_tuples(R, ax["tok"])
    assert all(len(v) == len(ax["levels"]) for v in tup.values())
    M = metrics_table(R, summary)
    print(f"[{axis_name}] {len(tup)} matched tuples", flush=True)
    fig_slopegraph(ax, out, R, tup, M)
    fig_ratios(ax, out, tup, M)
    fig_elasticity(ax, out, tup, M)
    fig_sizedist_by_level(ax, out, R)
    fig_time_ratio(ax, out, R, tup)
    fig_banana(ax, out)
    fig_banana(ax, out, sa=True)
    fig_banana_ratio(ax, out)
    fig_banana_ratio(ax, out, sa=True)
    fig_snapshots(ax, out)
    fig_snapshots(ax, out, linear=True)
    fig_totals(ax, out)
    print(f"[{axis_name}] 12 figures -> {out}", flush=True)


if __name__ == "__main__":
    targets = sys.argv[1:] or list(AXES)
    for name in targets:
        run_axis(name)
