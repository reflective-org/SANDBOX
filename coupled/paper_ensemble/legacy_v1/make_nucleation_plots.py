# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Nucleation-impact figures from the 810-case ensemble (-> runs/plots/nucleation/).

The full-factorial design contains 270 matched triplets identical except for the nucleation
rate scale (x0.01 / x1 / x100); every comparison here is within-triplet, so all other axes
cancel exactly.

End-state ("t*") metrics are NOT all taken at day 10: fast-diluting regimes relax to the
background aerosol before then, so a day-10 comparison would measure background air. Each
regime is instead evaluated at the last whole day before ANY of its runs falls to within 10%
of the background surface area (24-h-smoothed): day 10 for D1/D2/burst (never reach
background), day 5 for D3 (earliest crossing 5.44), day 3 for D5 (earliest crossing 3.59).
Whole days keep the diurnal phase identical across regimes. The days are insensitive to the
threshold anywhere in 5-10% above background.

  1. slopegraph.png          -- per-triplet lines of t* N / SA / r_eff vs nucleation scale.
  2. amplification_ratios.png-- boxplots of metric(x100)/metric(x0.01) per dilution regime.
  3. elasticity.png          -- dlog(metric)/dlog(nuc scale) per metric x regime.
  4. sizedist_by_nuc.png     -- median t* dN/dlogDp per nucleation level, faceted by regime.
  5. time_resolved_ratio.png -- median per-pair ratio (x100/x0.01) of N(t) and SA(t) per regime,
                                with the per-regime evaluation days marked.

Run (from SANDBOX/): python -m coupled.paper_ensemble.make_nucleation_plots
Reuses runs/plots/_reduced_cache.npz from make_paper_candidate_plots; the evaluation-day
size distributions are cached in runs/plots/nucleation/_nd_eval_cache.npz.
"""
import glob
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coupled.paper_ensemble.make_paper_candidate_plots import (
    _RUNS, REGIMES, REGIME_LABEL, REGIME_COLOR, NUC_LEVELS, _tokens, reduce_runs, load_summary,
    case_dir, SAI_SUFFIX)

_OUT = os.path.join(_RUNS, "plots", f"nucleation{SAI_SUFFIX}")

plt.rcParams.update({"font.family": "Helvetica", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.35,
                     "grid.linewidth": 0.5, "grid.color": "#e1e0d9", "axes.axisbelow": True,
                     "figure.dpi": 130, "legend.frameon": False})

# ordinal ramp for the ordered nucleation levels (palette steps 250 / 450 / 650)
NUC_COLOR = {0.01: "#86b6ef", 1.0: "#2a78d6", 100.0: "#104281"}
NUC_LAB = {0.01: "×0.01", 1.0: "×1", 100.0: "×100"}

# per-regime evaluation day: last whole day before any run in the regime relaxes to within
# 10% of the background surface area (see module docstring)
EVAL_DAY = {"D1low": 10, "D2med": 10, "D3high": 5, "D5vhigh": 3, "burst": 10}
EVAL_NOTE = "evaluated at day 10 (D1/D2/burst), day 5 (D3), day 3 (D5)"


def eval_index(R):
    """Per-run time index of its regime's evaluation day."""
    t = R["t_days"]
    return np.array([np.searchsorted(t, EVAL_DAY[_tokens(c)[2]] - 1e-9)
                     for c in R["case"]])


def nd_eval(R):
    """Day-of-evaluation dN/dlogDp per run (day-10 rows come from the main cache)."""
    cache = os.path.join(_OUT, "_nd_eval_cache_cesm_amb.npz")
    if os.path.exists(cache):
        return np.load(cache)["nd"]
    ie = eval_index(R)
    nd = R["nd10"].copy()
    for i, cid in enumerate(R["case"]):
        if EVAL_DAY[_tokens(cid)[2]] != 10:
            nd[i] = np.load(os.path.join(case_dir(cid), "state.npz"))["dNdlogDp"][ie[i]]
    np.savez_compressed(cache, nd=nd)
    return nd


def build_triplets(R):
    """(site, bg, regime, alpha, coag) -> {nuc_scale: run_index}; 270 complete triplets."""
    trip = {}
    for i, c in enumerate(R["case"]):
        site, bg, reg, alpha, nuc, cg = _tokens(c)
        trip.setdefault((site, bg, reg, alpha, cg), {})[nuc] = i
    assert len(trip) == 270 and all(len(v) == 3 for v in trip.values())
    return trip


def metrics_table(R, summary):
    """Per-run response metrics for the ratio/elasticity/slopegraph figures."""
    n = np.arange(len(R["case"]))
    ie = eval_index(R)
    return {
        "number at t* [cm$^{-3}$]": R["Ntot"][n, ie],
        "surface area at t* [µm$^2$ cm$^{-3}$]": R["SA"][n, ie],
        "r$_{eff}$ (wet) at t* [µm]": R["reff_um"][n, ie],
        "peak number [cm$^{-3}$]": np.array([float(r["N_max"]) for r in summary]),
        "peak surface area [µm$^2$ cm$^{-3}$]": np.array([float(r["SA_max"]) for r in summary]),
        "peak gas H$_2$SO$_4$ [ppt]": np.array([float(r["H2SO4_max_ppt"]) for r in summary]),
    }


# --------------------------------------------------------------- 1. slopegraph
def fig_slopegraph(R, trip, M):
    keys = ["number at t* [cm$^{-3}$]", "surface area at t* [µm$^2$ cm$^{-3}$]",
            "r$_{eff}$ (wet) at t* [µm]"]
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 4.6))
    for ax, key in zip(axs, keys):
        v = M[key]
        for (site, bg, reg, alpha, cg), idx in trip.items():
            y = [v[idx[s]] for s in NUC_LEVELS]
            ax.plot(NUC_LEVELS, y, color=REGIME_COLOR[reg], lw=0.55, alpha=0.16)
        for reg in REGIMES:                       # bold per-regime median of the triplet lines
            sel = {s: [v[idx[s]] for k, idx in trip.items() if k[2] == reg] for s in NUC_LEVELS}
            ax.plot(NUC_LEVELS, [np.median(sel[s]) for s in NUC_LEVELS],
                    color=REGIME_COLOR[reg], lw=2.4, label=REGIME_LABEL[reg])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(NUC_LEVELS, [NUC_LAB[s] for s in NUC_LEVELS])
        ax.set_xlabel("nucleation rate scale")
        ax.set_title(key, fontsize=10.5)
    axs[0].legend(fontsize=8, loc="lower right", title="dilution", title_fontsize=8)
    fig.suptitle("Nucleation impact within matched triplets (thin: 270 triplets; "
                 f"bold: per-regime median)\nt* = end of plume life: {EVAL_NOTE}",
                 y=1.06, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "slopegraph.png"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------ 2. amplification ratio
def fig_ratios(trip, M):
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 7.0), sharex=True)
    for ax, (key, v) in zip(axs.ravel(), M.items()):
        for k, reg in enumerate(REGIMES):
            r = [v[idx[100.0]] / v[idx[0.01]] for tk, idx in trip.items() if tk[2] == reg]
            bp = ax.boxplot(r, positions=[k], widths=0.6, patch_artist=True, showfliers=False,
                            medianprops={"color": "#0b0b0b"})
            bp["boxes"][0].set(facecolor=REGIME_COLOR[reg], alpha=0.8, lw=0.6)
        ax.axhline(1.0, color="#898781", lw=0.9, ls="--")
        ax.set_yscale("log")
        ax.set_title(key, fontsize=10)
        ax.set_xticks(range(len(REGIMES)),
                      [REGIME_LABEL[r].replace(" ", "\n") for r in REGIMES], fontsize=8.5)
        if ax in axs[:, 0]:
            ax.set_ylabel("metric(×100) / metric(×0.01)")
    fig.suptitle("Nucleation amplification: within-triplet response to ×10$^4$ in nucleation "
                 f"rate (54 triplets per box)\nt* metrics {EVAL_NOTE}", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "amplification_ratios.png"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- 3. elasticity
def fig_elasticity(trip, M):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    names = list(M)
    for m, key in enumerate(names):
        v = M[key]
        for k, reg in enumerate(REGIMES):
            e = np.array([np.log10(v[idx[100.0]] / v[idx[0.01]]) / 4.0
                          for tk, idx in trip.items() if tk[2] == reg])
            lo, med, hi = np.percentile(e, [25, 50, 75])
            y = m + (k - 2) * 0.13
            ax.plot([lo, hi], [y, y], color=REGIME_COLOR[reg], lw=1.6, alpha=0.85)
            ax.plot(med, y, "o", color=REGIME_COLOR[reg], ms=6,
                    label=REGIME_LABEL[reg] if m == 0 else None)
    ax.axvline(0.0, color="#898781", lw=0.9, ls="--")
    ax.axvline(1.0, color="#898781", lw=0.9, ls=":")
    ax.text(0.985, 0.03, "1:1 response ", fontsize=8, color="#898781", ha="right", va="bottom",
            rotation=90, transform=ax.get_xaxis_transform())
    ax.set_xlim(right=1.06)
    ax.set_yticks(range(len(names)), names, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("elasticity  d log(metric) / d log(nucleation scale)")
    ax.legend(fontsize=8.5, title="dilution", title_fontsize=8.5, loc="upper right",
              bbox_to_anchor=(0.985, 0.97))
    ax.set_title("How buffered is each response? (median + IQR over matched triplets)\n"
                 f"t* metrics {EVAL_NOTE}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "elasticity.png"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------ 4. size dist by nuc
def fig_sizedist_by_nuc(R):
    toks = [_tokens(c) for c in R["case"]]
    nd = nd_eval(R)
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 6.6), sharex=True, sharey=True)
    for p, reg in enumerate(REGIMES):
        ax = axs.ravel()[p]
        for s in NUC_LEVELS:
            sel = np.array([tk[2] == reg and tk[4] == s for tk in toks])
            med = np.median(nd[sel], axis=0)
            ax.plot(R["dp_um"], np.maximum(med, 1e-6), color=NUC_COLOR[s], lw=1.9,
                    label=f"nucleation {NUC_LAB[s]}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 3e5)
        ax.set_xlim(2e-3, 20)
        ax.set_title(f"{REGIME_LABEL[reg]} (day {EVAL_DAY[reg]})", fontsize=10.5,
                     color=REGIME_COLOR[reg])
        if p % 3 == 0:
            ax.set_ylabel("dN/dlogD$_p$ [cm$^{-3}$]")
        if p >= 2:
            ax.set_xlabel("dry diameter [µm]")
    axs[0, 0].legend(fontsize=8.5, loc="lower left")
    axs[1, 2].axis("off")
    axs[1, 2].text(0.03, 0.8, "Median end-of-plume-life distribution\nover the 54 matched runs "
                   "per regime ×\nnucleation level (site / background /\nα / coag pooled). "
                   "Each regime is shown\nat its evaluation day (panel title).",
                   fontsize=9.5, va="top", color="#52514e")
    fig.suptitle("Where the extra particles live: end-of-life size distribution vs "
                 "nucleation rate", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "sizedist_by_nuc.png"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------ 5. time-resolved ratio
def fig_time_ratio(R, trip):
    t = R["t_days"]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)
    for ax, (key, lab) in zip(axs, [("Ntot", "total number"), ("SA", "surface area")]):
        for reg in REGIMES:
            ratios = np.array([R[key][idx[100.0]] / R[key][idx[0.01]]
                               for tk, idx in trip.items() if tk[2] == reg])
            lo, med, hi = np.percentile(ratios, [25, 50, 75], axis=0)
            ax.fill_between(t, lo, hi, color=REGIME_COLOR[reg], alpha=0.14, lw=0)
            ax.plot(t, med, color=REGIME_COLOR[reg], lw=1.8, label=REGIME_LABEL[reg])
        ax.axhline(1.0, color="#898781", lw=0.9, ls="--")
        for reg in ("D5vhigh", "D3high"):        # per-regime evaluation days (t*)
            ax.axvline(EVAL_DAY[reg], color=REGIME_COLOR[reg], lw=1.1, ls=":")
            ax.text(EVAL_DAY[reg], 0.985, f" t* {REGIME_LABEL[reg].split()[0]}", fontsize=7.5,
                    color=REGIME_COLOR[reg], ha="left", va="top",
                    transform=ax.get_xaxis_transform())
        ax.set_yscale("log")
        ax.set_xlim(0, 10)
        ax.set_xlabel("day")
        ax.set_ylabel(f"{lab} ratio  (nuc ×100 / ×0.01)")
        ax.set_title(lab, fontsize=11)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncols=5, fontsize=9, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("When nucleation matters: within-pair ratio vs time (median + IQR, "
                 "54 pairs per regime)", y=1.0, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(os.path.join(_OUT, "time_resolved_ratio.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(_OUT, exist_ok=True)
    R = reduce_runs()
    summary = load_summary()
    assert list(R["case"]) == [r["case_id"] for r in summary]
    trip = build_triplets(R)
    M = metrics_table(R, summary)
    fig_slopegraph(R, trip, M)
    print("  slopegraph.png", flush=True)
    fig_ratios(trip, M)
    print("  amplification_ratios.png", flush=True)
    fig_elasticity(trip, M)
    print("  elasticity.png", flush=True)
    fig_sizedist_by_nuc(R)
    print("  sizedist_by_nuc.png", flush=True)
    fig_time_ratio(R, trip)
    print("  time_resolved_ratio.png", flush=True)


if __name__ == "__main__":
    main()
