# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Five paper-candidate figures from the 810-case SAI ensemble (-> runs/plots/).

  1. tornado_sensitivity.png   -- main-effect range (in decades) of each design axis on the five
     summary metrics (median over the other axes).
  2. sizedist_day10_grid.png   -- day-10 dN/dlogDp, dilution regime (rows) x background (cols);
     54 runs per panel as faint lines + panel median.
  3. timeseries_by_regime.png  -- total N / surface area / effective radius vs time, per-regime
     median + IQR, one column per site.
  4. so2_conversion.png        -- SO2 fraction of in-box sulfur vs time (median per regime) and
     the day-10 converted fraction by regime x site.
  5. npf_outcome_map.png       -- N_max and day-10 N vs nucleation scale, colored by dilution
     regime, one column per background.

Run (from SANDBOX/): python -m coupled.paper_ensemble.make_paper_candidate_plots
The reduction over the 810 state.npz files is cached in runs/plots/_reduced_cache.npz.
"""
import csv
import glob
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.figure
import matplotlib.pyplot as plt

# Save every .png figure as a vector .pdf twin as well. Patched here because all the
# paper-ensemble figure modules import this one.
_savefig_orig = matplotlib.figure.Figure.savefig


def _savefig_both(self, fname, *args, **kwargs):
    res = _savefig_orig(self, fname, *args, **kwargs)
    if isinstance(fname, (str, os.PathLike)) and str(fname).endswith(".png"):
        _savefig_orig(self, str(fname)[:-4] + ".pdf", *args, **kwargs)
    return res


matplotlib.figure.Figure.savefig = _savefig_both

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNS = os.path.join(_HERE, "runs")
_RUNS_GEO = os.path.join(_HERE, "runs_geo")
_OUT = os.path.join(_RUNS, "plots")

# Which geoengineering background plays the "SAI" role in the figures:
#   'cesm'   -> cesm_g6_amb (ambient CESM re-runs, runs_geo/)          [default]
#   'aergeo' -> aer_geo (Pierce AER-2D geoengineered stratosphere, runs_geo/)
# The aergeo variant writes to suffixed plot dirs (e.g. plots/nucleation_aergeo/).
SAI_BG = os.environ.get("PAPER_SAI_BG", "cesm")
assert SAI_BG in ("cesm", "aergeo")
SAI_SUFFIX = "" if SAI_BG == "cesm" else "_aergeo"
SAI_LABEL = ("CESM-G6 (SAI, ambient) background" if SAI_BG == "cesm"
             else "AER-2D geoengineered background (Pierce)")
SAI_SHORT = "CESM-G6 SAI" if SAI_BG == "cesm" else "AER-2D geo"
# cache bumped when the cesm source switched to the ambient re-runs (runs_geo)
_CACHE = os.path.join(_OUT, f"_reduced_cache_cesm_amb{SAI_SUFFIX}.npz")


def case_dir(cid):
    """Source directory for a case. Geoengineered backgrounds (cesm = ambient CESM re-runs,
    aergeo = Pierce AER-2D) come from runs_geo/; the original runs/ cesm cases used the
    erroneous STP->ambient factor and are kept on disk for reproducibility, not used in
    figures."""
    root = _RUNS_GEO if cid.split("__")[1] in ("cesm", "aergeo") else _RUNS
    return os.path.join(root, cid)
_SO2, _SO3, _H2SO4 = 32, 34, 35

plt.rcParams.update({"font.family": "Helvetica", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.35,
                     "grid.linewidth": 0.5, "grid.color": "#e1e0d9", "axes.axisbelow": True,
                     "figure.dpi": 130, "legend.frameon": False})

# dilution regimes are ordered (ordinal blue ramp) + the qualitatively different "burst" (red)
REGIMES = ["D1low", "D2med", "D3high", "D5vhigh", "burst"]
REGIME_LABEL = {"D1low": "D1 low", "D2med": "D2 med", "D3high": "D3 high",
                "D5vhigh": "D5 very high", "burst": "burst"}
REGIME_COLOR = {"D1low": "#86b6ef", "D2med": "#3987e5", "D3high": "#1c5cab",
                "D5vhigh": "#0d366b", "burst": "#e34948"}
SITES = ["30N_20km", "30N_20km_213K", "60N_15km"]
SITE_LABEL = {"30N_20km": "30°N, 20 km, 210 K", "30N_20km_213K": "30°N, 20 km, 213 K",
              "60N_15km": "60°N, 15 km"}
BGS = ["sabr330", "sabr220", "cesm"]
BG_LABEL = {"sabr330": "SABR-330", "sabr220": "SABR-220", "cesm": "CESM-G6"}
CAT3 = ["#2a78d6", "#1baf7a", "#eda100"]          # validated 3-slot categorical (sites / backgrounds)
NUC_LEVELS = [0.01, 1.0, 100.0]


def _tokens(case_id):
    """case_id -> (site, bg, regime, alpha, nuc, coag)."""
    parts = case_id.split("__")
    site, bg, regime = parts[0], parts[1], parts[2]
    alpha = {"a0p5": 0.5, "a1p0": 1.0}[parts[3]]
    nuc = {"nuc0p01": 0.01, "nuc1": 1.0, "nuc100": 100.0}[parts[4]]
    coag = {"cg0p5": 0.5, "cg1": 1.0, "cg2": 2.0}[parts[5]]
    return site, bg, regime, alpha, nuc, coag


def reduce_runs():
    if os.path.exists(_CACHE):
        z = np.load(_CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}
    cids = sorted(os.path.basename(d) for d in glob.glob(os.path.join(_RUNS, "*"))
                  if os.path.isdir(d) and os.path.exists(os.path.join(d, "state.npz")))
    if SAI_BG != "cesm":                         # swap the SAI third of the pool to aer_geo
        cids = sorted(c.replace("__cesm__", f"__{SAI_BG}__") for c in cids)
    dirs = [case_dir(c) for c in cids]           # geoengineered cases live in runs_geo/
    assert all(os.path.exists(os.path.join(d, "state.npz")) for d in dirs)
    n = len(dirs)
    d0 = np.load(os.path.join(dirs[0], "state.npz"))
    t_days = d0["t"] / 86400.0
    dp = d0["dp_mid_um"]
    nt, nb = len(t_days), len(dp)
    R = {"t_days": t_days, "dp_um": dp,
         "case": np.array([os.path.basename(d) for d in dirs]),
         "Ntot": np.full((n, nt), np.nan), "SA": np.full((n, nt), np.nan),
         "reff_um": np.full((n, nt), np.nan), "fSO2": np.full((n, nt), np.nan),
         "nd10": np.full((n, nb), np.nan)}
    for i, dd in enumerate(dirs):
        z = np.load(os.path.join(dd, "state.npz"))
        x = z["x"]
        stot = x[:, _SO2] + x[:, _SO3] + x[:, _H2SO4] + z["particulate_S"]
        R["fSO2"][i] = x[:, _SO2] / stot
        R["Ntot"][i] = z["total_n"]
        R["SA"][i] = z["SA"]
        R["reff_um"][i] = z["radius_cm"] * 1e4
        R["nd10"][i] = z["dNdlogDp"][-1]
        if (i + 1) % 100 == 0:
            print(f"  reduced {i + 1}/{n}", flush=True)
    os.makedirs(_OUT, exist_ok=True)
    np.savez_compressed(_CACHE, **R)
    return R


def load_summary():
    rows = list(csv.DictReader(open(os.path.join(_RUNS, "summary.csv"))))
    # geoengineered-background metrics come from runs_geo/summary.csv (cesm rows are the
    # ambient re-runs; in the aergeo variant the cesm slot is swapped to the aer_geo cases)
    geo = {r["case_id"]: r for r in csv.DictReader(open(os.path.join(_RUNS_GEO, "summary.csv")))}
    out = []
    for r in rows:
        if r["case_id"].split("__")[1] == "cesm":
            gid = r["case_id"].replace("__cesm__", f"__{SAI_BG}__")
            assert gid in geo, f"missing geo run: {gid}"
            r = geo[gid]
        out.append(r)
    out.sort(key=lambda r: r["case_id"])
    return out


# ------------------------------------------------------------------ 1. tornado
METRICS = [("SO2_end_ppt", "SO$_2$ at day 10 [ppt]"),
           ("H2SO4_max_ppt", "peak gas H$_2$SO$_4$ [ppt]"),
           ("N_max", "peak number [cm$^{-3}$]"),
           ("SA_max", "peak surface area [µm$^2$ cm$^{-3}$]"),
           ("sulfate_end", "day-10 particulate S [cm$^{-3}$]")]
AXES = [("site", "site (lat/alt/T)"), ("bg", "background aerosol"), ("regime", "dilution"),
        ("alpha", "condensation α"), ("nuc", "nucleation ×"), ("coag", "coagulation ×")]


def fig_tornado(summary):
    toks = [_tokens(r["case_id"]) for r in summary]
    keys = {"site": [t[0] for t in toks], "bg": [t[1] for t in toks],
            "regime": [t[2] for t in toks], "alpha": [t[3] for t in toks],
            "nuc": [t[4] for t in toks], "coag": [t[5] for t in toks]}
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 6.2))
    for (mk, mlab), ax in zip(METRICS, axs.ravel()):
        v = np.log10(np.maximum([float(r[mk]) for r in summary], 1e-30))
        effects = []
        for ak, alab in AXES:
            g = np.array(keys[ak])
            med = [np.median(v[g == lev]) for lev in np.unique(g)]
            effects.append((alab, max(med) - min(med)))
        effects.sort(key=lambda e: e[1])
        names, spans = zip(*effects)
        ax.barh(range(len(spans)), spans, color="#2a78d6", height=0.62)
        for j, s in enumerate(spans):
            ax.text(s + 0.02 * max(spans), j, f"×{10 ** s:.2g}", va="center", fontsize=8.5,
                    color="#52514e")
        ax.set_yticks(range(len(names)), names, fontsize=9)
        ax.set_xlim(0, max(spans) * 1.22)
        ax.set_title(mlab, fontsize=10.5)
        ax.set_xlabel("main-effect range [decades]", fontsize=9)
    axs[1, 2].axis("off")
    axs[1, 2].text(0.02, 0.85, "Main effect per axis:\nspread of the per-level medians\n"
                   "of log$_{10}$(metric), all other axes\npooled (810 runs). ×N = fold\n"
                   "change between extreme levels.", fontsize=9.5, va="top", color="#52514e")
    fig.suptitle("Which design axis moves each headline metric", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "tornado_sensitivity.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------- 2. day-10 size dist grid
def fig_sizedist_grid(R):
    case = R["case"]
    toks = [_tokens(c) for c in case]
    fig, axs = plt.subplots(len(REGIMES), len(BGS), figsize=(9.5, 11.5),
                            sharex=True, sharey=True)
    for i, reg in enumerate(REGIMES):
        for j, bg in enumerate(BGS):
            ax = axs[i, j]
            sel = np.array([t[2] == reg and t[1] == bg for t in toks])
            Y = R["nd10"][sel]
            for y in Y:
                ax.plot(R["dp_um"], np.maximum(y, 1e-6), color="0.35", lw=0.5, alpha=0.10,
                        rasterized=True)
            ax.plot(R["dp_um"], np.maximum(np.median(Y, axis=0), 1e-6),
                    color=REGIME_COLOR[reg], lw=1.9)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(1e-4, 3e5)
            ax.set_xlim(2e-3, 20)
            if i == 0:
                ax.set_title(BG_LABEL[bg], fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{REGIME_LABEL[reg]}\ndN/dlogD$_p$ [cm$^{{-3}}$]", fontsize=9.5)
            if j == 2:
                ax.text(0.96, 0.88, f"n={sel.sum()}", transform=ax.transAxes, ha="right",
                        fontsize=8, color="#898781")
            if i == len(REGIMES) - 1:
                ax.set_xlabel("dry diameter [µm]", fontsize=10)
    fig.suptitle("Day-10 size distribution — dilution regime × background "
                 "(faint: individual runs; bold: median)", y=0.995, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "sizedist_day10_grid.png"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------ 3. N / SA / r_eff time series
def fig_timeseries(R):
    toks = [_tokens(c) for c in R["case"]]
    t = R["t_days"]
    rows = [("Ntot", "total number [cm$^{-3}$]", True),
            ("SA", "surface area [µm$^2$ cm$^{-3}$]", True),
            ("reff_um", "effective radius (wet) [µm]", False)]
    fig, axs = plt.subplots(3, 3, figsize=(11.5, 8.6), sharex=True, sharey="row")
    for j, site in enumerate(SITES):
        for i, (key, ylab, logy) in enumerate(rows):
            ax = axs[i, j]
            for reg in REGIMES:
                sel = np.array([tk[2] == reg and tk[0] == site for tk in toks])
                Y = R[key][sel]
                lo, med, hi = np.percentile(Y, [25, 50, 75], axis=0)
                ax.fill_between(t, lo, hi, color=REGIME_COLOR[reg], alpha=0.16, lw=0)
                ax.plot(t, med, color=REGIME_COLOR[reg], lw=1.7, label=REGIME_LABEL[reg])
            if logy:
                ax.set_yscale("log")
            if i == 0:
                ax.set_title(SITE_LABEL[site], fontsize=11)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=10)
            if i == 2:
                ax.set_xlabel("day", fontsize=10)
            ax.set_xlim(0, 10)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncols=5, fontsize=9, bbox_to_anchor=(0.5, 0.965),
               title="dilution", title_fontsize=9)
    fig.suptitle("Aerosol evolution by dilution regime (median + IQR over the 54 runs "
                 "per regime × site)", y=1.0, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(_OUT, "timeseries_by_regime.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------- 4. SO2 conversion
def fig_so2_conversion(R):
    toks = [_tokens(c) for c in R["case"]]
    t = R["t_days"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4), width_ratios=[1.15, 1])
    for reg in REGIMES:
        sel = np.array([tk[2] == reg for tk in toks])
        lo, med, hi = np.percentile(R["fSO2"][sel], [25, 50, 75], axis=0)
        a1.fill_between(t, lo, hi, color=REGIME_COLOR[reg], alpha=0.15, lw=0)
        a1.plot(t, med, color=REGIME_COLOR[reg], lw=1.8, label=REGIME_LABEL[reg])
    a1.set_xlabel("day")
    a1.set_ylabel("SO$_2$ / total in-box sulfur")
    a1.set_xlim(0, 10)
    a1.set_ylim(0, 1.02)
    a1.legend(fontsize=8.5, title="dilution", title_fontsize=8.5)
    a1.set_title("SO$_2$ fraction of in-box sulfur (median + IQR)", fontsize=11)

    conv10 = 1.0 - R["fSO2"][:, -1]
    pos, width = np.arange(len(REGIMES), dtype=float), 0.26
    for k, site in enumerate(SITES):
        data = [conv10[np.array([tk[2] == reg and tk[0] == site for tk in toks])]
                for reg in REGIMES]
        bp = a2.boxplot(data, positions=pos + (k - 1) * width, widths=width * 0.86,
                        patch_artist=True, showfliers=False, medianprops={"color": "#0b0b0b"})
        for b in bp["boxes"]:
            b.set(facecolor=CAT3[k], alpha=0.75, lw=0.6)
    a2.set_xticks(pos, [REGIME_LABEL[r] for r in REGIMES], fontsize=9)
    a2.set_ylabel("converted fraction at day 10")
    a2.set_title("1 − SO$_2$/S$_{tot}$ at day 10, by site", fontsize=11)
    a2.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=CAT3[k], alpha=0.75)
                       for k in range(3)],
              labels=[SITE_LABEL[s] for s in SITES], fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "so2_conversion.png"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- 5. NPF outcome
def fig_npf_map(R, summary):
    toks = [_tokens(c) for c in R["case"]]
    nmax = np.array([float(r["N_max"]) for r in summary])
    n10 = R["Ntot"][:, -1]
    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 7.2), sharex=True, sharey="row")
    for j, bg in enumerate(BGS):
        for i, (vals, ylab) in enumerate([(nmax, "peak number [cm$^{-3}$]"),
                                          (n10, "day-10 number [cm$^{-3}$]")]):
            ax = axs[i, j]
            for reg in REGIMES:
                sel = np.array([tk[2] == reg and tk[1] == bg for tk in toks])
                x = np.array([tk[4] for tk in toks])[sel]
                jit = 10 ** (rng.uniform(-0.16, 0.16, size=sel.sum()))
                ax.scatter(x * jit, vals[sel], s=14, color=REGIME_COLOR[reg], alpha=0.75,
                           edgecolors="white", linewidths=0.4, label=REGIME_LABEL[reg])
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(NUC_LEVELS, ["×0.01", "×1", "×100"])
            if i == 0:
                ax.set_title(BG_LABEL[bg], fontsize=11)
            if i == 1:
                ax.set_xlabel("nucleation rate scale", fontsize=10)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=10)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h[:len(REGIMES)], l[:len(REGIMES)], loc="upper center", ncols=5, fontsize=9,
               bbox_to_anchor=(0.5, 0.955), title="dilution", title_fontsize=9)
    fig.suptitle("New-particle-formation outcome vs nucleation rate scale", y=1.0, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(os.path.join(_OUT, "npf_outcome_map.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(_OUT, exist_ok=True)
    print("reducing 810 runs (cached after first pass)...", flush=True)
    R = reduce_runs()
    summary = load_summary()
    assert list(R["case"]) == [r["case_id"] for r in summary], "summary/case order mismatch"
    fig_tornado(summary)
    print("  tornado_sensitivity.png", flush=True)
    fig_sizedist_grid(R)
    print("  sizedist_day10_grid.png", flush=True)
    fig_timeseries(R)
    print("  timeseries_by_regime.png", flush=True)
    fig_so2_conversion(R)
    print("  so2_conversion.png", flush=True)
    fig_npf_map(R, summary)
    print("  npf_outcome_map.png", flush=True)


if __name__ == "__main__":
    main()
