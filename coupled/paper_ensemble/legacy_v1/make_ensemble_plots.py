# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Cross-run plots for the 810-case SAI ensemble (coupled/paper_ensemble/runs/).

Figures (-> runs/plots/):
  1. sizedist_NAV.png   -- number / area / volume dX/dlogDp snapshots at 2, 5, 10 d (3x3), all 810 as
     faint gray lines + ensemble median (shows the range/uncertainty envelope).
  2. bin_boxplots.png   -- per-size-bin boxplots of dN/dlogDp at 2/5/10 d (range per bin).
  3. OH.png, HO2.png    -- radical concentration vs time, LINEAR y, all 810 faint gray + median + mean.
  4. SO2_sulfur_budget.png -- SO2 as a fraction of total in-box sulfur (gray+median) + median sulfur
     partitioning (SO2 / SO3 / gas-H2SO4 / particulate sulfate) over time.

Number/area/volume use the DRY diameter grid (dp_mid_um): A = pi*Dp^2, V = (pi/6)*Dp^3 per particle.
Run (from SANDBOX/): python -m coupled.paper_ensemble.make_ensemble_plots
"""
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNS = os.path.join(_HERE, "runs")
_OUT = os.path.join(_RUNS, "plots")
_SO2, _SO3, _H2SO4, _OH, _HO2 = 32, 34, 35, 18, 19
_DAY_IDX = {2: 292, 5: 730, 10: 1460}
_GRAY = dict(color="0.35", lw=0.5, alpha=0.04, rasterized=True)

plt.rcParams.update({"font.family": "Helvetica", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
                     "grid.linewidth": 0.5, "axes.axisbelow": True, "figure.dpi": 130})


_BG_SO2 = {"sabr330": 20.0, "sabr220": 20.0, "cesm": 100.0}   # background SO2 [pptv] by background


def load_all():
    dirs = sorted(d for d in glob.glob(os.path.join(_RUNS, "*")) if os.path.isdir(d)
                  and os.path.exists(os.path.join(d, "state.npz")))
    n = len(dirs)
    d0 = np.load(os.path.join(dirs[0], "state.npz"), allow_pickle=True)
    t_days = d0["t"] / 86400.0
    dp = d0["dp_mid_um"]; nb = len(dp)
    ND = np.full((n, 3, nb), np.nan)          # dN/dlogDp at 2/5/10 d
    OH = np.full((n, len(t_days)), np.nan); HO2 = OH.copy()
    S = {k: OH.copy() for k in ("SO2", "SO3", "H2SO4", "sulf")}
    V = OH.copy()                             # V(t)/V0
    regime = []; bg_so2 = np.full(n, np.nan)  # dilution-regime token + background SO2 [molec/cm^3]
    for i, dd in enumerate(dirs):
        z = np.load(os.path.join(dd, "state.npz"), allow_pickle=True)
        dnd = z["dNdlogDp"]
        for j, day in enumerate((2, 5, 10)):
            ND[i, j] = dnd[_DAY_IDX[day]]
        x = z["x"]
        OH[i] = x[:, _OH]; HO2[i] = x[:, _HO2]
        S["SO2"][i] = x[:, _SO2]; S["SO3"][i] = x[:, _SO3]; S["H2SO4"][i] = x[:, _H2SO4]
        S["sulf"][i] = z["particulate_S"]; V[i] = z["V_ratio"]
        toks = os.path.basename(dd).split("__"); regime.append(toks[2])
        bg_so2[i] = _BG_SO2[toks[1]] * float(z["M"]) / 1e12
    return dict(dirs=dirs, n=n, t=t_days, dp=dp, ND=ND, OH=OH, HO2=HO2, S=S,
                V=V, regime=np.array(regime), bg_so2=bg_so2)


def fig_so2_consumed(D):
    """Fraction of INJECTED SO2 chemically oxidized vs time (dilution removed via plume-integrated
    excess (SO2-SO2_bg)*V(t), which dilution conserves). Noise-robust: hold the value once the plume
    is no longer resolvable above background (SO2 <= 10*bg), and enforce monotonicity. Colored by
    dilution regime."""
    t = D["t"]; so2 = D["S"]["SO2"]; V = D["V"]; bg = D["bg_so2"][:, None]
    excessV = (so2 - bg) * V
    cons = 1.0 - excessV / excessV[:, :1]
    resolvable = so2 > 10 * bg
    C = np.empty_like(cons)
    for i in range(cons.shape[0]):
        c = cons[i].copy(); m = resolvable[i]
        if m.any():                          # forward-fill after the plume disperses
            last = np.where(m)[0][-1]; c[last + 1:] = c[last]
        C[i] = np.clip(np.maximum.accumulate(np.clip(c, 0, 1)), 0, 1)   # monotone, [0,1]
    order = ["D1low", "D2med", "burst", "D3high", "D5vhigh"]
    labels = {"D1low": "D1 Low Kz", "D2med": "D2 Med Kz", "burst": "Burst",
              "D3high": "D3 High Kz", "D5vhigh": "D5 Very high"}
    cols = {"D1low": "#2c7fb8", "D2med": "#31a354", "burst": "#756bb1",
            "D3high": "#e6550d", "D5vhigh": "#c0392b"}
    fig, ax = plt.subplots(figsize=(10, 6))
    for reg in order:
        idx = np.where(D["regime"] == reg)[0]
        for i in idx:
            ax.plot(t, C[i] * 100, color=cols[reg], lw=0.5, alpha=0.05, rasterized=True)
    for reg in order:
        idx = np.where(D["regime"] == reg)[0]
        med = np.median(C[idx] * 100, axis=0)
        ax.plot(t, med, color=cols[reg], lw=2.5, label=f"{labels[reg]} (median {med[-1]:.0f}%)", zorder=5)
    ax.plot(t, np.median(C * 100, axis=0), color="k", lw=2.0, ls="--",
            label=f"all (median {np.median(C[:, -1]*100):.0f}%)", zorder=6)
    ax.set_xlabel("time (days)"); ax.set_ylabel("% of injected SO$_2$ chemically oxidized")
    ax.set_title("Injected SO$_2$ consumed by chemistry (dilution removed) — 810 runs by dilution regime")
    ax.legend(frameon=False, fontsize=9, loc="upper left"); ax.set_ylim(0, None)
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "SO2_consumed_fraction.png"), bbox_inches="tight")
    plt.close(fig)


def _overlay(ax, t, Y, ylabel, title, logy=False):
    for row in Y:
        ax.plot(t, row, **_GRAY)
    med = np.nanmedian(Y, axis=0); mean = np.nanmean(Y, axis=0)
    ax.plot(t, med, color="#c0392b", lw=2.0, label="median", zorder=5)
    ax.plot(t, mean, color="#2c7fb8", lw=1.5, ls="--", label="mean", zorder=5)
    ax.set_xlabel("time (days)"); ax.set_ylabel(ylabel); ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9)


def fig_sizedist(D, logy=False):
    dp = D["dp"]; ND = D["ND"]
    area = ND * (np.pi * dp ** 2)[None, None, :]          # dA/dlogDp [um^2 cm^-3]
    vol = ND * (np.pi / 6.0 * dp ** 3)[None, None, :]     # dV/dlogDp [um^3 cm^-3]
    moments = [("Number  dN/dlogDp [cm$^{-3}$]", ND, 1e0),
               ("Area  dA/dlogDp [$\\mu$m$^2$ cm$^{-3}$]", area, 1e-3),
               ("Volume  dV/dlogDp [$\\mu$m$^3$ cm$^{-3}$]", vol, 1e-4)]
    fig, axs = plt.subplots(3, 3, figsize=(14, 11), sharex=True)
    for r, day in enumerate((2, 5, 10)):
        for c, (lab, M, floor) in enumerate(moments):
            ax = axs[r, c]
            for row in M[:, r, :]:
                ax.plot(dp, row, **_GRAY)
            ax.plot(dp, np.nanmedian(M[:, r, :], axis=0), color="#c0392b", lw=2.0, zorder=5)
            ax.set_xscale("log"); ax.set_xlim(2e-3, 3)
            if logy:
                ax.set_yscale("log")
                ax.set_ylim(floor, np.nanmax(M[:, r, :]) * 2)
            if r == 0:
                ax.set_title(lab, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"day {day}", fontweight="bold")
            if r == 2:
                ax.set_xlabel("dry diameter ($\\mu$m)")
    scale = "log-y" if logy else "linear-y"
    fig.suptitle(f"Ensemble size distributions (810 runs, gray) + median (red) — N / A / V at "
                 f"2, 5, 10 d [{scale}]", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, f"sizedist_NAV{'_logy' if logy else ''}.png"), bbox_inches="tight")
    plt.close(fig)


def fig_boxplots(D):
    dp = D["dp"]; ND = D["ND"]; nb = len(dp)
    # limit to the populated size range for legibility
    m = (dp >= 3e-3) & (dp <= 2.0); idx = np.where(m)[0]
    fig, axs = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    for r, day in enumerate((2, 5, 10)):
        data = [ND[:, r, k] for k in idx]
        axs[r].boxplot(data, positions=np.arange(len(idx)), widths=0.6, showfliers=False,
                       medianprops=dict(color="#c0392b"), flierprops=dict(marker="."))
        axs[r].set_ylabel(f"day {day}\ndN/dlogDp [cm$^{{-3}}$]")
        axs[r].set_yscale("log")
    ticks = np.arange(len(idx))[::4]
    axs[2].set_xticks(ticks); axs[2].set_xticklabels([f"{dp[idx][i]:.3g}" for i in ticks], rotation=45)
    axs[2].set_xlabel("dry diameter ($\\mu$m)")
    fig.suptitle("Per-bin dN/dlogDp across 810 runs (boxplots) at 2, 5, 10 d", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "bin_boxplots.png"), bbox_inches="tight")
    plt.close(fig)


def fig_radical(D, key, label):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _overlay(ax, D["t"], D[key], f"{label} [molec cm$^{{-3}}$]",
             f"{label} — 810 runs (gray) with median & mean (linear axis)")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, f"{key}.png"), bbox_inches="tight")
    plt.close(fig)


_REG_ORDER = ["D1low", "D2med", "burst", "D3high", "D5vhigh"]
_REG_LABEL = {"D1low": "D1 Low Kz", "D2med": "D2 Med Kz", "burst": "Burst",
              "D3high": "D3 High Kz", "D5vhigh": "D5 Very high"}
_REG_COL = {"D1low": "#2c7fb8", "D2med": "#31a354", "burst": "#756bb1",
            "D3high": "#e6550d", "D5vhigh": "#c0392b"}


def fig_radical_by_regime(D, key, label):
    """Radical (OH/HO2) concentration vs time, 810 runs faint, COLORED BY DILUTION REGIME, with
    per-regime medians. Linear y with scientific (power) tick format."""
    import matplotlib.ticker as mticker
    t = D["t"]; Y = D[key]; reg = D["regime"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in _REG_ORDER:
        for i in np.where(reg == r)[0]:
            ax.plot(t, Y[i], color=_REG_COL[r], lw=0.4, alpha=0.04, rasterized=True)
    for r in _REG_ORDER:
        idx = np.where(reg == r)[0]
        ax.plot(t, np.nanmedian(Y[idx], axis=0), color=_REG_COL[r], lw=2.5,
                label=_REG_LABEL[r], zorder=5)
    ax.set_xlabel("time (days)"); ax.set_ylabel(f"{label} [molec cm$^{{-3}}$]")
    ax.set_title(f"{label} vs time by dilution regime (810 runs faint + per-regime median)")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
    ax.set_ylim(0, None); ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, f"{key}_by_regime.png"), bbox_inches="tight")
    plt.close(fig)


def fig_sulfur(D):
    S = D["S"]; t = D["t"]
    tot = S["SO2"] + S["SO3"] + S["H2SO4"] + S["sulf"]      # total in-box sulfur [molec/cm^3]
    tot = np.where(tot > 0, tot, np.nan)
    so2_frac = S["SO2"] / tot
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    # (a) SO2 as fraction of total sulfur, overlay
    _overlay(axs[0], t, so2_frac, "SO$_2$ / total sulfur", "SO$_2$ fraction of total S (810 runs)")
    axs[0].set_ylim(0, 1.02)
    # (b) median partitioning of sulfur reservoirs
    labs = [("SO2", "SO$_2$", "#c0392b"), ("SO3", "SO$_3$", "#e08214"),
            ("H2SO4", "gas H$_2$SO$_4$", "#2c7fb8"), ("sulf", "particulate sulfate", "#31a354")]
    for k, lab, col in labs:
        frac = S[k] / tot
        axs[1].plot(t, np.nanmedian(frac, axis=0), color=col, lw=2.0, label=lab)
        axs[1].fill_between(t, np.nanpercentile(frac, 10, axis=0), np.nanpercentile(frac, 90, axis=0),
                            color=col, alpha=0.15)
    axs[1].set_xlabel("time (days)"); axs[1].set_ylabel("fraction of total sulfur")
    axs[1].set_title("Sulfur partitioning (median + 10-90%)"); axs[1].set_ylim(0, 1.02)
    axs[1].legend(frameon=False, fontsize=9)
    fig.suptitle("Sulfur budget across the ensemble", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "SO2_sulfur_budget.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(_OUT, exist_ok=True)
    print("loading 810 runs ...", flush=True)
    D = load_all()
    print(f"loaded {D['n']} runs; plotting ...", flush=True)
    fig_sizedist(D); print("  sizedist_NAV.png", flush=True)
    fig_boxplots(D); print("  bin_boxplots.png", flush=True)
    fig_radical(D, "OH", "OH"); print("  OH.png", flush=True)
    fig_radical(D, "HO2", "HO$_2$"); print("  HO2.png", flush=True)
    fig_sulfur(D); print("  SO2_sulfur_budget.png", flush=True)
    print("done ->", _OUT, flush=True)


if __name__ == "__main__":
    main()
