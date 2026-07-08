# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""dRF/dSO2 for every ensemble run from its ACTUAL day-10 size distribution
(-> runs/plots/rf/).

Method (see FIGURES.md):
  * Plume-attributable particle count per bin: dN_k = [n_k(t*) - n_k(0)] x V(t*)/V0 x V0
    (plume-integrated EXCESS over background; negative bins = background particles consumed
    by the plume, a real forcing debit). Evaluated at the per-regime t* (day 10 D1/D2/burst,
    day 5 D3, day 3 D5): at day 10 the fast regimes have V/V0 up to 3e21, which amplifies
    per-cm^3 residuals (e.g. diurnal nucleation absent from the static t=0 background
    reference) into nonsense -- the excess method is only meaningful at moderate dilution.
  * Per-bin optics at the run's own WET bin diameters Dp_m(10 d): Bohren-Huffman Qsca and
    asymmetry g, solar-spectrum weighted (300-2500 nm); Wiscombe-Grams global-average
    upscatter beta(g).
  * Chylek & Wong (1995) / Pierce et al. (2010) scene: S0 = 1370, Tatm = 1, albedo 0.15,
    cloud fraction 0.6. Each run injected exactly 1 t SO2 -> RF per Mt-S = plume RF with
    1e6 such plumes spread over the Earth's area, x2 (SO2->S mass).
  * Reported as POSITIVE cooling [W m^-2 (Mt-S)^-1], comparable to the monodisperse
    optimum 0.657 in rf_efficiency_vs_size.png.

Figures:
  drf_boxplot.png       -- dRF/dS by dilution regime (all 810 runs) + pooled.
  drf_by_nucleation.png -- dRF/dS vs nucleation scale, regime-colored boxes.

Run (from SANDBOX/): python -m coupled.paper_ensemble.make_rf_runs [replot]
Per-run values cached in runs/plots/rf/_drf_cache.npz.
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coupled.paper_ensemble.make_paper_candidate_plots import (
    _RUNS, REGIMES, REGIME_LABEL, REGIME_COLOR, NUC_LEVELS, _tokens, reduce_runs, case_dir)
from coupled.paper_ensemble.make_nucleation_plots import EVAL_DAY, EVAL_NOTE

_OUT = os.path.join(_RUNS, "plots", "rf")
_CACHE = os.path.join(_OUT, "_drf_cache.npz")
V0_CM3 = 1.5e12
A_EARTH = 5.101e14
PREF = (1370.0 / 4.0) * 1.0 * (1.0 - 0.6) * (1.0 - 0.15) ** 2   # Chylek-Wong scene prefactor

plt.rcParams.update({"font.family": "Helvetica", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
                     "grid.linewidth": 0.5, "grid.color": "#e1e0d9", "axes.axisbelow": True,
                     "figure.dpi": 130, "legend.frameon": False})


def compute_all():
    if os.path.exists(_CACHE):
        z = np.load(_CACHE, allow_pickle=True)
        return list(z["case"]), z["drf"]
    import jax
    import jax.numpy as jnp
    from coupled import tomas_bridge  # noqa: F401
    from tomas_jax.physics.bhmie import bhmie_qsca_jax
    from tomas_jax.physics.radiative_forcing import (_solar_spectral_weights,
                                                     _compute_global_avg_upscatter,
                                                     REFINDEX_SULFATE)
    R = reduce_runs()                             # for the authoritative case list/order
    cases = list(R["case"])
    wls, wts = _solar_spectral_weights(n_wl=60)
    wls = np.asarray(wls)
    w = np.asarray(wts)

    @jax.jit
    def bin_optics(dp_wet):
        """Solar-weighted Qsca and scattering-weighted mean g per bin."""
        def per_wl(wl):
            x = jnp.pi * dp_wet / wl
            _, qs, g = jax.vmap(bhmie_qsca_jax, (0, None))(x, REFINDEX_SULFATE)
            return qs, g
        qs, g = jax.vmap(per_wl)(jnp.asarray(wls))           # (nwl, nbins)
        wj = jnp.asarray(w)[:, None]
        qsca = jnp.sum(qs * wj, axis=0) / jnp.sum(wj)
        gbar = jnp.sum(g * qs * wj, axis=0) / jnp.maximum(jnp.sum(qs * wj, axis=0), 1e-30)
        return qsca, gbar

    drf = np.full(len(cases), np.nan)
    for i, cid in enumerate(cases):
        z = np.load(os.path.join(case_dir(cid), "state.npz"))
        it = int(np.searchsorted(z["t"], EVAL_DAY[_tokens(cid)[2]] * 86400.0 - 1e-6))
        dN = (z["n_cm3"][it] - z["n_cm3"][0]) * float(z["V_ratio"][it]) * V0_CM3
        dp_wet = jnp.asarray(z["Dp_m"][it])
        qsca, gbar = bin_optics(dp_wet)
        beta = np.asarray(_compute_global_avg_upscatter(gbar))
        sigma = np.asarray(qsca) * np.pi * (np.asarray(dp_wet) / 2.0) ** 2   # m^2/particle
        tau = dN * sigma / A_EARTH * 1.0e6 * 2.0             # per Mt-S (1 t SO2 = 0.5 t S)
        drf[i] = np.sum(PREF * 2.0 * beta * tau)             # POSITIVE cooling
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(cases)}", flush=True)
    np.savez_compressed(_CACHE, case=np.array(cases), drf=drf)
    return cases, drf


def figs(cases, drf):
    toks = [_tokens(c) for c in cases]

    rng = np.random.default_rng(0)

    # 1) by dilution regime + pooled
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    data = [drf[[t[2] == reg for t in toks]] for reg in REGIMES] + [drf]
    bp = ax.boxplot(data, positions=range(len(REGIMES) + 1), widths=0.62, patch_artist=True,
                    showfliers=False, medianprops={"color": "#0b0b0b"})
    for b, col in zip(bp["boxes"], [REGIME_COLOR[r] for r in REGIMES] + ["#898781"]):
        b.set(facecolor=col, alpha=0.35, lw=0.6)
    for i, (v, col) in enumerate(zip(data, [REGIME_COLOR[r] for r in REGIMES] + ["#898781"])):
        ax.scatter(i + rng.uniform(-0.22, 0.22, len(v)), v, s=5, color=col, alpha=0.55,
                   edgecolors="none", zorder=3)
    ax.axhline(0.657, color="#898781", lw=1.0, ls="--")
    ax.text(0.02, 0.657, "monodisperse optimum (0.22 µm radius) ", fontsize=8,
            color="#52514e", va="bottom", transform=ax.get_yaxis_transform())
    ax.set_xticks(range(len(REGIMES) + 1),
                  [REGIME_LABEL[r].replace(" ", "\n") for r in REGIMES] + ["ALL\n810"])
    ax.set_ylabel("dRF/dS at t* [W m$^{-2}$ (Mt-S)$^{-1}$, cooling positive]")
    ax.set_title("Forcing per unit sulfur from the ACTUAL end-of-plume-life distributions — all 810 runs\n"
                 f"plume-integrated excess over background at t* ({EVAL_NOTE.split(': ')[-1]}); "
                 "Chylek & Wong scene as rf_efficiency_vs_size", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "drf_boxplot.png"), bbox_inches="tight")
    plt.close(fig)
    print("  drf_boxplot.png")

    # 2) by microphysics axis (nucleation / coagulation / condensation), regime-colored
    AXES = [(4, NUC_LEVELS, "nucleation", "drf_by_nucleation.png"),
            (5, [0.5, 1.0, 2.0], "coagulation kernel", "drf_by_coagulation.png"),
            (3, [0.5, 1.0], "condensation α", "drf_by_condensation.png")]
    for tok, levels, name, fname in AXES:
        n_per = 810 // (len(REGIMES) * len(levels))
        fig, ax = plt.subplots(figsize=(9.2, 4.8))
        pos, width = np.arange(len(levels), dtype=float), 0.15
        for k, reg in enumerate(REGIMES):
            data = [drf[[t[2] == reg and t[tok] == s for t in toks]] for s in levels]
            bp = ax.boxplot(data, positions=pos + (k - 2) * width, widths=width * 0.85,
                            patch_artist=True, showfliers=False,
                            medianprops={"color": "#0b0b0b"})
            for b in bp["boxes"]:
                b.set(facecolor=REGIME_COLOR[reg], alpha=0.35, lw=0.6)
            for j, v in enumerate(data):
                ax.scatter(pos[j] + (k - 2) * width
                           + rng.uniform(-0.32, 0.32, len(v)) * width, v, s=4,
                           color=REGIME_COLOR[reg], alpha=0.6, edgecolors="none", zorder=3)
        ax.set_xticks(pos, [f"{name} ×{s:g}" for s in levels])
        ax.set_ylabel("dRF/dS at t* [W m$^{-2}$ (Mt-S)$^{-1}$, cooling positive]")
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=REGIME_COLOR[r], alpha=0.8)
                           for r in REGIMES],
                  labels=[REGIME_LABEL[r] for r in REGIMES], fontsize=8.5, title="dilution",
                  title_fontsize=8.5, loc="upper left", ncols=2)
        ax.set_title(f"dRF/dS at t* vs {name} scale ({n_per} runs per box)", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(_OUT, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  {fname}")


if __name__ == "__main__":
    cases, drf = compute_all()
    figs(cases, np.asarray(drf))
