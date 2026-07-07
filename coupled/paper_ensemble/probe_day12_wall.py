# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Probe the ~day-12.14 gas-ODE stiffness wall WITHOUT the SciPy-BDF rescue: let Kvaerno5
fail raw, and show the state right up to the blow-up.

Two-stage, one case (30N_20km__sabr220__D2med__a1p0__nuc1__cg1, the paper case study):
  A. Run with the BDF fallback stubbed OUT (it captures the exact failing interval's start
     state y_pre and t_fail, then raises) -> locates the wall precisely.
  B. Re-run stopping exactly at t_fail -> the full clean time series up to the blow-up.
Plus a stiffness diagnosis: eigen-timescales of the gas-ODE Jacobian at y_pre vs a healthy
midday state the day before.

Outputs -> coupled/analyses/day12_wall/:
  state.npz, y_fail.npz, species_all.png, species_zoom.png, banana.png,
  sizedist_at_stall.png, jacobian_timescales.txt

Run (from SANDBOX/): python -m coupled.paper_ensemble.probe_day12_wall [replot]
"""
import dataclasses
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

_OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "analyses", "day12_wall"))
_CASE = "30N_20km__sabr220__D2med__a1p0__nuc1__cg1"
_DAYS_MAX = 14

plt.rcParams.update({"font.family": "Helvetica", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
                     "grid.linewidth": 0.5, "grid.color": "#e1e0d9", "axes.axisbelow": True,
                     "figure.dpi": 130, "legend.frameon": False})


class _Wall(RuntimeError):
    pass


def run():
    os.makedirs(_OUT, exist_ok=True)
    from coupled.paper_ensemble import run_ensemble as re_mod
    from coupled import driver
    from coupled.driver import run_coupled
    from config import IDX

    cases = dict(re_mod.all_cases())

    # --- stage A: fallback stubbed out; capture the failing interval and blow up ---------
    captured = {}

    def _no_rescue(gas_vf, y0, t0, t1, args, ts_grid, atol, h2so4_idx):
        captured.update(y_pre=np.asarray(y0), t_fail=float(t0), args=args)
        np.savez(os.path.join(_OUT, "y_fail.npz"), y_pre=np.asarray(y0), t_fail=float(t0))
        raise _Wall(f"Kvaerno5 stall at t0={t0:.1f}s ({t0/86400.0:.4f} d); no BDF rescue")

    orig = driver._bdf_gas_solve
    driver._bdf_gas_solve = _no_rescue
    sc = dataclasses.replace(re_mod.build_scenario(cases[_CASE]), days=_DAYS_MAX)
    try:
        run_coupled(sc, return_aerosol=True, return_state=True, return_size_dist=True)
        print("NO WALL HIT in", _DAYS_MAX, "days")
        return
    except _Wall as e:
        print("WALL:", e, flush=True)
    finally:
        driver._bdf_gas_solve = orig

    t_fail = captured["t_fail"]

    # --- stage B: clean rerun that STOPS at the failing interval -------------------------
    sc2 = dataclasses.replace(sc, days=t_fail / 86400.0)
    t, x, aero, st_final, sd, jrec = run_coupled(
        sc2, return_aerosol=True, return_state=True, return_size_dist=True,
        return_photolysis=True)
    from coupled.tomas_bridge import _bad
    dp_edges = _bad._xk_to_dp_um(np.asarray(st_final.xk))
    logdp = np.log10(dp_edges)
    dp_mid = 10 ** (0.5 * (logdp[:-1] + logdp[1:]))
    np.savez(os.path.join(_OUT, "state.npz"),
             t=t, x=x, species=list(IDX), SA=aero["SA"], radius_cm=aero["radius_cm"],
             h2so4wp=aero["h2so4wp"], particulate_S=aero["particulate_S"],
             n_cm3=sd["n_cm3"], dp_mid_um=dp_mid,
             dNdlogDp=sd["n_cm3"] / np.diff(logdp)[None, :], t_fail_d=t_fail / 86400.0)
    print(f"stage B saved: {len(t)} steps up to the wall at {t_fail/86400.0:.4f} d")

    # --- stiffness diagnosis: Jacobian eigen-timescales at y_pre vs 24 h earlier ---------
    import jax
    from coupled.driver import make_frozen_vf
    vf = make_frozen_vf(sc.opt if hasattr(sc, "opt") else 1)
    y_pre = captured["y_pre"]
    args = captured["args"]
    sp = list(IDX)
    i_healthy = np.searchsorted(t, t_fail / 86400.0 - 1.0)
    lines = []
    for tag, y in (("AT WALL  ", y_pre), ("24h EARLIER", x[min(i_healthy, len(t)-1)])):
        J = np.asarray(jax.jacfwd(lambda yy: vf(0.0, yy, args))(np.asarray(y)))
        ev = np.linalg.eigvals(J)
        tau = np.sort(1.0 / np.abs(np.real(ev[np.real(ev) < 0])))[:5]
        diag = np.abs(np.diag(J))
        top = np.argsort(diag)[::-1][:5]
        lines.append(f"{tag}: fastest eigen-timescales [s]: "
                     + ", ".join(f"{v:.2e}" for v in tau))
        lines.append(f"{tag}: stiffest species (1/|J_ii| [s]): "
                     + ", ".join(f"{sp[k]}={1.0/max(diag[k],1e-300):.2e}" for k in top))
    txt = "\n".join(lines)
    open(os.path.join(_OUT, "jacobian_timescales.txt"), "w").write(txt + "\n")
    print(txt)


def plots():
    z = np.load(os.path.join(_OUT, "state.npz"), allow_pickle=True)
    t, x, sp = z["t"] / 86400.0, z["x"], [str(s) for s in z["species"]]
    tf = float(z["t_fail_d"])

    fig, axs = plt.subplots(6, 6, figsize=(16, 12), sharex=True)
    for k, ax in enumerate(axs.ravel()):
        if k >= len(sp):
            ax.axis("off")
            continue
        ax.plot(t, np.maximum(x[:, k], 1e-12), lw=0.8, color="#2a78d6")
        ax.set_yscale("log")
        ax.set_title(sp[k], fontsize=8.5)
        ax.axvline(tf, color="#d03b3b", lw=1.0, ls="--")
        ax.tick_params(labelsize=7)
    for ax in axs[-1]:
        ax.set_xlabel("day", fontsize=8)
    fig.suptitle(f"All species up to the raw Kvaerno5 blow-up at day {tf:.4f} (red dashed) — "
                 f"{_CASE}", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "species_all.png"), bbox_inches="tight")
    plt.close(fig)

    fast = ["OH", "HO2", "O1D", "O", "NO3", "Cl", "ClO", "SO3", "H2SO4", "NO", "NO2", "H2O2"]
    m = t > tf - 0.75
    fig, axs = plt.subplots(3, 4, figsize=(13, 8), sharex=True)
    for name, ax in zip(fast, axs.ravel()):
        k = sp.index(name)
        ax.plot(t[m], np.maximum(x[m, k], 1e-12), lw=1.2, color="#2a78d6")
        ax.set_yscale("log")
        ax.set_title(name, fontsize=10)
        ax.axvline(tf, color="#d03b3b", lw=1.0, ls="--")
    for ax in axs[-1]:
        ax.set_xlabel("day")
    fig.suptitle(f"Fast species, final 0.75 d before the blow-up (day {tf:.4f})", y=1.0,
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "species_zoom.png"), bbox_inches="tight")
    plt.close(fig)

    dp = z["dp_mid_um"]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    pc = ax.pcolormesh(t, dp, np.maximum(z["dNdlogDp"], 1e-3).T, norm=LogNorm(1e2, 1e8),
                       cmap="viridis", rasterized=True, shading="nearest")
    ax.set_yscale("log")
    ax.set_ylim(dp[0], 2.0)
    ax.axvline(tf, color="#d03b3b", lw=1.2, ls="--")
    ax.set_xlabel("day")
    ax.set_ylabel("dry diameter [µm]")
    fig.colorbar(pc, ax=ax, label="dN/dlogD$_p$ [cm$^{-3}$]")
    ax.set_title(f"Size distribution up to the blow-up (day {tf:.4f})", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "banana.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for dt_h, col in ((-24, "#86b6ef"), (-3, "#2a78d6"), (0, "#d03b3b")):
        i = min(np.searchsorted(t, tf + dt_h / 24.0), len(t) - 1)
        ax.plot(dp, np.maximum(z["dNdlogDp"][i], 1e-6), lw=1.8, color=col,
                label=f"blow-up {dt_h:+d} h")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2e-3, 20)
    ax.set_ylim(1e-4, 1e6)
    ax.set_xlabel("dry diameter [µm]")
    ax.set_ylabel("dN/dlogD$_p$ [cm$^{-3}$]")
    ax.legend()
    ax.set_title(f"Size distribution approaching the blow-up (day {tf:.4f})", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "sizedist_at_stall.png"), bbox_inches="tight")
    plt.close(fig)
    print("plots ->", _OUT)


if __name__ == "__main__":
    if "replot" not in sys.argv:
        run()
    plots()
