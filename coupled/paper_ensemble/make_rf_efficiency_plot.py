# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Size-resolved shortwave radiative-forcing efficiency (-> runs/plots/rf/).

Uses tomas_jax.physics.radiative_forcing.scattering_efficiency_vs_radius (the Pierce et al.
2010 Fig. 1 diagnostic, Chylek & Wong 1995 forcing equation): direct SW forcing per unit
aerosol mass burden for a monodisperse population, as a function of particle radius.
Stratospheric (Pierce SI) parameters: S0 = 1370 W/m^2, Tatm = 1.0, surface albedo 0.15,
cloud fraction 0.6, wet H2SO4/H2O density 1700 kg/m^3.

Run (from SANDBOX/): python -m coupled.paper_ensemble.make_rf_efficiency_plot
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coupled.paper_ensemble.make_paper_candidate_plots import _RUNS  # PNG+PDF hook
from coupled import tomas_bridge  # noqa: F401  (sys.path for tomas_jax)
from tomas_jax.physics.radiative_forcing import scattering_efficiency_vs_radius

_OUT = os.path.join(_RUNS, "plots", "rf")
_PIERCE = dict(solar_constant=1370.0, Tatm=1.0, albedo=0.15, cloud_fraction=0.6,
               density=1700.0)

plt.rcParams.update({"font.family": "Helvetica", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
                     "grid.linewidth": 0.5, "grid.color": "#e1e0d9", "axes.axisbelow": True,
                     "figure.dpi": 130, "legend.frameon": False})


def main():
    from tomas_jax.physics.radiative_forcing import (h2so4_equilibrium_wt,
                                                     h2so4_solution_density)
    os.makedirs(_OUT, exist_ok=True)
    # Pierce et al. (2010) Fig. 1 convention: POSITIVE "scattering efficiency" per Mt of S
    # spread over the globe: 1 Mt S / A_Earth = 1e9 kg / 5.101e14 m^2 = 1.9604e-6 kg S m^-2
    MTS = 1.0e9 / 5.101e14
    T_STRAT = 210.0
    RHS = [(3.0, "#86b6ef"), (5.0, "#2a78d6"), (10.0, "#0d366b")]

    cache_path = os.path.join(_OUT, "_eff_curves_nw500_nr800.npz")
    cache = dict(np.load(cache_path)) if os.path.exists(cache_path) else {}
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    peak_note = None
    for rh, col in RHS:
        wt = float(h2so4_equilibrium_wt(T_STRAT, rh))            # wt% H2SO4 at 210 K, this RH
        rho = float(h2so4_solution_density(wt))                  # kg/m^3
        s_factor = (98.079 / 32.065) / (wt / 100.0)              # kg wet aerosol per kg S
        key = f"rh{rh:.0f}"
        if key in cache:                                          # Mie curves cached: style
            r_um, eff = cache[key]                                # tweaks re-render instantly
        else:
            params = dict(_PIERCE)
            params["density"] = rho
            r, rf = scattering_efficiency_vs_radius(spectral=True, n_wavelengths=500,
                                                    n_radii=800, **params)
            r_um = np.asarray(r) * 1e6
            eff = -np.asarray(rf) * s_factor * MTS
            cache[key] = np.stack([r_um, eff])
        d_um = 2.0 * r_um                                        # plot vs DIAMETER
        ax.plot(d_um, eff, lw=2.0, color=col,
                label=f"RH {rh:.0f}%  ({wt:.0f} wt%, ρ = {rho:.0f} kg m$^{{-3}}$)")
        if rh == 3.0:
            ipk = int(np.argmax(eff))
            peak_note = (d_um[ipk], eff[ipk])
    np.savez_compressed(cache_path, **cache)
    ax.axvline(peak_note[0], color="#898781", lw=1.1, ls="--")
    # right axis: gravitational settling velocity v_s = rho_p D^2 g Cc/(18 mu)
    # (= S&P relaxation time m_p*Cc/(3 pi mu Dp) times g) at 55 hPa / 210 K, RH 3% density
    T, P = 210.0, 5500.0
    mu = 1.458e-6 * T ** 1.5 / (T + 110.4)
    lam = 2.0 * mu / ((P / (287.05 * T)) * np.sqrt(8.0 * 287.05 * T / np.pi))
    d_m = 2.0 * cache["rh3"][0] * 1e-6
    Kn = 2.0 * lam / d_m
    cc_slip = 1.0 + Kn * (1.257 + 0.4 * np.exp(-1.1 / Kn))
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#d03b3b")
    # settling depends on the WET diameter (x-axis) and linearly on the WET density,
    # so one line per RH composition (they differ by <6%: 1567/1533/1479 kg m^-3)
    RED = {3.0: "#eda3a3", 5.0: "#d03b3b", 10.0: "#7a1c1c"}
    for rh, _ in RHS:
        rho_p = float(h2so4_solution_density(float(h2so4_equilibrium_wt(T_STRAT, rh))))
        v_s = rho_p * d_m ** 2 * 9.81 * cc_slip / (18.0 * mu)  # m/s
        ax2.plot(d_m * 1e6, v_s * 3.156e7 / 1e3, lw=1.6, ls="-.", color=RED[rh])
    ax2.set_yscale("log")
    ax2.set_ylabel("gravitational settling velocity [km yr$^{-1}$]", color="#d03b3b")
    ax2.tick_params(axis="y", colors="#d03b3b")
    ax2.grid(False)
    ax.set_ylim(bottom=0)
    ax.set_xscale("log")
    ax.set_xlim(2e-3, 20)
    ax.set_xlabel("particle diameter [µm]")
    ax.set_ylabel("scattering efficiency [W m$^{-2}$ Mt-S$^{-1}$]")
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))
    ax.grid(True, which="minor", axis="y", alpha=0.18, lw=0.4, color="#e1e0d9")
    # log x: minor gridlines at the 2..9 subdecades
    ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=range(2, 10),
                                                            numticks=100))
    ax.grid(True, which="minor", axis="x", alpha=0.18, lw=0.4, color="#e1e0d9")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=9, loc="upper left")
    ax.set_title("Size-resolved forcing efficiency of stratospheric sulfate — solar-spectrum "
                 "integrated (300–2500 nm)\n"
                 "Chylek & Wong 1995 / Pierce et al. 2010: S$_0$ = 1370 W m$^{-2}$, "
                 "T$_{atm}$ = 1.0, surface albedo 0.15, cloud fraction 0.6\n"
                 "monodisperse; m = 1.4 + 10$^{-8}$i; composition from 210 K equilibrium; "
                 "cooling shown positive, per Mt-S global burden", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "rf_efficiency_vs_size.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  rf_efficiency_vs_size.png; RH3% peak at r = {peak_note[0]:.3f} um, "
          f"{peak_note[1]:.3f} W/m2/MtS")


def fig_qsca():
    """Mie scattering efficiency Q_sca vs radius (dimensionless), 550 nm and solar-weighted."""
    import jax
    import jax.numpy as jnp
    from tomas_jax.physics.bhmie import bhmie_qsca_jax
    from tomas_jax.physics.radiative_forcing import (_solar_spectral_weights,
                                                     REFINDEX_SULFATE)
    radii = np.logspace(-3, 1, 300) * 1e-6                       # m
    wls, wts = _solar_spectral_weights(n_wl=30)
    qsca_wl = []
    for wl in np.asarray(wls):
        x = jnp.asarray(2.0 * np.pi * radii / wl)
        _, qs, _ = jax.vmap(bhmie_qsca_jax, (0, None))(x, REFINDEX_SULFATE)
        qsca_wl.append(np.asarray(qs))
    qsca_spec = np.average(np.array(qsca_wl), axis=0, weights=np.asarray(wts))
    x550 = jnp.asarray(2.0 * np.pi * radii / 550e-9)
    _, q550, _ = jax.vmap(bhmie_qsca_jax, (0, None))(x550, REFINDEX_SULFATE)

    r_um = radii * 1e6
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(r_um, qsca_spec, lw=2.2, color="#2a78d6",
            label="solar-spectrum weighted (300–2500 nm)")
    ax.plot(r_um, np.asarray(q550), lw=1.6, ls="--", color="#eda100",
            label="monochromatic 550 nm")
    ipk = int(np.argmax(np.asarray(q550)))
    ax.axvline(r_um[ipk], color="#898781", lw=0.9, ls=":")
    ax.annotate(f"first Mie maximum (550 nm)\nr = {r_um[ipk]:.2f} µm, "
                f"Q$_{{sca}}$ = {float(q550[ipk]):.2f}",
                xy=(r_um[ipk], float(q550[ipk])), xytext=(0.05, 0.8),
                textcoords="axes fraction", fontsize=9, color="#52514e",
                arrowprops=dict(arrowstyle="->", color="#898781", lw=0.9))
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 10)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("particle radius [µm]")
    ax.set_ylabel("scattering efficiency Q$_{sca}$ [–]")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("Mie scattering efficiency of sulfate vs particle size\n"
                 "(Bohren & Huffman; m = 1.4 + 10$^{-8}$i)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "qsca_vs_size.png"), bbox_inches="tight")
    plt.close(fig)
    print("  qsca_vs_size.png")


def fig_lifetime():
    """Sedimentation-limited lifetime and lifetime-weighted forcing efficiency vs diameter.

    Stokes settling with Cunningham slip at the 30N/20 km baseline (55 hPa / 210 K):
        v_s = rho_p D^2 g Cc(Kn) / (18 mu),  Cc = 1 + Kn(1.257 + 0.4 exp(-1.1/Kn))
    tau_sed = H / v_s with H = 4 km (20 km -> ~16 km tropopause at 30N);
    tau_eff = 1/(1/tau_dyn + 1/tau_sed) with tau_dyn = 1.5 yr (Brewer-Dobson residence).
    The weighted efficiency = efficiency x tau_eff/tau_dyn (time-integrated forcing per
    Mt-S INJECTED, relative to a sedimentation-free particle). Diagnostic only: the box
    model itself has no vertical transport.
    """
    from tomas_jax.physics.radiative_forcing import (h2so4_equilibrium_wt,
                                                     h2so4_solution_density)
    T, P = 210.0, 5500.0                                       # K, Pa (55 hPa)
    H_FALL = 4000.0                                            # m to the ~16 km tropopause
    TAU_DYN = 1.5 * 3.156e7                                    # s (1.5 yr)
    G = 9.81
    mu = 1.458e-6 * T ** 1.5 / (T + 110.4)                     # Sutherland viscosity
    rho_air = P / (287.05 * T)
    c_air = np.sqrt(8.0 * 287.05 * T / np.pi)
    lam = 2.0 * mu / (rho_air * c_air)                         # mean free path [m] (~0.8 um)

    cache = dict(np.load(os.path.join(_OUT, "_eff_curves_nw500_nr800.npz")))
    RHS = [(3.0, "#86b6ef"), (5.0, "#2a78d6"), (10.0, "#0d366b")]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.2, 8.0), sharex=True)
    yr = 3.156e7
    for rh, col in RHS:
        wt = float(h2so4_equilibrium_wt(T, rh))
        rho_p = float(h2so4_solution_density(wt))
        r_um, eff = cache[f"rh{rh:.0f}"]
        d_m = 2.0 * r_um * 1e-6
        Kn = 2.0 * lam / d_m
        cc = 1.0 + Kn * (1.257 + 0.4 * np.exp(-1.1 / Kn))
        v_s = rho_p * d_m ** 2 * G * cc / (18.0 * mu)          # m/s
        tau_sed = H_FALL / v_s                                 # s
        tau_eff = 1.0 / (1.0 / TAU_DYN + 1.0 / tau_sed)
        if rh == 3.0:                                          # lifetimes: show RH 3% only
            a1.plot(d_m * 1e6, tau_sed / yr, lw=1.6, ls="--", color="#898781",
                    label="sedimentation only  H/v$_s$")
            a1.axhline(TAU_DYN / yr, color="#898781", lw=1.0, ls=":",
                       label="dynamical (Brewer–Dobson) 1.5 yr")
            a1.plot(d_m * 1e6, tau_eff / yr, lw=2.2, color="#2a78d6",
                    label="effective lifetime")
            a2.plot(d_m * 1e6, eff, lw=1.4, ls="--", color="#898781",
                    label="RH 3%, no sedimentation")
        a2.plot(d_m * 1e6, eff * tau_eff / TAU_DYN, lw=2.0, color=col,
                label=f"RH {rh:.0f}%, lifetime-weighted")
    a1.set_yscale("log")
    a1.set_ylim(3e-2, 30)
    a1.set_ylabel("stratospheric lifetime [yr]")
    a1.legend(fontsize=8.5, loc="lower left")
    a1.set_title("particle lifetime at 30°N / 20 km (55 hPa / 210 K); fall depth H = 4 km",
                 fontsize=10)
    a2.set_ylim(bottom=0)
    a2.set_xscale("log")
    a2.set_xlim(2e-3, 20)
    a2.set_xlabel("particle diameter [µm]")
    a2.set_ylabel("scattering efficiency [W m$^{-2}$ Mt-S$^{-1}$]")
    a2.legend(fontsize=8.5, loc="upper left")
    a2.set_title("efficiency × τ$_{eff}$/τ$_{dyn}$: time-integrated forcing per Mt-S injected",
                 fontsize=10)
    fig.suptitle("Sedimentation-weighted forcing efficiency — same optics/assumptions as "
                 "rf_efficiency_vs_size\n(diagnostic: the box model has no vertical transport)",
                 y=0.995, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(_OUT, "rf_efficiency_lifetime.png"), bbox_inches="tight")
    plt.close(fig)
    print("  rf_efficiency_lifetime.png")


if __name__ == "__main__":
    main()
    fig_qsca()
    fig_lifetime()
