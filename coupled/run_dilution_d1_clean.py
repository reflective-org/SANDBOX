# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Dilution 1 -- Low Latitude, High Altitude, Clean Stratosphere (20 km, 30N, summer, 10 days).

Full coupling (gas chemistry + TUV-x photolysis + TOMAS nucleation/condensation/coagulation +
aerosol->J + heating->T + dilution). A concentrated SO2 injection plume (2.9e9 pptv = 0.29%,
~1.7 t SO2 in the initial V0 = 10 m x 10 m x 30 km track) dilutes (regime D1 'Low Kz', Schumann
volume expansion) into CLEAN stratosphere: the background has a natural 15 pptv SO2 and
5e5 molec/cm^3 OH but NO SO3/H2SO4 and NO other short-lived radicals. NO ammonia/ammonium anywhere
(nucleation is binary H2SO4-H2O, neutral + ion-induced at ion_pair_rate = 30 pairs/cm^3/s; water
uptake is the pure-H2SO4/H2O Tabazadeh scheme). Initial + background aerosol = the TOMAS Marianna
'redcircles' clean distribution.

V0 does not enter the intensive dynamics (dilution uses only V(t)/V0); it converts concentrations
to PLUME-INTEGRATED masses (d1_plume_mass plot + the injected-SO2 line in d1_inputs.md).

Each run writes to its own clearly-named directory, coupled/analyses/d1_clean_<nbins>bin/
(d1_clean.npz + d1_inputs.md + d1_observations.csv + plots).
Run: python -m coupled.run_dilution_d1_clean [nbins]     (nbins = 40 default, or 80)
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coupled import CoupledScenario
from coupled.coupled_scenario import Switches
from coupled.driver import run_coupled
from coupled import dilution as dl
from coupled import tomas_bridge as tb
from coupled.aerosol_props import het_inputs
from config import IDX, air_number_density
from reactions import MECHANISM, build_env

_OUT = os.path.join(os.path.dirname(__file__), "analyses")   # reassigned per run in main()

_T, _P, _WTR = 215.0, 55.0, 4.5                     # 20 km, 30N summer
_M_AIR = air_number_density(_P, _T)                 # molec/cm^3
_V0_M3 = 10.0 * 10.0 * 30000.0                      # initial plume volume: 10 m x 10 m x 30 km track
_AVOG = 6.02214076e23


def _ppt_from_conc(molec_cm3: float) -> float:
    return molec_cm3 / _M_AIR * 1.0e12


_SO2_PPT = 2.9e9                                    # injected SO2 (0.29%; ~1.7 t in V0)
# clean-stratosphere background: no plume sulfur, no short-lived radicals (regenerate photochemically)
_ZERO_BG = ("SO2", "SO3", "H2SO4", "OH", "HO2", "O1D", "O", "NO3", "Cl", "ClO", "Br", "BrO",
            "CH3", "OClO")
# ... except natural ambient levels in the entrained air (override the zeros above)
_BG_PPT = {"SO2": 15.0,                             # ambient stratospheric SO2
           "OH": _ppt_from_conc(5.0e5)}             # 5e5 molec/cm^3 (~0.27 pptv)


def _scenario(nbins=40):
    return CoupledScenario(
        T=_T, P=_P, WTR=_WTR,
        latitude=30.0, longitude=0.0, day_of_year=172, start_utc_hour=6.0,
        days=10, DT=600.0, dt_couple=600.0, photolysis="tuvx",
        dilution_regime="D1", dilution_zero_species=_ZERO_BG, dilution_background=_BG_PPT,
        ion_pair_rate=30.0,                          # Dunne ion-induced nucleation (GCR ~20 km)
        tomas_nbins=nbins,
        switches=Switches(sulfur=True, nucleation=True, condensation=True, coagulation=True,
                          aerosol_to_j=True, heating_to_t=True, dilution=True),
        concentrations={"O2": 2.1e11, "O3": 1.18e6, "SO2": _SO2_PPT, "OH": 0.5, "HO2": 3.0,
                        "NO": 450.0, "NO2": 450.0, "HCl": 777.0, "ClONO2": 127.0, "HNO3": 5000.0,
                        "H2SO4": _ppt_from_conc(1.0e5)})   # 1e5 molec/cm^3 (~0.054 pptv)


def _bin_diam_edges_um(xk):
    # diameter bin edges [um] from the TOMAS mass boundaries xk at the dry sulfate density (1770)
    xk = np.asarray(xk, dtype=float)                              # kg, (nbins+1,)
    Dp_m = (6.0 * xk / (np.pi * 1770.0)) ** (1.0 / 3.0)
    return Dp_m * 1e6


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, name), dpi=120)
    plt.close(fig)


def _write_inputs_md(sc, het0):
    """The full input table (for the group) -- everything the run consumes, in one file.

    ``het0`` = het_inputs(initial TomasState): the initial-distribution surface area / r_eff / wt%
    that the FIRST interval's heterogeneous chemistry actually uses (the scenario SA field is inert
    with TOMAS on).
    """
    so2_conc = _SO2_PPT * 1e-12 * _M_AIR
    so2_kg = so2_conc * 1e6 * _V0_M3 / _AVOG * 0.064
    conc_rows = "\n".join(
        f"| {n} | {v:.4g} | {v * 1e-12 * _M_AIR:.3e} |" for n, v in sc.concentrations.items())
    bg_rows = "\n".join(
        f"| {n} | {v:.4g} | {v * 1e-12 * _M_AIR:.3e} |" for n, v in sc.dilution_background.items())
    md = f"""# Dilution D1 -- input variables

## Environment (box at ~20 km)
| variable | value |
|---|---|
| Temperature T | {sc.T} K |
| Pressure P | {sc.P} mbar (box altitude from the USSA profile) |
| Water vapour | {sc.WTR} ppm  ->  RH = {100 * tb.rh_from_scenario(sc):.2f}% over liquid (Tabazadeh a_w; ~1.8% over ice) |
| Air density M | {_M_AIR:.4e} molec/cm^3 |
| N2O5+H2O uptake Yn2o5 | {sc.Yn2o5} |
| O3-photolysis mode opt | {sc.opt} |

## Location / date / schedule
| variable | value |
|---|---|
| Latitude, longitude | {sc.latitude} N, {sc.longitude} E |
| Day of year | {sc.day_of_year} (~June 21) |
| Start | {sc.start_utc_hour:.1f} UTC |
| Duration | {sc.days} days |
| Output step DT | {sc.DT:.0f} s |
| Outer coupling step dt_couple | {sc.dt_couple:.0f} s (TUV-x J + optics frozen per interval, midpoint) |
| Micro-step (gas<->TOMAS) | adaptive, eps={sc.micro_eps}, floor={sc.micro_floor_s:g} s, cap={sc.micro_cap_s:g} s |

## Plume geometry
| variable | value |
|---|---|
| Initial volume V0 | 10 m x 10 m x 30 km = {_V0_M3:.3g} m^3 |
| Injected SO2 | {_SO2_PPT:.3g} pptv = {so2_conc:.3e} molec/cm^3 = **{so2_kg:.0f} kg SO2 in V0** |
| Role of V0 | intensive dynamics use only V(t)/V0; V0 converts conc -> plume-integrated mass |

## Initial gas composition (species not listed start at 0; NO NH3/ammonium anywhere)
| species | pptv | molec/cm^3 |
|---|---|---|
{conc_rows}
| H2O | {sc.WTR * 1e6:.3g} (from WTR) | {sc.WTR * 1e-6 * _M_AIR:.3e} |

## Dilution (regime D1 'Low Kz', Schumann plume expansion)
| item | value |
|---|---|
| V(t)/V0 | max(1, t^0.8) for t < 1e4 s, then 1585 * exp(2.811e-9 (t-1e4)^1.5) |
| k_dil | interval-averaged d ln V / dt |
| Background zeroed species | {", ".join(sc.dilution_zero_species)} |
| Background overrides | table below; all other species keep their initial value |

| background species | pptv | molec/cm^3 |
|---|---|---|
{bg_rows}

## Aerosol (TOMAS)
| item | value |
|---|---|
| Bins | {sc.tomas_nbins} ({"mass-doubling" if sc.tomas_nbins == 40 else "sqrt(2) mass ratio"}), dry Dp 1.7 nm - 17.5 um |
| Initial + background distribution | Marianna 'redcircles' clean stratosphere (N ~ 3 cm^-3) |
| Initial surface area (from that distribution; used by the first het-chem interval) | {het0["SA"]:.3f} um^2/cm^3 (r_eff = {het0["radius_cm"] * 1e4:.3f} um, {het0["h2so4wp"]:.1f} wt%) |
| Nucleation | Dunne 2016 binary H2SO4-H2O, neutral + ion-induced; ion_pair_rate = {sc.ion_pair_rate:g} pairs/cm^3/s; NH3 = 0 (ternary off); no organics (Riccobono channel = 0); fn_scale = {sc.nucleation_rate_scale:g} |
| Condensation | PPM ('ppm_jit'), accommodation alpha = {sc.condensation_alpha:g} |
| Coagulation | Brownian + Fuchs non-continuum correction |
| Water uptake | Tabazadeh 1997 pure H2SO4/H2O ('h2so4_tabazadeh') |
| TOMAS SO2 chemistry | OFF (gas model owns sulfur) |

## Couplings (all ON)
sulfur chain (SO2->SO3->H2SO4 via OH), nucleation, condensation, coagulation,
aerosol->photolysis (plume layer {sc.aerosol_thickness_km:g} km thick, pressure-anchored at the box altitude),
heating->T (SW only, no LW cooling), dilution.

## Solvers / numerics
gas: Diffrax Kvaerno5, rtol 1e-3, per-species atol 1e-6 (jitted, LU root finder; bit-identical to reference);
photolysis: TUV-x port (machine-precision vs Fortran), config tuv_5_4_no_aerosol.json + dynamic TOMAS
aerosol radiator; heating and J share ONE radiation solve per interval.
"""
    with open(os.path.join(_OUT, "d1_inputs.md"), "w") as f:
        f.write(md)


def _reaction_rates(sc, x, aero, jrec, t):
    """Per-reaction rates [molec/cm^3/s] at every output time, using the recorded frozen J.

    Rebuilds the NumPy Env at each output time (T from the heating trajectory; SA/r_eff/wt% from the
    TOMAS diagnostics -- end-of-interval values, diagnostic-grade) and overwrites the photolysis
    coefficients with the EXACT frozen values the gas solve used (recorded per outer interval).
    """
    from coupled.model_bridge import to_model_config
    cfg = to_model_config(sc)
    photo_idx = [i for i, r in enumerate(MECHANISM.active) if r.kind == "photo"]
    rates = np.zeros((len(t), len(MECHANISM.active)))
    for k, tk in enumerate(t):
        cfg.T = float(aero["T"][k])
        cfg.M = air_number_density(cfg.P, cfg.T)
        if np.isfinite(aero["SA"][k]):
            cfg.SA = float(aero["SA"][k])
            cfg.particle_radius = float(aero["radius_cm"][k])
            cfg.h2so4wp = float(aero["h2so4wp"][k])
        env = build_env(cfg, x[k], 0.0, j_values=None, sulfur_chain=True)
        coeff = np.array([r.coeff_fn(env) for r in MECHANISM.active])
        j_row = jrec["J"][int(np.clip(np.searchsorted(jrec["t_mid"], tk) - 1, 0,
                                      len(jrec["t_mid"]) - 1))]
        coeff[photo_idx] = j_row                          # exact frozen J actually used
        for i, rxn in enumerate(MECHANISM.active):
            r = coeff[i]
            for idx, power in rxn._rate_idx:
                r *= x[k, idx] ** power
            rates[k, i] = r
    return rates


def _export_observations(sc):
    """Save the digitized Marianna observations (dN/dlogDp) as data + a comparison plot.

    The source figure reports cm^-3 STP; the simulation uses AMBIENT concentrations, so both are
    written (ambient = STP * (P/P_STP)*(T_STP/T), the same factor the initial state uses).
    """
    import background_aerosol_distribution as bad
    fac = bad.stp_to_ambient_factor(sc.T, sc.P * 100.0)
    obs = {"330-340 ppbv (triangles)": bad._DATA1,
           "310-320 ppbv (diamonds)": bad._DATA4,
           "220-230 ppbv (circles; THIS RUN)": bad._DATA3}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    lines = [f"# Marianna observations, dN/dlogDp; ambient = STP * {fac:.4f} "
             f"(T={sc.T} K, P={sc.P} mbar)", "# obs, Dp_um, dNdlogDp_cm3_STP, dNdlogDp_cm3_ambient"]
    for (label, d), color in zip(obs.items(), ("C3", "C4", "C0")):
        ax.plot(d[:, 0], d[:, 1], "o-", ms=3, lw=1.2, color=color, label=f"{label} (STP)")
        ax.plot(d[:, 0], d[:, 1] * fac, "--", lw=1.2, color=color, alpha=0.6)
        for row in d:
            lines.append(f"{label.split(' ')[0]}, {row[0]:.4g}, {row[1]:.4g}, {row[1] * fac:.4g}")
    ax.set_xscale("log")
    ax.set_xlabel("diameter [um]"); ax.set_ylabel("dN/dlogDp [cm$^{-3}$]")
    ax.set_title("Digitized observations: solid = as published (STP), dashed = ambient at 20 km")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    _save(fig, "d1_observations.png")
    with open(os.path.join(_OUT, "d1_observations.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main(nbins=40):
    global _OUT
    _OUT = os.path.join(os.path.dirname(__file__), "analyses", f"d1_clean_{nbins}bin")
    os.makedirs(_OUT, exist_ok=True)
    sc = _scenario(nbins)
    _write_inputs_md(sc, het_inputs(tb.initial_tomas_state(sc)))
    _export_observations(sc)
    print(f"Running Dilution-1 clean-stratosphere ({nbins} bins; 20 km, 30N, summer, 10 d, "
          f"full coupling) -> {_OUT}/ ...")
    try:
        # run_coupled appends extras in order: aerosol, state, size_dist, photolysis
        t, x, aero, st_final, sd, jrec = run_coupled(
            sc, return_aerosol=True, return_state=True, return_size_dist=True,
            return_photolysis=True)
    except RuntimeError as e:
        print(f"RUN STOPPED (reported, not swallowed): {e}")
        raise
    days = t / 86400.0
    M = x[0, IDX["O2"]] / 0.21

    def ppt(name):
        return x[:, IDX[name]] / M * 1e12

    def conc(name):
        return x[:, IDX[name]]                                    # molec/cm^3

    # dN/dlogDp on fixed diameter bins: n_cm3(t,bin) / dlogDp(bin)
    edges = _bin_diam_edges_um(st_final.xk)
    dp_mid = np.sqrt(edges[:-1] * edges[1:])
    dlogdp = np.log10(edges[1:] / edges[:-1])
    dNdlogDp = sd["n_cm3"] / dlogdp[None, :]                      # (n_t, nbins) cm^-3
    V = dl.volume_ratio(t, "D1")
    kdil = np.array([dl.kdil_from_regime("D1", t[i], t[i + 1]) for i in range(len(t) - 1)])
    total_n = sd["n_cm3"].sum(axis=1)                             # cm^-3
    # particulate_S is H2SO4-equivalent molec/cm^3 -> dry sulfate mass concentration [ug/m^3]
    mass_ug_m3 = aero["particulate_S"] * 98.0 / _AVOG * 1e12
    # plume-integrated masses [kg]: conc [molec/cm^3] x V0*V_ratio [cm^3] / NA x MW [kg/mol]
    Vt_cm3 = _V0_M3 * 1e6 * V
    so2_kg = x[:, IDX["SO2"]] * Vt_cm3 / _AVOG * 0.064
    h2so4_kg = x[:, IDX["H2SO4"]] * Vt_cm3 / _AVOG * 0.098
    sulfate_kg = aero["particulate_S"] * Vt_cm3 / _AVOG * 0.098

    print("computing per-reaction rates (offline, frozen-J exact)...")
    rates = _reaction_rates(sc, x, aero, jrec, t)

    np.savez(os.path.join(_OUT, "d1_clean.npz"), t=t, x=x, species=list(IDX),
             SA=aero["SA"], radius_cm=aero["radius_cm"], h2so4wp=aero["h2so4wp"],
             particulate_S=aero["particulate_S"], T=aero["T"],
             n_cm3=sd["n_cm3"], Dp_m=sd["Dp_m"], dp_mid_um=dp_mid, dNdlogDp=dNdlogDp,
             V_ratio=V, kdil=kdil, M=M, total_n=total_n, mass_ug_m3=mass_ug_m3,
             J_tmid=jrec["t_mid"], J=jrec["J"], J_equations=jrec["equations"],
             rates=rates, rate_equations=[r.equation for r in MECHANISM.active],
             V0_m3=_V0_M3, so2_kg=so2_kg, h2so4_kg=h2so4_kg, sulfate_kg=sulfate_kg)

    # 1) initial size distribution (Marianna 'redcircles' background, dry diameter)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(dp_mid, dNdlogDp[0], where="mid", lw=1.8, color="C0")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("dry diameter [um]"); ax.set_ylabel("dN/dlogDp [cm$^{-3}$]")
    ax.set_title(f"Initial aerosol (Marianna 'redcircles' background): "
                 f"N = {total_n[0]:.2f} cm$^{{-3}}$, m = {mass_ug_m3[0]:.3f} ug m$^{{-3}}$")
    ax.grid(alpha=0.3, which="both")
    _save(fig, "d1_initial_size_dist.png")

    # 2) banana plot
    fig, ax = plt.subplots(figsize=(9, 5))
    pm = ax.pcolormesh(days, dp_mid, np.maximum(dNdlogDp, 1e-3).T, shading="auto",
                       norm=matplotlib.colors.LogNorm(vmin=1e-1, vmax=max(1e2, dNdlogDp.max())),
                       cmap="turbo")
    ax.set_yscale("log"); ax.set_ylabel("dry diameter [um]"); ax.set_xlabel("day")
    ax.set_title("Dilution 1 (clean strat): aerosol size distribution dN/dlogDp [cm$^{-3}$]")
    fig.colorbar(pm, ax=ax, label="dN/dlogDp [cm$^{-3}$]")
    _save(fig, "d1_banana.png")

    # 3) size-distribution snapshots
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0):
        i = int(np.argmin(np.abs(days - d)))
        ax.step(dp_mid, dNdlogDp[i], where="mid", lw=1.4, label=f"{days[i]:.1f} d")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("dry diameter [um]"); ax.set_ylabel("dN/dlogDp [cm$^{-3}$]")
    ax.set_title("Size distributions"); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    _save(fig, "d1_size_distributions.png")

    # 4) EVERY gas-phase species separately (pptv)
    names = list(IDX)
    ncols, nrows = 6, int(np.ceil(len(names) / 6))
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.2 * nrows), sharex=True)
    for ax, name in zip(axs.flat, names):
        y = ppt(name)
        ax.plot(days, np.maximum(y, 1e-12), lw=1.0)
        if np.nanmax(y) > 0:
            ax.set_yscale("log")
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    for ax in axs.flat[len(names):]:
        ax.axis("off")
    for ax in axs[-1, :]:
        ax.set_xlabel("day", fontsize=8)
    fig.suptitle("Dilution 1: gas-phase species [pptv]", y=1.001)
    _save(fig, "d1_species_all.png")

    # 5) photolysis coefficients J(t) actually used, per reaction
    j_days = jrec["t_mid"] / 86400.0
    n_j = len(jrec["equations"])
    ncols = 5
    nrows = int(np.ceil(n_j / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.3 * nrows), sharex=True)
    for ax, k in zip(axs.flat, range(n_j)):
        ax.plot(j_days, np.maximum(jrec["J"][:, k], 1e-30), lw=0.9)
        ax.set_yscale("log")
        ax.set_title(jrec["equations"][k], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    for ax in axs.flat[n_j:]:
        ax.axis("off")
    for ax in axs[-1, :]:
        ax.set_xlabel("day", fontsize=8)
    fig.suptitle("Frozen photolysis coefficients J(t) used by the gas solve [s$^{-1}$]", y=1.001)
    _save(fig, "d1_photolysis_J.png")

    # 6) per-reaction rates (all active reactions)
    eqs = [r.equation for r in MECHANISM.active]
    ncols = 6
    nrows = int(np.ceil(len(eqs) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.2 * nrows), sharex=True)
    for ax, k in zip(axs.flat, range(len(eqs))):
        ax.plot(days, np.maximum(rates[:, k], 1e-12), lw=0.9)
        ax.set_yscale("log")
        ax.set_title(eqs[k], fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    for ax in axs.flat[len(eqs):]:
        ax.axis("off")
    for ax in axs[-1, :]:
        ax.set_xlabel("day", fontsize=8)
    fig.suptitle("Reaction rates [molec cm$^{-3}$ s$^{-1}$]", y=1.001)
    _save(fig, "d1_reaction_rates.png")

    # 7) dilution alone: plume volume + rate
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.plot(days, V, lw=1.6); a1.set_yscale("log")
    a1.set_title("plume volume V(t)/V$_0$ (D1 'Low Kz', Schumann)"); a1.set_xlabel("day")
    a2.plot(days[1:], kdil, lw=1.4, color="C1"); a2.set_yscale("log")
    a2.set_title("dilution rate k$_{dil}$(t) [s$^{-1}$]"); a2.set_xlabel("day")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    _save(fig, "d1_dilution.png")

    # 8) surface area alone
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(days, aero["SA"], lw=1.6, color="C2")
    ax.set_yscale("log"); ax.set_xlabel("day"); ax.set_ylabel("SA [um$^2$ cm$^{-3}$]")
    ax.set_title("Aerosol surface area (wet)"); ax.grid(alpha=0.3)
    _save(fig, "d1_surface_area.png")

    # 9) H2SO4: gas vs particle
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(days, np.maximum(ppt("H2SO4"), 1e-12), lw=1.5, label="gas H2SO4")
    ax.plot(days, np.maximum(aero["particulate_S"] / M * 1e12, 1e-12), lw=1.5,
            label="particulate sulfate (H2SO4-equiv)")
    ax.set_yscale("log"); ax.set_xlabel("day"); ax.set_ylabel("pptv")
    ax.set_title("Sulfuric acid: gas vs particle")
    ax2 = ax.secondary_yaxis("right", functions=(lambda p: p * 1e-12 * M, lambda c: c / M * 1e12))
    ax2.set_ylabel("molec cm$^{-3}$")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, "d1_h2so4_gas_particle.png")

    # 10) totals: number, dry sulfate mass, surface area
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    axs[0].plot(days, total_n, lw=1.5); axs[0].set_title("total number [cm$^{-3}$]")
    axs[1].plot(days, mass_ug_m3, lw=1.5, color="C3"); axs[1].set_title("dry sulfate mass [ug m$^{-3}$]")
    axs[2].plot(days, aero["SA"], lw=1.5, color="C2"); axs[2].set_title("surface area [um$^2$ cm$^{-3}$]")
    for a in axs:
        a.set_yscale("log"); a.set_xlabel("day"); a.grid(alpha=0.3)
    fig.suptitle("Aerosol totals (per cm$^3$ of plume air)")
    _save(fig, "d1_totals.png")

    # 11) plume-integrated masses (uses V0): conc x V0*V(t)/V0
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(days, so2_kg, lw=1.6, label="SO2 (gas)")
    ax.plot(days, np.maximum(h2so4_kg, 1e-12), lw=1.4, label="H2SO4 (gas)")
    ax.plot(days, sulfate_kg, lw=1.6, label="particulate sulfate (H2SO4-equiv)")
    ax.set_yscale("log"); ax.set_xlabel("day"); ax.set_ylabel("mass in plume [kg]")
    ax.set_title(f"Plume-integrated mass (V$_0$ = 10m x 10m x 30km = {_V0_M3:.1e} m$^3$)\n"
                 "note: dilution entrains background SO2/aerosol as the plume grows")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, "d1_plume_mass.png")

    # 12) OH alone [molec/cm^3]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(days, np.maximum(conc("OH"), 1e-2), lw=1.2, color="C0")
    ax.axhline(5.0e5, color="k", ls=":", lw=1.0, label="background (5e5)")
    ax.set_yscale("log"); ax.set_xlabel("day"); ax.set_ylabel("OH [molec cm$^{-3}$]")
    ax.set_title("OH"); ax.legend(); ax.grid(alpha=0.3)
    _save(fig, "d1_OH.png")

    # 13) SO2 alone
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(days, conc("SO2"), lw=1.5, color="C3")
    ax.axhline(15.0e-12 * M, color="k", ls=":", lw=1.0, label="background (15 pptv)")
    ax.set_yscale("log"); ax.set_xlabel("day"); ax.set_ylabel("SO2 [molec cm$^{-3}$]")
    ax2 = ax.secondary_yaxis("right", functions=(lambda c: c / M * 1e12, lambda p: p * 1e-12 * M))
    ax2.set_ylabel("pptv")
    ax.set_title("SO2"); ax.legend(); ax.grid(alpha=0.3)
    _save(fig, "d1_SO2.png")

    # 14) total H2SO4 (gas + particulate H2SO4-equivalent) [molec/cm^3]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    total_h2so4 = conc("H2SO4") + aero["particulate_S"]
    ax.plot(days, total_h2so4, lw=1.6, color="k", label="total (gas + particle)")
    ax.plot(days, np.maximum(conc("H2SO4"), 1e-2), lw=1.0, alpha=0.7, label="gas")
    ax.plot(days, aero["particulate_S"], lw=1.0, alpha=0.7, label="particulate (H2SO4-equiv)")
    ax.set_yscale("log"); ax.set_xlabel("day"); ax.set_ylabel("H2SO4 [molec cm$^{-3}$]")
    ax.set_title("Total sulfuric acid"); ax.legend(); ax.grid(alpha=0.3)
    _save(fig, "d1_H2SO4_total.png")

    # 15) ozone alone
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(days, ppt("O3") / 1e6, lw=1.4, color="C2")
    ax.set_xlabel("day"); ax.set_ylabel("O3 [ppmv]")
    ax2 = ax.secondary_yaxis("right", functions=(lambda p: p * 1e-6 * M / 1e12,
                                                 lambda c: c * 1e12 / (1e-6 * M)))
    ax2.set_ylabel("O3 [10$^{12}$ molec cm$^{-3}$]")
    ax.set_title("Ozone"); ax.grid(alpha=0.3)
    _save(fig, "d1_O3.png")

    # 16) aerosol properties + box temperature (heating)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    axs[0].plot(days, aero["radius_cm"] * 1e4, lw=1.5)
    axs[0].set_yscale("log"); axs[0].set_title("effective wet radius r$_{eff}$ [um]")
    axs[1].plot(days, aero["h2so4wp"], lw=1.5, color="C1"); axs[1].set_title("aerosol H2SO4 [wt%]")
    axs[2].plot(days, aero["T"], lw=1.5, color="C3"); axs[2].set_title("box temperature [K]")
    for a in axs:
        a.set_xlabel("day"); a.grid(alpha=0.3)
    fig.suptitle("Aerosol composition / heating response")
    _save(fig, "d1_aerosol_props.png")

    print(f"steps={len(t)}  SO2 {ppt('SO2')[0]:.3g}->{ppt('SO2')[-1]:.1f} pptv  "
          f"H2SO4 max {np.max(ppt('H2SO4')):.2f} pptv  total N max {total_n.max():.2e}/cm3  "
          f"SA max {np.nanmax(aero['SA']):.3f} um2/cm3  mass max {mass_ug_m3.max():.3f} ug/m3  "
          f"sulfate max {sulfate_kg.max():.1f} kg (of {so2_kg[0] * 98 / 64:.0f} kg potential)")
    print(f"wrote npz + d1_inputs.md + 16 plots to {_OUT}/")


if __name__ == "__main__":
    main(nbins=int(sys.argv[1]) if len(sys.argv) > 1 else 40)
