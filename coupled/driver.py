# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Operator-split coupling driver (Phase 2.4b).

Runs the coupled box on the JAX gas backend with an outer loop of ``dt_couple`` steps. Each outer
step: compute the photolysis J from the TUV-x port at the step MIDPOINT (2nd-order splitting), freeze
it, and integrate the gas ODE over the interval with a fresh solve (never stepping across a J jump).
The outer grid is snapped to sunrise/sunset (so no interval straddles the terminator; J is off at
night). One SZA source (gas ``solar.py``, the same the adapter uses) drives J, the day/night snap, and
the fallback ``j_scale``. ``switches.sulfur`` drives the gas sulfur gate.

TOMAS microphysics (Phase 3) slots into this same outer loop: after the gas sub-step, the gas-produced
H2SO4 is handed to TOMAS (``Gc[SRTSO4]``), TOMAS runs nucleation/condensation/coagulation (its own SO2
chemistry OFF) over the interval, the depleted H2SO4 is written back to the gas, and the new aerosol
surface area / wet radius / H2SO4 weight-percent feed the NEXT interval's heterogeneous chemistry.
TOMAS is active iff any of ``switches.{nucleation,condensation,coagulation}`` is on; otherwise the run
is the Phase-2 gas-only path with the prescribed ``cfg.SA``. Dilution / heating: Phases 5-6.

Δt_couple with TOMAS (AD-3, decision 3): a single TOMAS ``make_step`` per outer interval is used, so
``dt_couple`` must be small enough to resolve the sub-second H2SO4 condensation transient (highest in
the first minutes) -- pick a small ``dt_couple`` for TOMAS-active runs. The aerosol het inputs are held
frozen over each interval (sequential operator splitting, like J), using the state from the end of the
previous interval; the first interval uses the initial TOMAS state.

Gate source: ``switches.sulfur`` is THE sulfur-gate source here. It is passed to the JAX side
(``sulfur_chain``) and, for a NumPy comparison, must likewise be passed to ``build_env(sulfur_chain=...)``
so both backends read the same flag (see the 2.5 parity harness). ``reactions.sulfur_chain_active``
remains the default mode->gate derivation for STANDALONE NumPy runs only.

NOTE on ``photolysis="reference"``: this driver always uses the continuous SZA ``j_scale`` (like
``sza``); it does NOT reproduce the NumPy driver's fixed-45-degree, day/night-gated MATLAB reference
behavior. Use ``reference`` here only for continuous-SZA testing, not for MATLAB-faithful comparison.
"""

from __future__ import annotations

import numpy as np

# model_bridge puts gas_phase_chemistry on sys.path (interim; see DEFERRED) -- import it first.
from .model_bridge import initial_state, to_model_config
from .tomas_bridge import (initial_tomas_state, make_microphysics_step, SRTSO4,
                           BOXVOL_CM3, MW_H2SO4)
from .aerosol_props import het_inputs
from . import heating as _heating
from .dilution import dilute_gas, dilute_aerosol, kdil_from_regime as dl_kdil_from_regime
from .units import conc_to_mass, mass_to_conc

from config import IDX, air_number_density                   # noqa: E402  (gas model)
from driver import _abstol                                   # noqa: E402  (gas model)
from reactions import photolysis_coeffs                      # noqa: E402
from solar import cos_solar_zenith, photolysis_scale         # noqa: E402  (single SZA source)
from tuvx_photolysis_adapter import _compute_j_values, calculator_grids  # noqa: E402
from jaxmodel.model import make_frozen_step                  # noqa: E402

import jax.numpy as jnp                                      # noqa: E402


def _cosz(cfg, t_seconds: float) -> float:
    """cos(SZA) at model time t (s), via the gas solar source the adapter also uses."""
    total_hours = cfg.start_utc_hour + t_seconds / 3600.0
    doy = cfg.day_of_year + total_hours / 24.0
    return cos_solar_zenith(cfg.latitude, cfg.longitude, doy, total_hours % 24.0)


def _outer_intervals(cfg, days: int, dt_couple: float) -> list[tuple[float, float]]:
    """(t0, t1) intervals of length <= dt_couple, SNAPPED so none straddles sunrise/sunset.

    Uniform dt_couple edges plus the day/night terminator crossings (found by a coarse cos(SZA) scan
    then bisection). Because terminators only subdivide uniform intervals, every interval is <=
    dt_couple AND lies entirely in day or entirely in night.
    """
    total = float(days) * 86400.0
    edges = set(np.arange(0.0, total, dt_couple).tolist()) | {0.0, total}
    scan = np.arange(0.0, total + 1.0, min(dt_couple, 300.0))
    cz = np.array([_cosz(cfg, t) for t in scan])
    for i in range(len(scan) - 1):
        if cz[i] != 0.0 and cz[i + 1] != 0.0 and np.sign(cz[i]) != np.sign(cz[i + 1]):
            a, b, sa = scan[i], scan[i + 1], np.sign(cz[i])   # bisection for the crossing time
            for _ in range(40):
                m = 0.5 * (a + b)
                a, b = (m, b) if np.sign(_cosz(cfg, m)) == sa else (a, m)
            edges.add(0.5 * (a + b))
    E = sorted(e for e in edges if 0.0 <= e <= total)
    return list(zip(E[:-1], E[1:]))


def _frozen_j_values(cfg, t_mid: float, aerosol_props=None):
    """Absolute per-reaction J at the interval midpoint (uncached -> dt_couple drives the recompute,
    bypassing the adapter's 120 s cache). ``None`` in non-tuvx modes (photolysis_coeffs then uses
    j45*j_scale). ``aerosol_props`` (Phase 4) injects the TOMAS aerosol radiator for this solve. The
    adapter returns zeros at night."""
    return _compute_j_values(cfg, t_mid, aerosol_props=aerosol_props) if cfg.photolysis == "tuvx" else None


def run_coupled(scenario, return_aerosol=False, return_state=False, return_size_dist=False):
    """Integrate a CoupledScenario with operator splitting.

    Returns ``(t [s], states [n_t, n_species])``. With ``return_aerosol=True`` also returns a dict of
    per-output-time aerosol diagnostics (``SA`` [um^2/cm^3], ``radius_cm``, ``h2so4wp`` [wt%],
    ``particulate_S`` [molec/cm^3 as H2SO4-equiv]); the arrays are all-NaN when TOMAS is inactive.
    With ``return_state=True`` also appends the final ``TomasState`` (or ``None`` if TOMAS inactive),
    e.g. for plotting the evolved size distribution. Extra outputs are appended in that order.
    """
    cfg = to_model_config(scenario)
    y0 = initial_state(scenario)
    sulfur = float(scenario.switches.sulfur)   # single gate source (NumPy match: build_env(sulfur_chain=))
    params = dict(T=cfg.T, M=cfg.M, P=cfg.P, SA=cfg.SA, WTR=cfg.WTR, Yn2o5=cfg.Yn2o5)
    step = make_frozen_step(cfg.opt, atol=jnp.asarray(_abstol(cfg.opt)))

    tomas_step = make_microphysics_step(scenario.switches)
    tomas_active = tomas_step is not None
    tstate = initial_tomas_state(scenario) if tomas_active else None
    het = het_inputs(tstate) if tomas_active else None
    h2so4_idx = IDX["H2SO4"]

    nuc_scale = float(scenario.nucleation_rate_scale)   # Phase 7 knob -> TOMAS nucleation fn_scale
    # Phase 6: dilution -> relaxation toward a background (AD-6.3). Background gas = initial state with
    # the scenario's dilution_zero_species set to 0 (e.g. plume SO2/H2SO4 + radicals absent from clean
    # entrained air). Rate is constant (dilution_rate) or time-varying from a regime's V(t) expansion.
    dilution_active = bool(scenario.switches.dilution)
    regime = scenario.dilution_regime
    gas_bg = np.asarray(y0, dtype=float).copy()
    for name in scenario.dilution_zero_species:
        if name not in IDX:
            raise ValueError(f"dilution_zero_species has unknown species {name!r}")
        gas_bg[IDX[name]] = 0.0
    gas_bg = jnp.asarray(gas_bg)
    tstate_bg = tstate                  # background aerosol = initial TomasState (immutable)

    def _kdil(t0, t1):
        return dl_kdil_from_regime(regime, t0, t1) if regime else float(scenario.dilution_rate)

    # Phase 4: aerosol -> photolysis. Active only with TOMAS on, tuvx photolysis, and the switch.
    aerosol_to_j = bool(scenario.switches.aerosol_to_j) and tomas_active and cfg.photolysis == "tuvx"
    # Phase 5: radiative heating -> T. Requires tuvx (needs the real radiation solve).
    heating_active = bool(scenario.switches.heating_to_t) and cfg.photolysis == "tuvx"
    mie_table = None
    if aerosol_to_j or (heating_active and tomas_active):   # mie needed for aerosol optics/absorption
        from .aerosol_optics import MieTable, aerosol_optical_props
        _wl_nm, _height_edges = calculator_grids(cfg)
        mie_table = MieTable(_wl_nm)

    def _aerosol_props():   # frozen per interval (from the end-of-previous-interval TOMAS state)
        if not aerosol_to_j:
            return None
        return aerosol_optical_props(tstate, _wl_nm, _height_edges, scenario.aerosol_band_km,
                                     mie=mie_table)

    def _particulate_S(state):
        # aerosol sulfate mass (H2SO4-equiv kg, AD-3.8) -> molec/cm^3 of sulfur
        m_so4_kg = float(jnp.sum(state.Mk[:, SRTSO4]))
        return mass_to_conc(m_so4_kg, BOXVOL_CM3, MW_H2SO4)

    def _aero_record():
        aero = (np.nan, np.nan, np.nan, np.nan) if not tomas_active else \
            (het["SA"], het["radius_cm"], het["h2so4wp"], _particulate_S(tstate))
        return aero + (cfg.T,)   # current box temperature (evolves when heating_to_t is on)

    def _sizedist_record():
        # per-bin number [#/cm^3] and wet diameter [m] for the banana plot / size distributions
        from .aerosol_props import _wet_diameters_m
        Dpk, _ = _wet_diameters_m(tstate)
        return np.asarray(tstate.Nk) / float(tstate.boxvol), np.asarray(Dpk)

    t_list, x_list = [0.0], [np.asarray(y0)]
    aero_list = [_aero_record()]
    nk_list, dp_list = ([_sizedist_record()[0]], [_sizedist_record()[1]]) if (
        return_size_dist and tomas_active) else (None, None)
    yc = jnp.asarray(y0)
    for t0, t1 in _outer_intervals(cfg, scenario.days, scenario.dt_couple):
        t_mid = 0.5 * (t0 + t1)
        j_scale = photolysis_scale(_cosz(cfg, t_mid))        # for fallbacks; 0 at night
        jv = _frozen_j_values(cfg, t_mid, aerosol_props=_aerosol_props())
        override = jnp.asarray(photolysis_coeffs(cfg, j_scale, jv))
        args = {**params, "j_scale": j_scale, "sulfur_chain": sulfur, "photo_override": override}
        if tomas_active:   # freeze this interval's aerosol het inputs (end-of-previous-interval state)
            args = {**args, "SA": het["SA"], "particle_radius": het["radius_cm"],
                    "h2so4wp": het["h2so4wp"]}
        yc = step(yc, t0, t1, args)

        if tomas_active:
            # handoff: gas H2SO4 [molec/cm^3] -> Gc[SRTSO4] [kg]; TOMAS depletes it; write back.
            gas_h2so4_kg = conc_to_mass(float(yc[h2so4_idx]), BOXVOL_CM3, MW_H2SO4)
            Gc = tstate.Gc.at[SRTSO4].set(gas_h2so4_kg)
            Nk, Mk, Gc = tomas_step(tstate.Nk, tstate.Mk, Gc, tstate.xk, tstate.temp, tstate.pres,
                                    tstate.boxvol, tstate.rh, tstate.alpha, float(t1 - t0),
                                    fn_scale=nuc_scale)
            if not (bool(jnp.all(jnp.isfinite(Nk))) and bool(jnp.all(jnp.isfinite(Mk)))):
                raise RuntimeError(
                    f"TOMAS microphysics produced non-finite output on interval [{t0}, {t1}] "
                    f"(gas H2SO4 handed in = {float(yc[h2so4_idx]):.3e} molec/cm^3). This is TOMAS's "
                    f"nucleation-rate overflow for a large H2SO4 slug -- reduce dt_couple so the "
                    f"per-step gas H2SO4 stays below ~1e9 molec/cm^3 (see docs/CAVEATS.md, AD-3.9).")
            tstate = tstate._replace(Nk=Nk, Mk=Mk, Gc=Gc)
            yc = yc.at[h2so4_idx].set(mass_to_conc(float(Gc[SRTSO4]), BOXVOL_CM3, MW_H2SO4))
            het = het_inputs(tstate)                          # for the next interval

        if heating_active:
            # radiative heating -> box T (forward-Euler over the interval). The updated T feeds the
            # NEXT interval's chemistry (rate constants) and M (ideal gas at fixed P; species number
            # densities kept -- fixed-volume box, AD-5.3). Aerosol optics enter the heating solve iff
            # aerosol_to_j; aerosol SW absorption (b_abs) is included whenever TOMAS is active.
            aer = _aerosol_props() if aerosol_to_j else None
            dTdt = _heating.box_dTdt(cfg, np.asarray(yc), t_mid, aerosol_props=aer,
                                     tstate=(tstate if tomas_active else None),
                                     mie=(mie_table if tomas_active else None))
            cfg.T = float(cfg.T + dTdt * (t1 - t0))
            cfg.M = air_number_density(cfg.P, cfg.T)
            params["T"], params["M"] = cfg.T, cfg.M

        if dilution_active:   # relax gas + aerosol toward the background (operator-split, last)
            kdil = _kdil(t0, t1)
            yc = dilute_gas(yc, gas_bg, kdil, float(t1 - t0))
            if tomas_active:
                tstate = dilute_aerosol(tstate, tstate_bg, kdil, float(t1 - t0))
                het = het_inputs(tstate)   # het inputs reflect the diluted aerosol next interval

        t_list.append(t1)
        x_list.append(np.asarray(yc))
        aero_list.append(_aero_record())
        if nk_list is not None:
            nk, dp = _sizedist_record()
            nk_list.append(nk)
            dp_list.append(dp)

    t = np.asarray(t_list)
    x = np.asarray(x_list)
    out = [t, x]
    if return_aerosol:
        a = np.asarray(aero_list)
        out.append({"SA": a[:, 0], "radius_cm": a[:, 1], "h2so4wp": a[:, 2],
                    "particulate_S": a[:, 3], "T": a[:, 4]})
    if return_state:
        out.append(tstate)
    if return_size_dist:
        # dict with per-time per-bin number [#/cm^3] and wet diameter [m] (None if TOMAS inactive)
        sd = None if nk_list is None else {"n_cm3": np.asarray(nk_list), "Dp_m": np.asarray(dp_list)}
        out.append(sd)
    return tuple(out) if len(out) > 2 else (t, x)
