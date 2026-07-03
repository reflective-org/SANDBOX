# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Two-level operator-split coupling driver.

Runs the coupled box on the JAX gas backend with a TWO-LEVEL time integration (see
docs/time-integration-plan.md):

* OUTER (radiation) step ``dt_couple`` (== ``dt_rad``, ~minutes): compute the photolysis J from the
  TUV-x port at the step MIDPOINT (2nd-order splitting) and FREEZE it (plus the aerosol optics and the
  heterogeneous-chemistry inputs) for the whole outer step. The outer grid is snapped to sunrise/sunset
  (no interval straddles the terminator; J is off at night). One SZA source (gas ``solar.py``, the same
  the adapter uses) drives J, the day/night snap, and the fallback ``j_scale``. ``switches.sulfur``
  drives the gas sulfur gate.
* INNER (micro) step -- ADAPTIVE, fine early: within each outer step the gas ODE is solved ONCE with a
  DENSE output (H2SO4 accumulates, no aerosol sink), then TOMAS consumes that H2SO4 envelope in
  adaptive micro-steps (``_tomas_micro``). Each micro-step is sized so the fractional change in gaseous
  H2SO4 and aerosol number stays <= ``micro_eps``, bounded to [``micro_floor_s``, ``micro_cap_s``]. This
  is approach B: nucleation only ever sees one micro-step's worth of H2SO4 (so it self-limits instead of
  running away on a coarse step), while the expensive TUV-x solve stays on the coarse outer cadence.

TOMAS microphysics (Phase 3) runs nucleation/condensation/coagulation with its own SO2 chemistry OFF;
the gas model owns sulfur. TOMAS is active iff any of ``switches.{nucleation,condensation,coagulation}``
is on; otherwise the run is the gas-only path with the prescribed ``cfg.SA`` (one endpoint gas solve per
outer step, no micro-loop). The aerosol surface area / wet radius / H2SO4 weight-percent from the end of
each outer step feed the NEXT step's heterogeneous chemistry (frozen per outer step, like J); the first
step uses the initial TOMAS state. Heating (Phase 5) and dilution (Phase 6) are applied once per outer
step (heating is a slow ~0.1 K/day term; dilution's analytic relaxation is exact over the outer step).

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
from tuvx_photolysis_adapter import (compute_j_and_heating, calculator_grids,  # noqa: E402
                                     _box_altitude_km, _TUVX_ROOT)
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
    # plain floats throughout: terminator edges come out of the bisection as np.float64, and the
    # weak/strong scalar-dtype flip on (t0, t1) would retrace the jitted gas solve at each transition.
    E = sorted(float(e) for e in edges if 0.0 <= e <= total)
    return list(zip(E[:-1], E[1:]))


def _frozen_j_and_heating(cfg, t_mid: float, aerosol_props=None):
    """Frozen per-reaction J AND box-heating inputs at the interval midpoint, from ONE TUV-x radiation
    solve (uncached -> dt_couple drives the recompute, bypassing the adapter's 120 s cache).
    ``(None, None)`` in non-tuvx modes (photolysis_coeffs then uses j45*j_scale; heating requires
    tuvx). ``aerosol_props`` (Phase 4) injects the TOMAS aerosol radiator for this solve -- the SAME
    frozen (end-of-previous-interval) aerosol now feeds both J and heating. The adapter returns
    zeros / ``None`` at night."""
    if cfg.photolysis != "tuvx":
        return None, None
    return compute_j_and_heating(cfg, t_mid, aerosol_props=aerosol_props)


# H2SO4 number density [molec/cm^3] below which gaseous sulfuric acid is negligible for nucleation --
# used as the denominator floor in the production-fraction step controller (avoids div-by-~0 when the
# gas H2SO4 is essentially zero, e.g. at t=0 in a clean background, so the step is free to grow).
_H2SO4_NEGLIGIBLE = 1.0e5


def micro_consume(env_h2so4_conc, t0, t1, tstate, tomas_step, nuc_scale, eps, floor, cap, dt_start,
                  on_accept=None):
    """Adaptive TOMAS micro-loop over [t0, t1] consuming the gas H2SO4 envelope (approach B).

    ``env_h2so4_conc(t) -> gaseous H2SO4 [molec/cm^3]`` is the gas solution WITHOUT any aerosol sink
    (H2SO4 accumulates), evaluated at micro-step boundaries. TOMAS removes H2SO4 incrementally, so it
    only ever sees one micro-step's worth of *fresh* production.

    What the micro-step controls is the OPERATOR-SPLIT COUPLING error: each step lumps the gas H2SO4
    produced over ``dt`` and hands it to TOMAS, which then resolves nucleation/condensation/coagulation
    for that ``dt`` with its OWN internal (adaptive) sub-stepping. So the outer step must keep the
    *produced* H2SO4 small vs the standing amount -- ``production/standing <= eps`` -- but must NOT try
    to also resolve the nucleation kinetics (TOMAS does that; controlling the fractional change in
    particle number would over-refine to an infeasible step count when N is small, and controlling the
    fraction CONSUMED would double-resolve TOMAS's internal sub-stepping). ``dt`` is bounded to
    [``floor``, ``cap``] s. Non-finite TOMAS output (its internal sub-stepping overflowed) shrinks
    ``dt``; hitting the floor while STILL non-finite RAISES -- never silently swallow a blown-up step.
    If the floor is reached only because the production fraction is still > eps (but TOMAS is finite),
    the step is accepted and a one-line under-resolution notice is printed (the coupling is slightly
    under-resolved but the microphysics itself is valid).

    Shared verbatim by the JAX driver (``sol.evaluate``) and the NumPy mirror (SciPy ``sol.sol``) so the
    two-level stepping is identical across backends. Returns ``(tstate, removal_kg, next_dt, n_micro)``.
    """
    Nk, Mk, Gc = tstate.Nk, tstate.Mk, tstate.Gc
    removal_kg = 0.0
    t = t0
    env_prev = env_h2so4_conc(t0)                                    # H2SO4 envelope [molec/cm^3] at t
    dt = min(max(dt_start, floor), cap, t1 - t0)
    n_micro = 0
    while t < t1 - 1e-9:
        dt = min(dt, t1 - t)
        t_end = t + dt
        env_end = env_h2so4_conc(t_end)                             # envelope at step end
        produced = max(env_end - env_prev, 0.0)                     # gas H2SO4 made this step [/cm^3]
        frac_prod = produced / max(env_end, _H2SO4_NEGLIGIBLE)      # coupling error metric
        # standing H2SO4 available to TOMAS = envelope at step end minus what TOMAS already removed
        avail_kg = max(conc_to_mass(env_end, BOXVOL_CM3, MW_H2SO4) - removal_kg, 0.0)
        Gc_in = Gc.at[SRTSO4].set(avail_kg)
        Nk2, Mk2, Gc2 = tomas_step(Nk, Mk, Gc_in, tstate.xk, tstate.temp, tstate.pres,
                                   tstate.boxvol, tstate.rh, tstate.alpha, float(dt),
                                   fn_scale=nuc_scale)
        # ONE device->host transfer for the two scalars the loop needs (finite flag + leftover H2SO4),
        # instead of ~4 separate .item() syncs -- the per-step sync/dispatch was ~40 ms (vs ~1.6 ms for
        # the TOMAS step itself), so batching + the host-interpolated envelope is the whole speedup.
        _finite_flag = (jnp.all(jnp.isfinite(Nk2)) & jnp.all(jnp.isfinite(Mk2))
                        & jnp.all(jnp.isfinite(Gc2))).astype(jnp.float64)
        _flag, _gc_so4 = (float(v) for v in np.asarray(jnp.stack([_finite_flag, Gc2[SRTSO4]])))
        finite = _flag > 0.5
        at_floor = dt <= floor * (1.0 + 1e-9)
        if not finite:
            if at_floor:
                raise RuntimeError(
                    f"TOMAS produced non-finite output at the micro-step floor ({floor:g} s) on outer "
                    f"interval [{t0:.4g}, {t1:.4g}] at t={t:.4g} s (H2SO4 handed in "
                    f"{avail_kg:.3e} kg, nucleation_rate_scale={nuc_scale}). TOMAS's internal "
                    f"sub-stepping overflowed even at the floor -- lower micro_floor_s or the "
                    f"nucleation scale (see docs/CAVEATS.md). NOT swallowing this.")
            dt = max(dt * 0.5, floor)
            continue
        if frac_prod > eps and not at_floor:                        # refine to resolve the coupling
            dt = max(dt * 0.5, floor)
            continue
        if frac_prod > eps and at_floor:                            # finite but coupling under-resolved
            print(f"[coupled] NOTE: micro-step at floor ({floor:g} s) on [{t0:.4g},{t1:.4g}] t={t:.4g}s "
                  f"has production fraction {frac_prod:.3f} > micro_eps={eps} (H2SO4 rising fast); "
                  f"accepting -- microphysics finite but the gas->TOMAS coupling is under-resolved "
                  f"here. Lower micro_floor_s for a tighter split.")
        Nk, Mk, Gc = Nk2, Mk2, Gc2                                    # accept
        removal_kg += max(avail_kg - _gc_so4, 0.0)                     # leftover H2SO4 pulled above
        if on_accept is not None:                                     # diagnostics hook (t, dt, frac)
            on_accept(t, dt, frac_prod)
        t = t_end
        env_prev = env_end
        n_micro += 1
        if frac_prod < 0.5 * eps:                                     # relax when comfortably under eps
            dt = min(dt * 2.0, cap)
    return tstate._replace(Nk=Nk, Mk=Mk, Gc=Gc), removal_kg, max(dt, floor), n_micro


def _envelope_pts(dt_couple: float) -> int:
    """Point count for the per-outer-step H2SO4 envelope grid: ~2 s spacing at the run's dt_couple,
    in [301, 1025]. Constant PER RUN (not per interval) so the jitted grid-save solve keeps one input
    shape and compiles once; terminator-snapped (shorter) intervals just get a denser grid. The
    envelope is smooth/monotone, so ~2 s spacing interpolates it well within micro_eps -- the fine
    down-to-micro_floor_s resolution is needed for the TOMAS handoff, NOT for the envelope shape."""
    return int(np.clip(round(dt_couple / 2.0) + 1, 301, 1025))


def _envelope_grid(t0, t1, n_pts: int = 301):
    """Times [t0..t1] (inclusive, fixed length ``n_pts``) to save the gas H2SO4 envelope on for the
    micro-loop's host interpolation."""
    return np.linspace(t0, t1, n_pts)


def run_coupled(scenario, return_aerosol=False, return_state=False, return_size_dist=False,
                return_photolysis=False):
    """Integrate a CoupledScenario with operator splitting.

    Returns ``(t [s], states [n_t, n_species])``. With ``return_aerosol=True`` also returns a dict of
    per-output-time aerosol diagnostics (``SA`` [um^2/cm^3], ``radius_cm``, ``h2so4wp`` [wt%],
    ``particulate_S`` [molec/cm^3 as H2SO4-equiv]); the arrays are all-NaN when TOMAS is inactive.
    With ``return_state=True`` also appends the final ``TomasState`` (or ``None`` if TOMAS inactive),
    e.g. for plotting the evolved size distribution. With ``return_photolysis=True`` also appends a
    dict of the FROZEN per-interval photolysis coefficients actually used by the gas solve:
    ``{"t_mid" [n_int], "J" [n_int, n_photo] (s^-1), "equations" [n_photo]}`` (the ``photo_override``
    rows of ``reactions.photolysis_coeffs``, i.e. absolute TUV-x J or the j45*j_scale fallback).
    Extra outputs are appended in that order.
    """
    cfg = to_model_config(scenario)
    y0 = initial_state(scenario)
    sulfur = float(scenario.switches.sulfur)   # single gate source (NumPy match: build_env(sulfur_chain=))
    params = dict(T=cfg.T, M=cfg.M, P=cfg.P, SA=cfg.SA, WTR=cfg.WTR, Yn2o5=cfg.Yn2o5)

    tomas_step = make_microphysics_step(scenario.switches,
                                        ion_pair_rate=float(scenario.ion_pair_rate))
    tomas_active = tomas_step is not None
    tstate = initial_tomas_state(scenario) if tomas_active else None
    het = het_inputs(tstate) if tomas_active else None
    h2so4_idx = IDX["H2SO4"]

    # Two-level integration: the gas ODE is solved ONCE per outer step. With TOMAS active we need a
    # DENSE solution so the adaptive micro-loop can query the gas H2SO4 envelope at sub-interval times
    # while TOMAS consumes it (approach B). Without TOMAS, just the endpoint (Phase-2 gas-only path).
    # jit_grid + forward_only make the per-outer-step gas solve ~85x faster (eager diffeqsolve ~1.2 s
    # -> jitted+LU ~15 ms): the coupled forward run takes no gradients through the solve, and the
    # envelope grid has a fixed length so the jit compiles once. Bit-identical to the eager solve.
    step = make_frozen_step(cfg.opt, atol=jnp.asarray(_abstol(cfg.opt)),
                            jit_grid=True, forward_only=True)

    nuc_scale = float(scenario.nucleation_rate_scale)   # Phase 7 knob -> TOMAS nucleation fn_scale

    # Opt-in JOINT solver: one stiff ODE (chem + nucleation + coagulation + condensation gas-sink)
    # per outer step, then a PPM size remap -- eliminates the chem->nuc->cond ordering error. Requires
    # all three TOMAS processes on and heating off (the aerosol temperature is frozen in the joint
    # step); use the "split" path otherwise. See coupled/joint_solver.py.
    joint = scenario.microphysics_solver == "joint"
    joint_step = None
    if joint:
        if not tomas_active or not (scenario.switches.nucleation and scenario.switches.condensation
                                    and scenario.switches.coagulation):
            raise NotImplementedError("microphysics_solver='joint' needs nucleation+condensation+"
                                      "coagulation all ON (use 'split' for partial-process runs)")
        if bool(scenario.switches.heating_to_t) and cfg.photolysis == "tuvx":
            raise NotImplementedError("microphysics_solver='joint' does not support heating_to_t yet "
                                      "(aerosol T is frozen in the joint step); turn heating off")
        from .joint_solver import make_joint_step
        joint_step = make_joint_step(
            cfg.opt, int(scenario.tomas_nbins), tstate.temp, tstate.pres, tstate.boxvol,
            ion_pair_rate=float(scenario.ion_pair_rate), nuc_scale=nuc_scale,
            enable_inorganic=1.0, enable_organic=0.0)
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
    for name, bg_ppt in scenario.dilution_background.items():   # explicit pptv overrides (win over 0)
        if name not in IDX:
            raise ValueError(f"dilution_background has unknown species {name!r}")
        gas_bg[IDX[name]] = float(bg_ppt) * 1e-12 * cfg.M       # pptv -> molec/cm^3 (as initial_state)
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
        from .aerosol_optics import MieTable, aerosol_optical_props, placement_band_km, bin_radii_m
        _wl_nm, _height_edges = calculator_grids(cfg)
        # radii from the STATE's own bin grid (40- or 80-bin), not the module default
        mie_table = MieTable(_wl_nm, radii_m=bin_radii_m(np.asarray(tstate.xk)))
        # pressure-anchored plume: band centered on the box altitude (same USSA reference the J
        # interpolation uses), thickness = scenario.aerosol_thickness_km -- or the absolute override.
        _box_alt = _box_altitude_km(float(cfg.P), str(getattr(cfg, "tuvx_data_root", _TUVX_ROOT)))
        _band_km = placement_band_km(_box_alt, scenario)

    def _aerosol_props():   # frozen per interval (from the end-of-previous-interval TOMAS state)
        if not aerosol_to_j:
            return None
        return aerosol_optical_props(tstate, _wl_nm, _height_edges, _band_km, mie=mie_table)

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

    # --- adaptive inner (micro) step for the gas<->TOMAS handoff (two-level integration, approach B) --
    eps = float(scenario.micro_eps)
    floor = float(scenario.micro_floor_s)
    cap = float(scenario.micro_cap_s)
    dt_micro = floor                # start fine (t=0 burst); carried/relaxed across outer intervals
    micro_total = 0
    t_list, x_list = [0.0], [np.asarray(y0)]
    aero_list = [_aero_record()]
    nk_list, dp_list = ([_sizedist_record()[0]], [_sizedist_record()[1]]) if (
        return_size_dist and tomas_active) else (None, None)
    env_pts = _envelope_pts(scenario.dt_couple)     # per-run constant -> one jit compile
    if return_photolysis:                           # photo positions/equations of the override vector
        from reactions import MECHANISM
        _photo_idx = [i for i, r in enumerate(MECHANISM.active) if r.kind == "photo"]
        _photo_eqs = [MECHANISM.active[i].equation for i in _photo_idx]
        j_tmid, j_rows = [], []
    yc = jnp.asarray(y0)
    for t0, t1 in _outer_intervals(cfg, scenario.days, scenario.dt_couple):
        t_mid = 0.5 * (t0 + t1)
        # plain float: photolysis_scale returns 0.0 (float) at night but np.float64 by day, and the
        # weak/strong dtype flip would force an extra one-time trace of the jitted gas solve.
        j_scale = float(photolysis_scale(_cosz(cfg, t_mid)))   # for fallbacks; 0 at night
        # ONE TUV-x radiation solve per interval feeds BOTH the frozen J and the heating step (same
        # SZA, same frozen end-of-previous-interval aerosol; previously heating solved again).
        jv, heat_res = _frozen_j_and_heating(cfg, t_mid, aerosol_props=_aerosol_props())
        override = jnp.asarray(photolysis_coeffs(cfg, j_scale, jv))
        if return_photolysis:
            j_tmid.append(t_mid)
            j_rows.append(np.asarray(override)[_photo_idx])
        args = {**params, "j_scale": j_scale, "sulfur_chain": sulfur, "photo_override": override,
                "k_so2_ho2": float(scenario.so2_ho2_rate)}
        if tomas_active:   # freeze this interval's aerosol het inputs (end-of-previous-interval state)
            args = {**args, "SA": het["SA"], "particle_radius": het["radius_cm"],
                    "h2so4wp": het["h2so4wp"]}
        if joint:
            # JOINT solver: one stiff ODE co-evolves gas + nucleation + coagulation + condensation
            # gas-sink over [t0,t1] (nucleation sees the true pseudo-steady H2SO4), then a PPM remap
            # deposits the ODE-integrated condensed mass onto the bins. Replaces the envelope+micro-loop.
            conc1, Nk1, Mk1 = joint_step(yc, tstate.Nk, tstate.Mk, t0, t1, args)
            yc = jnp.asarray(conc1)
            tstate = tstate._replace(Nk=Nk1, Mk=Mk1)
            het = het_inputs(tstate)                          # for the next interval
        elif tomas_active:
            # ONE gas solve over the outer interval (H2SO4 accumulates, no aerosol sink), SAVED on a
            # fine grid; TOMAS then consumes that H2SO4 envelope in adaptive micro-steps (fine early),
            # interpolating the envelope on the HOST (np.interp) -- no per-micro-step device call. Final
            # gas H2SO4 = envelope(t1) minus cumulative TOMAS removal. Two-level split that keeps
            # nucleation from seeing a whole interval's H2SO4 at once (docs/time-integration-plan).
            ts_grid = _envelope_grid(t0, t1, env_pts)
            sol = step(yc, t0, t1, args, save_ts=ts_grid)
            env_grid = np.asarray(sol.ys[:, h2so4_idx])       # smooth H2SO4 envelope [molec/cm^3]
            tstate, removal_kg, dt_micro, n_micro = micro_consume(
                lambda tt: float(np.interp(tt, ts_grid, env_grid)), t0, t1, tstate, tomas_step,
                nuc_scale, eps, floor, cap, dt_micro)
            micro_total += n_micro
            yc = sol.ys[-1]
            env_end_kg = conc_to_mass(float(env_grid[-1]), BOXVOL_CM3, MW_H2SO4)   # envelope(t1)
            yc = yc.at[h2so4_idx].set(
                mass_to_conc(max(env_end_kg - removal_kg, 0.0), BOXVOL_CM3, MW_H2SO4))
            het = het_inputs(tstate)                          # for the next interval
        else:
            yc = step(yc, t0, t1, args)                       # gas-only endpoint (Phase-2 path)

        if heating_active:
            # radiative heating -> box T (forward-Euler over the interval). The updated T feeds the
            # NEXT interval's chemistry (rate constants) and M (ideal gas at fixed P; species number
            # densities kept -- fixed-volume box, AD-5.3). The actinic flux comes from THIS interval's
            # single radiation solve (heat_res, frozen midpoint J's field -- with the aerosol radiator
            # iff aerosol_to_j); aerosol SW absorption (b_abs) is included whenever TOMAS is active.
            dTdt = _heating.dTdt_from_heating(cfg, np.asarray(yc), heat_res,
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
    if return_photolysis:
        out.append({"t_mid": np.asarray(j_tmid), "J": np.asarray(j_rows), "equations": _photo_eqs})
    return tuple(out) if len(out) > 2 else (t, x)
