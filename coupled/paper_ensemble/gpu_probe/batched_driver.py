# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Batched (GPU) coupled driver: B ensemble members stepped together, one outer interval at a time.

A SEPARATE implementation for the GPU. It changes nothing in the CPU model: ``coupled/driver.py``,
``coupled/tomas_bridge.py``, ``run_v2.py`` and the submodules are only IMPORTED. It reproduces
``coupled.driver.run_coupled`` for the v2 ensemble configuration and is checked against CPU runs of the
same cases (``run_batched.py``, RESULTS.md).

What is shared and what is per member
-------------------------------------
Members of one batch must share everything that sets the outer-interval grid and the photolysis: site
(T, P, WTR, lat/lon, date), days, dt_couple, the dilution regime and its refinement, the micro-step
settings and the switches. They may differ in the initial gas (V0 / ctrl), ``condensation_alpha``,
``nucleation_rate_scale`` and ``coag_kernel_scale`` -- i.e. one batch = one dilution case of the v2
ensemble (5 V0 x 18 microphysics plume members + 18 ctrl twins). ``check_batchable`` enforces it.

Per outer interval (same order and formulas as run_coupled):
1. TUV-x J at the interval midpoint: ONE solve, shared (J does not depend on the member with
   aerosol_to_j off; checked at the first interval against the last member's config).
2. Gas solve for all members (vmapped Diffrax Kvaerno5, the driver's own vector field / solver /
   PID controller / save grid). The stall retry is per member with the driver's sticky semantics
   (probe dt0=1e-10 with a 5e4 step cap; failed or sticky members re-solved from dt0=1e-2, 1e6 cap;
   re-probe every 48 intervals).
3. TOMAS micro-loop for all members: ``micro_consume`` translated line by line into a jitted
   ``lax.while_loop`` and vmapped, so each member keeps its own adaptive dt; the loop runs until the
   last member reaches t1 (finished members are masked). Non-finite TOMAS output at the floor is
   reported per member (the CPU driver raises).
4. Gas H2SO4 = envelope(t1) - removal; SO2 chemistry counter; dilution (gas + aerosol, per member
   backgrounds); het inputs for the next interval (the het_inputs formulas, batched).

Performance: the TOMAS step on a GPU is dominated by ``mnfix_jax``'s 240 sequential bin iterations;
``fast_mnfix.install()`` (bitwise-identical replacement, this process only) is applied by default.
"""
from __future__ import annotations

import math
import time

import numpy as np

import coupled.tomas_bridge as tb                      # puts tomas_jax on sys.path
from coupled.paper_ensemble.gpu_probe import fast_mnfix

import jax
import jax.numpy as jnp
import diffrax
from diffrax import ODETerm, PIDController, SaveAt, diffeqsolve

import coupled.driver as drv
from coupled import aerosol_props as ap
from coupled.model_bridge import initial_state, to_model_config
from coupled.units import conc_to_mass, mass_to_conc
from coupled.tomas_bridge import (initial_tomas_state, active_processes, particle_sulfate_mw,
                                  SRTSO4, BOXVOL_CM3, MW_H2SO4)
from tomas_jax.solvers.condensation import make_step
from tomas_jax.physics.dilution import dilution_step
from tomas_jax.physics.properties import calc_particle_properties
from tomas_jax.physics.water_equilibrium import calc_equilibrium_water_h2so4

from config import IDX                                  # noqa: E402 (gas model)
from driver import _abstol                              # noqa: E402 (gas model)
from reactions import photolysis_coeffs, MECHANISM      # noqa: E402
from solar import photolysis_scale                      # noqa: E402
from jaxmodel.model import make_frozen_vf, _stiff_solver  # noqa: E402

#: scenario fields that must be identical across a batch (everything except the per-member knobs)
_PER_MEMBER = {"concentrations", "condensation_alpha", "nucleation_rate_scale", "coag_kernel_scale"}
_RETRY_EVERY = 48            # driver._REPROBE
_PROBE_MAXSTEPS = 50_000     # driver default COUPLED_GAS_MAXSTEPS
_RETRY_MAXSTEPS = 1_000_000


def check_batchable(scenarios):
    import dataclasses
    s0 = scenarios[0]
    for f in dataclasses.fields(s0):
        if f.name in _PER_MEMBER:
            continue
        v0 = getattr(s0, f.name)
        for s in scenarios[1:]:
            if getattr(s, f.name) != v0:
                raise ValueError(f"batch members differ in {f.name!r}: {v0!r} vs {getattr(s, f.name)!r}")
    sw = s0.switches
    if sw.aerosol_to_j or sw.heating_to_t or not (sw.nucleation and sw.condensation and sw.coagulation):
        raise ValueError("batched driver supports the v2 configuration only (all TOMAS processes on, "
                         "aerosol_to_j and heating_to_t off)")
    if s0.micro_nuc_eps != 0.0:
        raise ValueError("batched driver supports micro_nuc_eps == 0 only (production setting)")
    if s0.legacy_dilution:
        raise ValueError("legacy dilution regimes are not supported")


# ---------------------------------------------------------------- TOMAS step (batched knobs) -----
def make_tomas_step(sc):
    """The bridge's step (same make_step scheme choices and fixed kwargs), with ``fn_scale`` and
    ``coag_kernel_scale`` as per-member arguments instead of baked Python floats."""
    step = make_step(active_processes(sc.switches), cond_method=tb._COND_METHOD,
                     nucl_scheme=tb._NUCL_SCHEME, water_scheme=tb._WATER_SCHEME,
                     water_every_process=bool(sc.water_every_process),
                     molar_so4_transfer=bool(sc.molar_so4_transfer))
    fion = float(sc.ion_pair_rate)
    org = float(sc.nuc_org_mass_frac)

    def f(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, fns, cks):
        return step(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                    fion=fion, coag_kernel_scale=cks, nuc_org_mass_frac=org, fn_scale=fns)
    return f


# ---------------------------------------------------------------- micro-loop (one member) --------
def _np_interp(x, xp, fp):
    """numpy.interp's formula (slope * (x - xp[j]) + fp[j]) for a scalar x in [xp[0], xp[-1]]."""
    n = xp.shape[0]
    j = jnp.clip(jnp.searchsorted(xp, x, side="right") - 1, 0, n - 2)
    slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])
    v = slope * (x - xp[j]) + fp[j]
    v = jnp.where(x >= xp[n - 1], fp[n - 1], v)
    return jnp.where(x <= xp[0], fp[0], v)


def make_micro_loop(tomas_step, eps, floor, cap, prod_ref, handoff):
    """``driver.micro_consume`` (nuc_eps == 0, no on_accept hook) as a pure JAX function of ONE member."""
    standing = prod_ref == "standing"
    mid = handoff == "mid"
    NEG = drv._H2SO4_NEGLIGIBLE

    def run(ts_grid, env_grid, t0, t1, Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, fns, cks, dt_start):
        env = lambda tt: _np_interp(tt, ts_grid, env_grid)   # noqa: E731
        dt0 = jnp.minimum(jnp.minimum(jnp.maximum(dt_start, floor), cap), t1 - t0)
        # carry: t, dt, env_prev, removal_kg, Nk, Mk, Gc, n_micro, n_calls, err, n_floor_notes
        c0 = (t0, dt0, env(t0), jnp.float64(0.0), Nk, Mk, Gc, jnp.int64(0), jnp.int64(0),
              jnp.bool_(False), jnp.int64(0))

        def cond(c):
            return (c[0] < t1 - 1e-9) & (~c[9])

        def body(c):
            t, dt, env_prev, removal, Nk, Mk, Gc, n_micro, n_calls, err, notes = c
            dt = jnp.minimum(dt, t1 - t)
            t_end = t + dt
            env_end = env(t_end)
            produced = jnp.maximum(env_end - env_prev, 0.0)
            if standing:
                frac_prod = produced / jnp.maximum(env_end - mass_to_conc(removal, BOXVOL_CM3, MW_H2SO4), NEG)
            else:
                frac_prod = produced / jnp.maximum(env_end, NEG)
            env_hand = env(t + 0.5 * dt) if mid else env_end
            avail = jnp.maximum(conc_to_mass(env_hand, BOXVOL_CM3, MW_H2SO4) - removal, 0.0)
            Gc_in = Gc.at[SRTSO4].set(avail)
            Nk2, Mk2, Gc2 = tomas_step(Nk, Mk, Gc_in, xk, temp, pres, boxvol, rh, alpha, dt, fns, cks)
            finite = jnp.all(jnp.isfinite(Nk2)) & jnp.all(jnp.isfinite(Mk2)) & jnp.all(jnp.isfinite(Gc2))
            at_floor = dt <= floor * (1.0 + 1e-9)
            half = jnp.maximum(dt * 0.5, floor)
            new_err = (~finite) & at_floor
            reject = ((~finite) & (~at_floor)) | (finite & (frac_prod > eps) & (~at_floor))
            accept = finite & (~reject)
            notes = notes + (accept & (frac_prod > eps) & at_floor).astype(jnp.int64)
            relax = frac_prod < 0.5 * eps
            dt_acc = jnp.where(relax, jnp.minimum(dt * 2.0, cap), dt)
            return (jnp.where(accept, t_end, t),
                    jnp.where(accept, dt_acc, jnp.where(reject, half, dt)),
                    jnp.where(accept, env_end, env_prev),
                    jnp.where(accept, removal + jnp.maximum(avail - Gc2[SRTSO4], 0.0), removal),
                    jnp.where(accept, Nk2, Nk), jnp.where(accept, Mk2, Mk), jnp.where(accept, Gc2, Gc),
                    n_micro + accept.astype(jnp.int64), n_calls + 1, err | new_err, notes)

        t, dt, _, removal, Nk, Mk, Gc, n_micro, n_calls, err, notes = jax.lax.while_loop(cond, body, c0)
        return Nk, Mk, Gc, removal, jnp.maximum(dt, floor), n_micro, n_calls, err, notes
    return run


# ---------------------------------------------------------------- het inputs / records (batched) --
def make_aero_fns(molar_so4):
    mw_p = particle_sulfate_mw(molar_so4)

    def one(Nk, Mk, temp, pres, rh, boxvol):
        Mk_wet = calc_equilibrium_water_h2so4(Mk, rh, temp)          # ap._wet_diameters_m
        Dpk, _Dk, _ck = calc_particle_properties(Nk, Mk_wet, temp, pres)
        r_m = 0.5 * Dpk
        return dict(area=jnp.sum(Nk * math.pi * Dpk ** 2), m2=jnp.sum(Nk * r_m ** 2),
                    m3=jnp.sum(Nk * r_m ** 3), m_so4=jnp.sum(Mk[:, ap._SRTSO4]),
                    m_h2o=jnp.sum(Mk_wet[:, ap._SRTH2O]), Dpk=Dpk, n_cm3=Nk / boxvol)
    batched = jax.jit(jax.vmap(one, in_axes=(0, 0, None, None, None, None)))

    def het_and_records(Nk, Mk, temp, pres, rh, boxvol):
        """het_inputs (SA, radius_cm, h2so4wp) + particulate_S + size-dist rows, per member, with the
        aerosol_props / driver formulas finished on the host in float64."""
        d = jax.device_get(batched(Nk, Mk, temp, pres, rh, boxvol))
        bv = float(boxvol)
        SA = d["area"] * 1.0e12 / bv
        radius = np.where(d["m2"] <= 0.0, 0.1e-4, (d["m3"] / np.where(d["m2"] > 0, d["m2"], 1.0)) * 100.0)
        m_so4 = d["m_so4"] * (ap._SO4_TO_H2SO4 if molar_so4 else 1.0)
        denom = m_so4 + d["m_h2o"]
        wp = np.where(denom <= 0.0, 0.0, 100.0 * m_so4 / np.where(denom > 0, denom, 1.0))
        pS = mass_to_conc(d["m_so4"], BOXVOL_CM3, mw_p)
        return dict(SA=SA, radius_cm=radius, h2so4wp=wp, particulate_S=pS, n_cm3=d["n_cm3"], Dp_m=d["Dpk"])
    return het_and_records


# ---------------------------------------------------------------- the driver ----------------------
def run_batched(scenarios, use_fast_mnfix=True, log=print, progress_every_h=6.0):
    """Run the batch; returns one dict per member with run_coupled's outputs
    (t, x, aero{SA,radius_cm,h2so4wp,particulate_S,T,so2_chem_loss_cum,V_ratio_applied},
    sd{n_cm3,Dp_m}, jrec{t_mid,J,equations}, final TomasState) plus per-member micro-step counts."""
    check_batchable(scenarios)
    if use_fast_mnfix:
        fast_mnfix.install()
    B = len(scenarios)
    sc0 = scenarios[0]
    cfgs = [to_model_config(s) for s in scenarios]
    cfg = cfgs[0]
    y0 = np.stack([np.asarray(initial_state(s), dtype=float) for s in scenarios])     # (B, nsp)
    ts0 = [initial_tomas_state(s) for s in scenarios]
    for st in ts0[1:]:
        for k in ("xk", "temp", "pres", "boxvol", "rh"):
            assert np.array_equal(np.asarray(getattr(st, k)), np.asarray(getattr(ts0[0], k))), k
    xk, temp, pres, boxvol, rh = (ts0[0].xk, ts0[0].temp, ts0[0].pres, ts0[0].boxvol, ts0[0].rh)
    alpha = jnp.asarray([float(st.alpha) for st in ts0])
    Nk = jnp.stack([st.Nk for st in ts0]); Mk = jnp.stack([st.Mk for st in ts0]); Gc = jnp.stack([st.Gc for st in ts0])
    Nk_bg, Mk_bg, Gc_bg = Nk, Mk, Gc                                                   # tstate_bg
    fns = jnp.asarray([float(s.nucleation_rate_scale) for s in scenarios])
    cks = jnp.asarray([float(s.coag_kernel_scale) for s in scenarios])
    molar_so4 = bool(sc0.molar_so4_transfer)
    h2so4_idx, so2_idx = IDX["H2SO4"], IDX["SO2"]
    sulfur = float(sc0.switches.sulfur)

    # dilution: regime, per-member gas background (initial state with the scenario's overrides)
    regime = sc0.dilution_regime
    dl_mod = drv._dilution
    gas_bg = y0.copy()
    for name in sc0.dilution_zero_species:
        gas_bg[:, IDX[name]] = 0.0
    for name, bg_ppt in sc0.dilution_background.items():
        gas_bg[:, IDX[name]] = np.array([float(bg_ppt) * 1e-12 * c.M for c in cfgs])
    kdil_of = (lambda a, b: dl_mod.kdil_from_regime(regime, a, b)) if regime else (lambda a, b: float(sc0.dilution_rate))

    # gas solve: the driver's vector field / solver / controller / grid, vmapped, throw=False
    term = ODETerm(make_frozen_vf(cfg.opt))
    solver = _stiff_solver(forward_only=True)
    ctrl = PIDController(rtol=1e-3, atol=jnp.asarray(_abstol(cfg.opt)))

    def _solve(maxs):
        def one(y, args, t0, t1, ts, dt0):
            sol = diffeqsolve(term, solver, t0=t0, t1=t1, dt0=dt0, y0=y, args=args, stepsize_controller=ctrl,
                              saveat=SaveAt(ts=ts), max_steps=maxs, throw=False)
            return sol.ys, sol.result == diffrax.RESULTS.successful
        return one
    params = dict(T=cfg.T, M=cfg.M, P=cfg.P, WTR=cfg.WTR, Yn2o5=cfg.Yn2o5)
    batched_keys = ("SA", "particle_radius", "h2so4wp")

    def make_gas(maxs):
        one = _solve(maxs)

        def f(y, shared, per, t0, t1, ts, dt0):
            def g(y_, per_, dt0_):
                return one(y_, {**shared, **per_}, t0, t1, ts, dt0_)
            return jax.vmap(g)(y, per, dt0)
        return jax.jit(f)
    gas_probe, gas_retry = make_gas(_PROBE_MAXSTEPS), make_gas(_RETRY_MAXSTEPS)

    tomas_step = make_tomas_step(sc0)
    micro = make_micro_loop(tomas_step, float(sc0.micro_eps), float(sc0.micro_floor_s), float(sc0.micro_cap_s),
                            str(sc0.micro_prod_ref), str(sc0.micro_handoff))
    micro_b = jax.jit(jax.vmap(micro, in_axes=(None, 0, None, None, 0, 0, 0, None, None, None, None, None,
                                               0, 0, 0, 0)))
    dilute_b = jax.jit(jax.vmap(dilution_step, in_axes=(0, 0, 0, None, None, 0, 0, 0)))
    aero = make_aero_fns(molar_so4)

    # intervals (same function and arguments as run_coupled)
    _refine = bool(sc0.switches.dilution) and bool(regime) and sc0.dilution_max_dlnv > 0.0
    _gc = drv.DL_GAUSSIAN_CASES.get(regime) if _refine else None
    _burst = ((_gc.t_burst_start, _gc.t_burst_start + _gc.tau_burst)
              if _gc is not None and _gc.tau_burst is not None else ())
    intervals = drv._outer_intervals(cfg, sc0.days, sc0.dt_couple,
                                     vratio=(lambda t: dl_mod.volume_ratio(t, regime)) if _refine else None,
                                     max_dlnv=float(sc0.dilution_max_dlnv), extra_edges=_burst)
    env_pts = drv._envelope_pts(sc0.dt_couple)
    _photo_idx = [i for i, r in enumerate(MECHANISM.active) if r.kind == "photo"]
    _photo_eqs = [MECHANISM.active[i].equation for i in _photo_idx]

    het = aero(Nk, Mk, temp, pres, rh, boxvol)
    yc = jnp.asarray(y0)
    dt_micro = jnp.full(B, float(sc0.micro_floor_s))
    vr_applied, so2_loss = 1.0, np.zeros(B)
    T_box = float(cfg.T)
    rec_t, rec_x = [0.0], [y0.copy()]
    rec = {k: [het[k]] for k in ("SA", "radius_cm", "h2so4wp", "particulate_S")}
    rec_budget = [(so2_loss.copy(), vr_applied)]
    rec_n, rec_dp = [het["n_cm3"]], [het["Dp_m"]]
    j_tmid, j_rows = [], []
    use_retry = np.zeros(B, bool)
    gas_retries = [[] for _ in range(B)]
    n_micro_tot = np.zeros(B, np.int64)
    n_calls_tot = np.zeros(B, np.int64)
    n_notes = np.zeros(B, np.int64)
    timing = dict(tuvx=0.0, gas=0.0, micro=0.0, other=0.0)
    w0 = time.perf_counter()
    next_log = progress_every_h * 3600.0
    iters_since_log, micro_at_log = 0, 0.0
    for ivl, (t0, t1) in enumerate(intervals, start=1):
        ta = time.perf_counter()
        t_mid = 0.5 * (t0 + t1)
        j_scale = float(photolysis_scale(drv._cosz(cfg, t_mid)))
        jv, _heat = drv._frozen_j_and_heating(cfg, t_mid, aerosol_props=None)
        override = np.asarray(photolysis_coeffs(cfg, j_scale, jv))
        if ivl == 1 and B > 1:   # J must not depend on the member (same site/time; aerosol_to_j off)
            jv_last, _ = drv._frozen_j_and_heating(cfgs[-1], t_mid, aerosol_props=None)
            # (NaN = "no override" sentinel for non-photolysis reactions, hence equal_nan)
            assert np.array_equal(override, np.asarray(photolysis_coeffs(cfgs[-1], j_scale, jv_last)),
                                  equal_nan=True), "photolysis differs across batch members"
        j_tmid.append(t_mid)
        j_rows.append(override[_photo_idx])
        tb_ = time.perf_counter()
        shared = {**params, "j_scale": j_scale, "sulfur_chain": sulfur, "photo_override": jnp.asarray(override),
                  "k_so2_ho2": float(sc0.so2_ho2_rate)}
        per = {"SA": jnp.asarray(het["SA"]), "particle_radius": jnp.asarray(het["radius_cm"]),
               "h2so4wp": jnp.asarray(het["h2so4wp"])}
        so2_before = np.asarray(yc[:, so2_idx])
        ts_grid = jnp.asarray(drv._envelope_grid(t0, t1, env_pts))
        try_std = (~use_retry) | (ivl % _RETRY_EVERY == 0)
        ys = None
        need_retry = ~try_std
        if try_std.any():
            ys, ok = gas_probe(yc, shared, per, t0, t1, ts_grid, jnp.full(B, 1.0e-10))
            ok = np.asarray(ok)
            fresh = try_std & ~ok
            for m in np.nonzero(fresh)[0]:
                gas_retries[m].append(float(t0))
            recovered = try_std & ok & use_retry
            use_retry = np.where(fresh, True, np.where(recovered, False, use_retry))
            need_retry = need_retry | fresh
        if need_retry.any():
            ys_r, ok_r = gas_retry(yc, shared, per, t0, t1, ts_grid, jnp.full(B, drv._RETRY_FIRST_STEP))
            bad = need_retry & ~np.asarray(ok_r)
            if bad.any():
                raise RuntimeError(f"gas solve failed even with the retry solver for members "
                                   f"{np.nonzero(bad)[0].tolist()} on [{t0}, {t1}]")
            ys = ys_r if ys is None else jnp.where(jnp.asarray(need_retry)[:, None, None], ys_r, ys)
        env_grid = ys[:, :, h2so4_idx]
        yc_end = ys[:, -1, :]
        jax.block_until_ready(env_grid)
        tc = time.perf_counter()
        Nk, Mk, Gc, removal, dt_micro, n_micro, n_calls, err, notes = micro_b(
            ts_grid, env_grid, float(t0), float(t1), Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, fns, cks,
            dt_micro)
        err = np.asarray(err)
        if err.any():
            raise RuntimeError(f"TOMAS non-finite at the micro-step floor for members {np.nonzero(err)[0].tolist()} "
                               f"on [{t0}, {t1}] (the CPU driver raises here too)")
        n_micro_i, n_calls_i = np.asarray(n_micro), np.asarray(n_calls)
        n_micro_tot += n_micro_i; n_calls_tot += n_calls_i; n_notes += np.asarray(notes)
        td = time.perf_counter()
        ivl_iters = int(n_calls_i.max())            # lockstep iterations this interval (= slowest member)
        iters_since_log += ivl_iters
        env_end_kg = conc_to_mass(env_grid[:, -1], BOXVOL_CM3, MW_H2SO4)
        yc = yc_end.at[:, h2so4_idx].set(mass_to_conc(jnp.maximum(env_end_kg - removal, 0.0), BOXVOL_CM3, MW_H2SO4))
        so2_loss = so2_loss + (so2_before - np.asarray(yc[:, so2_idx])) * vr_applied
        kdil = kdil_of(t0, t1)
        dtt = float(t1 - t0)
        yc = jnp.asarray(gas_bg) + (yc - jnp.asarray(gas_bg)) * np.exp(-kdil * dtt)     # dilute_gas
        vr_applied *= float(np.exp(kdil * dtt))
        Nk, Mk, Gc = dilute_b(Nk, Mk, Gc, dtt, kdil, Nk_bg, Mk_bg, Gc_bg)
        het = aero(Nk, Mk, temp, pres, rh, boxvol)
        rec_t.append(t1)
        rec_x.append(np.asarray(yc))
        for k in rec:
            rec[k].append(het[k])
        rec_budget.append((so2_loss.copy(), vr_applied))
        rec_n.append(het["n_cm3"]); rec_dp.append(het["Dp_m"])
        te = time.perf_counter()
        timing["tuvx"] += tb_ - ta; timing["gas"] += tc - tb_; timing["micro"] += td - tc; timing["other"] += te - td
        if t1 >= next_log - 1e-6:
            slow = int(np.argmax(n_calls_tot))
            log(f"[batched] t={t1 / 86400:.3f} d ({t1 / 3600:.1f} h) interval {ivl}/{len(intervals)} "
                f"wall={te - w0:.0f}s micro max/median={int(n_micro_tot.max())}/{int(np.median(n_micro_tot))} "
                f"lockstep iters since last log={iters_since_log} "
                f"({1e3 * (timing['micro'] - micro_at_log) / max(iters_since_log, 1):.2f} ms/iter) "
                f"slowest member={slow} retry members={int(use_retry.sum())} " +
                " ".join(f"{k}={v:.0f}s" for k, v in timing.items()))
            iters_since_log = 0
            micro_at_log = timing["micro"]
            next_log += progress_every_h * 3600.0

    wall = time.perf_counter() - w0
    t = np.asarray(rec_t)
    X = np.stack(rec_x, axis=1)                                   # (B, nt, nsp)
    A = {k: np.stack(v, axis=1) for k, v in rec.items()}           # (B, nt)
    cum = np.stack([b[0] for b in rec_budget], axis=1)
    vra = np.asarray([b[1] for b in rec_budget])
    NN, DP = np.stack(rec_n, axis=1), np.stack(rec_dp, axis=1)
    jrec = {"t_mid": np.asarray(j_tmid), "J": np.asarray(j_rows), "equations": _photo_eqs}
    out = []
    for m in range(B):
        aero_m = {"SA": A["SA"][m], "radius_cm": A["radius_cm"][m], "h2so4wp": A["h2so4wp"][m],
                  "particulate_S": A["particulate_S"][m], "T": np.full(len(t), T_box),
                  "so2_chem_loss_cum": cum[m], "V_ratio_applied": vra}
        final = ts0[m]._replace(Nk=Nk[m], Mk=Mk[m], Gc=Gc[m])
        out.append(dict(t=t, x=X[m], aero=aero_m, state=final, sd={"n_cm3": NN[m], "Dp_m": DP[m]}, jrec=jrec,
                        n_micro=int(n_micro_tot[m]), n_tomas_calls=int(n_calls_tot[m]),
                        floor_notes=int(n_notes[m]), gas_retries=gas_retries[m]))
    return out, dict(wall_s=wall, timing=timing, n_intervals=len(intervals), B=B,
                     backend=jax.default_backend(), device=str(jax.devices()[0]))
