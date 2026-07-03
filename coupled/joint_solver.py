# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Joint stiff ODE solver: gas chemistry + nucleation + coagulation + condensation gas-sink.

Opt-in alternative to the operator-split micro-loop (approach B). One Diffrax ``Kvaerno5`` solve
co-evolves, over an outer interval:

  * the full gas chemistry (36-species dC/dt, reusing ``jaxmodel.model.make_frozen_vf``),
  * nucleation as a rate source (bin-0 number + sulfate mass) with its H2SO4 gas sink,
  * coagulation (``calc_coagulation_rates`` dNdt/dMdt), and
  * condensation's GAS SINK ``-CS*[H2SO4]`` (``calc_condensation_sink``),

so nucleation always sees the true (pseudo-steady) H2SO4 -- eliminating the chem->nuc->cond
operator-split ordering error. Condensation's SIZE redistribution stays a PPM remap (integrator-only)
applied AFTER the solve, driven by the ODE-integrated condensed mass (``m_cond`` accumulator).

State is carried in INTENSIVE per-cm^3 units so the result is boxvol-independent and the components
are reasonably scaled for the solver:

    y = [ conc(36) molec/cm^3 | n(nbins) #/cm^3 | q_so4(nbins) H2SO4-equiv molec/cm^3 | m_cond(1) ]

TOMAS functions want per-grid-cell Nk [#/cell] and Mk [kg/cell]; we convert (x boxvol,
conc_to_mass) inside the RHS. Sulfate-only: only the SO4 mass column is carried (organics/water are
zero in the clean-stratosphere scenarios); the nucleated cluster's 10% organic fraction is dropped
(no sulfur, so the sulfur budget is unaffected and matches the split path's SO4 accounting).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
from diffrax import Kvaerno5, ODETerm, PIDController, SaveAt, diffeqsolve

from . import tomas_bridge as tb   # puts tomas_jax + gas_phase_chemistry on sys.path
from .units import conc_to_mass, mass_to_conc

from config import IDX                                           # noqa: E402  (gas model)
from jaxmodel.model import make_frozen_vf                        # noqa: E402  (reuse gas chem RHS)

from tomas_jax.core.config import (SRTSO4, ICOMP, N_GAS_SPECIES,  # noqa: E402
                                   ICOMP_NODIAG)
from tomas_jax.physics.nucleation import ricco_dunne_nucleation_rate, _MNUC  # noqa: E402
from tomas_jax.physics.condensation_sink import calc_condensation_sink       # noqa: E402
from tomas_jax.physics.coagulation_rates import calc_coagulation_rates       # noqa: E402
from tomas_jax.physics.coagulation_kernel import calc_coagulation_kernel     # noqa: E402
from tomas_jax.physics.properties import calc_particle_properties            # noqa: E402
from tomas_jax.physics.ezcond_ppm_jax import ezcond_ppm_jax                  # noqa: E402

_H2SO4 = IDX["H2SO4"]
_MW = tb.MW_H2SO4
_BV = tb.BOXVOL_CM3
_NUC_SO4_FRAC = 0.9   # sulfate fraction of the nucleated cluster (matches nucleation_step)

# LU Newton (fast, well-posed): the coupled gas solver uses LU successfully; least-squares
# (well_posed=False) is available as a fallback if Newton fails to converge on the joint system.
_ROOT_FINDER = optx.Newton(rtol=1e-3, atol=1e-6, linear_solver=lx.LU())
_ROOT_FINDER_LSQ = optx.Newton(rtol=1e-3, atol=1e-6,
                               linear_solver=lx.AutoLinearSolver(well_posed=False))


def _pack(conc, n, q_so4, m_cond):
    return jnp.concatenate([conc, n, q_so4, jnp.reshape(m_cond, (1,))])


def _unpack(y, nbins):
    conc = y[:36]
    n = y[36:36 + nbins]
    q_so4 = y[36 + nbins:36 + 2 * nbins]
    m_cond = y[36 + 2 * nbins]
    return conc, n, q_so4, m_cond


def make_joint_vf(opt, nbins, temp, pres, boxvol, ion_pair_rate, nuc_scale,
                  enable_inorganic=1.0, enable_organic=0.0):
    """Combined gas+aerosol vector field. Gas chem via the frozen-J gas RHS; aerosol via rate-form
    nucleation/coagulation + condensation gas sink. ``args`` carries the same frozen params the gas
    solver uses (T, M, P, SA, WTR, Yn2o5, j_scale, sulfur_chain, photo_override, particle_radius,
    h2so4wp, k_so2_ho2) PLUS ``kij`` -- the coagulation kernel FROZEN per outer step (computed at t0),
    which removes the O(nbins^2) kernel recompute + its Jacobian from every implicit Newton eval (the
    dominant cost). The condensation sink CS stays LIVE (cheap, O(nbins)) since it must respond to the
    growing aerosol within the step -- the nucleation<->H2SO4 coupling this solver exists to resolve."""
    gas_vf = make_frozen_vf(opt)   # vf(t, conc36, args) -> dC/dt (chem only)
    xk = tb.tcfg.xk_boundaries() if nbins == 40 else tb.tcfg.make_grid_80bin()

    def vf(t, y, args):
        conc, n, q_so4, _m_cond = _unpack(y, nbins)

        # --- gas chemistry (produces H2SO4 at idx _H2SO4; no aerosol sink here) ---
        dconc = gas_vf(t, conc, args)

        # --- reconstruct TOMAS per-cell state (kg, #/cell) from the intensive state ---
        Nk = n * boxvol                                    # #/cell
        Mk = jnp.zeros((nbins, ICOMP)).at[:, SRTSO4].set(
            conc_to_mass(q_so4, boxvol, _MW))              # kg/cell (sulfate only)
        h2so4_kg = conc_to_mass(conc[_H2SO4], boxvol, _MW)
        Gc = jnp.zeros(N_GAS_SPECIES).at[SRTSO4].set(h2so4_kg)

        # --- nucleation (rate-form): bin-0 number + SO4 mass source, H2SO4 gas sink ---
        fn = ricco_dunne_nucleation_rate(Gc, temp, pres, boxvol, 0.0, 0.0, ion_pair_rate,
                                         enable_organic=enable_organic,
                                         enable_inorganic=enable_inorganic, fn_scale=nuc_scale)
        fn = jnp.maximum(fn, 0.0)
        nuc_so4_kgs = _NUC_SO4_FRAC * fn * _MNUC * boxvol   # kg/s of H2SO4 into bin 0
        nuc_gas_sink_conc = mass_to_conc(nuc_so4_kgs, boxvol, _MW)   # molec/cm^3/s

        # --- coagulation (rate-form; kernel FROZEN per outer step via args["kij"]) ---
        dNdt, dMdt, _ov = calc_coagulation_rates(Nk, Mk, args["kij"], xk, ICOMP_NODIAG)

        # --- condensation GAS SINK only (size growth deferred to the PPM remap) ---
        CS, _sinkfrac = calc_condensation_sink(Nk, Mk, temp, pres, boxvol, xk=xk)
        cond_gas_sink_conc = CS * conc[_H2SO4]              # molec/cm^3/s

        # --- assemble ---
        dconc = dconc.at[_H2SO4].add(-(nuc_gas_sink_conc + cond_gas_sink_conc))
        dn = dNdt / boxvol
        dn = dn.at[0].add(fn)                              # fn is #/cm^3/s
        dq_so4 = mass_to_conc(dMdt[:, SRTSO4], boxvol, _MW)
        dq_so4 = dq_so4.at[0].add(nuc_gas_sink_conc)       # nucleated SO4 into bin 0
        dm_cond = cond_gas_sink_conc                        # H2SO4 condensing [molec/cm^3/s]
        return _pack(dconc, dn, dq_so4, dm_cond)

    return vf, xk


def _atol_vector(nbins):
    # per-component absolute tolerance: molec/cm^3 blocks ~1e3, number/cm^3 block ~1e-4
    return jnp.concatenate([jnp.full(36, 1e3), jnp.full(nbins, 1e-4),
                            jnp.full(nbins, 1e3), jnp.array([1e3])])


def make_joint_step(opt, nbins, temp, pres, boxvol, ion_pair_rate, nuc_scale,
                    enable_inorganic=1.0, enable_organic=0.0,
                    rtol=1e-3, first_step=1e-6, max_steps=1_000_000):
    """Build ``step(conc, Nk, Mk_so4, t0, t1, args) -> (conc, Nk, Mk_so4)`` for one outer interval:
    the joint ODE solve followed by the PPM condensation size-remap of the integrated condensed
    mass. All state carried intensively (per cm^3); Nk/Mk are per-cell on the boundary."""
    vf, xk = make_joint_vf(opt, nbins, temp, pres, boxvol, ion_pair_rate, nuc_scale,
                           enable_inorganic, enable_organic)
    term = ODETerm(vf)
    solver = Kvaerno5(root_finder=_ROOT_FINDER)
    ctrl = PIDController(rtol=rtol, atol=_atol_vector(nbins))

    @jax.jit
    def _solve(y0, t0, t1, args):
        sol = diffeqsolve(term, solver, t0=t0, t1=t1, dt0=first_step, y0=y0, args=args,
                          stepsize_controller=ctrl, saveat=SaveAt(t1=True), max_steps=max_steps)
        return sol.ys[-1]

    @jax.jit
    def _frozen_kij(Nk, Mk):
        Dpk, Dk, ck = calc_particle_properties(Nk, Mk, temp, pres)
        return calc_coagulation_kernel(Dpk, Dk, ck, boxvol)

    def step(conc, Nk, Mk, t0, t1, args):
        # freeze the coagulation kernel at the interval start; pass through args (solve stays cached)
        args = {**args, "kij": _frozen_kij(Nk, Mk)}
        # to intensive state
        n0 = Nk / boxvol
        q0 = mass_to_conc(Mk[:, SRTSO4], boxvol, _MW)
        y0 = _pack(jnp.asarray(conc), n0, q0, 0.0)
        y1 = _solve(y0, jnp.asarray(t0), jnp.asarray(t1), args)
        conc1, n1, q1, m_cond = _unpack(y1, nbins)
        # positivity
        conc1 = jnp.maximum(conc1, 0.0); n1 = jnp.maximum(n1, 0.0); q1 = jnp.maximum(q1, 0.0)
        Nk1 = n1 * boxvol
        Mk1 = Mk.at[:, SRTSO4].set(conc_to_mass(q1, boxvol, _MW))
        # PPM size remap of the ODE-integrated condensed H2SO4 mass
        mcond_kg = conc_to_mass(jnp.maximum(m_cond, 0.0), boxvol, _MW)
        Nk1, Mk1 = ezcond_ppm_jax(Nk1, Mk1, mcond_kg, SRTSO4, xk, temp, pres, boxvol)
        return conc1, Nk1, Mk1

    return step
