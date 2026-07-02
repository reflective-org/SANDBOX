# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""NumPy operator-split mirror of the JAX coupled driver, for cross-backend parity (Phase 2.5).

``run_coupled_numpy`` reuses the JAX driver's OWN helpers (same outer intervals, same frozen
step-midpoint J, same ``switches.sulfur`` gate) but integrates each interval with SciPy BDF and the
NumPy RHS instead of diffrax. So a JAX-vs-this comparison isolates the SOLVER difference (BDF vs
Kvaerno5) with J-handling held identical -- the honest parity check the plan review asked for. Both
backends read the gate from ``switches.sulfur`` (NumPy via ``build_env(sulfur_chain=...)``).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from . import driver as cd                      # sets gas_phase_chemistry on sys.path
from .tomas_bridge import (initial_tomas_state, make_microphysics_step, SRTSO4,
                           BOXVOL_CM3, MW_H2SO4)
from .aerosol_props import het_inputs
from .units import conc_to_mass, mass_to_conc

from config import IDX                           # noqa: E402  (gas model)
from driver import _abstol                       # noqa: E402  (gas model)
from reactions import MECHANISM, build_env       # noqa: E402


def run_coupled_numpy(scenario):
    """NumPy/BDF operator-split run matching ``driver.run_coupled``. Returns ``(t, states)``.

    Mirrors the JAX driver exactly EXCEPT the gas integrator (SciPy BDF vs diffrax): same outer
    intervals, same frozen midpoint J, same ``switches.sulfur`` gate, and -- when TOMAS is active --
    the SAME TOMAS ``make_step`` (a JAX function used by both backends), gas H2SO4 handoff, and frozen
    per-interval aerosol het inputs. So a JAX-vs-this comparison isolates the GAS solver difference.
    """
    cfg = cd.to_model_config(scenario)
    y = cd.initial_state(scenario)
    sulfur = bool(scenario.switches.sulfur)
    atol = _abstol(cfg.opt)
    intervals = cd._outer_intervals(cfg, scenario.days, scenario.dt_couple)

    tomas_step = make_microphysics_step(scenario.switches)
    tomas_active = tomas_step is not None
    tstate = initial_tomas_state(scenario) if tomas_active else None
    het = het_inputs(tstate) if tomas_active else None
    h2so4_idx = IDX["H2SO4"]

    t_list, x_list, yc = [0.0], [y.copy()], y.copy()
    for t0, t1 in intervals:
        t_mid = 0.5 * (t0 + t1)
        j_scale = cd.photolysis_scale(cd._cosz(cfg, t_mid))
        jv = cd._frozen_j_values(cfg, t_mid)                 # identical to the JAX driver's J
        # freeze this interval's aerosol het inputs on the ModelConfig (TOMAS-derived, or leave
        # cfg.SA / defaults when TOMAS is inactive) -- mirrors the JAX driver's args overrides.
        if tomas_active:
            cfg.SA = het["SA"]
            cfg.particle_radius = het["radius_cm"]
            cfg.h2so4wp = het["h2so4wp"]

        def rhs(t, c):                                        # frozen J; gammas recomputed per eval
            env = build_env(cfg, c, j_scale, j_values=jv, sulfur_chain=sulfur)
            return MECHANISM.dCdt(env, c)

        sol = solve_ivp(rhs, (t0, t1), yc, method="BDF", atol=atol, rtol=1e-3, first_step=1e-10)
        if not sol.success:
            raise RuntimeError(f"NumPy mirror failed on [{t0}, {t1}]: {sol.message}")
        yc = sol.y[:, -1]

        if tomas_active:
            gas_h2so4_kg = conc_to_mass(float(yc[h2so4_idx]), BOXVOL_CM3, MW_H2SO4)
            Gc = tstate.Gc.at[SRTSO4].set(gas_h2so4_kg)
            Nk, Mk, Gc = tomas_step(tstate.Nk, tstate.Mk, Gc, tstate.xk, tstate.temp, tstate.pres,
                                    tstate.boxvol, tstate.rh, tstate.alpha, float(t1 - t0))
            if not (bool(np.all(np.isfinite(np.asarray(Nk)))) and
                    bool(np.all(np.isfinite(np.asarray(Mk))))):
                raise RuntimeError(
                    f"TOMAS produced non-finite output on [{t0}, {t1}] (gas H2SO4 = "
                    f"{float(yc[h2so4_idx]):.3e} molec/cm^3); reduce dt_couple (see AD-3.9).")
            tstate = tstate._replace(Nk=Nk, Mk=Mk, Gc=Gc)
            yc = yc.copy()
            yc[h2so4_idx] = mass_to_conc(float(Gc[SRTSO4]), BOXVOL_CM3, MW_H2SO4)
            het = het_inputs(tstate)

        t_list.append(t1)
        x_list.append(yc.copy())
    return np.asarray(t_list), np.asarray(x_list)
