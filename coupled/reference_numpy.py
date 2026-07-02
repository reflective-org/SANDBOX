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
from driver import _abstol                      # noqa: E402  (gas model)
from reactions import MECHANISM, build_env      # noqa: E402


def run_coupled_numpy(scenario):
    """NumPy/BDF operator-split run matching ``driver.run_coupled``. Returns ``(t, states)``."""
    cfg = cd.to_model_config(scenario)
    y = cd.initial_state(scenario)
    sulfur = bool(scenario.switches.sulfur)
    atol = _abstol(cfg.opt)
    intervals = cd._outer_intervals(cfg, scenario.days, scenario.dt_couple)

    t_list, x_list, yc = [0.0], [y.copy()], y.copy()
    for t0, t1 in intervals:
        t_mid = 0.5 * (t0 + t1)
        j_scale = cd.photolysis_scale(cd._cosz(cfg, t_mid))
        jv = cd._frozen_j_values(cfg, t_mid)                 # identical to the JAX driver's J

        def rhs(t, c):                                        # frozen J; gammas recomputed per eval
            env = build_env(cfg, c, j_scale, j_values=jv, sulfur_chain=sulfur)
            return MECHANISM.dCdt(env, c)

        sol = solve_ivp(rhs, (t0, t1), yc, method="BDF", atol=atol, rtol=1e-3, first_step=1e-10)
        if not sol.success:
            raise RuntimeError(f"NumPy mirror failed on [{t0}, {t1}]: {sol.message}")
        yc = sol.y[:, -1]
        t_list.append(t1)
        x_list.append(yc.copy())
    return np.asarray(t_list), np.asarray(x_list)
