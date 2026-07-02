# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Dilution: first-order relaxation of the box toward a background state (Phase 6).

Ports TOMAS's dilution formulation ``C_new = C_bg + (C - C_bg)*exp(-k_dil*dt)`` (exact for a constant
k_dil over the interval) and applies it to BOTH the gas species (frank's 36) and the aerosol
(``Nk``/``Mk``/``Gc`` via ``tomas_jax.physics.dilution.dilution_step``). The background is the initial
box state (AD-6.3); the rate is a constant ``dilution_rate`` [1/s] (AD-6.2; the V(t) schedule is
deferred). Applied once per outer interval, operator-split (AD-6.1/6.4).
"""

from __future__ import annotations

import numpy as np

from . import tomas_bridge as tb   # puts tomas_jax on sys.path

from tomas_jax.physics.dilution import dilution_step  # noqa: E402


def dilute_gas(conc, conc_bg, kdil, dt):
    """Relax a gas-species vector toward ``conc_bg`` at rate ``kdil`` [1/s] over ``dt`` [s].

    ``C_new = C_bg + (C - C_bg)*exp(-kdil*dt)`` -- the same analytic exponential as TOMAS's
    ``dilution_step`` (elementwise; works for numpy/JAX arrays).
    """
    decay = np.exp(-kdil * dt)
    return conc_bg + (conc - conc_bg) * decay


def dilute_aerosol(tstate, tstate_bg, kdil, dt):
    """Relax the aerosol (Nk, Mk, Gc) toward the background TomasState via TOMAS ``dilution_step``.

    Returns a new TomasState with diluted Nk/Mk/Gc (other fields unchanged).
    """
    Nk, Mk, Gc = dilution_step(tstate.Nk, tstate.Mk, tstate.Gc, dt, kdil,
                               tstate_bg.Nk, tstate_bg.Mk, tstate_bg.Gc)
    return tstate._replace(Nk=Nk, Mk=Mk, Gc=Gc)
