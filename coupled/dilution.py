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

# ---------------------------------------------------------------------------------------------
# Time-varying dilution regimes (ported from tomas-jax experimental_case/run_marianna_dilution.py):
# plume volume expansion V(t)/V0 -> k_dil(t) = d ln V / dt. Two-piece: early Schumann t^0.8 (clamped
# >=1) then a turbulent-growth exponential whose Kz coefficient sets the regime (Schumann et al. 1998).
# D1 'Low Kz' = low vertical diffusivity ~ high-altitude / low-latitude / clean stratosphere.
# ---------------------------------------------------------------------------------------------
_T_BREAK = 1.0e4
_BIG_T = 1.0e12


def _two_piece(k):
    return ((_T_BREAK, ("power", 0.8)), (_BIG_T, ("exp", 1585.0, k, _T_BREAK, 1.5)))


#: V(t)/V0 piecewise segments per regime (matches tomas-jax DILUTIONS).
DILUTION_REGIMES = {
    "D1": ("Low Kz",  _two_piece(2.811e-9)),
    "D2": ("Med Kz",  _two_piece(8.89e-9)),
    "D3": ("High Kz", _two_piece(2.811e-8)),
    "D5": ("Low Lx",  _two_piece(5.33e-8)),
}


def _eval_segment(t, seg):
    if seg[0] == "power":
        return np.maximum(1.0, t ** seg[1])
    _kind, A, k, t0, q = seg
    # clamp t-t0 >= 0: np.select evaluates every segment everywhere, so this branch is computed (and
    # discarded) for t < t0 where (t-t0)**q would be NaN. Clamping avoids the spurious warning.
    return A * np.exp(k * np.maximum(t - t0, 0.0) ** q)


def volume_ratio(t_seconds, regime):
    """V(t)/V0 for a named regime (scalar or array t). Continuous piecewise (Schumann scaling)."""
    segs = DILUTION_REGIMES[regime][1]
    t = np.maximum(np.asarray(t_seconds, dtype=float), 0.0)
    conds, vals, t0 = [], [], 0.0
    for t_end, seg in segs:
        conds.append((t >= t0) & (t < t_end))
        vals.append(_eval_segment(t, seg))
        t0 = t_end
    return np.select(conds, vals, default=vals[-1])


def kdil_from_regime(regime, t0, t1):
    """Interval-average dilution rate [1/s] from the volume expansion: ln(V(t1)/V(t0)) / (t1 - t0)."""
    v0 = float(volume_ratio(t0, regime))
    v1 = float(volume_ratio(t1, regime))
    return max(0.0, np.log(max(v1, 1e-300) / max(v0, 1e-300)) / (t1 - t0))


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
