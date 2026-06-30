"""JAX port of the sulfate aerosol composition (Phase A ``aerosol.h2so4wp_at``).

Same Tabazadeh (2000) parameterization, but written with ``jax.numpy`` and ``jnp.where`` for
the water-activity branches so it is traceable (jit / grad / vmap). Numerically identical to
the Phase A version in float64.
"""

from __future__ import annotations

import jax.numpy as jnp


def h2so4wp_at(T, P, H2O):
    """Return ``(h2so4wpAT, h2so4mlAT, a_WAT)`` -- weight %, molality, water activity."""
    # Saturation water-vapour pressure (mbar) and water activity.
    p0h2o = jnp.exp(18.452406985 - 3505.1578807 / T
                    - 330918.55082 / T**2 + 12725068.262 / T**3)
    a_W = (H2O * 1e-6 * P) / p0h2o

    # Three water-activity regimes -> select coefficients with nested where
    # (a_W <= 0.05 ; 0.05 < a_W < 0.85 ; a_W >= 0.85).
    def pick(low, mid, high):
        return jnp.where(a_W <= 0.05, low, jnp.where(a_W < 0.85, mid, high))

    a1 = pick(12.37208932, 11.820654354, -180.06541028)
    b1 = pick(-0.16125516114, -0.20786404244, -0.38601102592)
    c1 = pick(-30.490657554, -4.807306373, -93.317846778)
    d1 = pick(-2.1133114241, -5.1727540348, 273.88132245)
    a2 = pick(13.455394705, 12.891938068, -176.95814097)
    b2 = pick(-0.1921312255, -0.23233847708, -0.36257048154)
    c2 = pick(-34.285174607, -6.4261237757, -90.469744201)
    d2 = pick(-1.7620073078, -4.9005471319, 267.45509988)

    y1 = a1 * a_W**b1 + c1 * a_W + d1
    y2 = a2 * a_W**b2 + c2 * a_W + d2
    h2so4mlAT = y1 + (T - 190.0) * (y2 - y1) / 70.0
    h2so4wpAT = 9800.0 * h2so4mlAT / (98.0 * h2so4mlAT + 1000.0)
    return h2so4wpAT, h2so4mlAT, a_W
