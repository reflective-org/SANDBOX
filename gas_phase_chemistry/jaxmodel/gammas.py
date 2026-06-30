"""JAX port of the heterogeneous uptake coefficients (Phase A ``gammas.hetgammas_jpl00``).

Same JPL 2000 / Hanson parameterization, written with ``jax.numpy``. The ``real(sqrt(...))``
guards of the original become a NaN-safe sqrt (the "double where" trick) so both the value
and its gradient stay finite when the argument is non-positive.
"""

from __future__ import annotations

import jax.numpy as jnp


def _real_sqrt(x):
    """sqrt(x) for x>0 else 0, with finite gradients everywhere (mirrors MATLAB real(sqrt))."""
    safe = jnp.where(x > 0.0, x, 1.0)
    return jnp.where(x > 0.0, jnp.sqrt(safe), 0.0)


def _coth(x):
    return 1.0 / jnp.tanh(x)


def hetgammas_jpl00(T, P, h2so4wp, a_W, HCl, ClONO2, radius, sts=0):
    """Return ``(Yhocl, Yclnh2o, Yclnhcl)`` uptake coefficients (see Phase A docstring)."""
    p_hcl = HCl * 1e-9 * P / 1013.0
    p_clono2 = ClONO2 * 1e-9 * P / 1013.0

    h2so4ml = h2so4wp * 1000.0 / (9800.0 - 98.0 * h2so4wp)

    Z1 = 0.12364 - 5.6e-7 * T**2
    Z2 = -0.02954 + 1.814e-7 * T**2
    Z3 = 2.343e-3 - 1.487e-6 * T - 1.324e-8 * T**2
    density = 1.0 + Z1 * h2so4ml + Z2 * h2so4ml**1.5 + Z3 * h2so4ml**2

    h2so4M = density * h2so4wp / 9.8
    X = h2so4wp / (h2so4wp + (100.0 - h2so4wp) * 98.0 / 18.0)

    T0 = 144.11 + 0.166 * h2so4wp - 0.015 * h2so4wp**2 + 2.18e-4 * h2so4wp**3
    A = 169.5 + 5.18 * h2so4wp - 0.0825 * h2so4wp**2 + 3.27e-3 * h2so4wp**3
    viscosity = A * (T**-1.43) * jnp.exp(448.0 / (T - T0))

    a_H1 = 60.51 - 0.095 * h2so4wp + 0.0077 * h2so4wp**2 - 1.61e-5 * h2so4wp**3
    a_H2 = -(1.76 + 2.52e-4 * h2so4wp**2) * T**0.5
    a_H3 = (-805.89 + 253.05 * h2so4wp**0.076) / (T**0.5)
    a_H = jnp.exp(a_H1 + a_H2 + a_H3)

    k_H = 1.22e12 * jnp.exp(-6200.0 / T)
    k_h2o = 1.95e10 * jnp.exp(-2800.0 / T)
    k_hydr = k_h2o * a_W + k_H * a_H * a_W

    D_clono2 = 5e-8 * T / viscosity
    S_clono2 = 0.306 + 24.0 / T
    H_clono2 = 1.6e-6 * jnp.exp(4710.0 / T) * jnp.exp(-S_clono2 * h2so4M)
    C_clono2 = 1474.0 * T**0.5
    R = 0.082
    uptake_h2o_b = 4.0 * H_clono2 * R * T * _real_sqrt(D_clono2 * k_hydr) / C_clono2

    H_hcl = (0.094 - 0.61 * X + 1.2 * X**2) * jnp.exp(-8.68 + (8515.0 - 10718.0 * X**0.7) / T)
    M_hcl = H_hcl * p_hcl
    k_hcl = 7.9e11 * a_H * D_clono2 * M_hcl

    l_clono2 = _real_sqrt(D_clono2 / (k_hydr + k_hcl))
    f_clono2 = _coth(radius / l_clono2) - l_clono2 / radius
    uptake_rxn_clono2 = f_clono2 * uptake_h2o_b * _real_sqrt(1.0 + k_hcl / k_hydr)
    if sts == 1:
        uptake_rxn_clono2 = uptake_rxn_clono2 / 2.0

    uptake_hcl_b = uptake_rxn_clono2 * k_hcl / (k_hcl + k_hydr)
    uptake_s = 66.12 * H_clono2 * M_hcl * jnp.exp(-1374.0 / T)
    if sts == 1:
        uptake_s = uptake_s / 10.0

    F_hcl = 1.0 / (1.0 + 0.612 * (uptake_s + uptake_hcl_b) * p_clono2 / p_hcl)
    uptake_s_prime = F_hcl * uptake_s
    uptake_hcl_b_prime = F_hcl * uptake_hcl_b
    uptake_b = uptake_hcl_b_prime + uptake_rxn_clono2 * k_hydr / (k_hcl + k_hydr)

    Ycln = 1.0 / (1.0 + 1.0 / (uptake_s_prime + uptake_b))
    Yclnhcl = Ycln * (uptake_s_prime + uptake_hcl_b_prime) / (uptake_s_prime + uptake_b)
    Yclnh2o = Ycln - Yclnhcl

    D_hocl = 6.4e-8 * T / viscosity
    k_hocl = 1.25e9 * a_H * D_hocl * M_hcl
    S_hocl = 0.0776 + 59.18 / T
    H_hocl = 1.91e-6 * jnp.exp(5862.4 / T) * jnp.exp(-S_hocl * h2so4M)
    C_hocl = 2009.0 * T**0.5
    uptake_rxn_hocl = 4.0 * H_hocl * R * T * _real_sqrt(D_hocl * k_hocl) / C_hocl
    l_hocl = jnp.sqrt(D_hocl / k_hocl)
    f_hocl = _coth(radius / l_hocl) - l_hocl / radius
    Yhocl = 1.0 / (1.0 + 1.0 / (f_hocl * uptake_rxn_hocl * F_hcl))

    return Yhocl, Yclnh2o, Yclnhcl
