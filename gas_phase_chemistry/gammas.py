"""Heterogeneous reaction probabilities (reactive uptake coefficients, "gammas").

Python port of ``src-matlab/hetgammasJPL00.m`` (adapted from JPL 2000 and
Hanson, JPCA 102, p. 4794, and Tom's clono2h2o.m).

Computes the reactive uptake coefficients on liquid sulfate aerosol for:
    Yclnhcl : ClONO2 + HCl  -> Cl2  + HNO3
    Yclnh2o : ClONO2 + H2O  -> HOCl + HNO3
    Yhocl   : HOCl   + HCl  -> Cl2  + H2O

Inputs (same as the MATLAB function):
    T        : temperature (K)
    P        : pressure (mbar)
    h2so4wp  : H2SO4 weight percent of the aerosol (from aerosol.h2so4wp_at)
    a_W      : water activity (from aerosol.h2so4wp_at)
    HCl      : HCl mixing ratio (ppbv)
    ClONO2   : ClONO2 mixing ratio (ppbv)
    radius   : aerosol radius (cm)
    sts      : ternary-solution flag (0 or 1); halves/reduces some uptake terms

About ``real(...)``:
    The MATLAB code wraps several square-root expressions in ``real()`` to discard tiny
    imaginary parts that can appear from round-off. For physical inputs the arguments are
    positive, so these are no-ops. We mirror the semantics with ``_real_sqrt`` (returns 0
    for a negative argument, matching ``real(sqrt(neg)) == 0`` in MATLAB) so behaviour is
    identical at the edges.
"""

from __future__ import annotations

import math


def _real_sqrt(x: float) -> float:
    """sqrt(x), but 0 for negative x -- mirrors MATLAB ``real(x.^0.5)``."""
    return math.sqrt(x) if x > 0.0 else 0.0


def _coth(x: float) -> float:
    """Hyperbolic cotangent, 1/tanh(x) (as written in the MATLAB f-factor terms)."""
    return 1.0 / math.tanh(x)


def hetgammas_jpl00(
    T: float,
    P: float,
    h2so4wp: float,
    a_W: float,
    HCl: float,
    ClONO2: float,
    radius: float,
    sts: float,
) -> tuple[float, float, float]:
    """Return ``(Yhocl, Yclnh2o, Yclnhcl)`` uptake coefficients."""
    # NOTE: the MATLAB source also computes an air number density `M` here, but it is
    # never used downstream, so it is intentionally omitted from this port.

    # Gas partial pressures (atm). HCl/ClONO2 are in ppbv; 1013 mbar ~ 1 atm.
    p_hcl = HCl * 1e-9 * P / 1013.0
    p_clono2 = ClONO2 * 1e-9 * P / 1013.0

    # H2SO4 molality back out of weight percent (98 = molar mass of H2SO4).
    h2so4ml = h2so4wp * 1000.0 / (9800.0 - 98.0 * h2so4wp)

    # Aerosol mass density from molality (empirical polynomial).
    Z1 = 0.12364 - 5.6e-7 * T**2
    Z2 = -0.02954 + 1.814e-7 * T**2
    Z3 = 2.343e-3 - 1.487e-6 * T - 1.324e-8 * T**2
    density = 1.0 + Z1 * h2so4ml + Z2 * h2so4ml**1.5 + Z3 * h2so4ml**2

    # Molarity (mol/L) and H2SO4 mole fraction.
    h2so4M = density * h2so4wp / 9.8
    X = h2so4wp / (h2so4wp + (100.0 - h2so4wp) * 98.0 / 18.0)

    # Viscosity (poise) -- VTF-like form with composition-dependent T0 and A.
    T0 = 144.11 + 0.166 * h2so4wp - 0.015 * h2so4wp**2 + 2.18e-4 * h2so4wp**3
    A = 169.5 + 5.18 * h2so4wp - 0.0825 * h2so4wp**2 + 3.27e-3 * h2so4wp**3
    viscosity = A * (T**-1.43) * math.exp(448.0 / (T - T0))

    # Acid activity a_H.
    a_H1 = 60.51 - 0.095 * h2so4wp + 0.0077 * h2so4wp**2 - 1.61e-5 * h2so4wp**3
    a_H2 = -(1.76 + 2.52e-4 * h2so4wp**2) * T**0.5
    a_H3 = (-805.89 + 253.05 * h2so4wp**0.076) / (T**0.5)
    a_H = math.exp(a_H1 + a_H2 + a_H3)

    # Hydrolysis rate (ClONO2 + H2O and acid-catalysed channel).
    k_H = 1.22e12 * math.exp(-6200.0 / T)
    k_h2o = 1.95e10 * math.exp(-2800.0 / T)
    k_hydr = k_h2o * a_W + k_H * a_H * a_W

    # ClONO2 diffusion, solubility, Henry's law, thermal speed.
    D_clono2 = 5e-8 * T / viscosity
    S_clono2 = 0.306 + 24.0 / T
    H_clono2 = 1.6e-6 * math.exp(4710.0 / T) * math.exp(-S_clono2 * h2so4M)
    C_clono2 = 1474.0 * T**0.5
    R = 0.082  # gas constant (L*atm / mol / K)
    uptake_h2o_b = 4.0 * H_clono2 * R * T * _real_sqrt(D_clono2 * k_hydr) / C_clono2

    # HCl uptake. Effective Henry's law coefficient for HCl, then its rate contribution.
    H_hcl = (0.094 - 0.61 * X + 1.2 * X**2) * math.exp(
        -8.68 + (8515.0 - 10718.0 * X**0.7) / T
    )
    M_hcl = H_hcl * p_hcl
    k_hcl = 7.9e11 * a_H * D_clono2 * M_hcl

    # Reacto-diffusive length and the resulting in-particle f-factor for ClONO2.
    l_clono2 = _real_sqrt(D_clono2 / (k_hydr + k_hcl))
    f_clono2 = _coth(radius / l_clono2) - l_clono2 / radius
    uptake_rxn_clono2 = f_clono2 * uptake_h2o_b * _real_sqrt(1.0 + k_hcl / k_hydr)
    if sts == 1:
        uptake_rxn_clono2 = uptake_rxn_clono2 / 2.0

    uptake_hcl_b = uptake_rxn_clono2 * k_hcl / (k_hcl + k_hydr)
    uptake_s = 66.12 * H_clono2 * M_hcl * math.exp(-1374.0 / T)
    if sts == 1:
        uptake_s = uptake_s / 10.0

    # F_hcl accounts for HCl depletion in the particle by reaction with ClONO2.
    F_hcl = 1.0 / (1.0 + 0.612 * (uptake_s + uptake_hcl_b) * p_clono2 / p_hcl)
    uptake_s_prime = F_hcl * uptake_s
    uptake_hcl_b_prime = F_hcl * uptake_hcl_b
    uptake_b = uptake_hcl_b_prime + uptake_rxn_clono2 * k_hydr / (k_hcl + k_hydr)

    # ClONO2 gammas: total, then split into the +HCl and +H2O channels.
    Ycln = 1.0 / (1.0 + 1.0 / (uptake_s_prime + uptake_b))
    Yclnhcl = Ycln * (uptake_s_prime + uptake_hcl_b_prime) / (uptake_s_prime + uptake_b)
    Yclnh2o = Ycln - Yclnhcl

    # HOCl + HCl gamma.
    D_hocl = 6.4e-8 * T / viscosity
    k_hocl = 1.25e9 * a_H * D_hocl * M_hcl
    S_hocl = 0.0776 + 59.18 / T
    H_hocl = 1.91e-6 * math.exp(5862.4 / T) * math.exp(-S_hocl * h2so4M)
    C_hocl = 2009.0 * T**0.5
    uptake_rxn_hocl = 4.0 * H_hocl * R * T * _real_sqrt(D_hocl * k_hocl) / C_hocl
    # MATLAB precedence: real() wraps only (D_hocl/k_hocl), then ^0.5 is applied outside.
    l_hocl = math.sqrt(D_hocl / k_hocl)
    f_hocl = _coth(radius / l_hocl) - l_hocl / radius
    Yhocl = 1.0 / (1.0 + 1.0 / (f_hocl * uptake_rxn_hocl * F_hcl))

    return Yhocl, Yclnh2o, Yclnhcl
