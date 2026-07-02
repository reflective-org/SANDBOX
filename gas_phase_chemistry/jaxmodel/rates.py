"""JAX rate coefficients, in the exact order of ``reactions.REACTIONS``.

This mirrors the rate expressions in the Phase A declarative table (``reactions.py``) using
``jax.numpy`` so they are traceable. The ORDER must match ``reactions.REACTIONS`` one-to-one
(including the disabled reactions, which are filtered out downstream using the same mask),
so the coefficients line up with the stoichiometry matrix. A cross-check test compares the
resulting dC/dt against the Phase A right-hand side to guard against any divergence.

``p`` is a dict of (possibly traced) scalars: T, M, P, SA, WTR, j_scale, and the uptake
coefficients Yhocl/Yclnh2o/Yclnhcl/Yn2o5/Ybrono2. ``opt`` is a static int (the O3 mode).
"""

from __future__ import annotations

import math

import jax.numpy as jnp

_KB = 1.3807e-23
_M_PER_AMU = 1.0 / 6.02e23 / 1000.0


def falloff(T, M, k0_300, n0, kinf_300, ninf):
    """JPL 19-5 termolecular fall-off, **300 K reference**. Returns (k_f, kinf). Pass n0=-n, ninf=-m.

    JPL 19-5 tabulates k0(300)/kinf(300) (Table 2-1); see docs/jpl19-5-sulfur-crosscheck.md.
    """
    k0 = k0_300 * (T / 300.0) ** n0
    kinf = kinf_300 * (T / 300.0) ** ninf
    ratio = k0 * M / kinf
    k_f = (k0 * M / (1.0 + ratio)) * 0.6 ** (1.0 / (1.0 + jnp.log10(ratio) ** 2))
    return k_f, kinf


def troe(T, M, k0_300, n0, kinf_300, ninf):  # JPL 19-5 termolecular association (300 K ref)
    return falloff(T, M, k0_300, n0, kinf_300, ninf)[0]


def khet(gamma, gasmass, T, SA):
    v = 100.0 * jnp.sqrt(8.0 * _KB * T / (math.pi * gasmass * _M_PER_AMU))
    return 0.25 * gamma * SA * 1e-8 * v


def all_coefficients(p, opt):
    """List of rate coefficients aligned 1:1 with ``reactions.REACTIONS``.

    The Nth entry corresponds to reaction R<N> (see reactions.BY_RNUMBER and REACTIONS.md for
    the R-number / MATLAB k-label of each).

    LIMITATION: photolysis here uses only ``j_scale`` (the reference/sza path: j45 * j_scale, and O3
    -> 4.7e-5 * j_scale). It does NOT implement the absolute TUV-x J path (``j_values``) that
    ``rhs.concs_het`` uses in ``photolysis="tuvx"`` mode. The JAX Phase-B model therefore matches
    Phase A in reference/sza modes only; it is not valid for tuvx-coupled scenarios. See REVIEW_FINDINGS.md.
    """
    T, M, P, SA, WTR, H2O = p["T"], p["M"], p["P"], p["SA"], p["WTR"], p["H2O"]
    j = p["j_scale"]
    g = p  # gammas live in the same dict (Yhocl, Yclnh2o, ...)
    exp = jnp.exp

    # opt-dependent O3 photolysis + O1D quenching (opt is static -> plain Python if).
    if opt == 1:
        k20a = 4.7e-5 * (1.0 - 6.4e-6 * WTR) * j
        k20b = 4.7e-5 * (6.4e-6 * WTR) * j
        k40 = 0.0
        k41 = 0.0
    else:
        k20a = 0.0 * j
        k20b = 4.7e-5 * j
        k40 = 3.3e-11 * exp(55.0 / T)
        k41 = 2.15e-11 * exp(110.0 / T) * 0.79 * M

    # ClO+ClO+M association (reused for k2 and ClOOCl decomposition via Keq).
    k2 = troe(T, M, 1.9e-32, -3.6, 3.7e-12, -1.6)
    k3 = k2 / (2.16e-27 * exp(8537.0 / T))

    # OH + HNO3: JPL 19-5 chemical-activation form (Table 2-2, K2).
    kf22, kinf22 = falloff(T, M, 3.9e-31, -7.2, 1.5e-13, -4.8)
    k22 = kf22 + (3.7e-14 * exp(240.0 / T)) * (1.0 - kf22 / kinf22)

    # O + NO2: chemical-activation system (Table 2-2, K1); shared fall-off.
    kf_no2, kinf_no2 = falloff(T, M, 3.4e-31, -1.6, 2.3e-11, -0.2)
    k25 = kf_no2                                                   # -> NO3 (association)
    k26 = (5.3e-12 * exp(200.0 / T)) * (1.0 - kf_no2 / kinf_no2)   # -> NO + O2 (chem. activation)

    # SO2 + OH (+M): JPL 19-5 termolecular (now T-dependent, 300 K ref).
    k68 = troe(T, M, 2.9e-31, -4.1, 1.7e-12, 0.2)

    # HO2 + HO2: bimolecular + termolecular[M] + H2O enhancement (JPL 19-5 B13).
    k70 = (3.0e-13 * exp(460.0 / T) + 2.1e-33 * M * exp(920.0 / T)) \
        * (1.0 + 1.4e-21 * H2O * exp(2200.0 / T))

    return [
        6.4e-12 * exp(290.0 / T),                                  # ClO + NO
        k2,                                                        # ClO + ClO + M
        k3,                                                        # ClOOCl + M
        troe(T, M, 1.8e-31, -3.4, 1.5e-11, -1.9),               # ClO + NO2 + M
        6.9e-7 * exp(-10909.0 / T) * M,                            # ClONO2 + M  (Fahey, kept)
        2.3e-11 * exp(-200.0 / T),                                 # Cl + O3
        7.1e-12 * exp(-1270.0 / T),                                # Cl + CH4
        khet(g["Yclnhcl"], 97.0, T, SA),                           # het ClONO2 + HCl
        khet(g["Yclnh2o"], 97.0, T, SA),                           # het ClONO2 + H2O
        khet(g["Yhocl"], 52.0, T, SA),                             # het HOCl + HCl
        khet(g["Yn2o5"], 108.0, T, SA),                            # het N2O5 + H2O
        6.5e-5 * 0.9 / 1.5 * j,                                    # ClONO2 hv -> Cl + NO3
        6.5e-5 * 0.1 / 1.5 * j,                                    # ClONO2 hv -> ClO + NO2
        2.3e-3 * 0.9 * j,                                          # ClOOCl hv -> 2Cl
        2.3e-3 * 0.1 * j,                                          # ClOOCl hv -> 2ClO
        4.0e-3 * j,                                                # Cl2 hv
        4.5e-4 * j,                                                # HOCl hv
        1.0e-6 * j,                                                # HNO3 hv
        0.23 * 0.9 * j,                                            # NO3 hv -> NO2 + O
        0.23 * 0.1 * j,                                            # NO3 hv -> NO + O2
        0.014 * j,                                                 # NO2 hv
        2.7e-5 * j,                                                # N2O5 hv
        k20a,                                                      # O3 hv -> O2 + O
        k20b,                                                      # O3 hv -> O2 + O1D
        troe(T, M, 2.4e-30, -3.0, 1.6e-12, 0.1),                # NO2 + NO3 + M
        k22,                                                       # OH + HNO3 (chem. activation)
        troe(T, M, 1.8e-30, -3.0, 2.8e-11, 0.0),                # OH + NO2 + M
        troe(T, M, 9.1e-32, -1.5, 3.0e-11, 0.0),                # O + NO + M
        k25,                                                       # O + NO2 + M -> NO3 (assoc.)
        k26,                                                       # O + NO2 -> NO + O2 (chem. act.)
        3.0e-12 * exp(-1500.0 / T),                                # NO + O3
        1.2e-13 * exp(-2450.0 / T),                                # NO2 + O3
        troe(T, M, 7.1e-31, -2.6, 3.6e-11, -0.1),               # OH + NO + M
        3.0e-12 * exp(250.0 / T),                                  # OH + HONO
        1.7e-11 * exp(125.0 / T),                                  # NO + NO3
        4.5e-13 * exp(610.0 / T),                                  # OH + HNO4
        3.44e-12 * exp(260.0 / T),                                 # HO2 + NO
        troe(T, M, 1.9e-31, -3.4, 4.0e-12, -0.3),               # HO2 + NO2 + M
        6.1e-34 * (T / 300.0) ** -2.4 * M,                         # O + O2 + M
        8.0e-12 * exp(-2060.0 / T),                                # O + O3
        1.7e-12 * exp(-940.0 / T),                                 # OH + O3
        4.8e-11 * exp(250.0 / T),                                  # OH + HO2
        1.0e-14 * exp(-490.0 / T),                                 # HO2 + O3
        k40,                                                       # O1D + O2
        k41,                                                       # O1D + N2
        1.63e-10 * exp(60.0 / T),                                  # O1D + H2O
        1.31e-10 * exp(0.0 / T),                                   # O1D + CH4
        2.8e-11 * exp(85.0 / T),                                   # O + ClO
        7.4e-12 * exp(270.0 / T),                                  # OH + ClO -> Cl + HO2
        6.0e-13 * exp(230.0 / T),                                  # OH + ClO -> HCl + O2
        1.8e-12 * exp(-250.0 / T),                                 # OH + HCl
        2.6e-12 * exp(290.0 / T),                                  # HO2 + ClO -> HOCl
        2.6e-12 * exp(290.0 / T) * 0.03,                           # HO2 + ClO -> HCl (disabled)
        1.9e-11 * exp(230.0 / T),                                  # O + BrO
        1.6e-11 * exp(-780.0 / T),                                 # Br + O3
        8.8e-12 * exp(260.0 / T),                                  # BrO + NO
        2.3e-12 * exp(260.0 / T),                                  # BrO + ClO -> Br + Cl + O2
        4.1e-13 * exp(290.0 / T),                                  # BrO + ClO -> BrCl
        9.5e-13 * exp(550.0 / T),                                  # BrO + ClO -> Br + OClO
        troe(T, M, 5.5e-31, -3.1, 6.6e-12, -2.9),               # BrO + NO2 + M
        1.3e-5 * 0.8 * j,                                          # HNO4 hv -> NO2 + HO2
        1.3e-5 * 0.2 * j,                                          # HNO4 hv -> NO3 + OH
        0.013 * j,                                                 # OClO hv
        0.060 * j,                                                 # BrO hv
        1.8e-3 * 0.85 * j,                                         # BrONO2 hv -> Br + NO3
        1.8e-3 * 0.15 * j,                                         # BrONO2 hv -> BrO + NO2
        0.017 * j,                                                 # BrCl hv
        4.0e-13 * j,                                               # O2 hv (disabled)
        5.0e-4 * j,                                                # HONO hv
        1.0e-5,                                                    # HNO3aq -> HNO3
        7.2e-11 * exp(-70.0 / T),                                  # Cl + C2H6 (disabled)
        khet(g["Ybrono2"], 142.0, T, SA),                          # het BrONO2 + H2O
        1.8e-3 * j,                                                # HOBr hv
        k68,                                                       # SO2 + OH (JPL 19-5 termolecular)
        2.45e-12 * exp(-1775.0 / T),                               # CH4 + OH (JPL 19-5 D14)
        k70,                                                       # HO2 + HO2 (JPL 19-5 B13)
        1e-5,                                                      # H2O2 -> 2OH (FK, constant)
        1.0e-18,                                                   # SO2 + HO2 (upper limit; sens. test)
    ]
