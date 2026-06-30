"""Sulfate aerosol composition.

Python port of ``src-matlab/h2so4wpATfxn.m`` (adapted from Tabazadeh, JPL, 2000).

Given the temperature, pressure, and water-vapour mixing ratio, this returns the
composition of the stratospheric sulfate aerosol: its H2SO4 weight percent and molality,
together with the water activity. Those quantities feed the heterogeneous-uptake
calculation in ``gammas.py``.

Units (same as the MATLAB function):
    T   : temperature (K)
    P   : pressure (mbar)
    H2O : water-vapour mixing ratio (ppmv)

The model uses scalar inputs, so this is written as a plain scalar function with the same
three water-activity branches as the MATLAB code. (Vectorisation/where-based branching is
deferred to the JAX phase.)
"""

from __future__ import annotations

import math


def h2so4wp_at(T: float, P: float, H2O: float) -> tuple[float, float, float]:
    """Return ``(h2so4wpAT, h2so4mlAT, a_WAT)``.

    h2so4wpAT : H2SO4 weight percent of the aerosol
    h2so4mlAT : H2SO4 molality (mol/kg)
    a_WAT     : water activity (dimensionless)
    """
    # Saturation water-vapour pressure over the aerosol (mbar). Polynomial in 1/T from
    # the MATLAB source (h2so4wpATfxn.m lines 9-10).
    p0h2o = math.exp(
        18.452406985
        - 3505.1578807 / T
        - 330918.55082 / T**2
        + 12725068.262 / T**3
    )

    # Partial pressure of water (mbar), then water activity a_W = pH2O / p0H2O.
    ph2o = H2O * 1e-6 * P
    a_W = ph2o / p0h2o

    # Branch coefficients for the two reference-temperature fits (y1 at 190 K-ish, y2 at
    # 260 K-ish); selected by water-activity regime exactly as in the MATLAB code.
    if a_W <= 0.05:
        a1, b1, c1, d1 = 12.37208932, -0.16125516114, -30.490657554, -2.1133114241
        a2, b2, c2, d2 = 13.455394705, -0.1921312255, -34.285174607, -1.7620073078
    elif a_W < 0.85:  # 0.05 < a_W < 0.85
        a1, b1, c1, d1 = 11.820654354, -0.20786404244, -4.807306373, -5.1727540348
        a2, b2, c2, d2 = 12.891938068, -0.23233847708, -6.4261237757, -4.9005471319
    else:  # a_W >= 0.85
        a1, b1, c1, d1 = -180.06541028, -0.38601102592, -93.317846778, 273.88132245
        a2, b2, c2, d2 = -176.95814097, -0.36257048154, -90.469744201, 267.45509988

    # Two molality fits, then linearly interpolate in temperature between them.
    y1 = a1 * a_W**b1 + c1 * a_W + d1
    y2 = a2 * a_W**b2 + c2 * a_W + d2
    h2so4mlAT = y1 + (T - 190.0) * (y2 - y1) / 70.0

    # Convert molality (mol/kg) to weight percent. 98 = molar mass of H2SO4 (g/mol).
    h2so4wpAT = 9800.0 * h2so4mlAT / (98.0 * h2so4mlAT + 1000.0)

    return h2so4wpAT, h2so4mlAT, a_W
