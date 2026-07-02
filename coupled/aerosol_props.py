# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Aerosol diagnostics for the gas-phase heterogeneous chemistry (Phase 3.2).

Derive the three quantities the gas het chemistry needs from an evolving ``TomasState``:
  * ``surface_area_um2_cm3`` -- total aerosol surface area per cm^3 of air (feeds ``Env.SA``),
  * ``mean_wet_radius_cm``   -- number-weighted wet radius (feeds ``hetgammas_jpl00`` radius),
  * ``h2so4_weight_pct``     -- H2SO4 weight-percent of the aerosol (feeds the gamma composition).

All use the WET particle (equilibrium water at ``SRTH2O``): uptake happens on the deliquesced
particle (AD-3.4). TOMAS ``Mk[:,SRTSO4]`` is H2SO4-equivalent mass (AD-3.8), so the weight-percent
needs no 96/98 conversion. There is no prebuilt TOMAS helper for these; we compose the physics
functions (``calc_equilibrium_water`` + ``calc_particle_properties``) that TOMAS itself uses.
"""

from __future__ import annotations

import math

from . import tomas_bridge as tb   # sets tomas_jax on sys.path; exposes SRTSO4/SRTH2O

import jax.numpy as jnp  # noqa: E402
from tomas_jax.physics.properties import calc_particle_properties  # noqa: E402
from tomas_jax.physics.water_equilibrium import calc_equilibrium_water  # noqa: E402

_SRTSO4 = tb.SRTSO4
_SRTH2O = tb.SRTH2O


def _wet_diameters_m(state):
    """Per-bin WET particle diameter [m] (equilibrium water added at ``SRTH2O``)."""
    Mk_wet = calc_equilibrium_water(state.Mk, state.rh)
    Dpk, _Dk, _ck = calc_particle_properties(state.Nk, Mk_wet, state.temp, state.pres)
    return Dpk, Mk_wet


def surface_area_um2_cm3(state) -> float:
    """Total aerosol surface area [um^2 / cm^3 of air] = sum_k N_k * pi * Dp_wet,k^2 / boxvol.

    ``sum(Nk * pi * Dp^2)`` is [m^2 / grid-cell]; divide by ``boxvol`` [cm^3] and convert
    m^2 -> um^2 (x1e12).  (Returns 0 for an empty distribution.)
    """
    Dpk, _ = _wet_diameters_m(state)
    area_m2 = float(jnp.sum(state.Nk * math.pi * Dpk ** 2))     # m^2 per grid cell
    return area_m2 * 1.0e12 / float(state.boxvol)               # -> um^2 / cm^3


def effective_wet_radius_cm(state) -> float:
    """Surface-area-weighted (effective) WET radius [cm] = sum(Nk r^3) / sum(Nk r^2).

    This is the aerosol *effective radius* r_eff (3rd/2nd moment). It is the right single radius for
    the reacto-diffusive f-factor because heterogeneous uptake is carried by the surface-area-
    dominant (larger) particles -- a number-weighted mean instead collapses toward the ~1 nm
    nucleation mode when nucleation is active, which is both unphysical for uptake AND destabilises
    the f-factor ``coth(r/l) - l/r`` (r << l -> inf - inf). Returns ``0.1e-4`` cm (legacy 1 um) for an
    empty distribution. See AD-3.4.
    """
    Dpk, _ = _wet_diameters_m(state)
    r_m = 0.5 * Dpk
    m2 = float(jnp.sum(state.Nk * r_m ** 2))
    if m2 <= 0.0:
        return 0.1e-4
    m3 = float(jnp.sum(state.Nk * r_m ** 3))
    return (m3 / m2) * 100.0                                    # m -> cm


def h2so4_weight_pct(state) -> float:
    """H2SO4 weight-percent of the aerosol = 100 * M_H2SO4 / (M_H2SO4 + M_H2O).

    ``M_H2SO4`` is ``sum(Mk[:,SRTSO4])`` directly (H2SO4-equiv mass, AD-3.8); ``M_H2O`` is the
    equilibrium water at the state's RH. Returns 0 if there is no sulfate.
    """
    _Dpk, Mk_wet = _wet_diameters_m(state)
    m_so4 = float(jnp.sum(state.Mk[:, _SRTSO4]))
    m_h2o = float(jnp.sum(Mk_wet[:, _SRTH2O]))
    denom = m_so4 + m_h2o
    if denom <= 0.0:
        return 0.0
    return 100.0 * m_so4 / denom


def het_inputs(state) -> dict:
    """Bundle the three het-chem inputs for one outer step: ``{SA, radius_cm, h2so4wp}``."""
    return {"SA": surface_area_um2_cm3(state),
            "radius_cm": effective_wet_radius_cm(state),
            "h2so4wp": h2so4_weight_pct(state)}
