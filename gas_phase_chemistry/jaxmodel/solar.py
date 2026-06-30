"""JAX port of the solar geometry (Phase A ``solar.py``), for SZA-dependent photolysis.

Same Spencer (1971) formulas, written with jax.numpy so the photolysis scaling can depend
on the (traced) integration time inside a Diffrax vector field.
"""

from __future__ import annotations

import math

import jax.numpy as jnp

_DEG = math.pi / 180.0
_COS45 = math.cos(45.0 * _DEG)


def _fractional_year(day_of_year, utc_hour):
    return 2.0 * math.pi / 365.0 * (day_of_year - 1.0 + (utc_hour - 12.0) / 24.0)


def solar_declination(day_of_year, utc_hour=12.0):
    g = _fractional_year(day_of_year, utc_hour)
    return (0.006918
            - 0.399912 * jnp.cos(g) + 0.070257 * jnp.sin(g)
            - 0.006758 * jnp.cos(2 * g) + 0.000907 * jnp.sin(2 * g)
            - 0.002697 * jnp.cos(3 * g) + 0.001480 * jnp.sin(3 * g))


def equation_of_time(day_of_year, utc_hour=12.0):
    g = _fractional_year(day_of_year, utc_hour)
    return 229.18 * (0.000075
                     + 0.001868 * jnp.cos(g) - 0.032077 * jnp.sin(g)
                     - 0.014615 * jnp.cos(2 * g) - 0.040849 * jnp.sin(2 * g))


def cos_solar_zenith(latitude, longitude, day_of_year, utc_hour):
    dec = solar_declination(day_of_year, utc_hour)
    eot = equation_of_time(day_of_year, utc_hour)
    true_solar_time = utc_hour * 60.0 + 4.0 * longitude + eot
    hour_angle = (true_solar_time / 4.0 - 180.0) * _DEG
    latr = latitude * _DEG
    return (jnp.sin(latr) * jnp.sin(dec)
            + jnp.cos(latr) * jnp.cos(dec) * jnp.cos(hour_angle))


def photolysis_scale(cos_sza):
    """Normalized-cosine photolysis factor: max(cos SZA, 0) / cos(45 deg)."""
    return jnp.maximum(cos_sza, 0.0) / _COS45
